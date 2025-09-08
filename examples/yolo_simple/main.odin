package main

import "../../anvil/models/yolo"
import st "../../anvil/safetensors/"
import "../../anvil/tensor"
import "../../anvil/trace"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:time"

import "../../anvil/plot"

BBox :: struct {
	xmin, ymin, xmax, ymax: f32,
	conf:                   f32,
}

extract_bboxes :: proc(
	pred: ^tensor.Tensor($T),
	conf_thresh: f32,
	allocator := context.temp_allocator,
) -> [][dynamic]BBox {
	ensure(len(pred.shape) == 2, "`pred` must be 2d, representing a single image")
	pred_size, n_preds := pred.shape[0], pred.shape[1]
	n_classes := pred_size - 4

	bboxes := make([][dynamic]BBox, n_classes, allocator)
	for _, i in bboxes do bboxes[i] = make([dynamic]BBox, allocator)

	for index in 0 ..< n_preds {
		pred :=
			tensor.slice(pred, {{}, {int(index), int(index + 1), 1}}, context.temp_allocator).data
		confidence := slice.max(pred[4:])
		if confidence > conf_thresh {
			class_idx: uint = 0
			for i in 0 ..< n_classes {
				if pred[4 + i] > pred[4 + class_idx] do class_idx = i
			}
			if pred[4 + class_idx] > 0. {
				bbox := BBox {
					xmin = pred[0] - pred[2] / 2.,
					ymin = pred[1] - pred[3] / 2.,
					xmax = pred[0] + pred[2] / 2.,
					ymax = pred[1] + pred[3] / 2.,
					conf = confidence,
				}
				append(&bboxes[class_idx], bbox)
			}
		}
	}

	return bboxes
}

iou :: proc(b1, b2: BBox) -> f32 {
	b1_area := (b1.xmax - b1.xmin + 1) * (b1.ymax - b1.ymin + 1)
	b2_area := (b2.xmax - b2.xmin + 1) * (b2.ymax - b2.ymin + 1)
	i_xmin := max(b1.xmin, b2.xmin)
	i_xmax := min(b1.xmax, b2.xmax)
	i_ymin := max(b1.ymin, b2.ymin)
	i_ymax := min(b1.ymax, b2.ymax)
	i_area := max((i_xmax - i_xmin + 1), 0) * max(i_ymax - i_ymin + 1, 0)
	return i_area / (b1_area + b2_area - i_area)
}

non_maximum_suppression :: proc(bboxes: [][dynamic]BBox, threshold: f32) {
	for &bbox_per_class, bbox_idx in bboxes {
		slice.sort_by(bbox_per_class[:], proc(b1, b2: BBox) -> bool {return b1.conf > b2.conf})
		current_index := 0
		for index in 0 ..< len(bbox_per_class) {
			drop := false
			for prev_index in 0 ..< current_index {
				iou := iou(bbox_per_class[prev_index], bbox_per_class[index])
				if iou > threshold {
					drop = true
					break
				}
			}
			if !drop {
				slice.swap(bbox_per_class[:], current_index, index)
				current_index += 1
			}
		}
		resize_dynamic_array(&bbox_per_class, current_index)
	}
}

main :: proc() {
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	ensure(
		len(os.args) == 3,
		"Program requires two positional arguments: model path and image path",
	)
	model_path, image_path := os.args[1], os.args[2]

	trace.init_trace()
	defer trace.finish_trace()

	safetensors_ref, err_st_ref := st.read_from_file(f32, "weights/yolo_tensors.safetensors")
	ensure(err_st_ref == nil)
	defer st.free_safe_tensors(safetensors_ref)

	input_t := safetensors_ref.tensors["original_input"]

	safetensors, err_st := st.read_from_file(f32, model_path)
	ensure(err_st == nil)
	defer st.free_safe_tensors(safetensors)

	// NANO
	multiples := yolo.Multiples {
		depth = 0.33,
		width = 0.25,
		ratio = 2.0,
	}
	num_classes := uint(80)
	model := yolo.load_yolo(safetensors, multiples, num_classes, context.allocator)
	defer yolo.free_yolo(model)

	t := time.now()
	pred, _, _ := yolo.forward_yolo(model, input_t, context.allocator)
	fmt.println("inference time:", time.since(t))

	// postproc
	pred = tensor.squeeze(pred, context.temp_allocator)
	bboxes := extract_bboxes(pred, 0.25, context.temp_allocator)
	non_maximum_suppression(bboxes, 0.45)

	class_names := YOLO_Classes_80
	for bbox_per_class, i in bboxes {
		class_name := class_names[i]
		for bbox in bbox_per_class {
			fmt.printfln(
				"class_name:%s , xmin:%f, ymin:%f, xmax:%f, ymax:%f",
				class_name,
				bbox.xmin,
				bbox.ymin,
				bbox.xmax,
				bbox.ymax,
			)
		}

	}
}
