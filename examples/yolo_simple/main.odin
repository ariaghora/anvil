package main

import "../../anvil/models/yolo"
import st "../../anvil/safetensors/"
import "../../anvil/tensor"
import "../../anvil/trace"
import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import "core:time"
import "vendor:stb/image"

import "../../anvil/plot"

THRESHOLD_NMS :: 0.45
THRESHOLD_CONF :: 0.50
NUM_CLASSES :: 80
MODEL_PATH :: "weights/yolo_tensors.safetensors"

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

	safetensors_ref, err_st_ref := st.read_from_file(f32, MODEL_PATH)
	ensure(err_st_ref == nil)
	defer st.free_safe_tensors(safetensors_ref)

	f := libc.fopen(strings.clone_to_cstring(image_path, context.temp_allocator), "rb")
	ensure(f != nil, "cannot open file")
	defer libc.fclose(f)


	ori_w, ori_h, ori_chan: i32
	image_data := image.loadf_from_file(f, &ori_w, &ori_h, &ori_chan, 0)
	defer image.image_free(image_data)
	ensure(ori_chan == 3, "can only support RGB")

	// Sizes have to be divisible by 32.
	target_w, target_h: uint
	if ori_w < ori_h {
		w := ori_w * 640 / ori_h
		target_w, target_h = uint(w) / 32 * 32, 640
	} else {
		h := ori_h * 640 / ori_w
		target_w, target_h = 640, uint(h) / 32 * 32
	}

	image_data_resized := make([]f32, ori_chan * i32(target_w * target_h), context.temp_allocator)
	image.resize_float(
		image_data,
		ori_w,
		ori_h,
		0,
		raw_data(image_data_resized),
		i32(target_w),
		i32(target_h),
		0,
		ori_chan,
	)

	input_t := tensor.permute(
		tensor.new_with_init(
			image_data_resized,
			{1, target_h, target_w, uint(ori_chan)},
			context.temp_allocator,
		),
		{0, 3, 1, 2},
		context.temp_allocator,
	)

	safetensors, err_st := st.read_from_file(f32, model_path)
	ensure(err_st == nil)
	defer st.free_safe_tensors(safetensors)

	// Configuration for nano model
	multiples := yolo.Multiples {
		depth = 0.33,
		width = 0.25,
		ratio = 2.0,
	}
	num_classes := uint(NUM_CLASSES)
	model := yolo.load_yolo(safetensors, multiples, num_classes, context.allocator)
	defer yolo.free_yolo(model)

	t := time.now()
	pred, _, _ := yolo.forward_yolo(model, input_t, context.allocator)
	fmt.println("inference time:", time.since(t))

	// Postprocess the raw detection.
	// We need to filter out the detections with low confidence. Subsequently,
	// eliminate detection duplicates by using NMS
	pred = tensor.squeeze(pred, context.temp_allocator)
	bboxes := extract_bboxes(pred, THRESHOLD_CONF, context.temp_allocator)
	non_maximum_suppression(bboxes, THRESHOLD_NMS)

	// Print all detections
	class_names := YOLO_Classes_80
	for bbox_per_class, i in bboxes {
		class_name := class_names[i]
		for bbox in bbox_per_class {
			fmt.printfln(
				"class_name:%s , xmin:%f, ymin:%f, xmax:%f, ymax:%f, conf:%f",
				class_name,
				bbox.xmin,
				bbox.ymin,
				bbox.xmax,
				bbox.ymax,
				bbox.conf,
			)
		}

	}
}
