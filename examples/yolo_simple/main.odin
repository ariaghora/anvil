/*
This file shows example of how to use YOLOv8 model implemented in Anvil.

Obtain the safetensors model here: https://huggingface.co/lmz/candle-yolo-v8/tree/main
*/
package main

import "../../anvil/models/yolo"
import "../../anvil/plot"
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

// Adjust to your liking
THRESHOLD_NMS :: 0.45
THRESHOLD_CONF :: 0.50
NUM_CLASSES :: 80

// Convenience sturct to encode each bounding box
BBox :: struct {
	xmin, ymin, xmax, ymax: f32,
	conf:                   f32,
}

// Given raw model prediction, extract all bounding boxes with confidence exceeding
// some threshold.
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
					xmin = max(pred[0] - pred[2] / 2., 0),
					ymin = max(pred[1] - pred[3] / 2., 0),
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

// Intersection over union
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

load_image :: proc(
	image_path: string,
	allocator := context.allocator,
) -> (
	^tensor.Tensor(f32),
	uint,
	uint,
	uint,
	uint,
) {
	f := libc.fopen(strings.clone_to_cstring(image_path, context.temp_allocator), "rb")
	ensure(f != nil, "cannot open file")
	defer libc.fclose(f)

	orig_w, orig_h, orig_chan: i32
	image_data := image.loadf_from_file(f, &orig_w, &orig_h, &orig_chan, 0)
	defer image.image_free(image_data)
	ensure(orig_chan == 3, "can only support RGB")

	// Sizes have to be divisible by 32.
	target_w, target_h: uint
	if orig_w < orig_h {
		w := orig_w * 640 / orig_h
		target_w, target_h = uint(w) / 32 * 32, 640
	} else {
		h := orig_h * 640 / orig_w
		target_w, target_h = 640, uint(h) / 32 * 32
	}

	image_data_resized := make([]f32, orig_chan * i32(target_w * target_h), allocator)
	image.resize_float(
		image_data,
		orig_w,
		orig_h,
		0,
		raw_data(image_data_resized),
		i32(target_w),
		i32(target_h),
		0,
		orig_chan,
	)

	input_t := tensor.permute(
		tensor.new_with_init(
			image_data_resized,
			{1, target_h, target_w, uint(orig_chan)},
			allocator,
		),
		{0, 3, 1, 2},
		allocator,
	)
	return input_t, uint(orig_w), uint(orig_h), target_w, target_h
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
		len(os.args) == 4,
		"Program requires two positional arguments: model path, image path, and model scale (n, s, m, l)",
	)
	model_path, image_path, model_scale := os.args[1], os.args[2], os.args[3]


	// Configuration for different model scales
	// ref: https://github.com/huggingface/candle/blob/402782c6944993e20953839ed2fb41aab6c66cc2/candle-examples/examples/yolo-v8/model.rs#L12-L46
	multiples: yolo.Multiples
	switch model_scale {
	case "n":
		// for yolov8n.safetensors (nano)
		multiples = yolo.Multiples {
			depth = 0.33,
			width = 0.25,
			ratio = 2.0,
		}
	case "s":
		// for yolov8s.safetensors (small)
		multiples = yolo.Multiples {
			depth = 0.33,
			width = 0.50,
			ratio = 2.0,
		}
	case "m":
		// for yolov8m.safetensors (medium)
		multiples = yolo.Multiples {
			depth = 0.67,
			width = 0.75,
			ratio = 1.5,
		}
	case "l":
		// for yolov8l.safetensors (large)
		multiples = yolo.Multiples {
			depth = 1.0,
			width = 1.0,
			ratio = 1.0,
		}
	case "x":
		// for yolov8x.safetensors (extra large)
		multiples = yolo.Multiples {
			depth = 1.0,
			width = 1.25,
			ratio = 1.0,
		}
	case:
		fmt.panicf("supported scale are n, s, m, l, and x, but found %s", model_scale)
	}

	// Load safetensors file
	model_safetensors, err_st := st.read_from_file(f32, model_path)
	ensure(err_st == nil)
	defer st.free_safe_tensors(model_safetensors)
	// Then apply it to the model
	model := yolo.load_yolo(model_safetensors, multiples, NUM_CLASSES, context.allocator)
	defer yolo.free_yolo(model)

	// Load the input and the original sizes and resized sizes
	input_t, orig_w, orig_h, resize_w, resize_h := load_image(image_path, context.temp_allocator)

	// Do inference!
	t := time.now()

	pred, _, _ := yolo.forward_yolo(model, input_t, context.allocator)
	// Postprocess the raw detection.
	// We need to filter out the detections with low confidence. Subsequently,
	// eliminate detection duplicates by using NMS
	pred = tensor.squeeze(pred, context.temp_allocator)
	bboxes := extract_bboxes(pred, THRESHOLD_CONF, context.temp_allocator)
	non_maximum_suppression(bboxes, THRESHOLD_NMS)

	fmt.println("inference time:", time.since(t))

	// Calculate the scale to transform coordinates back to the original size space
	x_scale := f32(orig_w) / f32(resize_w)
	y_scale := f32(orig_h) / f32(resize_h)

	// Print all detections
	class_names := YOLO_Classes_80
	for bbox_per_class, i in bboxes {
		class_name := class_names[i]
		for bbox in bbox_per_class {
			fmt.printfln(
				"class_name:%s, xmin:%f, ymin:%f, xmax:%f, ymax:%f, conf:%f",
				class_name,
				bbox.xmin * x_scale,
				bbox.ymin * y_scale,
				bbox.xmax * x_scale,
				bbox.ymax * y_scale,
				bbox.conf,
			)
		}
	}
}
