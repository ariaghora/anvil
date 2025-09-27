package main

import "../../anvil/imageops"
import "../../anvil/io"
import yl "../../anvil/models/yolo"
import "../../anvil/onnx"
import "../../anvil/tensor"
import "core:fmt"
import "core:image/qoi"
import "core:math"
import "core:mem"
import "core:os/os2"
import "core:slice"
import "core:time"

THRESHOLD_NMS :: 0.45
THRESHOLD_CONF :: 0.50
NUM_CLASSES :: 80
T :: f32

BBox :: struct {
	xmin, ymin, xmax, ymax: f32,
	conf:                   f32,
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
		pred := tensor.slice(pred, {{}, int(index)}, allocator = context.temp_allocator).data
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

	ensure(len(os2.args) == 2, "first positional argument must be an image path")
	image_file_path := os2.args[1]

	// Get the ONNX file from this conversion script
	// https://colab.research.google.com/drive/1-yZg6hFg27uCPSycRCRtyezHhq_VAHxQ?usp=sharing#scrollTo=mq2SMmJ1ogLA
	model, err := onnx.read_from_file(T, "weights/yolov8m.onnx")
	defer onnx.free_onnx(model)

	// HWC
	img, err_im := io.read_image_from_file(T, image_file_path)
	ensure(err_im == nil, fmt.tprint(err_im))
	ori_height, ori_width := img.shape[0], img.shape[1]


	img_resized := imageops.resize(img, 480, 640, .Bilinear)
	defer tensor.free_tensor(img, img_resized)
	resize_w, resize_h := img_resized.shape[1], img_resized.shape[2]

	input_t := tensor.unsqueeze(
		tensor.permute(img_resized, {2, 0, 1}, context.temp_allocator),
		0,
		context.temp_allocator,
	)
	inputs := make(map[string]^tensor.Tensor(T), context.temp_allocator)
	inputs["images"] = input_t

	st := time.now()
	err_run := onnx.run(model, inputs)
	ensure(err_run == nil, fmt.tprint(err_run))

	out, _ := onnx.fetch_tensor(model, "output0")
	pred_single := tensor.squeeze(out, context.temp_allocator)
	bboxes := extract_bboxes(pred_single, THRESHOLD_CONF, context.temp_allocator)
	non_maximum_suppression(bboxes, THRESHOLD_NMS)
	fmt.println("Inference time   : ", time.since(st))


	x_scale := f32(ori_width) / f32(resize_w)
	y_scale := f32(ori_height) / f32(resize_h)
	class_names := yl.YOLO_Classes_80
	for bbox_per_class, i in bboxes {
		class_name := class_names[i]
		for bbox in bbox_per_class {
			fmt.printfln(
				"class_name:%s, xmin:%f, ymin:%f, xmax:%f, ymax:%f, conf:%f",
				class_name,
				bbox.xmin,
				bbox.ymin,
				bbox.xmax,
				bbox.ymax,
				bbox.conf,
			)

			x := uint(bbox.xmin)
			y := uint(bbox.ymin)
			w := uint((bbox.xmax - bbox.xmin))
			h := uint((bbox.ymax - bbox.ymin))
			imageops.draw_rectangle_line(img_resized, x, y, w, h, {1, 0, 0}, 3)
		}
	}
	io.write_image(img_resized, "detection.png")
}
