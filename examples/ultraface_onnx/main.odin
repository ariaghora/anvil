package main

import "../../anvil/imageops"
import "../../anvil/io"
import "../../anvil/onnx"
import "../../anvil/tensor"
import "core:fmt"
import "core:mem"
import "core:os/os2"
import "core:slice"
import "core:time"

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

non_maximum_suppression :: proc(bboxes: [dynamic]BBox, threshold: f32) -> [dynamic]BBox {
	bboxes := bboxes
	slice.sort_by(bboxes[:], proc(b1, b2: BBox) -> bool {return b1.conf > b2.conf})
	current_index := 0
	for index in 0 ..< len(bboxes) {
		drop := false
		for prev_index in 0 ..< current_index {
			iou := iou(bboxes[prev_index], bboxes[index])
			if iou > threshold {
				drop = true
				break
			}
		}
		if !drop {
			slice.swap(bboxes[:], current_index, index)
			current_index += 1
		}
	}
	resize_dynamic_array(&bboxes, current_index)
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

	// Get the ONNX file from
	// https://github.com/onnx/models/tree/main/validated/vision/body_analysis/ultraface, version-RFB-640
	model, err := onnx.read_from_file(T, "weights/ultraface.onnx")
	defer onnx.free_onnx(model)

	// HWC
	img, err_im := io.read_image_from_file(T, image_file_path)
	ensure(err_im == nil, fmt.tprint(err_im))
	ori_height, ori_width := img.shape[0], img.shape[1]

	img_resized := imageops.resize(img, 480, 640, .Bilinear)
	defer tensor.free_tensor(img, img_resized)
	for _, i in img_resized.data do img_resized.data[i] = ((img_resized.data[i] * 255) - 127) / 128

	input_t := tensor.unsqueeze(
		tensor.permute(img_resized, {2, 0, 1}, context.temp_allocator),
		0,
		context.temp_allocator,
	)
	inputs := make(map[string]^tensor.Tensor(T), context.temp_allocator)
	inputs["input"] = input_t

	st := time.now()
	err_run := onnx.run(model, inputs)
	ensure(err_run == nil, fmt.tprint(err_run))
	fmt.println("Inference time   : ", time.since(st))

	scores, _ := onnx.fetch_tensor(model, "scores")
	boxes, _ := onnx.fetch_tensor(model, "boxes")

	// Filter and scale. Reference:
	// https://github.com/onnx/models/blob/fcf08a81b6e1286d18ea45d178c5f97fb1476449/validated/vision/body_analysis/ultraface/demo.py#L21-L29
	bboxes := make([dynamic]BBox, context.temp_allocator)
	for i in 0 ..< scores.shape[1] {
		score := tensor.tensor_get(scores, 0, i, 1)
		if score > 0.80 {
			s := tensor.slice(boxes, {{}, int(i), {}}, true)
			defer tensor.free_tensor(s)
			xmin, ymin :=
				clamp(s.data[0], 0, 1) * f32(ori_width), clamp(s.data[1], 0, 1) * f32(ori_height)
			xmax, ymax :=
				clamp(s.data[2], 0, 1) * f32(ori_width), clamp(s.data[3], 0, 1) * f32(ori_height)
			w, h := xmax - xmin, ymax - ymin
			maximum := max(w, h)
			dx := ((maximum - w) / 2)
			dy := ((maximum - h) / 2)

			bbox := BBox {
				xmin = (xmin - dx),
				ymin = (ymin - dy),
				xmax = (xmax + dx),
				ymax = (ymax + dy),
				conf = score,
			}
			append(&bboxes, bbox)
		}
	}
	final_bboxes := non_maximum_suppression(bboxes, 0.45)

	// Draw red rectangles, in place
	for bbox in final_bboxes {
		imageops.draw_rectangle_line(
			img,
			uint(bbox.xmin),
			uint(bbox.ymin),
			uint(bbox.xmax - bbox.xmin),
			uint(bbox.ymax - bbox.ymin),
			{1, 0, 0},
			3,
		)
	}
	io.write_image(img, "test.png")
}
