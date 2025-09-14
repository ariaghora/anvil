package main

import "../../anvil/onnx"
import "../../anvil/tensor"
import "core:c/libc"
import "core:fmt"
import "core:mem"
import "core:os/os2"
import "core:slice"
import "core:strings"
import "core:time"
import "vendor:stb/image"

T :: f32

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
	// https://huggingface.co/onnx-community/resnet-50-ONNX/resolve/main/onnx/model.onnx?download=true
	model, err := onnx.read_from_file(T, "weights/model.onnx")
	defer onnx.free_onnx(model)

	ensure(err == nil, fmt.tprint(err))
	fmt.println("Producer Name    : ", model.producer_name)
	fmt.println("Producer Version : ", model.producer_version)
	fmt.println("Opset Version    : ", model.opset_version)

	f := libc.fopen(strings.clone_to_cstring(image_file_path, context.temp_allocator), "rb")
	ensure(f != nil, "cannot open image file")
	defer libc.fclose(f)

	orig_w, orig_h, orig_chan: i32
	image_data := image.loadf_from_file(f, &orig_w, &orig_h, &orig_chan, 0)
	defer image.image_free(image_data)
	ensure(orig_chan == 3, "can only support RGB")
	// Resize to 224x224, i.e., the standard imagenet dataset sizes, via stb image resize
	image_data_resized := make([]T, orig_chan * 224 * 224, context.temp_allocator)
	image.resize_float(
		image_data,
		orig_w,
		orig_h,
		0,
		raw_data(image_data_resized),
		224,
		224,
		0,
		orig_chan,
	)

	// The means and stdevs for normalization
	// https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
	means := tensor.new_with_init([]T{0.485, 0.456, 0.406}, {3}, context.temp_allocator)
	std := tensor.new_with_init([]T{0.229, 0.224, 0.225}, {3}, context.temp_allocator)
	// Image raw data to tensor and standardize
	input_t := tensor.new_with_init(
		image_data_resized,
		{1, 224, 224, uint(orig_chan)},
		context.temp_allocator,
	)
	input_t = tensor.div(
		tensor.sub(input_t, means, context.temp_allocator),
		std,
		context.temp_allocator,
	)
	// NHWC -> NCHW
	input_t = tensor.permute(input_t, {0, 3, 1, 2})
	defer tensor.free_tensor(input_t)

	// Set input for inference
	inputs := make(map[string]^tensor.Tensor(T), context.temp_allocator)
	// Input name from the downloaded ONNX is `pixel_values`. You can confirm using tool like netron.
	inputs["pixel_values"] = input_t

	// Launch!
	st := time.now()
	err_run := onnx.run(model, inputs)
	ensure(err_run == nil, fmt.tprint(err_run))
	fmt.println("Inference time   : ", time.since(st))


	// Output name from the downloaded ONNX is `logits`
	output, err_fetch := onnx.fetch_tensor(model, "logits")
	ensure(err_fetch == nil, fmt.tprint(err_fetch))

	// Simple Odin's builtin argmax is fine since we assume prediction only on 1 image
	argmax, ok := slice.max_index(output.data)
	ensure(ok)

	class_names := IMAGENET_CLASSES
	fmt.println("Predicted class  : ", class_names[argmax])

}
