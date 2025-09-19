package main

import "../../anvil/imageops"
import "../../anvil/io"
import "../../anvil/onnx"
import "../../anvil/tensor"
import "../../anvil/trace"

import "core:c/libc"
import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os/os2"
import "core:slice"
import "core:strings"
import "core:time"

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

	trace.init_trace()
	defer trace.finish_trace()

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

	img, err_im := io.read_image_from_file(T, image_file_path)
	ensure(err_im == nil, fmt.tprint(err_im))
	img_resized := imageops.resize(img, 224, 224, .Bilinear)
	defer tensor.free_tensor(img, img_resized)

	// The means and stdevs for normalization
	// https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
	means := tensor.new_with_init([]T{0.485, 0.456, 0.406}, {3}, context.temp_allocator)
	std := tensor.new_with_init([]T{0.229, 0.224, 0.225}, {3}, context.temp_allocator)
	img_norm := tensor.div(tensor.sub(img_resized, means, context.temp_allocator), std)
	defer tensor.free_tensor(img_norm)

	// HWC -> NHW, then unsqueeze to make a singleton batch
	input_t := tensor.unsqueeze(tensor.permute(img_resized, {2, 0, 1}, context.temp_allocator), 0)
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
