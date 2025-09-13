package main

import "../../anvil/onnx"
import "../../anvil/tensor"
import "core:fmt"

main :: proc() {
	// Get the ONNX file from
	// https://huggingface.co/onnx-community/resnet-50-ONNX/resolve/main/onnx/model.onnx?download=true
	model, err := onnx.read_from_file(f32, "weights/model.onnx", context.temp_allocator)
	ensure(err == nil, fmt.tprint(err))
	fmt.println("Producer Name    : ", model.producer_name)
	fmt.println("Producer Version : ", model.producer_version)
	fmt.println("Opset Version    : ", model.opset_version)

	T :: f32
	inputs := make(map[string]^tensor.Tensor(T), context.temp_allocator)
	inputs["pixel_values"] = tensor.ones(T, {1, 3, 512, 512}, context.temp_allocator)
	err_run := onnx.run(model, inputs)
	ensure(err_run == nil, fmt.tprint(err_run))
}
