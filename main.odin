package main

import "core:fmt"
import "core:math/rand"
import "libs/tensor"
import tf "libs/transformer"

main :: proc() {
	imgsz := uint(1024)
	model := tf.new_tiny_vit_5m(f32, imgsz, context.temp_allocator)

	input_size := 1 * 3 * imgsz * imgsz
	input_data := make([]f32, input_size, context.temp_allocator)

	for i in 0 ..< len(input_data) {
		input_data[i] = (rand.float32() - 0.5) * 0.1 // Small values [-0.05, 0.05]
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, imgsz, imgsz}, context.temp_allocator)

	fmt.printf("Starting TinyViT-5M forward pass...\n")
	output := tf.forward_tiny_vit_5m(model, input, context.temp_allocator)

	fmt.printf("Forward pass completed!\n")

	// tensor.print(output)

}
