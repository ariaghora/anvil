package main

import "core:fmt"
import "core:math/rand"
import "libs/tensor"
import tf "libs/transformer"

main :: proc() {
	fmt.printf("=== TinyViT-5M Integration Test ===\n")

	// Create model
	fmt.printf("Creating TinyViT-5M model...\n")
	model := tf.new_tiny_vit_5m(f32, 256, context.temp_allocator)

	// Create input: (1, 3, 256, 256)
	fmt.printf("Creating random input (1, 3, 256, 256)...\n")
	input_size := 1 * 3 * 256 * 256
	input_data := make([]f32, input_size, context.temp_allocator)

	// Initialize with small random values to avoid numerical issues
	for i in 0 ..< len(input_data) {
		input_data[i] = (rand.float32() - 0.5) * 0.1 // Small values [-0.05, 0.05]
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, 256, 256}, context.temp_allocator)

	fmt.printf(
		"Input created successfully: shape [%d, %d, %d, %d]\n",
		input.shape[0],
		input.shape[1],
		input.shape[2],
		input.shape[3],
	)

	// Forward pass using the function with all our debug timing
	fmt.printf("Starting TinyViT-5M forward pass...\n")
	output := tf.forward_tiny_vit_5m(model, input, context.temp_allocator)

	// Check output
	fmt.printf("Forward pass completed!\n")

	tensor.print(output)

}
