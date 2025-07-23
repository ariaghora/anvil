package main

import "core:fmt"
import "core:math/rand"
import "libs/tensor"
import tf "libs/transformer"

main :: proc() {
	fmt.printf("=== TinyViT-5M Integration Test ===\n")

	// Create model
	fmt.printf("Creating TinyViT-5M model...\n")
	model := tf.new_tiny_vit_5m(f32, context.temp_allocator)

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
	fmt.printf(
		"Output shape: [%d, %d, %d, %d]\n",
		output.shape[0],
		output.shape[1],
		output.shape[2],
		output.shape[3],
	)

	// Check that output contains reasonable values (not NaN/Inf)
	has_valid_values := true
	sample_count := min(100, len(output.data))
	for i in 0 ..< sample_count {
		val := output.data[i]
		// Check for NaN (val != val) or if value is too extreme
		if val != val || val > 1e10 || val < -1e10 {
			has_valid_values = false
			break
		}
	}


	// Print some statistics
	if len(output.data) > 0 {
		min_val := output.data[0]
		max_val := output.data[0]
		sum_val: f32 = 0

		for val in output.data {
			if val < min_val do min_val = val
			if val > max_val do max_val = val
			sum_val += val
		}
		mean_val := sum_val / f32(len(output.data))

		fmt.printf("Output stats: min=%.6f, max=%.6f, mean=%.6f\n", min_val, max_val, mean_val)
		fmt.printf("Total elements: %d input -> %d output\n", len(input.data), len(output.data))
	}

	fmt.printf("✅ TinyViT-5M Integration Test COMPLETED!\n")
}
