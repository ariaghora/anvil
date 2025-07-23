package tensor

import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_im2col :: proc(t: ^testing.T) {
	x := ones(f32, []uint{10, 3, 5, 5}, context.temp_allocator)
	_ = im2col(x, 3, 3, 1, 1, 0, context.temp_allocator)
}

@(test)
test_conv2d :: proc(t: ^testing.T) {
	// Simple 2D convolution test with known values
	// Input: 1x1x3x3 tensor (batch=1, channels=1, height=3, width=3)
	input_data := []f32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	input := new_with_init(input_data, []uint{1, 1, 3, 3}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	// Kernel: 1x1x2x2 (out_channels=1, in_channels=1, height=2, width=2)
	kernel_data := []f32{1, 0, 0, 1}
	kernel := new_with_init(kernel_data, []uint{1, 1, 2, 2}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	// Perform convolution with stride=1, dilation=1, padding=0
	result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected output shape: (1, 1, 2, 2) - (3-2+1, 3-2+1) = (2, 2)
	expected_shape := []uint{1, 1, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Conv2d output shape incorrect")

	// Expected values: kernel [1,0;0,1] applied to input
	// Top-left: 1*1 + 2*0 + 4*0 + 5*1 = 6
	// Top-right: 2*1 + 3*0 + 5*0 + 6*1 = 8
	// Bottom-left: 4*1 + 5*0 + 7*0 + 8*1 = 12
	// Bottom-right: 5*1 + 6*0 + 8*0 + 9*1 = 14
	expected_values := []f32{6, 8, 12, 14}

	for i in 0 ..< len(expected_values) {
		testing.expect(
			t,
			result.data[i] == expected_values[i],
			fmt.tprintf(
				"Conv2d value at index %d incorrect: got %f, expected %f",
				i,
				result.data[i],
				expected_values[i],
			),
		)
	}
}
@(test)
test_conv2d_grouped :: proc(t: ^testing.T) {
	// Test grouped convolution with 2 groups
	// Input: 1x4x3x3 (batch=1, channels=4, height=3, width=3)
	input_data := make([]f32, 36, context.temp_allocator) // 1*4*3*3 = 36
	for i in 0 ..< 36 {
		input_data[i] = f32(i + 1)
	}
	input := new_with_init(input_data, []uint{1, 4, 3, 3}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	// Kernel: 4x2x2x2 (out_channels=4, in_channels_per_group=2, height=2, width=2)
	// For 2 groups: each group has 2 input channels and 2 output channels
	kernel_data := make([]f32, 32, context.temp_allocator) // 4*2*2*2 = 32
	for i in 0 ..< 32 {
		kernel_data[i] = 1.0 // Simple kernel for easier testing
	}
	kernel := new_with_init(kernel_data, []uint{4, 2, 2, 2}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	// Perform grouped convolution with 2 groups
	result := conv2d_grouped(input, kernel, 2, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected output shape: (1, 4, 2, 2) - same as regular conv2d but processed in groups
	expected_shape := []uint{1, 4, 2, 2}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		"Grouped conv2d output shape incorrect",
	)

	// The exact values depend on the kernel and input, but we can check that the operation completed
	testing.expect(t, len(result.data) == 16, "Grouped conv2d output size incorrect")
}

@(test)
test_conv2d_grouped_vs_single :: proc(t: ^testing.T) {
	// Test that grouped convolution with groups=1 gives same result as single convolution
	input_data := []f32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	input := new_with_init(input_data, []uint{1, 1, 3, 3}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	kernel_data := []f32{1, 0, 0, 1}
	kernel := new_with_init(kernel_data, []uint{1, 1, 2, 2}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	// Single convolution
	result_single := conv2d_single(input, kernel, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result_single, context.temp_allocator)

	// Grouped convolution with groups=1
	result_grouped := conv2d_grouped(input, kernel, 1, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result_grouped, context.temp_allocator)

	// Results should be identical
	testing.expect(
		t,
		slice.equal(result_single.shape, result_grouped.shape),
		"Single vs grouped shape mismatch",
	)
	for i in 0 ..< len(result_single.data) {
		testing.expect(
			t,
			result_single.data[i] == result_grouped.data[i],
			fmt.tprintf(
				"Single vs grouped data mismatch at %d: %f != %f",
				i,
				result_single.data[i],
				result_grouped.data[i],
			),
		)
	}
}
