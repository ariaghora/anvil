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

@(test)
test_conv2d_1x1 :: proc(t: ^testing.T) {
   // Test 1x1 convolution (pointwise convolution)
   // Input: 2x3x4x4 (batch=2, channels=3, height=4, width=4)
   input_data := make([]f32, 96, context.temp_allocator) // 2*3*4*4
   for i in 0 ..< 96 {
   	input_data[i] = f32(i)
   }
   input := new_with_init(input_data, []uint{2, 3, 4, 4}, context.temp_allocator)
   defer free_tensor(input, context.temp_allocator)

   // 1x1 kernel: 2x3x1x1 (out_channels=2, in_channels=3, h=1, w=1)
   // This is essentially a linear transformation
   kernel_data := []f32{
   	1, 0, 0,  // First output channel weights for 3 input channels
   	0, 1, 0,  // Second output channel weights
   }
   kernel := new_with_init(kernel_data, []uint{2, 3, 1, 1}, context.temp_allocator)
   defer free_tensor(kernel, context.temp_allocator)

   // Perform 1x1 convolution
   result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
   defer free_tensor(result, context.temp_allocator)

   // Output shape should be (2, 2, 4, 4) - spatial dims unchanged
   expected_shape := []uint{2, 2, 4, 4}
   testing.expect(t, slice.equal(result.shape, expected_shape), "1x1 conv output shape incorrect")

   // First output channel should equal first input channel (weights: 1,0,0)
   // Second output channel should equal second input channel (weights: 0,1,0)
   // Check a few values
   ch_size := uint(16) // 4*4
   
   // Batch 0, Output channel 0, position (0,0) should equal input ch0
   testing.expect(t, result.data[0] == input_data[0], "1x1 conv incorrect at batch0,ch0,pos0")
   
   // Batch 0, Output channel 1, position (0,0) should equal input ch1
   testing.expect(t, result.data[ch_size] == input_data[ch_size], "1x1 conv incorrect at batch0,ch1,pos0")
}

@(test)
test_conv2d_3x3 :: proc(t: ^testing.T) {
   // Test 3x3 convolution with edge detection kernel
   // Input: 1x1x5x5 
   input_data := []f32{
   	0, 1, 2, 3, 4,
   	5, 6, 7, 8, 9,
   	10, 11, 12, 13, 14,
   	15, 16, 17, 18, 19,
   	20, 21, 22, 23, 24,
   }
   input := new_with_init(input_data, []uint{1, 1, 5, 5}, context.temp_allocator)
   defer free_tensor(input, context.temp_allocator)

   // 3x3 Sobel-like kernel for edge detection
   kernel_data := []f32{
   	-1, 0, 1,
   	-2, 0, 2,
   	-1, 0, 1,
   }
   kernel := new_with_init(kernel_data, []uint{1, 1, 3, 3}, context.temp_allocator)
   defer free_tensor(kernel, context.temp_allocator)

   // Perform 3x3 convolution
   result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
   defer free_tensor(result, context.temp_allocator)

   // Output shape should be (1, 1, 3, 3) - (5-3+1, 5-3+1)
   expected_shape := []uint{1, 1, 3, 3}
   testing.expect(t, slice.equal(result.shape, expected_shape), "3x3 conv output shape incorrect")

   // Calculate expected value at position (1,1) - center of output
   // Input patch centered at (2,2):
   // 6  7  8
   // 11 12 13  
   // 16 17 18
   // 
   // Convolution: (-1*6 + 0*7 + 1*8) + (-2*11 + 0*12 + 2*13) + (-1*16 + 0*17 + 1*18)
   // = -6 + 8 + -22 + 26 + -16 + 18 = 8
   expected_center := f32(8)
   center_idx := 4 // Position (1,1) in 3x3 output
   testing.expect(
   	t,
   	result.data[center_idx] == expected_center,
   	fmt.tprintf("3x3 conv center value incorrect: got %f, expected %f", 
   		result.data[center_idx], expected_center),
   )
}

@(test)
test_conv2d_3x3_with_padding :: proc(t: ^testing.T) {
   // Test 3x3 convolution with padding=1 (same padding)
   // Input: 1x2x4x4
   input_data := make([]f32, 32, context.temp_allocator) // 1*2*4*4
   for i in 0 ..< 32 {
   	input_data[i] = f32(i + 1)
   }
   input := new_with_init(input_data, []uint{1, 2, 4, 4}, context.temp_allocator)
   defer free_tensor(input, context.temp_allocator)

   // 3x3 kernel: 1x2x3x3 (averaging kernel)
   kernel_data := make([]f32, 18, context.temp_allocator) // 1*2*3*3
   for i in 0 ..< 18 {
   	kernel_data[i] = 1.0 / 18.0 // Average over 2 channels * 9 positions
   }
   kernel := new_with_init(kernel_data, []uint{1, 2, 3, 3}, context.temp_allocator)
   defer free_tensor(kernel, context.temp_allocator)

   // Perform 3x3 convolution with padding=1
   result := conv2d(input, kernel, 1, 1, 1, context.temp_allocator)
   defer free_tensor(result, context.temp_allocator)

   // With padding=1, output shape should match input spatial dims: (1, 1, 4, 4)
   expected_shape := []uint{1, 1, 4, 4}
   testing.expect(t, slice.equal(result.shape, expected_shape), 
   	"3x3 conv with padding output shape incorrect")

   // Check that we got 16 output values
   testing.expect(t, len(result.data) == 16, "3x3 conv with padding output size incorrect")
   
   // Corner values should be smaller (less overlap due to padding)
   // Center values should be larger (full kernel overlap)
   testing.expect(t, result.data[0] < result.data[5], 
   	"Padded corner should have smaller value than center")
}


@(test)
test_conv2d_1x1_expected_fixed :: proc(t: ^testing.T) {
	// Input: 1x2x2x2 (batch=1, channels=2, height=2, width=2)
	// Channel-major layout: [b, c, h, w]
	// ch0: 1 3 5 7   => shape [1, 2, 2, 2]
	// ch1: 2 4 6 8
	input_data := []f32{
		1, 3, 5, 7,  // channel 0
		2, 4, 6, 8,  // channel 1
	}
	input := new_with_init(input_data, []uint{1, 2, 2, 2}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	// Kernel: 1x2x1x1 (1 output channel, 2 input channels)
	kernel_data := []f32{0.5, 1.5} // output = 0.5 * ch0 + 1.5 * ch1
	kernel := new_with_init(kernel_data, []uint{1, 2, 1, 1}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected output:
	// pos0: 0.5*1 + 1.5*2 = 3.5
	// pos1: 0.5*3 + 1.5*4 = 7.5
	// pos2: 0.5*5 + 1.5*6 = 11.5
	// pos3: 0.5*7 + 1.5*8 = 15.5
	expected := []f32{3.5, 7.5, 11.5, 15.5}

	for i in 0..<4 {
		testing.expect(t, result.data[i] == expected[i],
			fmt.tprintf("1x1 conv value mismatch at %d: got %f, expected %f", i, result.data[i], expected[i]))
	}
}

@(test)
test_conv2d_3x3_expected :: proc(t: ^testing.T) {
	// Input: 1x1x3x3
	input_data := []f32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	input := new_with_init(input_data, []uint{1, 1, 3, 3}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	// Kernel: 1x1x3x3 (simple averaging kernel)
	kernel_data := []f32{
		1, 0, -1,
		1, 0, -1,
		1, 0, -1,
	}
	kernel := new_with_init(kernel_data, []uint{1, 1, 3, 3}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	// No padding, stride=1 -> output: (1,1,1,1)
	result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Manual convolution:
	// conv = 1*1 + 0*2 + (-1)*3 + 1*4 + 0*5 + (-1)*6 + 1*7 + 0*8 + (-1)*9
	//      = 1 - 3 + 4 - 6 + 7 - 9 = -6
	expected := f32(-6)
	testing.expect(t, result.data[0] == expected,
		fmt.tprintf("3x3 conv value mismatch: got %f, expected %f", result.data[0], expected))
}

@(test)
test_conv2d_3x3_multichannel :: proc(t: ^testing.T) {
	// Construct 2x5x5 input: channel-major
	ch0 := []f32{
		1, 2, 3, 4, 5,
		6, 7, 8, 9,10,
	   11,12,13,14,15,
	   16,17,18,19,20,
	   21,22,23,24,25,
	}
	ch1 := []f32{
	   26,27,28,29,30,
	   31,32,33,34,35,
	   36,37,38,39,40,
	   41,42,43,44,45,
	   46,47,48,49,50,
	}
	input_data := make([]f32, 50*2, context.temp_allocator)
	for i in 0..<25 {
		input_data[i] = ch0[i]
		input_data[25 + i] = ch1[i]
	}
	input := new_with_init(input_data, []uint{1, 2, 5, 5}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)

	// Kernel: 1x2x3x3 (all ones)
	kernel_data := make([]f32, 18, context.temp_allocator)
	for i in 0..<18 {
		kernel_data[i] = 1.0
	}
	kernel := new_with_init(kernel_data, []uint{1, 2, 3, 3}, context.temp_allocator)
	defer free_tensor(kernel, context.temp_allocator)

	result := conv2d(input, kernel, 1, 1, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected 3x3 result:
	expected := []f32{
		351, 378, 405,
		486, 513, 540,
		621, 648, 675,
	}

	for i in 0..<9 {
		testing.expect(t, result.data[i] == expected[i],
			fmt.tprintf("3x3 multi-channel conv mismatch at %d: got %f, expected %f", i, result.data[i], expected[i]))
	}
}
