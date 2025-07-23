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
	input_data := []f32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	input := new_with_init(input_data, []uint{1, 1, 3, 3}, context.temp_allocator)
	defer free_tensor(input, context.temp_allocator)
	
	// Kernel: 1x1x2x2 (out_channels=1, in_channels=1, height=2, width=2)
	kernel_data := []f32{
		1, 0,
		0, 1,
	}
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
	
	for i in 0..<len(expected_values) {
		testing.expect(t, result.data[i] == expected_values[i], 
			fmt.tprintf("Conv2d value at index %d incorrect: got %f, expected %f", 
			i, result.data[i], expected_values[i]))
	}
}
