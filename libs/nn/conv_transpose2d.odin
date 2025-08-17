package nn

import "../tensor"

get_transpose_hw :: proc(h_in, w_in, h_k, w_k, stride, dilation, padding: uint) -> (uint, uint) {
	// For transposed convolution, output size calculation is different
	h_out := (h_in - 1) * stride - 2 * padding + dilation * (h_k - 1) + 1
	w_out := (w_in - 1) * stride - 2 * padding + dilation * (w_k - 1) + 1
	return h_out, w_out
}

col2im :: proc(
	col: ^tensor.Tensor($T), // (B, H_in * W_in, C_out * K_h * K_w)
	h_out, w_out, c_out: uint,
	h_k, w_k: uint,
	stride, dilation, padding: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	b := col.shape[0]
	hw_in := col.shape[1]
	h_in := (h_out + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1
	w_in := (w_out + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1

	// Allocate output tensor and zero-initialize
	output := tensor.tensor_alloc(T, []uint{b, c_out, h_out, w_out}, true, allocator, loc)

	// Zero out the data (important for accumulation)
	for i in 0 ..< len(output.data) {
		output.data[i] = 0
	}

	col_data := col.data
	if !col.contiguous {
		col_data, _ = tensor.get_strided_data(col, allocator = context.temp_allocator)
	}

	#no_bounds_check {
		for b_idx in 0 ..< b {
			col_b := col_data[b_idx * hw_in * c_out * h_k * w_k:]
			out_b := output.data[b_idx * c_out * h_out * w_out:]

			col_idx := 0

			// Iterate over input spatial positions
			for h_idx in 0 ..< h_in {
				for w_idx in 0 ..< w_in {
					// For each position in input, scatter to output
					for c_idx in 0 ..< c_out {
						for kh_idx in 0 ..< h_k {
							for kw_idx in 0 ..< w_k {
								// Calculate output position
								h_out_idx := h_idx * stride + kh_idx * dilation
								w_out_idx := w_idx * stride + kw_idx * dilation

								// Check padding bounds
								if h_out_idx >= padding &&
								   h_out_idx < h_out + padding &&
								   w_out_idx >= padding &&
								   w_out_idx < w_out + padding {
									h_out_actual := h_out_idx - padding
									w_out_actual := w_out_idx - padding

									out_idx :=
										c_idx * h_out * w_out + h_out_actual * w_out + w_out_actual

									// Accumulate (scatter-add)
									out_b[out_idx] += col_b[col_idx]
								}

								col_idx += 1
							}
						}
					}
				}
			}
		}
	}

	return output
}

// Optimized col2im for common cases
col2im_fast :: proc(
	col: ^tensor.Tensor($T),
	h_out, w_out, c_out: uint,
	h_k, w_k: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Fast path for stride=1, dilation=1, padding=0
	b := col.shape[0]
	hw_in := col.shape[1]
	h_in := h_out - h_k + 1
	w_in := w_out - w_k + 1

	output := tensor.tensor_alloc(T, []uint{b, c_out, h_out, w_out}, true, allocator, loc)

	// Zero initialization
	for i in 0 ..< len(output.data) {
		output.data[i] = 0
	}

	col_data := col.data

	#no_bounds_check {
		for b_idx in 0 ..< b {
			col_b := col_data[b_idx * hw_in * c_out * h_k * w_k:]
			out_b := output.data[b_idx * c_out * h_out * w_out:]

			col_idx := 0

			for h_idx in 0 ..< h_in {
				for w_idx in 0 ..< w_in {
					for c_idx in 0 ..< c_out {
						out_base := c_idx * h_out * w_out + h_idx * w_out + w_idx

						// Unroll for 3x3 kernels
						if h_k == 3 && w_k == 3 {
							#unroll for kh in 0 ..< 3 {
								#unroll for kw in 0 ..< 3 {
									out_b[out_base + uint(kh) * w_out + uint(kw)] += col_b[col_idx]
									col_idx += 1
								}
							}
						} else {
							for kh in 0 ..< h_k {
								for kw in 0 ..< w_k {
									out_b[out_base + kh * w_out + kw] += col_b[col_idx]
									col_idx += 1
								}
							}
						}
					}
				}
			}
		}
	}

	return output
}

conv_transpose_2d :: proc(
	input: ^tensor.Tensor($T), // (B, C_in, H_in, W_in)
	kernel: ^tensor.Tensor(T), // (C_in, C_out, K_h, K_w) - note: different from conv2d!
	stride: uint = 1,
	dilation: uint = 1,
	padding: uint = 0,
	output_padding: uint = 0, // Additional padding to match desired output size
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Extract dimensions
	b, c_in, h_in, w_in := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	_, c_out, k_h, k_w := kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

	// Calculate output dimensions
	h_out, w_out := get_transpose_hw(h_in, w_in, k_h, k_w, stride, dilation, padding)
	h_out += output_padding
	w_out += output_padding

	// Step 1: Reshape input - (B, C_in, H_in, W_in) -> (B, H_in * W_in, C_in)
	input_reshaped := tensor.reshape(input, []uint{b, c_in, h_in * w_in}, context.temp_allocator)
	input_transposed := tensor.transpose(input_reshaped, 1, 2, context.temp_allocator) // (B, H_in * W_in, C_in)

	// Step 2: Reshape kernel - (C_in, C_out, K_h, K_w) -> (C_in, C_out * K_h * K_w)
	kernel_reshaped := tensor.reshape(
		kernel,
		[]uint{c_in, c_out * k_h * k_w},
		context.temp_allocator,
	)

	// Step 3: Matrix multiply - (B, H_in * W_in, C_in) @ (C_in, C_out * K_h * K_w)
	//                        -> (B, H_in * W_in, C_out * K_h * K_w)
	col := tensor.matmul(input_transposed, kernel_reshaped, context.temp_allocator)

	// Step 4: col2im to scatter the result
	result: ^tensor.Tensor(T)
	if stride == 1 && dilation == 1 && padding == 0 {
		result = col2im_fast(col, h_out, w_out, c_out, k_h, k_w, allocator, loc)
	} else {
		result = col2im(
			col,
			h_out,
			w_out,
			c_out,
			k_h,
			k_w,
			stride,
			dilation,
			padding,
			allocator,
			loc,
		)
	}

	return result
}

// Grouped transposed convolution
conv_transpose_2d_grouped :: proc(
	input: ^tensor.Tensor($T), // (B, C_in, H_in, W_in)
	kernel: ^tensor.Tensor(T), // (C_in, C_out, K_h, K_w)
	stride: uint = 1,
	dilation: uint = 1,
	padding: uint = 0,
	output_padding: uint = 0,
	groups: uint = 1,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	if groups == 1 {
		return conv_transpose_2d(
			input,
			kernel,
			stride,
			dilation,
			padding,
			output_padding,
			allocator,
			loc,
		)
	}

	b, c_in, h_in, w_in := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	_, c_out, k_h, k_w := kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

	assert(c_in % groups == 0, "Input channels must be divisible by groups")
	assert(c_out % groups == 0, "Output channels must be divisible by groups")

	c_in_per_group := c_in / groups
	c_out_per_group := c_out / groups

	h_out, w_out := get_transpose_hw(h_in, w_in, k_h, k_w, stride, dilation, padding)
	h_out += output_padding
	w_out += output_padding

	// Allocate output
	output := tensor.tensor_alloc(T, []uint{b, c_out, h_out, w_out}, true, allocator, loc)

	// Zero out
	for i in 0 ..< len(output.data) {
		output.data[i] = 0
	}

	// Process each group
	for g in 0 ..< groups {
		// Extract group from input
		input_group := slice_channel_group(input, g, c_in_per_group, context.temp_allocator)

		// Extract group from kernel  
		kernel_group := slice_kernel_group(
			kernel,
			g,
			c_in_per_group,
			c_out_per_group,
			context.temp_allocator,
		)

		// Perform grouped transposed convolution
		group_result := conv_transpose_2d(
			input_group,
			kernel_group,
			stride,
			dilation,
			padding,
			0, // output_padding handled at parent level
			context.temp_allocator,
		)

		// Copy result to output
		copy_group_to_output(output, group_result, g, c_out_per_group)
	}

	return output
}

// Helper to extract channel group for transposed conv
slice_channel_group :: proc(
	input: ^tensor.Tensor($T),
	group_idx: uint,
	channels_per_group: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	b, c, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	start_c := group_idx * channels_per_group

	result := tensor.tensor_alloc(T, []uint{b, channels_per_group, h, w}, true, allocator)

	hw := h * w
	#no_bounds_check {
		for b_idx in 0 ..< b {
			for c_idx in 0 ..< channels_per_group {
				src_offset := b_idx * c * hw + (start_c + c_idx) * hw
				dst_offset := b_idx * channels_per_group * hw + c_idx * hw

				// Copy spatial data for this channel
				for i in 0 ..< hw {
					result.data[dst_offset + i] = input.data[src_offset + i]
				}
			}
		}
	}

	return result
}

// Helper to extract kernel group for transposed conv
slice_kernel_group :: proc(
	kernel: ^tensor.Tensor($T),
	group_idx: uint,
	in_channels_per_group: uint,
	out_channels_per_group: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_, _, k_h, k_w := kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]
	start_in := group_idx * in_channels_per_group
	start_out := group_idx * out_channels_per_group

	result := tensor.tensor_alloc(
		T,
		[]uint{in_channels_per_group, out_channels_per_group, k_h, k_w},
		true,
		allocator,
	)

	kw_size := k_h * k_w
	#no_bounds_check {
		for in_c in 0 ..< in_channels_per_group {
			for out_c in 0 ..< out_channels_per_group {
				src_offset :=
					(start_in + in_c) * kernel.shape[1] * kw_size + (start_out + out_c) * kw_size
				dst_offset := in_c * out_channels_per_group * kw_size + out_c * kw_size

				// Copy kernel weights
				for i in 0 ..< kw_size {
					result.data[dst_offset + i] = kernel.data[src_offset + i]
				}
			}
		}
	}

	return result
}

// Helper to copy group result to output
copy_group_to_output :: proc(
	output: ^tensor.Tensor($T),
	group_result: ^tensor.Tensor(T),
	group_idx: uint,
	channels_per_group: uint,
) {
	b := output.shape[0]
	h_out, w_out := output.shape[2], output.shape[3]
	start_c := group_idx * channels_per_group

	hw := h_out * w_out
	#no_bounds_check {
		for b_idx in 0 ..< b {
			for c_idx in 0 ..< channels_per_group {
				src_offset := b_idx * channels_per_group * hw + c_idx * hw
				dst_offset := b_idx * output.shape[1] * hw + (start_c + c_idx) * hw

				for i in 0 ..< hw {
					output.data[dst_offset + i] = group_result.data[src_offset + i]
				}
			}
		}
	}
}

import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_conv_transpose_2d_basic :: proc(t: ^testing.T) {
	// Test basic transposed convolution
	// Input: 1x1x2x2 (batch=1, channels=1, height=2, width=2)
	input_data := []f32{1, 2, 3, 4}
	input := tensor.new_with_init(input_data, []uint{1, 1, 2, 2}, context.temp_allocator)

	// Kernel: 1x1x2x2 (in_channels=1, out_channels=1, height=2, width=2)
	kernel_data := []f32{1, 0, 0, 1}
	kernel := tensor.new_with_init(kernel_data, []uint{1, 1, 2, 2}, context.temp_allocator)

	// Perform transposed convolution with stride=1, padding=0
	result := conv_transpose_2d(input, kernel, 1, 1, 0, 0, context.temp_allocator)

	// Expected output shape: (1, 1, 3, 3)
	// Output size = (input - 1) * stride - 2 * padding + kernel = (2-1)*1 - 0 + 2 = 3
	expected_shape := []uint{1, 1, 3, 3}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		"Conv_transpose_2d output shape incorrect",
	)

	// Expected values with kernel [1,0;0,1]:
	// Input [[1,2],[3,4]] expanded with kernel becomes:
	// [[1,0,2,0],
	//  [0,1,0,2],
	//  [3,0,4,0],
	//  [0,3,0,4]]
	// Which sums overlapping regions to:
	// [[1,2,0],
	//  [3,5,2],
	//  [0,3,4]]
	expected_values := []f32{1, 2, 0, 3, 5, 2, 0, 3, 4}

	for i in 0 ..< len(expected_values) {
		testing.expect(
			t,
			result.data[i] == expected_values[i],
			fmt.tprintf(
				"Conv_transpose_2d value at index %d incorrect: got %f, expected %f",
				i,
				result.data[i],
				expected_values[i],
			),
		)
	}
}

@(test)
test_conv_transpose_2d_stride :: proc(t: ^testing.T) {
	// Test transposed convolution with stride=2
	// Input: 1x1x2x2
	input_data := []f32{1, 2, 3, 4}
	input := tensor.new_with_init(input_data, []uint{1, 1, 2, 2}, context.temp_allocator)

	// Kernel: 1x1x3x3 
	kernel_data := []f32{1, 1, 1, 1, 1, 1, 1, 1, 1}
	kernel := tensor.new_with_init(kernel_data, []uint{1, 1, 3, 3}, context.temp_allocator)

	// Perform transposed convolution with stride=2
	result := conv_transpose_2d(input, kernel, 2, 1, 0, 0, context.temp_allocator)

	// Output size = (2-1)*2 - 0 + 3 = 5
	expected_shape := []uint{1, 1, 5, 5}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		fmt.tprintf(
			"Strided conv_transpose shape incorrect: got %v, expected %v",
			result.shape,
			expected_shape,
		),
	)

	// With stride=2, each input pixel spreads its influence over a 3x3 area
	// but pixels are placed at stride=2 intervals
	// Verify corner values at least
	testing.expect(t, result.data[0] == 1, "Top-left corner should be 1")
	testing.expect(t, result.data[4] == 2, "Top-right region should be 2")
	testing.expect(t, result.data[20] == 3, "Bottom-left region should be 3")
	testing.expect(t, result.data[24] == 4, "Bottom-right corner should be 4")
}

@(test)
test_conv_transpose_2d_grouped :: proc(t: ^testing.T) {
	// Test grouped transposed convolution
	// Input: 1x4x2x2 (batch=1, channels=4, height=2, width=2)
	input_data := make([]f32, 16, context.temp_allocator)
	for i in 0 ..< 16 {
		input_data[i] = f32(i + 1)
	}
	input := tensor.new_with_init(input_data, []uint{1, 4, 2, 2}, context.temp_allocator)

	// Kernel: 4x4x2x2 with groups=2
	// This means 2 input channels map to 2 output channels per group
	kernel_data := make([]f32, 64, context.temp_allocator)
	// Initialize with simple pattern
	for i in 0 ..< 64 {
		kernel_data[i] = f32(i % 4 == 0 ? 1 : 0) // Simple diagonal-like pattern
	}
	kernel := tensor.new_with_init(kernel_data, []uint{4, 4, 2, 2}, context.temp_allocator)

	// Perform grouped transposed convolution
	result := conv_transpose_2d_grouped(input, kernel, 1, 1, 0, 0, 2, context.temp_allocator)

	// Output shape should be (1, 4, 3, 3)
	expected_shape := []uint{1, 4, 3, 3}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		fmt.tprintf(
			"Grouped conv_transpose shape incorrect: got %v, expected %v",
			result.shape,
			expected_shape,
		),
	)

	// Verify that result has reasonable values (non-zero, properly computed)
	has_nonzero := false
	for i in 0 ..< len(result.data) {
		if result.data[i] != 0 {
			has_nonzero = true
			break
		}
	}
	testing.expect(t, has_nonzero, "Grouped conv_transpose should produce non-zero output")
}
