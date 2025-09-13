package tensor

import "../trace"
import "core:math"
import "core:simd"

// Calculate output dimensions for pooling
get_pool_hw :: proc(h_in, w_in, k_h, k_w, stride, padding: uint) -> (uint, uint) {
	h_out := (h_in + 2 * padding - k_h) / stride + 1
	w_out := (w_in + 2 * padding - k_w) / stride + 1
	return h_out, w_out
}

max_pool_2d :: proc(
	input: ^Tensor($T), // (B, C, H, W)
	kernel_size: [2]uint, // (K_h, K_w)
	stride: uint = 1,
	padding: uint = 0, // Added with default 0 to maintain backward compatibility
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	pool_trace := trace.TRACE_FUNCTION("max_pool_2d")
	defer trace.end_scoped_trace(pool_trace)

	// Extract dimensions
	b, c, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	k_h, k_w := kernel_size[0], kernel_size[1]

	// Calculate output dimensions
	h_out, w_out := get_pool_hw(h, w, k_h, k_w, stride, padding)

	// Allocate output tensor
	output := tensor_alloc(T, []uint{b, c, h_out, w_out}, true, allocator, loc)

	// Get input data (handle non-contiguous case)
	src := input.data
	allocated := false
	if !input.contiguous {
		src, allocated = get_strided_data(input, allocator = context.temp_allocator)
	}
	defer if allocated do delete(src, context.temp_allocator)

	// Perform max pooling
	#no_bounds_check {
		when T == f32 {
			max_pool_2d_f32_simd(
				src,
				output.data,
				b,
				c,
				h,
				w,
				k_h,
				k_w,
				h_out,
				w_out,
				stride,
				padding,
			)
		} else {
			max_pool_2d_scalar(
				src,
				output.data,
				b,
				c,
				h,
				w,
				k_h,
				k_w,
				h_out,
				w_out,
				stride,
				padding,
			)
		}
	}

	return output
}

@(private)
max_pool_2d_f32_simd :: proc(
	src, dst: []f32,
	b, c, h, w, k_h, k_w, h_out, w_out, stride, padding: uint,
) {
	hw_in := h * w
	hw_out := h_out * w_out

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * hw_in:]
			dst_channel := dst[(b_idx * c + c_idx) * hw_out:]

			// Special case for 2x2 pooling with stride 2 and no padding (common case)
			if k_h == 2 && k_w == 2 && stride == 2 && padding == 0 {
				dst_idx := uint(0)
				for y_out in 0 ..< h_out {
					y_in := y_out * 2
					for x_out in 0 ..< w_out {
						x_in := x_out * 2

						idx0 := y_in * w + x_in
						idx1 := idx0 + 1
						idx2 := idx0 + w
						idx3 := idx2 + 1

						val0 := src_channel[idx0]
						val1 := src_channel[idx1]
						val2 := src_channel[idx2]
						val3 := src_channel[idx3]

						max_val := simd.reduce_max(#simd[4]f32{val0, val1, val2, val3})
						dst_channel[dst_idx] = max_val
						dst_idx += 1
					}
				}
			} else {
				// General case with padding support
				dst_idx := uint(0)
				for y_out in 0 ..< h_out {
					y_start := int(y_out * stride) - int(padding)
					y_end := min(y_start + int(k_h), int(h))
					y_start_clamped := max(y_start, 0)

					for x_out in 0 ..< w_out {
						x_start := int(x_out * stride) - int(padding)
						x_end := min(x_start + int(k_w), int(w))
						x_start_clamped := max(x_start, 0)

						// Find max in window using SIMD where possible
						max_val := math.inf_f32(-1)

						for y in y_start_clamped ..< y_end {
							row_start := uint(y) * w + uint(x_start_clamped)
							window_width := uint(x_end - x_start_clamped)

							x := uint(0)
							max_vec := #simd[4]f32 {
								math.inf_f32(-1),
								math.inf_f32(-1),
								math.inf_f32(-1),
								math.inf_f32(-1),
							}

							// SIMD processing for groups of 4
							for ; x + 4 <= window_width; x += 4 {
								vals := #simd[4]f32 {
									src_channel[row_start + x],
									src_channel[row_start + x + 1],
									src_channel[row_start + x + 2],
									src_channel[row_start + x + 3],
								}
								max_vec = simd.max(max_vec, vals)
							}

							// Reduce SIMD vector
							max_val = max(max_val, simd.reduce_max(max_vec))

							// Handle remainder
							for ; x < window_width; x += 1 {
								max_val = max(max_val, src_channel[row_start + x])
							}
						}

						dst_channel[dst_idx] = max_val
						dst_idx += 1
					}
				}
			}
		}
	}
}

@(private)
max_pool_2d_scalar :: proc(
	src, dst: []$T,
	b, c, h, w, k_h, k_w, h_out, w_out, stride, padding: uint,
) {
	hw_in := h * w
	hw_out := h_out * w_out

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * hw_in:]
			dst_channel := dst[(b_idx * c + c_idx) * hw_out:]

			dst_idx := uint(0)
			for y_out in 0 ..< h_out {
				y_start := int(y_out * stride) - int(padding)
				y_end := min(y_start + int(k_h), int(h))
				y_start_clamped := max(y_start, 0)

				for x_out in 0 ..< w_out {
					x_start := int(x_out * stride) - int(padding)
					x_end := min(x_start + int(k_w), int(w))
					x_start_clamped := max(x_start, 0)

					// Find max in window, padded regions contribute -inf
					max_val: T
					when T == f32 || T == f64 {
						max_val = math.inf_f32(-1) if T == f32 else math.inf_f64(-1)
					} else {
						max_val = min(T)
					}

					for y in y_start_clamped ..< y_end {
						for x in x_start_clamped ..< x_end {
							val := src_channel[uint(y) * w + uint(x)]
							if val > max_val {
								max_val = val
							}
						}
					}

					dst_channel[dst_idx] = max_val
					dst_idx += 1
				}
			}
		}
	}
}

import "core:slice"
import "core:testing"

@(test)
test_max_pool_2d :: proc(t: ^testing.T) {
	// Test 2x2 pooling with stride 2, no padding (backward compatibility)
	{
		input := new_with_init(
			[]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint{1, 1, 4, 4}, // BCHW
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{2, 2}, 2, 0, context.temp_allocator)

		expected := []f32{6, 8, 14, 16}
		testing.expect(t, slice.equal(output.data, expected), "2x2 max pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 2, 2}), "Output shape mismatch")
	}

	// Test 3x3 pooling with stride 1, no padding
	{
		input := new_with_init(
			[]f32 {
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9,
				10,
				11,
				12,
				13,
				14,
				15,
				16,
				17,
				18,
				19,
				20,
				21,
				22,
				23,
				24,
				25,
			},
			[]uint{1, 1, 5, 5},
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{3, 3}, 1, 0, context.temp_allocator)

		expected := []f32{13, 14, 15, 18, 19, 20, 23, 24, 25}
		testing.expect(t, slice.equal(output.data, expected), "3x3 max pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 3, 3}), "Output shape mismatch")
	}

	// Test multi-channel pooling
	{
		input := new_with_init(
			[]f32 {
				// Channel 0
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9,
				10,
				11,
				12,
				13,
				14,
				15,
				16,
				// Channel 1
				16,
				15,
				14,
				13,
				12,
				11,
				10,
				9,
				8,
				7,
				6,
				5,
				4,
				3,
				2,
				1,
			},
			[]uint{1, 2, 4, 4},
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{2, 2}, 2, 0, context.temp_allocator)

		expected := []f32{6, 8, 14, 16, 16, 14, 8, 6}
		testing.expect(t, slice.equal(output.data, expected), "Multi-channel pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 2, 2, 2}), "Output shape mismatch")
	}

	// Test with padding=1
	{
		input := new_with_init([]f32{1, 2, 3, 4}, []uint{1, 1, 2, 2}, context.temp_allocator)

		output := max_pool_2d(input, [2]uint{2, 2}, 1, 1, context.temp_allocator)

		// With padding=1, the 2x2 input becomes effectively 4x4 padded with -inf
		// Output should be 3x3
		expected := []f32{1, 2, 2, 3, 4, 4, 3, 4, 4}
		testing.expect(t, slice.equal(output.data, expected), "Pooling with padding failed")
		testing.expect(
			t,
			slice.equal(output.shape, []uint{1, 1, 3, 3}),
			"Output shape mismatch with padding",
		)
	}
}
