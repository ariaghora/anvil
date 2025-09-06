package tensor

import "../trace"
import "core:math"
import "core:simd"

// Calculate output dimensions for pooling (same formula as conv)
get_pool_hw :: proc(h_in, w_in, k_h, k_w, stride: uint) -> (uint, uint) {
	h_out := (h_in - k_h) / stride + 1
	w_out := (w_in - k_w) / stride + 1
	return h_out, w_out
}

max_pool_2d :: proc(
	input: ^Tensor($T), // (B, C, H, W)
	kernel_size: [2]uint, // (K_h, K_w)
	stride: uint = 1,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	pool_trace := trace.TRACE_FUNCTION("max_pool_2d")
	defer trace.end_scoped_trace(pool_trace)

	// Extract dimensions
	b, c, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	k_h, k_w := kernel_size[0], kernel_size[1]

	// Calculate output dimensions
	h_out, w_out := get_pool_hw(h, w, k_h, k_w, stride)

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
			max_pool_2d_f32_simd(src, output.data, b, c, h, w, k_h, k_w, h_out, w_out, stride)
		} else when T == f64 {
			max_pool_2d_f64_simd(src, output.data, b, c, h, w, k_h, k_w, h_out, w_out, stride)
		} else {
			max_pool_2d_scalar(src, output.data, b, c, h, w, k_h, k_w, h_out, w_out, stride)
		}
	}

	return output
}

@(private)
max_pool_2d_f32_simd :: proc(src, dst: []f32, b, c, h, w, k_h, k_w, h_out, w_out, stride: uint) {
	hw_in := h * w
	hw_out := h_out * w_out

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * hw_in:]
			dst_channel := dst[(b_idx * c + c_idx) * hw_out:]

			// Special case for 2x2 pooling with stride 2 (common case)
			if k_h == 2 && k_w == 2 && stride == 2 {
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

						max_val := max(max(val0, val1), max(val2, val3))
						dst_channel[dst_idx] = max_val
						dst_idx += 1
					}
				}
			} else {
				// General case
				dst_idx := uint(0)
				for y_out in 0 ..< h_out {
					y_start := y_out * stride
					y_end := min(y_start + k_h, h)

					for x_out in 0 ..< w_out {
						x_start := x_out * stride
						x_end := min(x_start + k_w, w)

						// Find max in window using SIMD where possible
						max_val := math.inf_f32(-1)

						for y in y_start ..< y_end {
							row_start := y * w + x_start
							window_width := x_end - x_start

							x := uint(0)
							max_vec := #simd[4]f32 {
								math.inf_f32(-1),
								math.inf_f32(-1),
								math.inf_f32(-1),
								math.inf_f32(-1),
							}

							// SIMD processing for groups of 4
							for ; x + 4 <= window_width; x += 4 {
								vals := (^#simd[4]f32)(&src_channel[row_start + x])^
								max_vec = simd.max(max_vec, vals)
							}

							// Reduce SIMD vector
							max_val = max(max_val, simd.extract(max_vec, 0))
							max_val = max(max_val, simd.extract(max_vec, 1))
							max_val = max(max_val, simd.extract(max_vec, 2))
							max_val = max(max_val, simd.extract(max_vec, 3))

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
max_pool_2d_f64_simd :: proc(src, dst: []f64, b, c, h, w, k_h, k_w, h_out, w_out, stride: uint) {
	hw_in := h * w
	hw_out := h_out * w_out

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * hw_in:]
			dst_channel := dst[(b_idx * c + c_idx) * hw_out:]

			dst_idx := uint(0)
			for y_out in 0 ..< h_out {
				y_start := y_out * stride
				y_end := min(y_start + k_h, h)

				for x_out in 0 ..< w_out {
					x_start := x_out * stride
					x_end := min(x_start + k_w, w)

					max_val := math.inf_f64(-1)

					for y in y_start ..< y_end {
						row_start := y * w + x_start
						window_width := x_end - x_start

						x := uint(0)
						max_vec := #simd[2]f64{math.inf_f64(-1), math.inf_f64(-1)}

						// SIMD processing for groups of 2
						for ; x + 2 <= window_width; x += 2 {
							vals := (^#simd[2]f64)(&src_channel[row_start + x])^
							max_vec = simd.max(max_vec, vals)
						}

						// Reduce SIMD vector
						max_val = max(max_val, simd.extract(max_vec, 0))
						max_val = max(max_val, simd.extract(max_vec, 1))

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

@(private)
max_pool_2d_scalar :: proc(src, dst: []$T, b, c, h, w, k_h, k_w, h_out, w_out, stride: uint) {
	hw_in := h * w
	hw_out := h_out * w_out

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * hw_in:]
			dst_channel := dst[(b_idx * c + c_idx) * hw_out:]

			dst_idx := uint(0)
			for y_out in 0 ..< h_out {
				y_start := y_out * stride
				y_end := min(y_start + k_h, h)

				for x_out in 0 ..< w_out {
					x_start := x_out * stride
					x_end := min(x_start + k_w, w)

					// Find max in window
					max_val := src_channel[y_start * w + x_start]
					for y in y_start ..< y_end {
						for x in x_start ..< x_end {
							val := src_channel[y * w + x]
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
	// Test 2x2 pooling with stride 2
	{
		input := new_with_init(
			[]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint{1, 1, 4, 4}, // BCHW
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{2, 2}, 2, context.temp_allocator)

		expected := []f32{6, 8, 14, 16}
		testing.expect(t, slice.equal(output.data, expected), "2x2 max pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 2, 2}), "Output shape mismatch")
	}

	// Test 3x3 pooling with stride 1
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

		output := max_pool_2d(input, [2]uint{3, 3}, 1, context.temp_allocator)

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

		output := max_pool_2d(input, [2]uint{2, 2}, 2, context.temp_allocator)

		expected := []f32{6, 8, 14, 16, 16, 14, 8, 6}
		testing.expect(t, slice.equal(output.data, expected), "Multi-channel pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 2, 2, 2}), "Output shape mismatch")
	}
}
