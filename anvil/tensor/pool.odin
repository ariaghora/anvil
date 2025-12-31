package tensor

import "../simd_backend"
import "../trace"
import "core:math"
import "core:simd"
import "core:slice"

// Calculate output dimensions
get_pool_hw :: proc(h_in, w_in, k_h, k_w, stride, padding: uint) -> (uint, uint) {
	h_out := (h_in + 2 * padding - k_h) / stride + 1
	w_out := (w_in + 2 * padding - k_w) / stride + 1
	return h_out, w_out
}

max_pool_2d :: proc(
	input: ^Tensor($T), // (B, C, H, W)
	kernel_size: [2]uint, // (K_h, K_w)
	stride: uint = 1,
	padding: uint = 0,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	pool_trace := trace.global_scoped("max_pool_2d")
	defer trace.global_end_scoped(pool_trace)

	b, c, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	k_h, k_w := kernel_size[0], kernel_size[1]

	h_out, w_out := get_pool_hw(h, w, k_h, k_w, stride, padding)
	output := tensor_alloc(T, []uint{b, c, h_out, w_out}, true, allocator, loc)

	src := input.data
	allocated := false
	if !input.contiguous {
		src, allocated = get_strided_data(input, allocator = allocator)
	}
	defer if allocated do delete(src, allocator)

	when T == f32 {
		max_pool_2d_f32_simd(src, output.data, b, c, h, w, k_h, k_w, h_out, w_out, stride, padding)
	} else {
		max_pool_2d_scalar(src, output.data, b, c, h, w, k_h, k_w, h_out, w_out, stride, padding)
	}

	return output
}

@(private)
max_pool_2d_f32_simd :: proc(
	src, dst: []f32,
	b, c, h, w, k_h, k_w, h_out, w_out, stride, padding: uint,
) {
	using simd_backend
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

						max_val := math.inf_f32(-1)
						for y in y_start_clamped ..< y_end {
							row_start := uint(y) * w + uint(x_start_clamped)
							window_width := uint(x_end - x_start_clamped)

							x := uint(0)
							// {-inf, -inf, ..., -inf}
							max_vec := splat(SIMD_F32, math.inf_f32(-1))

							for ; x + 4 <= window_width; x += 4 {
								vals := (^SIMD_F32)(&src_channel[row_start + x])^
								max_vec = simd_backend.max_f32(max_vec, vals)
							}

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

global_avg_pool_2d :: proc(
	input: ^Tensor($T), // (B, C, H, W)
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	pool_trace := trace.global_scoped("global_avg_pool_2d")
	defer trace.global_end_scoped(pool_trace)

	// Extract dimensions
	b, c, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	hw := h * w

	// Output is (B, C, 1, 1)
	output := tensor_alloc(T, []uint{b, c, 1, 1}, true, allocator, loc)

	src := input.data
	allocated := false
	if !input.contiguous {
		src, allocated = get_strided_data(input, allocator = context.temp_allocator)
	}
	defer if allocated do delete(src, context.temp_allocator)

	// Perform global average pooling
	#no_bounds_check {
		when T == f32 {
			global_avg_pool_2d_f32_simd(src, output.data, b, c, hw)
		} else {
			global_avg_pool_2d_scalar(src, output.data, b, c, hw)
		}
	}

	return output
}

@(private = "file")
global_avg_pool_2d_f32_simd :: proc(src, dst: []f32, b, c, hw: uint) {
	scale := f32(1.0) / f32(hw)

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			channel_offset := (b_idx * c + c_idx) * hw
			src_channel := src[channel_offset:channel_offset + hw]

			when ODIN_OS == .Darwin {
				sum: f32
				simd_backend.vDSP_sve(&src_channel[0], 1, &sum, u32(hw))
				dst[b_idx * c + c_idx] = sum * scale
			} else {
				sum_vec := #simd[4]f32{0, 0, 0, 0}
				i := uint(0)

				for ; i + 4 <= hw; i += 4 {
					vals := #simd[4]f32 {
						src_channel[i],
						src_channel[i + 1],
						src_channel[i + 2],
						src_channel[i + 3],
					}
					sum_vec += vals
				}

				sum := simd.reduce_add_bisect(sum_vec)

				// Handle remainder
				for ; i < hw; i += 1 {
					sum += src_channel[i]
				}
				// Store average
				dst[b_idx * c + c_idx] = sum * scale
			}

		}
	}
}

@(private = "file")
global_avg_pool_2d_scalar :: proc(src, dst: []$T, b, c, hw: uint) {
	scale := T(1.0) / T(hw)

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			channel_offset := (b_idx * c + c_idx) * hw
			src_channel := src[channel_offset:channel_offset + hw]

			sum := T(0)
			for i in 0 ..< hw {
				sum += src_channel[i]
			}

			dst[b_idx * c + c_idx] = sum * scale
		}
	}
}
