package tensor

import "../trace"
import "core:math"
import "core:simd"

upsample_nearest_2d :: proc(
	input: ^Tensor($T), // (B, C, H, W)
	target_h: uint,
	target_w: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	upsample_trace := trace.TRACE_FUNCTION("upsample_nearest_2d")
	defer trace.end_scoped_trace(upsample_trace)

	// Extract dimensions
	b, c, src_h, src_w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]

	// Allocate output tensor
	output := tensor_alloc(T, []uint{b, c, target_h, target_w}, true, allocator, loc)

	src := input.data
	allocated := false
	if !input.contiguous {
		src, allocated = get_strided_data(input, allocator = allocator)
	}
	defer if allocated do delete(src, allocator)

	// Special case for 2x upsampling
	if target_h == src_h * 2 && target_w == src_w * 2 {
		upsample_2x(src, output.data, b, c, src_h, src_w)
	} else {
		// General case with pre-computed indices
		scale_h := f64(src_h) / f64(target_h)
		scale_w := f64(src_w) / f64(target_w)

		// Pre-compute source indices
		src_h_idxs := make([]uint, target_h, context.temp_allocator)
		src_w_idxs := make([]uint, target_w, context.temp_allocator)

		for h_idx in 0 ..< target_h {
			src_h_idxs[h_idx] = min(src_h - 1, uint(f64(h_idx) * scale_h))
		}
		for w_idx in 0 ..< target_w {
			src_w_idxs[w_idx] = min(src_w - 1, uint(f64(w_idx) * scale_w))
		}

		#no_bounds_check {
			upsample_general(
				src,
				output.data,
				b,
				c,
				src_h,
				src_w,
				target_h,
				target_w,
				src_h_idxs,
				src_w_idxs,
			)
		}
	}

	return output
}

@(private)
upsample_2x :: proc(src, dst: []$T, b, c, src_h, src_w: uint) {
	dst_h := src_h * 2
	dst_w := src_w * 2
	src_hw := src_h * src_w
	dst_hw := dst_h * dst_w

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * src_hw:]
			dst_channel := dst[(b_idx * c + c_idx) * dst_hw:]

			for y in 0 ..< src_h {
				for x in 0 ..< src_w {
					val := src_channel[y * src_w + x]
					dst_y := y * 2
					dst_x := x * 2

					// Write 2x2 block
					dst_channel[dst_y * dst_w + dst_x] = val
					dst_channel[dst_y * dst_w + dst_x + 1] = val
					dst_channel[(dst_y + 1) * dst_w + dst_x] = val
					dst_channel[(dst_y + 1) * dst_w + dst_x + 1] = val
				}
			}
		}
	}
}

@(private)
upsample_general :: proc(
	src, dst: []$T,
	b, c, src_h, src_w, dst_h, dst_w: uint,
	src_h_idxs, src_w_idxs: []uint,
) {
	src_hw := src_h * src_w
	dst_hw := dst_h * dst_w

	for b_idx in 0 ..< b {
		for c_idx in 0 ..< c {
			src_channel := src[(b_idx * c + c_idx) * src_hw:]
			dst_channel := dst[(b_idx * c + c_idx) * dst_hw:]

			dst_idx := uint(0)
			for h_idx in 0 ..< dst_h {
				src_h_idx := src_h_idxs[h_idx]
				src_row := src_channel[src_h_idx * src_w:]

				when T == f32 {
					// Check for runs of same source indices for vectorization
					w_idx := uint(0)
					for w_idx < dst_w {
						src_w_idx := src_w_idxs[w_idx]
						val := src_row[src_w_idx]

						// Count consecutive pixels from same source
						run_length := uint(1)
						for w_idx + run_length < dst_w &&
						    src_w_idxs[w_idx + run_length] == src_w_idx {
							run_length += 1
						}

						// Fill run with same value
						if run_length >= 4 {
							val_vec := #simd[4]f32{val, val, val, val}
							for i := uint(0); i + 4 <= run_length; i += 4 {
								(^#simd[4]f32)(&dst_channel[dst_idx + i])^ = val_vec
							}
							for i := (run_length / 4) * 4; i < run_length; i += 1 {
								dst_channel[dst_idx + i] = val
							}
						} else {
							for i in 0 ..< run_length {
								dst_channel[dst_idx + i] = val
							}
						}

						dst_idx += run_length
						w_idx += run_length
					}
				} else {
					// Scalar path for other types
					for w_idx in 0 ..< dst_w {
						src_w_idx := src_w_idxs[w_idx]
						dst_channel[dst_idx] = src_row[src_w_idx]
						dst_idx += 1
					}
				}
			}
		}
	}
}

import "core:slice"
import "core:testing"

@(test)
test_upsample_nearest_2d :: proc(t: ^testing.T) {
	// Test 2x upsampling
	{
		input := new_with_init([]f32{1, 2, 3, 4}, []uint{1, 1, 2, 2}, context.temp_allocator)
		defer free_tensor(input, context.temp_allocator)

		output := upsample_nearest_2d(input, 4, 4, context.temp_allocator)
		defer free_tensor(output, context.temp_allocator)

		expected := []f32{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}
		testing.expect(t, slice.equal(output.data, expected), "2x upsampling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 4, 4}), "Output shape mismatch")
	}

	// Test arbitrary upsampling
	{
		input := new_with_init([]f32{1, 2, 3, 4}, []uint{1, 1, 2, 2}, context.temp_allocator)
		defer free_tensor(input, context.temp_allocator)

		output := upsample_nearest_2d(input, 3, 3, context.temp_allocator)
		defer free_tensor(output, context.temp_allocator)

		// With scale 2/3, we get:
		// (0,0) -> (0,0), (0,1) -> (0,0), (0,2) -> (0,1)
		// (1,0) -> (0,0), (1,1) -> (0,0), (1,2) -> (0,1)
		// (2,0) -> (1,0), (2,1) -> (1,0), (2,2) -> (1,1)
		expected := []f32{1, 1, 2, 1, 1, 2, 3, 3, 4}
		testing.expect(t, slice.equal(output.data, expected), "Arbitrary upsampling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 3, 3}), "Output shape mismatch")
	}

	// Test multi-channel upsampling
	{
		input := new_with_init(
			[]f32 {
				// Channel 0
				1,
				2,
				3,
				4,
				// Channel 1
				5,
				6,
				7,
				8,
			},
			[]uint{1, 2, 2, 2},
			context.temp_allocator,
		)
		defer free_tensor(input, context.temp_allocator)

		output := upsample_nearest_2d(input, 4, 4, context.temp_allocator)
		defer free_tensor(output, context.temp_allocator)

		expected := []f32 {
			// Channel 0
			1,
			1,
			2,
			2,
			1,
			1,
			2,
			2,
			3,
			3,
			4,
			4,
			3,
			3,
			4,
			4,
			// Channel 1
			5,
			5,
			6,
			6,
			5,
			5,
			6,
			6,
			7,
			7,
			8,
			8,
			7,
			7,
			8,
			8,
		}
		testing.expect(t, slice.equal(output.data, expected), "Multi-channel upsampling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 2, 4, 4}), "Output shape mismatch")
	}
}
