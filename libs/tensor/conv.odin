package tensor

import "../trace"
import "core:simd"

// NOTE(Aria): tuned for M1 chips. Basically this depends on cache size
TILE_H :: 64
TILE_W :: 64
TILE_C :: 16
TILE_SIZE :: 8

get_hw :: proc(h_im, w_im, h_k, w_k, stride, dilation, padding: uint) -> (uint, uint) {
	h_out := (h_im + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1
	w_out := (w_im + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1
	return h_out, w_out
}

im2col :: proc(
	t: ^Tensor($T),
	h_k, w_k, stride, dilation, padding: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	conv_bn_trace := trace.TRACE_FUNCTION("im2col")
	defer trace.end_scoped_trace(conv_bn_trace)

	b, c, h, w := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
	h_out, w_out := get_hw(h, w, h_k, w_k, stride, dilation, padding)

	src := t.data
	if !t.contiguous {
		src, _ = get_strided_data(t, allocator = context.temp_allocator)
	}

	dst_size := b * h_out * w_out * c * h_k * w_k
	dst := make([]T, dst_size, allocator, loc)

	im2col_building := trace.TRACE_SECTION("im2col_building")

	if stride == 1 && dilation == 1 && padding == 0 && t.contiguous {
		if h_k == 1 && w_k == 1 {
			im2col_fast_1x1(src, dst, b, c, h, w, h_out, w_out)
		} else if h_k == 3 && w_k == 3 {
			im2col_fast_3x3(src, dst, b, c, h, w, h_out, w_out)
		} else {
			im2col_general(
				src,
				dst,
				b,
				c,
				h,
				w,
				h_k,
				w_k,
				h_out,
				w_out,
				stride,
				dilation,
				padding,
				t.strides,
			)
		}
	} else {
		im2col_general(
			src,
			dst,
			b,
			c,
			h,
			w,
			h_k,
			w_k,
			h_out,
			w_out,
			stride,
			dilation,
			padding,
			t.strides,
		)
	}

	trace.end_scoped_trace(im2col_building)

	t_out := tensor_alloc(T, []uint{b, h_out * w_out, c * h_k * w_k}, false, allocator, loc)
	t_out.data = dst
	return t_out
}

im2col_fast_1x1 :: proc(src, dst: []$T, b, c, h, w, h_out, w_out: uint) {
	im2col_trace := trace.TRACE_FUNCTION("im2col_1x1")
	defer trace.end_scoped_trace(im2col_trace)

	hw := h * w
	chw := c * hw

	#no_bounds_check {
		for b_idx in 0 ..< b {
			src_b := src[b_idx * chw:]
			dst_b := dst[b_idx * hw * c:]

			when T == f32 {
				// Process multiple spatial positions AND channels with SIMD
				// This processes a 4x4 block: 4 spatial positions × 4 channels
				hw_idx := uint(0)

				// Main loop: process 4 spatial positions at a time
				for ; hw_idx + 4 <= hw; hw_idx += 4 {
					dst_base := hw_idx * c

					c_idx := uint(0)
					// Process 4 channels at a time
					for ; c_idx + 4 <= c; c_idx += 4 {
						// Load 4x4 block: 4 spatial × 4 channels
						row0 := #simd[4]f32 {
							src_b[c_idx * hw + hw_idx],
							src_b[(c_idx + 1) * hw + hw_idx],
							src_b[(c_idx + 2) * hw + hw_idx],
							src_b[(c_idx + 3) * hw + hw_idx],
						}
						row1 := #simd[4]f32 {
							src_b[c_idx * hw + hw_idx + 1],
							src_b[(c_idx + 1) * hw + hw_idx + 1],
							src_b[(c_idx + 2) * hw + hw_idx + 1],
							src_b[(c_idx + 3) * hw + hw_idx + 1],
						}
						row2 := #simd[4]f32 {
							src_b[c_idx * hw + hw_idx + 2],
							src_b[(c_idx + 1) * hw + hw_idx + 2],
							src_b[(c_idx + 2) * hw + hw_idx + 2],
							src_b[(c_idx + 3) * hw + hw_idx + 2],
						}
						row3 := #simd[4]f32 {
							src_b[c_idx * hw + hw_idx + 3],
							src_b[(c_idx + 1) * hw + hw_idx + 3],
							src_b[(c_idx + 2) * hw + hw_idx + 3],
							src_b[(c_idx + 3) * hw + hw_idx + 3],
						}

						// Store 4 rows of 4 channels each
						(^#simd[4]f32)(&dst_b[dst_base + c_idx])^ = row0
						(^#simd[4]f32)(&dst_b[dst_base + c + c_idx])^ = row1
						(^#simd[4]f32)(&dst_b[dst_base + 2 * c + c_idx])^ = row2
						(^#simd[4]f32)(&dst_b[dst_base + 3 * c + c_idx])^ = row3
					}

					// Handle remainder channels for these 4 spatial positions
					for ; c_idx < c; c_idx += 1 {
						dst_b[dst_base + c_idx] = src_b[c_idx * hw + hw_idx]
						dst_b[dst_base + c + c_idx] = src_b[c_idx * hw + hw_idx + 1]
						dst_b[dst_base + 2 * c + c_idx] = src_b[c_idx * hw + hw_idx + 2]
						dst_b[dst_base + 3 * c + c_idx] = src_b[c_idx * hw + hw_idx + 3]
					}
				}

				// Handle remainder spatial positions
				for ; hw_idx < hw; hw_idx += 1 {
					dst_offset := hw_idx * c

					c_idx := uint(0)
					for ; c_idx + 4 <= c; c_idx += 4 {
						vals := #simd[4]f32 {
							src_b[c_idx * hw + hw_idx],
							src_b[(c_idx + 1) * hw + hw_idx],
							src_b[(c_idx + 2) * hw + hw_idx],
							src_b[(c_idx + 3) * hw + hw_idx],
						}
						(^#simd[4]f32)(&dst_b[dst_offset + c_idx])^ = vals
					}

					for ; c_idx < c; c_idx += 1 {
						dst_b[dst_offset + c_idx] = src_b[c_idx * hw + hw_idx]
					}
				}
			} else when T == f64 {
				// Process 2x2 blocks for f64
				hw_idx := uint(0)

				for ; hw_idx + 2 <= hw; hw_idx += 2 {
					dst_base := hw_idx * c

					c_idx := uint(0)
					for ; c_idx + 2 <= c; c_idx += 2 {
						row0 := #simd[2]f64 {
							src_b[c_idx * hw + hw_idx],
							src_b[(c_idx + 1) * hw + hw_idx],
						}
						row1 := #simd[2]f64 {
							src_b[c_idx * hw + hw_idx + 1],
							src_b[(c_idx + 1) * hw + hw_idx + 1],
						}

						(^#simd[2]f64)(&dst_b[dst_base + c_idx])^ = row0
						(^#simd[2]f64)(&dst_b[dst_base + c + c_idx])^ = row1
					}

					for ; c_idx < c; c_idx += 1 {
						dst_b[dst_base + c_idx] = src_b[c_idx * hw + hw_idx]
						dst_b[dst_base + c + c_idx] = src_b[c_idx * hw + hw_idx + 1]
					}
				}

				for ; hw_idx < hw; hw_idx += 1 {
					dst_offset := hw_idx * c

					c_idx := uint(0)
					for ; c_idx + 2 <= c; c_idx += 2 {
						vals := #simd[2]f64 {
							src_b[c_idx * hw + hw_idx],
							src_b[(c_idx + 1) * hw + hw_idx],
						}
						(^#simd[2]f64)(&dst_b[dst_offset + c_idx])^ = vals
					}

					for ; c_idx < c; c_idx += 1 {
						dst_b[dst_offset + c_idx] = src_b[c_idx * hw + hw_idx]
					}
				}
			} else {
				// Original tiled implementation for other types
				for hw_tile := uint(0); hw_tile < hw; hw_tile += TILE_SIZE * TILE_SIZE {
					hw_tile_end := min(hw_tile + TILE_SIZE * TILE_SIZE, hw)

					for c_tile := uint(0); c_tile < c; c_tile += TILE_C {
						c_tile_end := min(c_tile + TILE_C, c)

						for hw_idx := hw_tile; hw_idx < hw_tile_end; hw_idx += 1 {
							dst_offset := hw_idx * c + c_tile

							c_idx := c_tile
							for ; c_idx + TILE_SIZE - 1 < c_tile_end; c_idx += TILE_SIZE {
								#unroll for i in 0 ..< TILE_SIZE {
									dst_b[dst_offset + uint(i)] =
										src_b[(c_idx + uint(i)) * hw + hw_idx]
								}
								dst_offset += TILE_SIZE
							}

							for ; c_idx < c_tile_end; c_idx += 1 {
								dst_b[dst_offset] = src_b[c_idx * hw + hw_idx]
								dst_offset += 1
							}
						}
					}
				}
			}
		}
	}
}

im2col_fast_3x3 :: proc(src, dst: []$T, b, c, h, w, h_out, w_out: uint) {
	im2col_trace := trace.TRACE_FUNCTION("im2col_3x3")
	defer trace.end_scoped_trace(im2col_trace)

	dst_idx := 0
	hw := h * w
	chw := c * hw

	#no_bounds_check {
		for b_idx in 0 ..< b {
			src_b := src[b_idx * chw:]

			for h_idx in 0 ..< h_out {
				for w_idx in 0 ..< w_out {
					for c_idx in 0 ..< c {
						#unroll for h_k_idx in 0 ..< 3 {
							#unroll for w_k_idx in 0 ..< 3 {
								src_offset := (h_idx + uint(h_k_idx)) * w + w_idx + uint(w_k_idx)
								dst[dst_idx] = src_b[c_idx * hw + src_offset]
								dst_idx += 1
							}
						}
					}
				}
			}
		}
	}
}

im2col_general :: proc(
	src, dst: []$T,
	b, c, h, w, h_k, w_k, h_out, w_out: uint,
	stride, dilation, padding: uint,
	strides: []uint,
) {
	im2col_trace := trace.TRACE_FUNCTION("im2col_general")
	defer trace.end_scoped_trace(im2col_trace)
	src_s0, src_s1, src_s2, src_s3 := strides[0], strides[1], strides[2], strides[3]

	for b_idx in 0 ..< b {
		src_idx_b := b_idx * src_s0
		dst_idx_b := b_idx * h_out * w_out * c * h_k * w_k

		// Tile over channels
		for c_tile_start := uint(0); c_tile_start < c; c_tile_start += TILE_C {
			c_tile_end := min(c_tile_start + TILE_C, c)

			// Tile over output height
			for h_tile_start := uint(0); h_tile_start < h_out; h_tile_start += TILE_H {
				h_tile_end := min(h_tile_start + TILE_H, h_out)

				// Tile over output width
				for w_tile_start := uint(0); w_tile_start < w_out; w_tile_start += TILE_W {
					w_tile_end := min(w_tile_start + TILE_W, w_out)

					// Now process this tile completely
					for c_idx in c_tile_start ..< c_tile_end {
						src_idx_c := src_idx_b + c_idx * src_s1

						for h_k_idx in 0 ..< h_k {
							// Pre-calculate valid h range for this tile
							h_k_offset := h_k_idx * dilation
							h_idx_start := h_tile_start
							h_idx_end := h_tile_end

							if padding != 0 {
								if h_k_offset < padding {
									h_idx_start = max(
										h_tile_start,
										(padding - h_k_offset + stride - 1) / stride,
									)
								}
								max_src_h := h + padding - 1
								if h_k_offset > max_src_h {
									continue // Skip this kernel position entirely
								}
								h_idx_end = min(h_idx_end, (max_src_h - h_k_offset) / stride + 1)
							}

							for w_k_idx in 0 ..< w_k {
								// Pre-calculate valid w range for this tile
								w_k_offset := w_k_idx * dilation
								w_idx_start := w_tile_start
								w_idx_end := w_tile_end

								if padding != 0 {
									if w_k_offset < padding {
										w_idx_start = max(
											w_tile_start,
											(padding - w_k_offset + stride - 1) / stride,
										)
									}
									max_src_w := w + padding - 1
									if w_k_offset > max_src_w {
										continue // Skip this kernel position entirely
									}
									w_idx_end = min(
										w_idx_end,
										(max_src_w - w_k_offset) / stride + 1,
									)
								}

								// Inner loops process only the tile
								for h_idx in h_idx_start ..< h_idx_end {
									src_h := h_idx * stride + h_k_offset - padding
									src_idx_h := src_idx_c + src_h * src_s2

									// Pre-calculate some destination indices
									dst_idx_h_base :=
										dst_idx_b +
										h_idx * w_out * (c * h_k * w_k) +
										c_idx * h_k * w_k +
										h_k_idx * w_k +
										w_k_idx

									for w_idx in w_idx_start ..< w_idx_end {
										src_w := w_idx * stride + w_k_offset - padding
										src_idx := src_idx_h + src_w * src_s3

										dst_idx := dst_idx_h_base + w_idx * (c * h_k * w_k)

										dst[dst_idx] = src[src_idx]
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

conv2d_grouped :: proc(
	input: ^Tensor($T), // (B, C_in, H, W)
	kernel: ^Tensor(T), // (C_out, C_in_per_group, K_h, K_w)
	groups: uint,
	stride, dilation, padding: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Extract dimensions
	b, c_in, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	c_out, c_in_k, k_h, k_w := kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

	// Validate group constraints
	if c_in % groups != 0 {
		panic("Input channels must be divisible by groups")
	}
	if c_out % groups != 0 {
		panic("Output channels must be divisible by groups")
	}
	if c_in_k != c_in / groups {
		panic("Kernel input channels must equal input channels per group")
	}

	if groups == 1 {
		return conv2d_single(input, kernel, stride, dilation, padding, allocator, loc)
	}

	// Split input and kernel into groups
	input_chunks := chunk(input, groups, 1, context.temp_allocator) // Split along channel dimension
	kernel_chunks := chunk(kernel, groups, 0, context.temp_allocator) // Split along output channel dimension

	// Apply convolution to each group
	results := make([]^Tensor(T), groups, context.temp_allocator)
	defer delete(results, context.temp_allocator)

	grouped_conv_trace := trace.TRACE_SECTION("grouped_conv")
	for i in 0 ..< groups {
		results[i] = conv2d_single(
			input_chunks[i],
			kernel_chunks[i],
			stride,
			dilation,
			padding,
			context.temp_allocator,
		)
	}
	trace.end_scoped_trace(grouped_conv_trace)

	// Concatenate results along output channel dimension (dim=1)
	grouped_conv_cat_trace := trace.TRACE_SECTION("grouped_conv_cat")
	final_result := cat(results, 1, allocator, loc)
	trace.end_scoped_trace(grouped_conv_cat_trace)

	return final_result
}

conv2d :: proc {
	conv2d_single,
	conv2d_grouped,
}

// Specialized for conv2d: (B*H*W, C) -> (B, C, H, W)
reshape_bhwc_to_bchw :: proc(
	tensor: ^Tensor($T),
	batch, height, width, channels: uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	result := tensor_alloc(T, []uint{batch, channels, height, width}, true, allocator)

	src := tensor.data
	dst := result.data
	hw := height * width

	#no_bounds_check {
		when T == f32 {
			for b in 0 ..< batch {
				src_batch := src[b * hw * channels:]
				dst_batch := dst[b * channels * hw:]

				// Process 4 spatial positions at once
				hw_idx := uint(0)
				for ; hw_idx + 4 <= hw; hw_idx += 4 {
					src_base := hw_idx * channels

					// Process 4 channels at a time
					c := uint(0)
					for ; c + 4 <= channels; c += 4 {
						// Load 4x4 block: 4 spatial × 4 channels
						row0 := #simd[4]f32 {
							src_batch[src_base + c],
							src_batch[src_base + c + 1],
							src_batch[src_base + c + 2],
							src_batch[src_base + c + 3],
						}
						row1 := #simd[4]f32 {
							src_batch[src_base + channels + c],
							src_batch[src_base + channels + c + 1],
							src_batch[src_base + channels + c + 2],
							src_batch[src_base + channels + c + 3],
						}
						row2 := #simd[4]f32 {
							src_batch[src_base + 2 * channels + c],
							src_batch[src_base + 2 * channels + c + 1],
							src_batch[src_base + 2 * channels + c + 2],
							src_batch[src_base + 2 * channels + c + 3],
						}
						row3 := #simd[4]f32 {
							src_batch[src_base + 3 * channels + c],
							src_batch[src_base + 3 * channels + c + 1],
							src_batch[src_base + 3 * channels + c + 2],
							src_batch[src_base + 3 * channels + c + 3],
						}

						// Transpose and store: each channel gets 4 spatial values
						dst_c0 := #simd[4]f32 {
							simd.extract(row0, 0),
							simd.extract(row1, 0),
							simd.extract(row2, 0),
							simd.extract(row3, 0),
						}
						dst_c1 := #simd[4]f32 {
							simd.extract(row0, 1),
							simd.extract(row1, 1),
							simd.extract(row2, 1),
							simd.extract(row3, 1),
						}
						dst_c2 := #simd[4]f32 {
							simd.extract(row0, 2),
							simd.extract(row1, 2),
							simd.extract(row2, 2),
							simd.extract(row3, 2),
						}
						dst_c3 := #simd[4]f32 {
							simd.extract(row0, 3),
							simd.extract(row1, 3),
							simd.extract(row2, 3),
							simd.extract(row3, 3),
						}

						(^#simd[4]f32)(&dst_batch[c * hw + hw_idx])^ = dst_c0
						(^#simd[4]f32)(&dst_batch[(c + 1) * hw + hw_idx])^ = dst_c1
						(^#simd[4]f32)(&dst_batch[(c + 2) * hw + hw_idx])^ = dst_c2
						(^#simd[4]f32)(&dst_batch[(c + 3) * hw + hw_idx])^ = dst_c3
					}

					// Handle remainder channels
					for ; c < channels; c += 1 {
						vals := #simd[4]f32 {
							src_batch[src_base + c],
							src_batch[src_base + channels + c],
							src_batch[src_base + 2 * channels + c],
							src_batch[src_base + 3 * channels + c],
						}
						(^#simd[4]f32)(&dst_batch[c * hw + hw_idx])^ = vals
					}
				}

				// Handle remainder spatial positions
				for ; hw_idx < hw; hw_idx += 1 {
					src_offset := hw_idx * channels

					c := uint(0)
					for ; c + 4 <= channels; c += 4 {
						vals := (^#simd[4]f32)(&src_batch[src_offset + c])^

						dst_batch[c * hw + hw_idx] = simd.extract(vals, 0)
						dst_batch[(c + 1) * hw + hw_idx] = simd.extract(vals, 1)
						dst_batch[(c + 2) * hw + hw_idx] = simd.extract(vals, 2)
						dst_batch[(c + 3) * hw + hw_idx] = simd.extract(vals, 3)
					}

					for ; c < channels; c += 1 {
						dst_batch[c * hw + hw_idx] = src_batch[src_offset + c]
					}
				}
			}
		} else when T == f64 {
			for b in 0 ..< batch {
				src_batch := src[b * hw * channels:]
				dst_batch := dst[b * channels * hw:]

				hw_idx := uint(0)
				for ; hw_idx + 2 <= hw; hw_idx += 2 {
					src_base := hw_idx * channels

					c := uint(0)
					for ; c + 2 <= channels; c += 2 {
						row0 := #simd[2]f64{src_batch[src_base + c], src_batch[src_base + c + 1]}
						row1 := #simd[2]f64 {
							src_batch[src_base + channels + c],
							src_batch[src_base + channels + c + 1],
						}

						dst_c0 := #simd[2]f64{simd.extract(row0, 0), simd.extract(row1, 0)}
						dst_c1 := #simd[2]f64{simd.extract(row0, 1), simd.extract(row1, 1)}

						(^#simd[2]f64)(&dst_batch[c * hw + hw_idx])^ = dst_c0
						(^#simd[2]f64)(&dst_batch[(c + 1) * hw + hw_idx])^ = dst_c1
					}

					for ; c < channels; c += 1 {
						vals := #simd[2]f64 {
							src_batch[src_base + c],
							src_batch[src_base + channels + c],
						}
						(^#simd[2]f64)(&dst_batch[c * hw + hw_idx])^ = vals
					}
				}

				for ; hw_idx < hw; hw_idx += 1 {
					src_offset := hw_idx * channels

					c := uint(0)
					for ; c + 2 <= channels; c += 2 {
						vals := (^#simd[2]f64)(&src_batch[src_offset + c])^

						dst_batch[c * hw + hw_idx] = simd.extract(vals, 0)
						dst_batch[(c + 1) * hw + hw_idx] = simd.extract(vals, 1)
					}

					for ; c < channels; c += 1 {
						dst_batch[c * hw + hw_idx] = src_batch[src_offset + c]
					}
				}
			}
		} else {
			// Original tiled code
			for b in 0 ..< batch {
				for c_tile := uint(0); c_tile < channels; c_tile += TILE_SIZE {
					c_end := min(c_tile + TILE_SIZE, channels)

					for h in 0 ..< height {
						for w in 0 ..< width {
							src_base := b * height * width * channels + (h * width + w) * channels
							dst_base := b * channels * height * width + h * width + w

							if c_end - c_tile == TILE_SIZE {
								#unroll for i in 0 ..< TILE_SIZE {
									dst[dst_base + (c_tile + uint(i)) * height * width] =
										src[src_base + c_tile + uint(i)]
								}
							} else {
								for c := c_tile; c < c_end; c += 1 {
									dst[dst_base + c * height * width] = src[src_base + c]
								}
							}
						}
					}
				}
			}
		}
	}

	return result
}

conv2d_single :: proc(
	input: ^Tensor($T), // (B, C_in, H, W)
	kernel: ^Tensor(T), // (C_out, C_in, K_h, K_w)
	stride, dilation, padding: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Extract dimensions
	b, c_in, h, w := input.shape[0], input.shape[1], input.shape[2], input.shape[3]
	c_out, _, k_h, k_w := kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

	// Calculate output dimensions
	h_out, w_out := get_hw(h, w, k_h, k_w, stride, dilation, padding)

	// Step 1: im2col - (B, C_in, H, W) -> (B, H_out * W_out, C_in * K_h * K_w)
	col := im2col(input, k_h, k_w, stride, dilation, padding, allocator)

	// Step 2: Reshape kernel - (C_out, C_in, K_h, K_w) -> (C_in * K_h * K_w, C_out)
	im2col_transpose_kernel_trace := trace.TRACE_SECTION("im2col_transpose_kernel")
	kernel_2d := reshape(kernel, []uint{c_out, c_in * k_h * k_w}, allocator)
	kernel_transposed := transpose(kernel_2d, 0, 1, allocator, loc) // -> (C_in * K_h * K_w, C_out)
	trace.end_scoped_trace(im2col_transpose_kernel_trace)

	// Step 3: Batched matrix multiplication
	// (B, H_out * W_out, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out) -> (B, H_out * W_out, C_out)
	im2col_matmul_trace := trace.TRACE_SECTION("im2col_matmul")
	result := matmul(col, kernel_transposed, allocator, loc)
	trace.end_scoped_trace(im2col_matmul_trace)

	// Step 4: Reshape back to (B, C_out, H_out, W_out)
	reshape_back_get_strided_data_trace := trace.TRACE_SECTION("reshape_back_get_strided_data")
	final := reshape_bhwc_to_bchw(result, b, h_out, w_out, c_out, allocator)
	trace.end_scoped_trace(reshape_back_get_strided_data_trace)

	return final
}
