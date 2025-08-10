package tensor

import "../tensor"
import "../trace"
import "core:os"
import "core:simd"
import "core:sync"
import "core:thread"


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
	// conv_bn_trace := trace.TRACE_FUNCTION("im2col")
	// defer trace.end_scoped_trace(conv_bn_trace)

	b, c, h, w := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
	h_out, w_out := get_hw(h, w, h_k, w_k, stride, dilation, padding)

	src := t.data
	if !t.contiguous {
		src, _ = get_strided_data(t, allocator = context.temp_allocator)
	}

	dst_size := b * h_out * w_out * c * h_k * w_k
	dst := make([]T, dst_size, allocator, loc)

	// im2col_building := trace.TRACE_SECTION("im2col_building")
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
	// trace.end_scoped_trace(im2col_building)

	t_out := tensor_alloc(T, []uint{b, h_out * w_out, c * h_k * w_k}, false, allocator, loc)
	t_out.data = dst
	return t_out
}

im2col_fast_1x1 :: proc(src, dst: []$T, b, c, h, w, h_out, w_out: uint) {
	// im2col_trace := trace.TRACE_FUNCTION("im2col_1x1")
	// defer trace.end_scoped_trace(im2col_trace)

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
	// im2col_trace := trace.TRACE_FUNCTION("im2col_3x3")
	// defer trace.end_scoped_trace(im2col_trace)

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
	// im2col_trace := trace.TRACE_FUNCTION("im2col_general")
	// defer trace.end_scoped_trace(im2col_trace)
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

	// Calculate output dimensions
	h_out, w_out := get_hw(h, w, k_h, k_w, stride, dilation, padding)

	// Allocate output tensor once
	result := tensor_alloc(T, []uint{b, c_out, h_out, w_out}, true, allocator, loc)

	c_in_per_group := c_in / groups
	c_out_per_group := c_out / groups

	// Decide whether to parallelize based on workload
	work_per_group := c_in_per_group * c_out_per_group * h * w * k_h * k_w
	// MIN_WORK_FOR_PARALLEL :: 100000 // Tune this threshold

	if groups >= 4 {
		// Parallel path using thread pool
		grouped_conv_trace := trace.TRACE_SECTION("grouped_conv_parallel")
		defer trace.end_scoped_trace(grouped_conv_trace)

		// Create a thread pool with optimal number of threads
		num_threads := min(int(groups), os.processor_core_count())
		pool: thread.Pool
		thread.pool_init(&pool, context.allocator, num_threads)
		defer thread.pool_destroy(&pool)
		thread.pool_start(&pool)

		// Work data for each group
		Group_Work_Data :: struct {
			input:                     ^Tensor(T),
			kernel:                    ^Tensor(T),
			result:                    ^Tensor(T),
			group_idx:                 uint,
			c_in_per_group:            uint,
			c_out_per_group:           uint,
			b, h, w, h_out, w_out:     uint,
			k_h, k_w:                  uint,
			stride, dilation, padding: uint,
		}

		// Allocate work items
		work_items := make([]Group_Work_Data, groups, context.temp_allocator)
		// defer delete(work_items)

		// Process group function
		process_group_task :: proc(t: thread.Task) {
			work := cast(^Group_Work_Data)t.data

			// Create views for this group
			input_offset := work.group_idx * work.c_in_per_group
			kernel_offset := work.group_idx * work.c_out_per_group
			output_offset := work.group_idx * work.c_out_per_group

			// Calculate offsets for views
			input_data_offset := work.b * input_offset * work.h * work.w
			kernel_data_offset := kernel_offset * work.c_in_per_group * work.k_h * work.k_w

			input_view := Tensor(T) {
				data       = work.input.data[input_data_offset:],
				shape      = []uint{work.b, work.c_in_per_group, work.h, work.w},
				strides    = work.input.strides,
				contiguous = work.input.contiguous,
			}

			kernel_view := Tensor(T) {
				data       = work.kernel.data[kernel_data_offset:],
				shape      = []uint{work.c_out_per_group, work.c_in_per_group, work.k_h, work.k_w},
				strides    = work.kernel.strides,
				contiguous = work.kernel.contiguous,
			}

			// Run convolution for this group
			group_output := conv2d_single(
				&input_view,
				&kernel_view,
				work.stride,
				work.dilation,
				work.padding,
				context.temp_allocator,
			)
			// defer tensor.free_tensor(group_output)

			// Copy to result (thread-safe as each group writes to different channels)
			copy_group_output_parallel(
				work.result,
				group_output,
				work.b,
				output_offset,
				work.c_out_per_group,
				work.h_out,
				work.w_out,
			)
		}

		// Create work items and submit tasks
		for g in 0 ..< groups {
			work_items[g] = Group_Work_Data {
				input           = input,
				kernel          = kernel,
				result          = result,
				group_idx       = g,
				c_in_per_group  = c_in_per_group,
				c_out_per_group = c_out_per_group,
				b               = b,
				h               = h,
				w               = w,
				h_out           = h_out,
				w_out           = w_out,
				k_h             = k_h,
				k_w             = k_w,
				stride          = stride,
				dilation        = dilation,
				padding         = padding,
			}

			// Add task to pool
			thread.pool_add_task(&pool, context.temp_allocator, process_group_task, &work_items[g])

		}

		// Wait for all tasks to complete
		thread.pool_finish(&pool)

	} else {
		// Sequential path for small workloads
		grouped_conv_trace := trace.TRACE_SECTION("grouped_conv_sequential")
		defer trace.end_scoped_trace(grouped_conv_trace)

		for g in 0 ..< groups {
			input_offset := g * c_in_per_group
			kernel_offset := g * c_out_per_group
			output_offset := g * c_out_per_group

			input_data_offset := b * input_offset * h * w
			kernel_data_offset := kernel_offset * c_in_per_group * k_h * k_w

			input_view := Tensor(T) {
				data       = input.data[input_data_offset:],
				shape      = []uint{b, c_in_per_group, h, w},
				strides    = input.strides,
				contiguous = input.contiguous,
			}

			kernel_view := Tensor(T) {
				data       = kernel.data[kernel_data_offset:],
				shape      = []uint{c_out_per_group, c_in_per_group, k_h, k_w},
				strides    = kernel.strides,
				contiguous = kernel.contiguous,
			}

			group_output := conv2d_single(
				&input_view,
				&kernel_view,
				stride,
				dilation,
				padding,
				context.temp_allocator,
			)
			// defer tensor.free_tensor(group_output)

			copy_group_output_parallel(
				result,
				group_output,
				b,
				output_offset,
				c_out_per_group,
				h_out,
				w_out,
			)
		}
	}

	return result
}

@(private)
copy_group_output_parallel :: proc(
	dst: ^Tensor($T),
	src: ^Tensor(T),
	batch: uint,
	channel_offset: uint,
	channels_per_group: uint,
	h_out: uint,
	w_out: uint,
) {
	hw_out := h_out * w_out
	dst_channels_total := dst.shape[1]

	#no_bounds_check {
		when T == f32 {
			for b in 0 ..< batch {
				for c in 0 ..< channels_per_group {
					src_offset := (b * channels_per_group + c) * hw_out
					dst_offset := (b * dst_channels_total + channel_offset + c) * hw_out

					// Copy spatial data with SIMD
					i := uint(0)
					for ; i + 8 <= hw_out; i += 8 {
						vals1 := (^#simd[4]f32)(&src.data[src_offset + i])^
						vals2 := (^#simd[4]f32)(&src.data[src_offset + i + 4])^
						(^#simd[4]f32)(&dst.data[dst_offset + i])^ = vals1
						(^#simd[4]f32)(&dst.data[dst_offset + i + 4])^ = vals2
					}

					for ; i + 4 <= hw_out; i += 4 {
						vals := (^#simd[4]f32)(&src.data[src_offset + i])^
						(^#simd[4]f32)(&dst.data[dst_offset + i])^ = vals
					}

					for ; i < hw_out; i += 1 {
						dst.data[dst_offset + i] = src.data[src_offset + i]
					}
				}
			}
		} else when T == f64 {
			for b in 0 ..< batch {
				for c in 0 ..< channels_per_group {
					src_offset := (b * channels_per_group + c) * hw_out
					dst_offset := (b * dst_channels_total + channel_offset + c) * hw_out

					i := uint(0)
					for ; i + 4 <= hw_out; i += 4 {
						vals1 := (^#simd[2]f64)(&src.data[src_offset + i])^
						vals2 := (^#simd[2]f64)(&src.data[src_offset + i + 2])^
						(^#simd[2]f64)(&dst.data[dst_offset + i])^ = vals1
						(^#simd[2]f64)(&dst.data[dst_offset + i + 2])^ = vals2
					}

					for ; i + 2 <= hw_out; i += 2 {
						vals := (^#simd[2]f64)(&src.data[src_offset + i])^
						(^#simd[2]f64)(&dst.data[dst_offset + i])^ = vals
					}

					for ; i < hw_out; i += 1 {
						dst.data[dst_offset + i] = src.data[src_offset + i]
					}
				}
			}
		} else {
			// Scalar copy for other types
			for b in 0 ..< batch {
				for c in 0 ..< channels_per_group {
					src_offset := (b * channels_per_group + c) * hw_out
					dst_offset := (b * dst_channels_total + channel_offset + c) * hw_out

					for i in 0 ..< hw_out {
						dst.data[dst_offset + i] = src.data[src_offset + i]
					}
				}
			}
		}
	}
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
			// Non-SIMD, tiled
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
	// im2col_transpose_kernel_trace := trace.TRACE_SECTION("im2col_transpose_kernel")
	kernel_2d := reshape(kernel, []uint{c_out, c_in * k_h * k_w}, allocator)
	kernel_transposed := transpose(kernel_2d, 0, 1, allocator, loc) // -> (C_in * K_h * K_w, C_out)
	// trace.end_scoped_trace(im2col_transpose_kernel_trace)

	// Step 3: Batched matrix multiplication
	// (B, H_out * W_out, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out) -> (B, H_out * W_out, C_out)
	// im2col_matmul_trace := trace.TRACE_SECTION("im2col_matmul")
	result := matmul(col, kernel_transposed, allocator, loc)
	// trace.end_scoped_trace(im2col_matmul_trace)

	// Step 4: Reshape back to (B, C_out, H_out, W_out)
	// reshape_back_get_strided_data_trace := trace.TRACE_SECTION("reshape_back_get_strided_data")
	final := reshape_bhwc_to_bchw(result, b, h_out, w_out, c_out, allocator)
	// trace.end_scoped_trace(reshape_back_get_strided_data_trace)

	return final
}
