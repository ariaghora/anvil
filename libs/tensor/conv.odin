package tensor

import "../trace"

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
	src_s0, src_s1, src_s2, src_s3 := t.strides[0], t.strides[1], t.strides[2], t.strides[3]

	// Use original tensor data directly if contiguous
	src := t.data
	if !t.contiguous {
		src, _ = get_strided_data(t, allocator = context.temp_allocator)
	}

	dst_size := b * h_out * w_out * c * h_k * w_k
	dst := make([]T, dst_size, allocator, loc)

	im2col_building := trace.TRACE_SECTION("im2col_building")
	for b_idx in 0 ..< b {
		src_idx_b := b_idx * src_s0
		dst_idx_b := b_idx * h_out * w_out * c * h_k * w_k

		for c_idx in 0 ..< c {
			src_idx_c := src_idx_b + c_idx * src_s1

			for h_k_idx in 0 ..< h_k {
				// Pre-calculate valid h_idx range for this kernel position
				h_k_offset := h_k_idx * dilation
				h_idx_start := uint(0)
				h_idx_end := h_out

				if padding != 0 {
					// Find first valid h_idx
					if h_k_offset < padding {
						h_idx_start = (padding - h_k_offset + stride - 1) / stride
					}
					// Find last valid h_idx
					max_src_h := h + padding - 1
					if h_k_offset > max_src_h {
						h_idx_end = 0 // No valid indices
					} else {
						h_idx_end = min(h_out, (max_src_h - h_k_offset) / stride + 1)
					}
				}

				for w_k_idx in 0 ..< w_k {
					// Pre-calculate valid w_idx range for this kernel position
					w_k_offset := w_k_idx * dilation
					w_idx_start := uint(0)
					w_idx_end := w_out

					if padding != 0 {
						// Find first valid w_idx
						if w_k_offset < padding {
							w_idx_start = (padding - w_k_offset + stride - 1) / stride
						}
						// Find last valid w_idx
						max_src_w := w + padding - 1
						if w_k_offset > max_src_w {
							w_idx_end = 0 // No valid indices
						} else {
							w_idx_end = min(w_out, (max_src_w - w_k_offset) / stride + 1)
						}
					}

					for h_idx in h_idx_start ..< h_idx_end {
						src_h := h_idx * stride + h_k_offset - padding
						src_idx_h := src_idx_c + src_h * src_s2

						for w_idx in w_idx_start ..< w_idx_end {
							src_w := w_idx * stride + w_k_offset - padding
							src_idx := src_idx_h + src_w * src_s3

							dst_idx :=
								dst_idx_b +
								(h_idx * w_out + w_idx) * (c * h_k * w_k) +
								c_idx * h_k * w_k +
								h_k_idx * w_k +
								w_k_idx

							dst[dst_idx] = src[src_idx]
						}
					}
				}
			}
		}
	}
	trace.end_scoped_trace(im2col_building)

	im2col_output_alloc_trace := trace.TRACE_SECTION("im2col_output_alloc")
	t_out := tensor_alloc(T, []uint{b, h_out * w_out, c * h_k * w_k}, false, allocator, loc)
	t_out.data = dst
	trace.end_scoped_trace(im2col_output_alloc_trace)

	return t_out
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
	// Note: temp_allocator chunks will be cleaned up automatically

	kernel_chunks := chunk(kernel, groups, 0, context.temp_allocator) // Split along output channel dimension
	// Note: temp_allocator chunks will be cleaned up automatically

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
	final_result := cat(results, 1, allocator, loc)

	return final_result
}

conv2d :: proc {
	conv2d_single,
	conv2d_grouped,
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
	output := reshape(result, []uint{b, h_out, w_out, c_out}, allocator)
	final := permute(output, []uint{0, 3, 1, 2}, allocator, loc) // BHWC -> BCHW

	return final
}
