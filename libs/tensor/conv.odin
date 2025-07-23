package tensor

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
	b, c, h, w := t.shape[0], t.shape[1], t.shape[2], t.shape[3]
	h_out, w_out := get_hw(h, w, h_k, w_k, stride, dilation, padding)
	src_s0, src_s1, src_s2, src_s3 := t.strides[0], t.strides[1], t.strides[2], t.strides[3]

	src, src_allocated := get_strided_data(t)
	dst := make([]T, b * h_out * w_out * c * h_k * w_k, allocator, loc)

	for b_idx in 0 ..< b {
		for h_idx in 0 ..< h_out {
			for w_idx in 0 ..< w_out {
				for c_idx in 0 ..< c {
					for h_k_idx in 0 ..< h_k {
						src_h := h_idx * stride + h_k_idx * dilation
						if padding != 0 && (src_h < padding || src_h >= h + padding) {
							continue
						}
						src_h = src_h - padding

						for w_k_idx in 0 ..< w_k {
							src_w := w_idx * stride + w_k_idx * dilation
							if padding != 0 && (src_w < padding || src_w >= w + padding) {
								continue
							}
							src_w = src_w - padding

							// Calculate indices fresh each time
							src_idx :=
								b_idx * src_s0 + c_idx * src_s1 + src_h * src_s2 + src_w * src_s3
							dst_idx :=
								b_idx * h_out * w_out * c * h_k * w_k +
								h_idx * w_out * c * h_k * w_k +
								w_idx * c * h_k * w_k +
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
	if src_allocated do delete(src)

	t_out := tensor_alloc(T, []uint{b, h_out * w_out, c * h_k * w_k}, false, allocator, loc)
	t_out.data = dst

	return t_out
}

conv2d :: proc {
	conv2d_single,
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
	kernel_2d := reshape(kernel, []uint{c_out, c_in * k_h * k_w}, allocator)
	kernel_transposed := transpose(kernel_2d, 0, 1, allocator, loc) // -> (C_in * K_h * K_w, C_out)

	// Step 3: Batched matrix multiplication
	// (B, H_out * W_out, C_in * K_h * K_w) @ (C_in * K_h * K_w, C_out) -> (B, H_out * W_out, C_out)
	result := matmul(col, kernel_transposed, allocator, loc)

	// Step 4: Reshape back to (B, C_out, H_out, W_out)
	output := reshape(result, []uint{b, h_out, w_out, c_out}, allocator)
	final := permute(output, []uint{0, 3, 1, 2}, allocator, loc) // BHWC -> BCHW

	return final
}
