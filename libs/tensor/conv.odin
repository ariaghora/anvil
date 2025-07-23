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

	// Use original tensor data directly if contiguous
	src := t.data
	if !t.contiguous {
		// Always use temp_allocator for temporary strided data copy
		src, _ = get_strided_data(t, allocator = context.temp_allocator)
	}
	
	dst_size := b * h_out * w_out * c * h_k * w_k
	dst := make([]T, dst_size, allocator, loc)

	for b_idx in 0 ..< b {
		src_idx_b := b_idx * src_s0
		dst_idx_b := b_idx * h_out * w_out * c * h_k * w_k
		for h_idx in 0 ..< h_out {
			dst_idx_h := dst_idx_b + h_idx * w_out * c * h_k * w_k
			for w_idx in 0 ..< w_out {
				dst_idx_w := dst_idx_h + w_idx * c * h_k * w_k
				for c_idx in 0 ..< c {
					dst_idx_c := dst_idx_w + c_idx * h_k * w_k
					src_idx_c := c_idx * src_s1 + src_idx_b
					for h_k_idx in 0 ..< h_k {
						src_h_raw := h_idx * stride + h_k_idx * dilation
						if padding != 0 && (src_h_raw < padding || src_h_raw >= h + padding) {
							continue
						}
						src_h := src_h_raw - padding
						src_idx_h := src_idx_c + src_h * src_s2  
						dst_idx_hk := dst_idx_c + h_k_idx * w_k
						for w_k_idx in 0 ..< w_k {
							src_w_raw := w_idx * stride + w_k_idx * dilation
							if padding != 0 && (src_w_raw < padding || src_w_raw >= w + padding) {
								continue
							}
							src_w := src_w_raw - padding
							src_idx := src_idx_h + src_w * src_s3
							dst_idx := dst_idx_hk + w_k_idx
							
							dst[dst_idx] = src[src_idx]
						}
					}
				}
			}
		}
	}
	// temp_allocator memory is managed automatically, no manual free needed

	t_out := tensor_alloc(T, []uint{b, h_out * w_out, c * h_k * w_k}, false, allocator, loc)
	t_out.data = dst

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
