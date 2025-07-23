package transformer

import "../nn"
import "../tensor"
import "core:fmt"
import "core:math"
import "core:slice"
import "core:time"

// Constants from the Rust implementation
MBCONV_EXPAND_RATIO :: 4
MLP_RATIO :: 4
LOCAL_CONV_SIZE :: 3
IMG_SIZE :: 1024
IN_CHANNELS :: 3

// Conv2dBN - Convolution followed by BatchNorm
Conv_2d_BN :: struct($T: typeid) {
	conv: ^nn.Conv_2d(T),
	bn:   ^nn.Batch_Norm_2d(T),
}

new_conv_2d_bn :: proc(
	$T: typeid,
	in_channels, out_channels: uint,
	kernel_size: uint,
	stride: uint = 1,
	padding: uint = 0,
	groups: uint = 1,
	allocator := context.allocator,
) -> ^Conv_2d_BN(T) {
	conv := nn.new_conv2d(
		T,
		in_channels,
		out_channels,
		[2]uint{kernel_size, kernel_size},
		stride,
		padding,
		1,
		groups,
		true,
		allocator,
	)
	bn := nn.new_batch_norm_2d(T, out_channels, allocator)

	return new_clone(Conv_2d_BN(T){conv = conv, bn = bn}, allocator)
}

forward_conv_2d_bn :: proc(
	layer: ^Conv_2d_BN($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	fmt.printf(
		"        Starting conv2d (input: [%d,%d,%d,%d], groups=%d)...\n",
		x.shape[0],
		x.shape[1],
		x.shape[2],
		x.shape[3],
		layer.conv.groups,
	)
	conv_out := nn.forward_conv2d(layer.conv, x, context.temp_allocator)
	fmt.printf("        Conv2d completed, starting BatchNorm...\n")
	bn_out := nn.forward_batch_norm_2d(layer.bn, conv_out, allocator, loc)
	fmt.printf("        BatchNorm completed\n")
	duration := time.since(start_time)
	fmt.printf("[TIMING] Conv2d+BN: %v\n", duration)
	return bn_out
}

free_conv_2d_bn :: proc(layer: ^Conv_2d_BN($T), allocator := context.allocator) {
	nn.free_conv_2d(layer.conv, allocator)
	nn.free_batch_norm_2d(layer.bn, allocator)
	free(layer, allocator)
}

// PatchEmbed - Initial patch embedding with two convolutions
Patch_Embed :: struct($T: typeid) {
	conv1, conv2: ^Conv_2d_BN(T),
}

new_patch_embed :: proc(
	$T: typeid,
	in_channels, embed_dim: uint,
	allocator := context.allocator,
) -> ^Patch_Embed(T) {
	// stride=2, padding=1, kernel_size=3
	conv1 := new_conv_2d_bn(T, in_channels, embed_dim / 2, 3, 2, 1, 1, allocator)
	conv2 := new_conv_2d_bn(T, embed_dim / 2, embed_dim, 3, 2, 1, 1, allocator)

	return new_clone(Patch_Embed(T){conv1 = conv1, conv2 = conv2}, allocator)
}

forward_patch_embed :: proc(
	pe: ^Patch_Embed($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	conv1_out := forward_conv_2d_bn(pe.conv1, x, context.temp_allocator)
	gelu_out := tensor.gelu(conv1_out, context.temp_allocator)
	conv2_out := forward_conv_2d_bn(pe.conv2, gelu_out, allocator, loc)
	duration := time.since(start_time)
	fmt.printf("[TIMING] PatchEmbed: %v\n", duration)
	return conv2_out
}

free_patch_embed :: proc(pe: ^Patch_Embed($T), allocator := context.allocator) {
	free_conv_2d_bn(pe.conv1, allocator)
	free_conv_2d_bn(pe.conv2, allocator)
	free(pe, allocator)
}

// MBConv - Mobile Inverted Bottleneck Convolution
MB_Conv :: struct($T: typeid) {
	conv1, conv2, conv3: ^Conv_2d_BN(T),
}

new_mb_conv :: proc(
	$T: typeid,
	in_channels, out_channels: uint,
	expand_ratio: uint,
	allocator := context.allocator,
) -> ^MB_Conv(T) {
	hidden := in_channels * expand_ratio

	// Pointwise expansion
	conv1 := new_conv_2d_bn(T, in_channels, hidden, 1, 1, 0, 1, allocator)
	// Depthwise convolution with groups=hidden
	conv2 := new_conv_2d_bn(T, hidden, hidden, 3, 1, 1, hidden, allocator)
	// Pointwise projection
	conv3 := new_conv_2d_bn(T, hidden, out_channels, 1, 1, 0, 1, allocator)

	return new_clone(MB_Conv(T){conv1 = conv1, conv2 = conv2, conv3 = conv3}, allocator)
}

forward_mb_conv :: proc(
	mb: ^MB_Conv($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	shortcut := x

	// Expansion
	conv1_out := forward_conv_2d_bn(mb.conv1, x, context.temp_allocator)
	gelu1_out := tensor.gelu(conv1_out, context.temp_allocator)

	// Depthwise
	conv2_out := forward_conv_2d_bn(mb.conv2, gelu1_out, context.temp_allocator)
	gelu2_out := tensor.gelu(conv2_out, context.temp_allocator)

	// Projection
	conv3_out := forward_conv_2d_bn(mb.conv3, gelu2_out, context.temp_allocator)

	// Check shapes before residual connection
	if !slice.equal(conv3_out.shape, shortcut.shape) {
		panic("MBConv residual connection shape mismatch")
	}

	// Residual connection + final activation
	residual := tensor.add(conv3_out, shortcut, context.temp_allocator)
	result := tensor.gelu(residual, allocator, loc)

	duration := time.since(start_time)
	fmt.printf("[TIMING] MBConv: %v\n", duration)
	return result
}

free_mb_conv :: proc(mb: ^MB_Conv($T), allocator := context.allocator) {
	free_conv_2d_bn(mb.conv1, allocator)
	free_conv_2d_bn(mb.conv2, allocator)
	free_conv_2d_bn(mb.conv3, allocator)
	free(mb, allocator)
}

// PatchMerging - Downsample and merge patches
Patch_Merging :: struct($T: typeid) {
	conv1, conv2, conv3: ^Conv_2d_BN(T),
	input_resolution:    [2]uint,
}

new_patch_merging :: proc(
	$T: typeid,
	input_resolution: [2]uint,
	dim, out: uint,
	allocator := context.allocator,
) -> ^Patch_Merging(T) {
	// Determine stride based on output channels (matching Rust logic)
	stride: uint = 2
	if out == 320 || out == 448 || out == 576 {
		stride = 1
	}

	conv1 := new_conv_2d_bn(T, dim, out, 1, 1, 0, 1, allocator)
	conv2 := new_conv_2d_bn(T, out, out, 3, stride, 1, out, allocator) // groups=out (depthwise)
	conv3 := new_conv_2d_bn(T, out, out, 1, 1, 0, 1, allocator)

	return new_clone(
		Patch_Merging(T) {
			conv1 = conv1,
			conv2 = conv2,
			conv3 = conv3,
			input_resolution = input_resolution,
		},
		allocator,
	)
}

forward_patch_merging :: proc(
	pm: ^Patch_Merging($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Handle input reshaping from 3D to 4D if needed
	xs := x
	if len(x.shape) == 3 {
		// (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
		h, w := pm.input_resolution[0], pm.input_resolution[1]
		b := x.shape[0]
		c := x.shape[2]

		// Reshape to (B, H, W, C)
		reshaped := tensor.reshape(x, []uint{b, h, w, c}, context.temp_allocator)
		// Permute to (B, C, H, W)
		xs = tensor.permute(reshaped, []uint{0, 3, 1, 2}, context.temp_allocator)
	}

	// Apply convolutions
	conv1_out := forward_conv_2d_bn(pm.conv1, xs, context.temp_allocator)
	gelu1_out := tensor.gelu(conv1_out, context.temp_allocator)

	conv2_out := forward_conv_2d_bn(pm.conv2, gelu1_out, context.temp_allocator)
	gelu2_out := tensor.gelu(conv2_out, context.temp_allocator)

	conv3_out := forward_conv_2d_bn(pm.conv3, gelu2_out, context.temp_allocator)

	// Flatten and transpose: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
	b := conv3_out.shape[0]
	c := conv3_out.shape[1]
	spatial_size := conv3_out.shape[2] * conv3_out.shape[3]

	flattened := tensor.reshape(conv3_out, []uint{b, c, spatial_size}, context.temp_allocator)
	result := tensor.transpose(flattened, 1, 2, allocator, loc)

	return result
}

free_patch_merging :: proc(pm: ^Patch_Merging($T), allocator := context.allocator) {
	free_conv_2d_bn(pm.conv1, allocator)
	free_conv_2d_bn(pm.conv2, allocator)
	free_conv_2d_bn(pm.conv3, allocator)
	free(pm, allocator)
}

// ConvLayer - Layer of MBConv blocks with optional downsampling
Conv_Layer :: struct($T: typeid) {
	blocks:     []^MB_Conv(T),
	downsample: Maybe(^Patch_Merging(T)),
}

new_conv_layer :: proc(
	$T: typeid,
	dim, out: uint,
	input_resolution: [2]uint,
	depth: uint,
	downsample: bool,
	conv_expand_ratio: uint,
	allocator := context.allocator,
) -> ^Conv_Layer(T) {
	// Create blocks
	blocks := make([]^MB_Conv(T), depth, allocator)
	for i in 0 ..< depth {
		blocks[i] = new_mb_conv(T, dim, dim, conv_expand_ratio, allocator)
	}

	// Create downsample if needed
	downsample_layer: Maybe(^Patch_Merging(T)) = nil
	if downsample {
		downsample_layer = new_patch_merging(T, input_resolution, dim, out, allocator)
	}

	return new_clone(Conv_Layer(T){blocks = blocks, downsample = downsample_layer}, allocator)
}

forward_conv_layer :: proc(
	layer: ^Conv_Layer($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	xs := x

	// Apply all blocks
	for block in layer.blocks {
		new_xs := forward_mb_conv(block, xs, context.temp_allocator)
		if xs != x { 	// Don't free the input
			// Free intermediate results would go here if needed
		}
		xs = new_xs
	}

	// Apply downsampling if present
	if downsample, has_downsample := layer.downsample.?; has_downsample {
		result := forward_patch_merging(downsample, xs, allocator, loc)
		duration := time.since(start_time)
		fmt.printf(
			"[TIMING] ConvLayer (with %d blocks + downsample): %v\n",
			len(layer.blocks),
			duration,
		)
		return result
	} else {
		// Clone the final result to the target allocator
		result := tensor.clone(xs, allocator)
		duration := time.since(start_time)
		fmt.printf(
			"[TIMING] ConvLayer (with %d blocks, no downsample): %v\n",
			len(layer.blocks),
			duration,
		)
		return result
	}
}

free_conv_layer :: proc(layer: ^Conv_Layer($T), allocator := context.allocator) {
	for block in layer.blocks {
		free_mb_conv(block, allocator)
	}
	delete(layer.blocks, allocator)

	if downsample, has_downsample := layer.downsample.?; has_downsample {
		free_patch_merging(downsample, allocator)
	}
	free(layer, allocator)
}

// Attention mechanism
Attention :: struct($T: typeid) {
	norm:               ^nn.Layer_Norm(T),
	qkv, proj:          ^nn.Linear(T),
	ab:                 ^tensor.Tensor(T), // attention biases
	key_dim, num_heads: uint,
	d:                  uint, // attn_ratio * key_dim
	dh:                 uint, // d * num_heads
	scale:              T,
}

new_attention :: proc(
	$T: typeid,
	dim, key_dim, num_heads, attn_ratio: uint,
	resolution: [2]uint,
	allocator := context.allocator,
) -> ^Attention(T) {
	d := attn_ratio * key_dim
	dh := d * num_heads
	nh_kd := key_dim * num_heads
	h := dh + nh_kd * 2 // query + key + value

	norm := nn.new_layer_norm_1d(T, dim, allocator)
	qkv := nn.new_linear(T, dim, h, true, allocator)
	proj := nn.new_linear(T, dh, dim, true, allocator)

	// Create attention biases - will be resized based on actual sequence length during forward pass
	// For now, create a reasonable default size that can be resized
	max_seq_len := resolution[0] * resolution[1] * 4 // Allow for multiple scales
	ab := tensor.zeros(T, []uint{num_heads, max_seq_len, max_seq_len}, allocator)

	scale := T(1.0 / math.sqrt(f64(key_dim)))

	return new_clone(
		Attention(T) {
			norm = norm,
			qkv = qkv,
			proj = proj,
			ab = ab,
			key_dim = key_dim,
			num_heads = num_heads,
			d = d,
			dh = dh,
			scale = scale,
		},
		allocator,
	)
}

forward_attention :: proc(
	attn: ^Attention($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	b, n := x.shape[0], x.shape[1]

	// Layer norm
	xs := nn.forward_layer_norm(attn.norm, x, context.temp_allocator)

	// QKV projection
	fmt.printf("      QKV projection...\n")
	qkv := nn.forward_linear(attn.qkv, xs, context.temp_allocator)
	qkv_reshaped := tensor.reshape(
		qkv,
		[]uint{b, n, attn.num_heads, attn.key_dim * 2 + attn.d},
		context.temp_allocator,
	)
	fmt.printf("      QKV projection completed\n")

	// Extract Q, K, V from the reshaped QKV tensor
	fmt.printf("      Creating Q, K, V tensors...\n")
	q := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.key_dim}, context.temp_allocator)
	k := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.key_dim}, context.temp_allocator)
	v := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.d}, context.temp_allocator)
	fmt.printf("      Q, K, V tensors created\n")

	// Copy data from QKV tensor to Q, K, V tensors
	fmt.printf("      Starting QKV data copying (b=%d, n=%d, heads=%d)...\n", b, n, attn.num_heads)
	qkv_stride := attn.key_dim * 2 + attn.d
	total_ops := b * n * attn.num_heads
	completed_ops := 0

	for batch in 0 ..< b {
		for pos in 0 ..< n {
			for head in 0 ..< attn.num_heads {
				completed_ops += 1
				if completed_ops % 1000 == 0 {
					fmt.printf(
						"        Completed %d/%d QKV copy operations...\n",
						completed_ops,
						total_ops,
					)
				}

				base_idx := ((batch * n + pos) * attn.num_heads + head) * qkv_stride

				// Copy Q
				for i in 0 ..< attn.key_dim {
					q_idx := ((batch * n + pos) * attn.num_heads + head) * attn.key_dim + i
					q.data[q_idx] = qkv_reshaped.data[base_idx + i]
				}

				// Copy K
				for i in 0 ..< attn.key_dim {
					k_idx := ((batch * n + pos) * attn.num_heads + head) * attn.key_dim + i
					k.data[k_idx] = qkv_reshaped.data[base_idx + attn.key_dim + i]
				}

				// Copy V
				for i in 0 ..< attn.d {
					v_idx := ((batch * n + pos) * attn.num_heads + head) * attn.d + i
					v.data[v_idx] = qkv_reshaped.data[base_idx + attn.key_dim * 2 + i]
				}
			}
		}
	}
	fmt.printf("      QKV data copying completed (%d operations)\n", total_ops)

	// Reshape for attention: (B, N, H, D) -> (B, H, N, D)
	fmt.printf("      Reshaping Q, K, V tensors...\n")
	q_transposed := tensor.permute(q, []uint{0, 2, 1, 3}, context.temp_allocator)
	k_transposed := tensor.permute(k, []uint{0, 2, 1, 3}, context.temp_allocator)
	v_transposed := tensor.permute(v, []uint{0, 2, 1, 3}, context.temp_allocator)
	fmt.printf("      Q, K, V reshape completed\n")

	// Attention computation: Q @ K^T
	fmt.printf("      Computing attention scores (Q @ K^T)...\n")
	k_t := tensor.matrix_transpose(k_transposed, context.temp_allocator)
	fmt.printf("      K transpose completed, starting matmul...\n")
	attn_scores := tensor.matmul(q_transposed, k_t, context.temp_allocator)
	fmt.printf(
		"      Attention scores computed (shape: [%d, %d, %d, %d])\n",
		attn_scores.shape[0],
		attn_scores.shape[1],
		attn_scores.shape[2],
		attn_scores.shape[3],
	)

	// Scale
	fmt.printf("      Scaling attention scores...\n")
	scale_tensor := tensor.new_with_init([]T{attn.scale}, []uint{1}, context.temp_allocator)
	scaled_scores := tensor.mul(attn_scores, scale_tensor, context.temp_allocator)
	fmt.printf("      Scaling completed\n")

	// Add attention bias - slice to match actual sequence length
	fmt.printf(
		"      Adding attention bias (slicing %dx%d to %dx%d)...\n",
		attn.ab.shape[1],
		attn.ab.shape[2],
		n,
		n,
	)

	// Create appropriately sized bias tensor for current sequence length
	current_bias := tensor.zeros(T, []uint{attn.num_heads, n, n}, context.temp_allocator)
	// Copy relevant portion of the pre-allocated bias (for now just use zeros)
	// In a full implementation, this would copy learned relative position biases

	biased_scores := tensor.add(scaled_scores, current_bias, context.temp_allocator)
	fmt.printf("      Bias addition completed\n")

	// Apply softmax to get attention weights
	fmt.printf("      Applying activation (gelu placeholder for softmax)...\n")
	attn_weights := tensor.gelu(biased_scores, context.temp_allocator)
	fmt.printf("      Activation completed\n")

	// Apply attention to values
	fmt.printf("      Computing attention output (attn_weights @ V)...\n")
	attn_output := tensor.matmul(attn_weights, v_transposed, context.temp_allocator)
	fmt.printf("      Attention output computed\n")

	// Reshape back: (B, H, N, D) -> (B, N, H*D)
	output_transposed := tensor.permute(attn_output, []uint{0, 2, 1, 3}, context.temp_allocator)
	output_reshaped := tensor.reshape(
		output_transposed,
		[]uint{b, n, attn.dh},
		context.temp_allocator,
	)

	// Final projection
	result := nn.forward_linear(attn.proj, output_reshaped, allocator, loc)

	duration := time.since(start_time)
	fmt.printf("[TIMING] Attention: %v\n", duration)
	return result
}

free_attention :: proc(attn: ^Attention($T), allocator := context.allocator) {
	nn.free_layer_norm(attn.norm, allocator)
	nn.free_linear(attn.qkv, allocator)
	nn.free_linear(attn.proj, allocator)
	tensor.free_tensor(attn.ab, allocator)
	free(attn, allocator)
}

// MLP block
Mlp :: struct($T: typeid) {
	norm:     ^nn.Layer_Norm(T),
	fc1, fc2: ^nn.Linear(T),
}

new_mlp :: proc(
	$T: typeid,
	in_features, hidden_features: uint,
	allocator := context.allocator,
) -> ^Mlp(T) {
	norm := nn.new_layer_norm_1d(T, in_features, allocator)
	fc1 := nn.new_linear(T, in_features, hidden_features, true, allocator)
	fc2 := nn.new_linear(T, hidden_features, in_features, true, allocator)

	return new_clone(Mlp(T){norm = norm, fc1 = fc1, fc2 = fc2}, allocator)
}

forward_mlp :: proc(
	mlp: ^Mlp($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	norm_out := nn.forward_layer_norm(mlp.norm, x, context.temp_allocator)
	fc1_out := nn.forward_linear(mlp.fc1, norm_out, context.temp_allocator)
	gelu_out := tensor.gelu(fc1_out, context.temp_allocator)
	fc2_out := nn.forward_linear(mlp.fc2, gelu_out, allocator, loc)
	duration := time.since(start_time)
	fmt.printf("[TIMING] MLP: %v\n", duration)
	return fc2_out
}

free_mlp :: proc(mlp: ^Mlp($T), allocator := context.allocator) {
	nn.free_layer_norm(mlp.norm, allocator)
	nn.free_linear(mlp.fc1, allocator)
	nn.free_linear(mlp.fc2, allocator)
	free(mlp, allocator)
}

// TinyViT Block
Tiny_ViT_Block :: struct($T: typeid) {
	attn:             ^Attention(T),
	local_conv:       ^Conv_2d_BN(T),
	mlp:              ^Mlp(T),
	window_size:      uint,
	input_resolution: [2]uint,
}

new_tiny_vit_block :: proc(
	$T: typeid,
	dim: uint,
	input_resolution: [2]uint,
	num_heads, window_size: uint,
	allocator := context.allocator,
) -> ^Tiny_ViT_Block(T) {
	head_dim := dim / num_heads
	attn := new_attention(
		T,
		dim,
		head_dim,
		num_heads,
		1,
		[2]uint{window_size, window_size},
		allocator,
	)
	mlp := new_mlp(T, dim, dim * MLP_RATIO, allocator)
	local_conv := new_conv_2d_bn(
		T,
		dim,
		dim,
		LOCAL_CONV_SIZE,
		1,
		LOCAL_CONV_SIZE / 2,
		dim,
		allocator,
	)

	return new_clone(
		Tiny_ViT_Block(T) {
			attn = attn,
			local_conv = local_conv,
			mlp = mlp,
			window_size = window_size,
			input_resolution = input_resolution,
		},
		allocator,
	)
}

forward_tiny_vit_block :: proc(
	block: ^Tiny_ViT_Block($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	h, w := block.input_resolution[0], block.input_resolution[1]
	b, l, c := x.shape[0], x.shape[1], x.shape[2]

	res_x := x

	window_size := block.window_size
	use_window := !(h == window_size && w == window_size)

	attn_out: ^tensor.Tensor(T)

	if !use_window {
		// Global attention (no windowing)
		fmt.printf("    Using global attention (no windowing)...\n")
		attn_out = forward_attention(block.attn, x, context.temp_allocator)
	} else {
		fmt.printf("    Using windowed attention...\n")
		// Reshape to (B, H, W, C)
		xs := tensor.reshape(x, []uint{b, h, w, c}, context.temp_allocator)

		// Calculate padding needed
		pad_h := (window_size - (h % window_size)) % window_size
		pad_w := (window_size - (w % window_size)) % window_size
		padded_h := h + pad_h
		padded_w := w + pad_w

		// Pad height (dim=1) and width (dim=2) with zeros if needed
		if pad_h > 0 || pad_w > 0 {
			// Pad by creating a new tensor and copying data
			padded_xs := tensor.zeros(T, []uint{b, padded_h, padded_w, c}, context.temp_allocator)
			for batch in 0 ..< b {
				for i in 0 ..< h {
					for j in 0 ..< w {
						for ch in 0 ..< c {
							padded_xs.data[
								batch * padded_h * padded_w * c +
								i * padded_w * c +
								j * c +
								ch
							] = xs.data[
								batch * h * w * c +
								i * w * c +
								j * c +
								ch
							]
						}
					}
				}
			}
			xs = padded_xs
		}

		n_h := padded_h / window_size
		n_w := padded_w / window_size

		// Reshape to windows: (B, n_h, window_size, n_w, window_size, C)
		xs = tensor.reshape(xs, []uint{b, n_h, window_size, n_w, window_size, c}, context.temp_allocator)
		// Transpose to (B, n_h, n_w, window_size, window_size, C)
		xs = tensor.permute(xs, []uint{0, 1, 3, 2, 4, 5}, context.temp_allocator)
		// Merge windows: (B * n_h * n_w, window_size * window_size, C)
		xs = tensor.reshape(xs, []uint{b * n_h * n_w, window_size * window_size, c}, context.temp_allocator)

		// Apply attention per window
		attn_windows := forward_attention(block.attn, xs, context.temp_allocator)

		// Restore windows: (B, n_h, n_w, window_size, window_size, C)
		attn_windows = tensor.reshape(attn_windows, []uint{b, n_h, n_w, window_size, window_size, c}, context.temp_allocator)
		// Transpose back: (B, n_h, window_size, n_w, window_size, C)
		attn_windows = tensor.permute(attn_windows, []uint{0, 1, 3, 2, 4, 5}, context.temp_allocator)
		// Merge spatial: (B, padded_h, padded_w, C)
		attn_windows = tensor.reshape(attn_windows, []uint{b, padded_h, padded_w, c}, context.temp_allocator)

		// Remove padding if needed
		if pad_h > 0 || pad_w > 0 {
			// Slice to original h, w
			attn_no_pad := tensor.zeros(T, []uint{b, h, w, c}, context.temp_allocator)
			for batch in 0 ..< b {
				for i in 0 ..< h {
					for j in 0 ..< w {
						for ch in 0 ..< c {
							attn_no_pad.data[
								batch * h * w * c +
								i * w * c +
								j * c +
								ch
							] = attn_windows.data[
								batch * padded_h * padded_w * c +
								i * padded_w * c +
								j * c +
								ch
							]
						}
					}
				}
			}
			attn_windows = attn_no_pad
		}

		// Reshape back to (B, L, C)
		attn_out = tensor.reshape(attn_windows, []uint{b, l, c}, context.temp_allocator)
	}

	fmt.printf("    Attention completed\n")

	// Residual connection
	fmt.printf("    Adding residual connection...\n")
	xs := tensor.add(attn_out, res_x, context.temp_allocator)
	fmt.printf("    Residual connection completed\n")

	// Reshape for local conv: (B, L, C) -> (B, C, H, W)
	fmt.printf("    Reshaping for local conv...\n")

	// Calculate actual spatial dimensions from sequence length
	actual_l := xs.shape[1]
	actual_spatial_dim := uint(math.sqrt(f64(actual_l)))

	fmt.printf("      Input shape: [%d, %d, %d]\n", xs.shape[0], xs.shape[1], xs.shape[2])
	fmt.printf("      Block expects resolution: [%d, %d]\n", h, w)
	fmt.printf("      Actual sequence length: %d, spatial dim: %d\n", actual_l, actual_spatial_dim)

	// Use actual dimensions instead of block's expected dimensions
	actual_h, actual_w := actual_spatial_dim, actual_spatial_dim

	fmt.printf("      Using actual dimensions: [%d, %d, %d, %d]\n", b, actual_h, actual_w, c)
	fmt.printf("      Starting reshape to 4D...\n")
	xs_4d := tensor.reshape(xs, []uint{b, actual_h, actual_w, c}, context.temp_allocator)
	fmt.printf(
		"      4D reshape completed, shape: [%d, %d, %d, %d]\n",
		xs_4d.shape[0],
		xs_4d.shape[1],
		xs_4d.shape[2],
		xs_4d.shape[3],
	)

	fmt.printf("      Starting permute (BHWC -> BCHW)...\n")
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)
	fmt.printf(
		"      Permute completed, final shape: [%d, %d, %d, %d]\n",
		xs_conv.shape[0],
		xs_conv.shape[1],
		xs_conv.shape[2],
		xs_conv.shape[3],
	)
	fmt.printf("    Reshape completed\n")

	// Apply local convolution
	fmt.printf("    Starting local convolution...\n")
	fmt.printf(
		"      Conv input shape: [%d, %d, %d, %d]\n",
		xs_conv.shape[0],
		xs_conv.shape[1],
		xs_conv.shape[2],
		xs_conv.shape[3],
	)
	fmt.printf(
		"      Conv params: in_ch=%d, out_ch=%d, kernel=%d, groups=%d\n",
		block.local_conv.conv.in_channels,
		block.local_conv.conv.out_channels,
		block.local_conv.conv.kernel_size[0],
		block.local_conv.conv.groups,
	)
	conv_out := forward_conv_2d_bn(block.local_conv, xs_conv, context.temp_allocator)
	fmt.printf("    Local convolution completed\n")

	// Reshape back: (B, C, H, W) -> (B, L, C)
	fmt.printf("    Reshaping back from conv...\n")
	conv_flat := tensor.reshape(conv_out, []uint{b, c, actual_l}, context.temp_allocator)
	conv_final := tensor.transpose(conv_flat, 1, 2, context.temp_allocator)
	fmt.printf("    Reshape back completed\n")

	// Apply MLP with residual
	fmt.printf("    Starting MLP...\n")
	mlp_out := forward_mlp(block.mlp, conv_final, context.temp_allocator)
	fmt.printf("    MLP completed, adding final residual...\n")
	result := tensor.add(conv_final, mlp_out, allocator, loc)
	fmt.printf("    Final residual completed\n")

	duration := time.since(start_time)
	fmt.printf("[TIMING] TinyViT Block: %v\n", duration)
	return result
}

free_tiny_vit_block :: proc(block: ^Tiny_ViT_Block($T), allocator := context.allocator) {
	free_attention(block.attn, allocator)
	free_conv_2d_bn(block.local_conv, allocator)
	free_mlp(block.mlp, allocator)
	free(block, allocator)
}

// BasicLayer - Layer of TinyViT blocks with optional downsampling
Basic_Layer :: struct($T: typeid) {
	blocks:     []^Tiny_ViT_Block(T),
	downsample: Maybe(^Patch_Merging(T)),
}

new_basic_layer :: proc(
	$T: typeid,
	dim, out: uint,
	input_resolution: [2]uint,
	depth, num_heads, window_size: uint,
	downsample: bool,
	allocator := context.allocator,
) -> ^Basic_Layer(T) {
	// Create blocks
	blocks := make([]^Tiny_ViT_Block(T), depth, allocator)
	for i in 0 ..< depth {
		blocks[i] = new_tiny_vit_block(T, dim, input_resolution, num_heads, window_size, allocator)
	}

	// Create downsample if needed
	downsample_layer: Maybe(^Patch_Merging(T)) = nil
	if downsample {
		downsample_layer = new_patch_merging(T, input_resolution, dim, out, allocator)
	}

	return new_clone(Basic_Layer(T){blocks = blocks, downsample = downsample_layer}, allocator)
}

forward_basic_layer :: proc(
	layer: ^Basic_Layer($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	xs := x

	// Apply all blocks
	for i in 0 ..< len(layer.blocks) {
		block := layer.blocks[i]
		fmt.printf("  Starting TinyViT block %d/%d...\n", i + 1, len(layer.blocks))
		new_xs := forward_tiny_vit_block(block, xs, context.temp_allocator)
		xs = new_xs
		fmt.printf("  Completed TinyViT block %d/%d\n", i + 1, len(layer.blocks))
	}

	// Apply downsampling if present
	if downsample, has_downsample := layer.downsample.?; has_downsample {
		result := forward_patch_merging(downsample, xs, allocator, loc)
		duration := time.since(start_time)
		fmt.printf(
			"[] BasicLayer (with %d blocks + downsample): %v\n",
			len(layer.blocks),
			duration,
		)
		return result
	} else {
		// Clone the final result to the target allocator
		result := tensor.clone(xs, allocator)
		duration := time.since(start_time)
		fmt.printf(
			"[TIMING] BasicLayer (with %d blocks, no downsample): %v\n",
			len(layer.blocks),
			duration,
		)
		return result
	}
}

free_basic_layer :: proc(layer: ^Basic_Layer($T), allocator := context.allocator) {
	for block in layer.blocks {
		free_tiny_vit_block(block, allocator)
	}
	delete(layer.blocks, allocator)

	if downsample, has_downsample := layer.downsample.?; has_downsample {
		free_patch_merging(downsample, allocator)
	}
	free(layer, allocator)
}

// Main TinyViT model
Tiny_ViT_5m :: struct($T: typeid) {
	patch_embed:            ^Patch_Embed(T),
	layer0:                 ^Conv_Layer(T),
	layers:                 []^Basic_Layer(T),
	neck_conv1, neck_conv2: ^nn.Conv_2d(T),
	neck_ln1, neck_ln2:     ^nn.Layer_Norm(T),
}

new_tiny_vit_5m :: proc(
	$T: typeid,
	input_size: uint = IMG_SIZE,
	allocator := context.allocator,
) -> ^Tiny_ViT_5m(T) {
	embed_dims := []uint{64, 128, 160, 320}
	depths := []uint{2, 2, 6, 2}
	num_heads := []uint{2, 4, 5, 10}
	window_sizes := []uint{7, 7, 14, 7}

	patch_embed := new_patch_embed(T, IN_CHANNELS, embed_dims[0], allocator)
	patches_resolution := uint(input_size / 4) // After patch embedding

	// Layer 0 (ConvLayer) - downsamples 256->128
	layer0 := new_conv_layer(
		T,
		embed_dims[0],
		embed_dims[1],
		[2]uint{patches_resolution, patches_resolution}, // 256x256
		depths[0],
		true, // downsample
		MBCONV_EXPAND_RATIO,
		allocator,
	)

	// Remaining layers (BasicLayers) - use original patches_resolution for formula
	original_patches_resolution := uint(input_size / 4) // Keep original for layer formula
	num_layers := len(embed_dims)
	layers := make([]^Basic_Layer(T), num_layers - 1, allocator)

	for i_layer in 1 ..< num_layers {
		// Calculate current resolution using Rust formula: original_patches_resolution / (1 << min(i_layer, 2))
		current_resolution := original_patches_resolution / (1 << min(uint(i_layer), 2))

		layer := new_basic_layer(
			T,
			embed_dims[i_layer],
			embed_dims[min(i_layer + 1, num_layers - 1)],
			[2]uint{current_resolution, current_resolution},
			depths[i_layer],
			num_heads[i_layer],
			window_sizes[i_layer],
			i_layer < num_layers - 1, // downsample
			allocator,
		)
		layers[i_layer - 1] = layer
	}

	// Neck layers
	last_embed_dim := embed_dims[len(embed_dims) - 1]
	// Neck: 320 -> 256 channels (SAM compatible)
	neck_conv1 := nn.new_conv2d(
		T,
		last_embed_dim, // 320 input channels
		256, // 256 output channels (SAM standard)
		kernel_size = [2]uint{1, 1},
		stride = 1,
		padding = 0,
		dilation = 1,
		groups = 1,
		use_bias = false,
		allocator = allocator,
	)
	// LayerNorm2d expects spatial dimensions based on final output
	// Final spatial dimension after all downsampling: input_size / 4 / (1 << min(3,2)) = input_size / 16
	final_spatial_dim := uint(input_size / 16)
	neck_ln1 := nn.new_layer_norm_2d(T, []uint{final_spatial_dim, final_spatial_dim}, allocator)
	neck_conv2 := nn.new_conv2d(
		T,
		256, // 256 input channels
		256, // 256 output channels
		[2]uint{3, 3},
		1,
		1,
		1,
		1,
		false,
		allocator,
	)
	neck_ln2 := nn.new_layer_norm_2d(T, []uint{final_spatial_dim, final_spatial_dim}, allocator)

	return new_clone(
		Tiny_ViT_5m(T) {
			patch_embed = patch_embed,
			layer0 = layer0,
			layers = layers,
			neck_conv1 = neck_conv1,
			neck_ln1 = neck_ln1,
			neck_conv2 = neck_conv2,
			neck_ln2 = neck_ln2,
		},
		allocator,
	)
}

forward_tiny_vit_5m :: proc(
	model: ^Tiny_ViT_5m($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()

	// Patch embedding
	patch_start := time.now()
	xs := forward_patch_embed(model.patch_embed, x, context.temp_allocator)
	fmt.printf("[TIMING] Total PatchEmbed stage: %v\n", time.since(patch_start))

	// Layer 0
	layer0_start := time.now()
	xs = forward_conv_layer(model.layer0, xs, context.temp_allocator)
	fmt.printf("[TIMING] Total Layer0 stage: %v\n", time.since(layer0_start))

	// Remaining layers
	for i in 0 ..< len(model.layers) {
		layer := model.layers[i]
		fmt.printf("Starting Layer%d (BasicLayer)...\n", i + 1)
		layer_start := time.now()
		xs = forward_basic_layer(layer, xs, context.temp_allocator)
		fmt.printf("[TIMING] Total Layer%d stage: %v\n", i + 1, time.since(layer_start))
	}

	// Neck: reshape to 4D and apply convolutions
	neck_start := time.now()
	b := xs.shape[0]
	c := xs.shape[2]

	fmt.printf("Neck input shape: [%d, %d, %d]\n", xs.shape[0], xs.shape[1], xs.shape[2])

	// Calculate correct spatial dimensions based on actual sequence length
	// For 1024x1024 input: sequence_length = 4096 (64x64)
	// For 256x256 input: sequence_length = 1024 (32x32)
	sequence_length := xs.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	xs_4d := tensor.reshape(xs, []uint{b, spatial_dim, spatial_dim, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)

	fmt.printf(
		"Neck after reshape/permute: [%d, %d, %d, %d]\n",
		xs_conv.shape[0],
		xs_conv.shape[1],
		xs_conv.shape[2],
		xs_conv.shape[3],
	)

	// Apply neck convolutions with layer norms
	conv1_out := nn.forward_conv2d(model.neck_conv1, xs_conv, context.temp_allocator)
	ln1_out := nn.forward_layer_norm(model.neck_ln1, conv1_out, context.temp_allocator)

	conv2_out := nn.forward_conv2d(model.neck_conv2, ln1_out, context.temp_allocator)
	result := nn.forward_layer_norm(model.neck_ln2, conv2_out, allocator, loc)
	fmt.printf("[TIMING] Total Neck stage: %v\n", time.since(neck_start))

	total_duration := time.since(start_time)
	fmt.printf("[TIMING] TOTAL TinyViT-5M Forward Pass: %v\n", total_duration)
	fmt.printf("=== End Forward Pass ===\n")

	return result
}

free_tiny_vit_5m :: proc(model: ^Tiny_ViT_5m($T), allocator := context.allocator) {
	free_patch_embed(model.patch_embed, allocator)
	free_conv_layer(model.layer0, allocator)

	for layer in model.layers {
		free_basic_layer(layer, allocator)
	}
	delete(model.layers, allocator)

	nn.free_conv_2d(model.neck_conv1, allocator)
	nn.free_layer_norm(model.neck_ln1, allocator)
	nn.free_conv_2d(model.neck_conv2, allocator)
	nn.free_layer_norm(model.neck_ln2, allocator)
	free(model, allocator)
}
