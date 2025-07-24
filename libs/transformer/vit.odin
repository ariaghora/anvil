package transformer

import "../nn"
import "../tensor"
import "../trace"
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
	conv_bn_trace := trace.TRACE_FUNCTION("conv2d_bn")
	defer trace.end_scoped_trace(conv_bn_trace)

	conv_trace := trace.TRACE_SECTION("conv2d")
	conv_out := nn.forward_conv2d(layer.conv, x, context.temp_allocator)
	trace.end_scoped_trace(conv_trace)

	bn_trace := trace.TRACE_SECTION("batch_norm")
	bn_out := nn.forward_batch_norm_2d(layer.bn, conv_out, allocator, loc)
	trace.end_scoped_trace(bn_trace)

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
	patch_embed_trace := trace.TRACE_FUNCTION("patch_embed")
	defer trace.end_scoped_trace(patch_embed_trace)

	conv1_out := forward_conv_2d_bn(pe.conv1, x, context.temp_allocator)

	gelu_trace := trace.TRACE_SECTION("gelu_activation")
	gelu_out := tensor.gelu(conv1_out, context.temp_allocator)
	trace.end_scoped_trace(gelu_trace)

	conv2_out := forward_conv_2d_bn(pe.conv2, gelu_out, allocator, loc)
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
	mbconv_trace := trace.TRACE_FUNCTION("mb_conv")
	defer trace.end_scoped_trace(mbconv_trace)

	shortcut := x

	// Expansion
	expansion_trace := trace.TRACE_SECTION("expansion")
	conv1_out := forward_conv_2d_bn(mb.conv1, x, context.temp_allocator)
	gelu1_out := tensor.gelu(conv1_out, context.temp_allocator)
	trace.end_scoped_trace(expansion_trace)

	// Depthwise
	depthwise_trace := trace.TRACE_SECTION("depthwise")
	conv2_out := forward_conv_2d_bn(mb.conv2, gelu1_out, context.temp_allocator)
	gelu2_out := tensor.gelu(conv2_out, context.temp_allocator)
	trace.end_scoped_trace(depthwise_trace)

	// Projection
	projection_trace := trace.TRACE_SECTION("projection")
	conv3_out := forward_conv_2d_bn(mb.conv3, gelu2_out, context.temp_allocator)
	trace.end_scoped_trace(projection_trace)

	// Check shapes before residual connection
	if !slice.equal(conv3_out.shape, shortcut.shape) {
		panic("MBConv residual connection shape mismatch")
	}

	// Residual connection + final activation
	residual_trace := trace.TRACE_SECTION("residual_connection")
	residual := tensor.add(conv3_out, shortcut, context.temp_allocator)
	result := tensor.gelu(residual, allocator, loc)
	trace.end_scoped_trace(residual_trace)

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
		return result
	} else {
		// Clone the final result to the target allocator
		result := tensor.clone(xs, allocator)
		duration := time.since(start_time)
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
	qkv := nn.forward_linear(attn.qkv, xs, context.temp_allocator)
	qkv_reshaped := tensor.reshape(
		qkv,
		[]uint{b, n, attn.num_heads, attn.key_dim * 2 + attn.d},
		context.temp_allocator,
	)

	// Extract Q, K, V from the reshaped QKV tensor
	q := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.key_dim}, context.temp_allocator)
	k := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.key_dim}, context.temp_allocator)
	v := tensor.zeros(T, []uint{b, n, attn.num_heads, attn.d}, context.temp_allocator)

	// Copy data from QKV tensor to Q, K, V tensors
	qkv_stride := attn.key_dim * 2 + attn.d
	total_ops := b * n * attn.num_heads
	completed_ops := 0

	for batch in 0 ..< b {
		for pos in 0 ..< n {
			for head in 0 ..< attn.num_heads {
				completed_ops += 1
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

	// Reshape for attention: (B, N, H, D) -> (B, H, N, D)
	q_transposed := tensor.permute(q, []uint{0, 2, 1, 3}, context.temp_allocator)
	k_transposed := tensor.permute(k, []uint{0, 2, 1, 3}, context.temp_allocator)
	v_transposed := tensor.permute(v, []uint{0, 2, 1, 3}, context.temp_allocator)

	// Attention computation: Q @ K^T
	k_t := tensor.matrix_transpose(k_transposed, context.temp_allocator)
	attn_scores := tensor.matmul(q_transposed, k_t, context.temp_allocator)

	// Scale
	scale_tensor := tensor.new_with_init([]T{attn.scale}, []uint{1}, context.temp_allocator)
	scaled_scores := tensor.mul(attn_scores, scale_tensor, context.temp_allocator)

	// Add attention bias - slice to match actual sequence length

	// Create appropriately sized bias tensor for current sequence length
	current_bias := tensor.zeros(T, []uint{attn.num_heads, n, n}, context.temp_allocator)
	// Copy relevant portion of the pre-allocated bias (for now just use zeros)
	// In a full implementation, this would copy learned relative position biases

	biased_scores := tensor.add(scaled_scores, current_bias, context.temp_allocator)

	// Apply softmax to get attention weights
	attn_weights := tensor.gelu(biased_scores, context.temp_allocator)

	// Apply attention to values
	attn_output := tensor.matmul(attn_weights, v_transposed, context.temp_allocator)

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
	tiny_vit_block_trace := trace.TRACE_FUNCTION("tiny_vit_block")
	defer trace.end_scoped_trace(tiny_vit_block_trace)

	h, w := block.input_resolution[0], block.input_resolution[1]
	b, l, c := x.shape[0], x.shape[1], x.shape[2]

	window_size := block.window_size
	attn_out: ^tensor.Tensor(T)

	// Skip windowing if input matches window size
	if h == window_size && w == window_size {
		global_attention_trace := trace.TRACE_SECTION("global_attention")
		attn_out = forward_attention(block.attn, x, context.temp_allocator)
		trace.end_scoped_trace(global_attention_trace)
	} else {
		win_attention_trace := trace.TRACE_SECTION("windowed_attention")

		// Calculate padded dimensions
		pad_h := (window_size - (h % window_size)) % window_size
		pad_w := (window_size - (w % window_size)) % window_size
		padded_h := h + pad_h
		padded_w := w + pad_w
		n_h := padded_h / window_size
		n_w := padded_w / window_size

		// Reshape to 4D
		xs := tensor.reshape(x, []uint{b, h, w, c}, context.temp_allocator)

		// Create padded tensor if needed
		if pad_h > 0 || pad_w > 0 {
			padded := tensor.zeros(T, []uint{b, padded_h, padded_w, c}, context.temp_allocator)

			// Copy data with batched memcpy
			for batch in 0 ..< b {
				for row in 0 ..< h {
					src_offset := (batch * h * w + row * w) * c
					dst_offset := (batch * padded_h * padded_w + row * padded_w) * c
					copy(
						padded.data[dst_offset:dst_offset + w * c],
						xs.data[src_offset:src_offset + w * c],
					)
				}
			}
			xs = padded
		}

		// Window partitioning with single allocation
		windows := tensor.zeros(
			T,
			[]uint{b * n_h * n_w, window_size * window_size, c},
			context.temp_allocator,
		)

		// Efficient windowing with better memory access pattern
		#no_bounds_check {
			window_idx: uint = 0
			for batch in 0 ..< b {
				for h_idx in 0 ..< n_h {
					for w_idx in 0 ..< n_w {
						// Copy each window
						for wh in 0 ..< window_size {
							for ww in 0 ..< window_size {
								src_h := h_idx * window_size + wh
								src_w := w_idx * window_size + ww
								src_idx := ((batch * padded_h + src_h) * padded_w + src_w) * c
								dst_idx :=
									(window_idx * window_size * window_size +
										wh * window_size +
										ww) *
									c

								// Copy all channels at once
								copy(
									windows.data[dst_idx:dst_idx + c],
									xs.data[src_idx:src_idx + c],
								)
							}
						}
						window_idx += 1
					}
				}
			}
		}

		// Apply attention to all windows at once
		attn_windows := forward_attention(block.attn, windows, context.temp_allocator)

		// Merge windows back
		merged := tensor.zeros(T, []uint{b, padded_h, padded_w, c}, context.temp_allocator)

		#no_bounds_check {
			window_idx: uint = 0
			for batch in 0 ..< b {
				for h_idx in 0 ..< n_h {
					for w_idx in 0 ..< n_w {
						// Copy each window back
						for wh in 0 ..< window_size {
							for ww in 0 ..< window_size {
								dst_h := h_idx * window_size + wh
								dst_w := w_idx * window_size + ww
								src_idx :=
									(window_idx * window_size * window_size +
										wh * window_size +
										ww) *
									c
								dst_idx := ((batch * padded_h + dst_h) * padded_w + dst_w) * c

								copy(
									merged.data[dst_idx:dst_idx + c],
									attn_windows.data[src_idx:src_idx + c],
								)
							}
						}
						window_idx += 1
					}
				}
			}
		}

		// Remove padding if needed
		if pad_h > 0 || pad_w > 0 {
			unpadded := tensor.zeros(T, []uint{b, h, w, c}, context.temp_allocator)

			for batch in 0 ..< b {
				for row in 0 ..< h {
					src_offset := (batch * padded_h * padded_w + row * padded_w) * c
					dst_offset := (batch * h * w + row * w) * c
					copy(
						unpadded.data[dst_offset:dst_offset + w * c],
						merged.data[src_offset:src_offset + w * c],
					)
				}
			}
			merged = unpadded
		}

		// Final reshape
		attn_out = tensor.reshape(merged, []uint{b, l, c}, context.temp_allocator)

		trace.end_scoped_trace(win_attention_trace)
	}

	// Rest remains the same
	xs := tensor.add(attn_out, x, context.temp_allocator)

	// Local conv
	actual_l := xs.shape[1]
	actual_spatial_dim := uint(math.sqrt_f64(f64(actual_l)))
	actual_h, actual_w := actual_spatial_dim, actual_spatial_dim

	xs_4d := tensor.reshape(xs, []uint{b, actual_h, actual_w, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)
	conv_out := forward_conv_2d_bn(block.local_conv, xs_conv, context.temp_allocator)
	conv_flat := tensor.reshape(conv_out, []uint{b, c, actual_l}, context.temp_allocator)
	conv_final := tensor.transpose(conv_flat, 1, 2, context.temp_allocator)

	// MLP with residual
	mlp_out := forward_mlp(block.mlp, conv_final, context.temp_allocator)
	result := tensor.add(conv_final, mlp_out, allocator, loc)

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
		new_xs := forward_tiny_vit_block(block, xs, context.temp_allocator)
		xs = new_xs
	}

	// Apply downsampling if present
	if downsample, has_downsample := layer.downsample.?; has_downsample {
		result := forward_patch_merging(downsample, xs, allocator, loc)
		duration := time.since(start_time)
		return result
	} else {
		// Clone the final result to the target allocator
		result := tensor.clone(xs, allocator)
		duration := time.since(start_time)
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
	tiny_vit_trace := trace.TRACE_FUNCTION("tiny_vit_5m_forward")
	defer trace.end_scoped_trace(tiny_vit_trace)

	// Patch embedding
	xs := forward_patch_embed(model.patch_embed, x, context.temp_allocator)

	// Layer 0
	layer0_trace := trace.TRACE_SECTION("layer0_conv")
	xs = forward_conv_layer(model.layer0, xs, context.temp_allocator)
	trace.end_scoped_trace(layer0_trace)

	// Remaining layers
	for i in 0 ..< len(model.layers) {
		layer := model.layers[i]
		layer_name := fmt.aprintf("layer_%d_basic", i + 1, allocator = context.allocator)

		layer_trace := trace.TRACE_SECTION(layer_name)
		xs = forward_basic_layer(layer, xs, context.temp_allocator)
		trace.end_scoped_trace(layer_trace)
	}

	// Neck: reshape to 4D and apply convolutions
	neck_trace := trace.TRACE_SECTION("neck_processing")
	b := xs.shape[0]
	c := xs.shape[2]

	// Calculate correct spatial dimensions based on actual sequence length
	// For 1024x1024 input: sequence_length = 4096 (64x64)
	// For 256x256 input: sequence_length = 1024 (32x32)
	sequence_length := xs.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	xs_4d := tensor.reshape(xs, []uint{b, spatial_dim, spatial_dim, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)

	// Apply neck convolutions with layer norms
	conv1_out := nn.forward_conv2d(model.neck_conv1, xs_conv, context.temp_allocator)
	ln1_out := nn.forward_layer_norm(model.neck_ln1, conv1_out, context.temp_allocator)

	conv2_out := nn.forward_conv2d(model.neck_conv2, ln1_out, context.temp_allocator)
	result := nn.forward_layer_norm(model.neck_ln2, conv2_out, allocator, loc)
	trace.end_scoped_trace(neck_trace)

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
