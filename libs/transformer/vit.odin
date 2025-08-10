package transformer

import "../nn"
import "../tensor"
import "../trace"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:simd"
import "core:slice"
import "core:time"

// Constants from the Rust implementation
MBCONV_EXPAND_RATIO :: 4
MLP_RATIO :: 4
LOCAL_CONV_SIZE :: 3
IMG_SIZE :: 1024
IN_CHANNELS :: 3
GELU_UNFOLD_FACTOR :: 8

// Conv2dBN - Convolution followed by BatchNorm
Conv_2d_BN :: struct($T: typeid) {
	conv: ^nn.Conv_2d(T),
	bn:   ^nn.Batch_Norm_2d(T),
}

tanh_fast :: proc(x: $T) -> T where T == f32 || T == f64 {
	// Pade approximation of tanh
	x2 := x * x
	a := x * (T(135135) + x2 * (T(17325) + x2 * (T(378) + x2)))
	b := T(135135) + x2 * (T(62370) + x2 * (T(3150) + x2 * T(28)))
	return a / b
}

// GELU approximation, inspired from from BERT
gelu_fast_no_simd :: proc(
	x: ^tensor.Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) where T == f32 ||
	T == f64 {
	gelu_fast_trace := trace.TRACE_FUNCTION("gelu_fast")
	defer trace.end_scoped_trace(gelu_fast_trace)

	result := tensor.tensor_alloc(T, x.shape, true, allocator, loc)

	// Constants for sigmoid approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	// Can be approximated as: x * sigmoid(1.702 * x)
	scale := T(1.702)

	#no_bounds_check {
		i := uint(0)
		n := uint(len(x.data))

		for ; i + GELU_UNFOLD_FACTOR <= n; i += GELU_UNFOLD_FACTOR {
			#unroll for j in 0 ..< GELU_UNFOLD_FACTOR {
				val := x.data[i + uint(j)]
				sigmoid_input := scale * val
				sigmoid := T(0.5) * (T(1.0) + tanh_fast(sigmoid_input * T(0.5)))
				result.data[i + uint(j)] = val * sigmoid
			}
		}

		// Handle remainder
		for ; i < n; i += 1 {
			val := x.data[i]
			sigmoid_input := scale * val
			sigmoid := T(0.5) * (T(1.0) + tanh_fast(sigmoid_input * T(0.5)))
			result.data[i] = val * sigmoid
		}
	}

	return result
}

gelu_fast :: proc(
	x: ^tensor.Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	result := tensor.tensor_alloc(T, x.shape, true, allocator, loc)
	total_elements := len(x.data)

	when T == f32 {
		// In the name of God...
		// Look. I'm sorry if this is fucked up. But if anything, see tanh_fast_simd_4xf32.
		// This is just my attempt to be fast.
		sqrt_2_over_pi := f32(0.7978845608028654)
		coeff := f32(0.044715)
		half := f32(0.5)
		one := f32(1.0)

		i := 0

		// SIMD path for chunks of 4
		for ; i + 4 <= total_elements; i += 4 {
			v := (^#simd[4]f32)(&x.data[i])^

			// Compute argument to tanh: sqrt_2_over_pi * v * (1.0 + 0.044715 * v * v)
			v2 := simd.mul(v, v)
			inner := simd.fma(
				v2,
				#simd[4]f32{coeff, coeff, coeff, coeff},
				#simd[4]f32{one, one, one, one},
			)
			arg := simd.mul(
				simd.mul(v, inner),
				#simd[4]f32{sqrt_2_over_pi, sqrt_2_over_pi, sqrt_2_over_pi, sqrt_2_over_pi},
			)

			// Apply tanh approximation
			tanh_result := tanh_fast_simd_4xf32(arg)

			// Final GELU: 0.5 * v * (1.0 + tanh_result)
			gelu_result := simd.mul(
				simd.mul(v, #simd[4]f32{half, half, half, half}),
				simd.add(#simd[4]f32{one, one, one, one}, tanh_result),
			)

			(^#simd[4]f32)(&result.data[i])^ = gelu_result
		}

		// Scalar fallback for remainder
		for ; i < total_elements; i += 1 {
			v := x.data[i]
			result.data[i] =
				0.5 * v * (1.0 + math.tanh(sqrt_2_over_pi * v * (1.0 + 0.044715 * v * v)))
		}

	} else {
		// Non-SIMD path for other types
		sqrt_2_over_pi := T(0.7978845608028654)
		for i in 0 ..< total_elements {
			v := x.data[i]
			result.data[i] =
				0.5 * v * (1.0 + math.tanh(sqrt_2_over_pi * v * (1.0 + 0.044715 * v * v)))
		}
	}

	return result
}

tanh_fast_simd_4xf32 :: proc(x: #simd[4]f32) -> #simd[4]f32 {
	// NOTE(Aria): idk why this works okay-ish. Especially compared to huggingface's 
	// SAM with tiny ViT.
	max_val := #simd[4]f32{3.0, 3.0, 3.0, 3.0}
	min_val := #simd[4]f32{-3.0, -3.0, -3.0, -3.0}
	x_clamped := simd.min(simd.max(x, min_val), max_val)

	x2 := simd.mul(x_clamped, x_clamped)
	c27 := #simd[4]f32{27.0, 27.0, 27.0, 27.0}
	c9 := #simd[4]f32{9.0, 9.0, 9.0, 9.0}

	numerator := simd.mul(x_clamped, simd.add(c27, x2))
	denominator := simd.fma(x2, c9, c27)

	return simd.div(numerator, denominator)
}


new_conv_2d_bn :: proc(
	$T: typeid,
	in_channels, out_channels: uint,
	kernel_size: uint,
	stride: uint = 1,
	padding: uint = 0,
	groups: uint = 1,
	init := true,
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
		use_bias = false,
		init = init,
		allocator = allocator,
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
	init := true,
	allocator := context.allocator,
) -> ^Patch_Embed(T) {
	// stride=2, padding=1, kernel_size=3
	conv1 := new_conv_2d_bn(T, in_channels, embed_dim / 2, 3, 2, 1, 1, init, allocator)
	conv2 := new_conv_2d_bn(T, embed_dim / 2, embed_dim, 3, 2, 1, 1, init, allocator)

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
	gelu_out := gelu_fast(conv1_out, context.temp_allocator)
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
	init := true,
	allocator := context.allocator,
) -> ^MB_Conv(T) {
	hidden := in_channels * expand_ratio

	// Pointwise expansion
	conv1 := new_conv_2d_bn(T, in_channels, hidden, 1, 1, 0, 1, init, allocator)
	// Depthwise convolution with groups=hidden
	conv2 := new_conv_2d_bn(T, hidden, hidden, 3, 1, 1, hidden, init, allocator)
	// Pointwise projection
	conv3 := new_conv_2d_bn(T, hidden, out_channels, 1, 1, 0, 1, init, allocator)

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
	gelu1_out := gelu_fast(conv1_out, context.temp_allocator)
	trace.end_scoped_trace(expansion_trace)

	// Depthwise
	depthwise_trace := trace.TRACE_SECTION("depthwise")
	conv2_out := forward_conv_2d_bn(mb.conv2, gelu1_out, context.temp_allocator)
	gelu2_out := gelu_fast(conv2_out, context.temp_allocator)
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
	result := gelu_fast(residual, allocator, loc)
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
	init := true,
	allocator := context.allocator,
) -> ^Patch_Merging(T) {
	// Determine stride based on output channels (matching Rust logic)
	stride: uint = 2
	if out == 320 || out == 448 || out == 576 {
		stride = 1
	}

	conv1 := new_conv_2d_bn(T, dim, out, 1, 1, 0, 1, init, allocator)
	conv2 := new_conv_2d_bn(T, out, out, 3, stride, 1, out, init, allocator) // groups=out (depthwise)
	conv3 := new_conv_2d_bn(T, out, out, 1, 1, 0, 1, init, allocator)

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
	gelu1_out := gelu_fast(conv1_out, context.temp_allocator)

	conv2_out := forward_conv_2d_bn(pm.conv2, gelu1_out, context.temp_allocator)
	gelu2_out := gelu_fast(conv2_out, context.temp_allocator)

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
	init := true,
	allocator := context.allocator,
) -> ^Conv_Layer(T) {
	// Create blocks
	blocks := make([]^MB_Conv(T), depth, allocator)
	for i in 0 ..< depth {
		blocks[i] = new_mb_conv(T, dim, dim, conv_expand_ratio, init, allocator)
	}

	// Create downsample if needed
	downsample_layer: Maybe(^Patch_Merging(T)) = nil
	if downsample {
		downsample_layer = new_patch_merging(T, input_resolution, dim, out, init, allocator)
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
	init := true,
	allocator := context.allocator,
) -> ^Attention(T) {
	d := attn_ratio * key_dim
	dh := d * num_heads
	nh_kd := key_dim * num_heads
	h := dh + nh_kd * 2 // query + key + value

	norm := nn.new_layer_norm_1d(T, dim, allocator)
	qkv := nn.new_linear(T, dim, h, true, init, allocator)
	proj := nn.new_linear(T, dh, dim, true, init, allocator)

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
	forward_attention_trace := trace.TRACE_FUNCTION("forward_attention")
	defer trace.end_scoped_trace(forward_attention_trace)

	b, n := x.shape[0], x.shape[1]
	h := attn.num_heads
	d_k := attn.key_dim
	d_v := attn.d

	// Layer norm
	xs := nn.forward_layer_norm_1d(attn.norm, x, context.temp_allocator)

	// QKV projection
	qkv := nn.forward_linear(attn.qkv, xs, context.temp_allocator)

	// Reshape QKV for easier slicing
	qkv_reshaped := tensor.reshape(qkv, []uint{b, n, h, 2 * d_k + d_v}, context.temp_allocator)

	// Create views for Q, K, V without copying data
	// This is the key optimization - use tensor slicing/views instead of copying
	q_shape := []uint{b, n, h, d_k}
	k_shape := []uint{b, n, h, d_k}
	v_shape := []uint{b, n, h, d_v}

	// Create Q, K, V as views into the QKV tensor (if your tensor library supports it)
	// Otherwise, we need to copy but can do it more efficiently
	q := tensor.zeros(T, q_shape, context.temp_allocator)
	k := tensor.zeros(T, k_shape, context.temp_allocator)
	v := tensor.zeros(T, v_shape, context.temp_allocator)

	// Optimized copy - do it in one pass with better memory access
	qkv_copy_trace := trace.TRACE_FUNCTION("qkv_copy_matmul")
	total_elements := b * n * h
	#no_bounds_check for i in 0 ..< total_elements {
		batch_idx := i / (n * h)
		remainder := i % (n * h)
		pos_idx := remainder / h
		head_idx := remainder % h

		base_src := i * (2 * d_k + d_v)
		base_q := i * d_k
		base_k := i * d_k
		base_v := i * d_v

		// Copy Q
		copy(q.data[base_q:base_q + d_k], qkv_reshaped.data[base_src:base_src + d_k])

		// Copy K
		copy(k.data[base_k:base_k + d_k], qkv_reshaped.data[base_src + d_k:base_src + 2 * d_k])

		// Copy V
		copy(
			v.data[base_v:base_v + d_v],
			qkv_reshaped.data[base_src + 2 * d_k:base_src + 2 * d_k + d_v],
		)
	}
	trace.end_scoped_trace(qkv_copy_trace)

	qkv_permute_transpose_matmul_trace := trace.TRACE_FUNCTION("qkv_permute_transpose_matmul")
	// Reshape for attention: (B, N, H, D) -> (B, H, N, D)
	q_transposed := tensor.permute(q, []uint{0, 2, 1, 3}, context.temp_allocator)
	k_transposed := tensor.permute(k, []uint{0, 2, 1, 3}, context.temp_allocator)
	v_transposed := tensor.permute(v, []uint{0, 2, 1, 3}, context.temp_allocator)

	// Attention computation: Q @ K^T - USE BLAS!
	k_t := tensor.matrix_transpose(k_transposed, context.temp_allocator)
	attn_scores := tensor.matmul(q_transposed, k_t, context.temp_allocator)
	trace.end_scoped_trace(qkv_permute_transpose_matmul_trace)

	// Scale scores in-place
	scale := attn.scale
	#no_bounds_check for i in 0 ..< b * h * n * n {
		attn_scores.data[i] *= scale
	}

	// NOTE(Claude): Skip the bias tensor allocation
	// NOTE(Aria): Fuck it, implement after this

	// Softmax - do it in-place to avoid allocation
	// Process each (batch, head) separately
	attention_softmax_trace := trace.TRACE_FUNCTION("attention_softmax")
	#no_bounds_check for bh in 0 ..< b * h {
		for row in 0 ..< n {
			row_offset := bh * n * n + row * n
			row_data := attn_scores.data[row_offset:][:n]

			when T == f32 {
				// Find max using SIMD
				col := uint(0)
				max_vec := #simd[4]f32 {
					math.inf_f32(-1),
					math.inf_f32(-1),
					math.inf_f32(-1),
					math.inf_f32(-1),
				}

				for ; col + 4 <= n; col += 4 {
					vals := (^#simd[4]f32)(&row_data[col])^
					max_vec = simd.max(max_vec, vals)
				}

				// Reduce max_vec to scalar
				max_val := max(
					max(simd.extract(max_vec, 0), simd.extract(max_vec, 1)),
					max(simd.extract(max_vec, 2), simd.extract(max_vec, 3)),
				)

				// Handle remainder
				for ; col < n; col += 1 {
					max_val = max(max_val, row_data[col])
				}

				sum := f32(0)
				col = 0
				sum_vec := #simd[4]f32{0, 0, 0, 0}

				for ; col + 4 <= n; col += 4 {
					vals := (^#simd[4]f32)(&row_data[col])^

					// Have to extract for exp :(
					exp_vals: #simd[4]f32
					exp_vals = simd.replace(exp_vals, 0, math.exp(simd.extract(vals, 0) - max_val))
					exp_vals = simd.replace(exp_vals, 1, math.exp(simd.extract(vals, 1) - max_val))
					exp_vals = simd.replace(exp_vals, 2, math.exp(simd.extract(vals, 2) - max_val))
					exp_vals = simd.replace(exp_vals, 3, math.exp(simd.extract(vals, 3) - max_val))


					(^#simd[4]f32)(&row_data[col])^ = exp_vals
					sum_vec = simd.add(sum_vec, exp_vals)
				}

				// Sum the vector elements
				sum =
					simd.extract(sum_vec, 0) +
					simd.extract(sum_vec, 1) +
					simd.extract(sum_vec, 2) +
					simd.extract(sum_vec, 3)

				// Handle remainder
				for ; col < n; col += 1 {
					val := math.exp(row_data[col] - max_val)
					row_data[col] = val
					sum += val
				}

				// Normalize with SIMD - this part is fully SIMD
				inv_sum := f32(1) / sum
				inv_sum_vec := #simd[4]f32{inv_sum, inv_sum, inv_sum, inv_sum}

				col = 0
				for ; col + 4 <= n; col += 4 {
					vals := (^#simd[4]f32)(&row_data[col])^
					vals = simd.mul(vals, inv_sum_vec)
					(^#simd[4]f32)(&row_data[col])^ = vals
				}

				// Handle remainder
				for ; col < n; col += 1 {
					row_data[col] *= inv_sum
				}

			} else when T == f64 {
				// Similar with 2-wide SIMD
				col := uint(0)
				max_vec := #simd[2]f64{math.inf_f64(-1), math.inf_f64(-1)}

				for ; col + 2 <= n; col += 2 {
					vals := (^#simd[2]f64)(&row_data[col])^
					max_vec = simd.max(max_vec, vals)
				}

				max_val := max(simd.extract(max_vec, 0), simd.extract(max_vec, 1))

				for ; col < n; col += 1 {
					max_val = max(max_val, row_data[col])
				}

				sum := f64(0)
				col = 0
				sum_vec := #simd[2]f64{0, 0}

				for ; col + 2 <= n; col += 2 {
					vals := (^#simd[2]f64)(&row_data[col])^

					exp_vals: #simd[2]f64
					exp_vals = simd.replace(exp_vals, 0, math.exp(simd.extract(vals, 0) - max_val))
					exp_vals = simd.replace(exp_vals, 1, math.exp(simd.extract(vals, 1) - max_val))


					(^#simd[2]f64)(&row_data[col])^ = exp_vals
					sum_vec = simd.add(sum_vec, exp_vals)
				}

				sum = simd.extract(sum_vec, 0) + simd.extract(sum_vec, 1)

				for ; col < n; col += 1 {
					val := math.exp(row_data[col] - max_val)
					row_data[col] = val
					sum += val
				}

				inv_sum := f64(1) / sum
				inv_sum_vec := #simd[2]f64{inv_sum, inv_sum}

				col = 0
				for ; col + 2 <= n; col += 2 {
					vals := (^#simd[2]f64)(&row_data[col])^
					vals = simd.mul(vals, inv_sum_vec)
					(^#simd[2]f64)(&row_data[col])^ = vals
				}

				for ; col < n; col += 1 {
					row_data[col] *= inv_sum
				}
			} else {
				// Original scalar code
				max_val := row_data[0]
				for col in 1 ..< n {
					if row_data[col] > max_val {
						max_val = row_data[col]
					}
				}

				sum := T(0)
				for col in 0 ..< n {
					val := math.exp(row_data[col] - max_val)
					row_data[col] = val
					sum += val
				}

				inv_sum := T(1) / sum
				for col in 0 ..< n {
					row_data[col] *= inv_sum
				}
			}
		}
	}
	trace.end_scoped_trace(attention_softmax_trace)

	// Apply attention to values - USE BLAS!
	attn_output := tensor.matmul(attn_scores, v_transposed, context.temp_allocator)

	// Reshape back: (B, H, N, D) -> (B, N, H*D)
	output_transposed := tensor.permute(attn_output, []uint{0, 2, 1, 3}, context.temp_allocator)
	output_reshaped := tensor.reshape(
		output_transposed,
		[]uint{b, n, h * d_v},
		context.temp_allocator,
	)

	// Final projection
	result := nn.forward_linear(attn.proj, output_reshaped, allocator, loc)

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
	init := true,
	allocator := context.allocator,
) -> ^Mlp(T) {
	norm := nn.new_layer_norm_1d(T, in_features, allocator)
	fc1 := nn.new_linear(T, in_features, hidden_features, true, init, allocator)
	fc2 := nn.new_linear(T, hidden_features, in_features, true, init, allocator)

	return new_clone(Mlp(T){norm = norm, fc1 = fc1, fc2 = fc2}, allocator)
}

forward_mlp :: proc(
	mlp: ^Mlp($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	start_time := time.now()
	norm_out := nn.forward_layer_norm_1d(mlp.norm, x, context.temp_allocator)
	fc1_out := nn.forward_linear(mlp.fc1, norm_out, context.temp_allocator)
	gelu_out := gelu_fast(fc1_out, context.temp_allocator)
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
	init := true,
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
		init,
		allocator,
	)
	mlp := new_mlp(T, dim, dim * MLP_RATIO, init, allocator)
	local_conv := new_conv_2d_bn(
		T,
		dim,
		dim,
		LOCAL_CONV_SIZE,
		1,
		LOCAL_CONV_SIZE / 2,
		dim,
		init,
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

	// Skip windowing if input matches window size
	if h == window_size && w == window_size {
		global_attention_trace := trace.TRACE_SECTION("global_attention")
		attn_out := forward_attention(block.attn, x, context.temp_allocator)
		trace.end_scoped_trace(global_attention_trace)

		// Add residual
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

	win_attention_trace := trace.TRACE_SECTION("windowed_attention")

	// Calculate padding
	pad_h := (window_size - (h % window_size)) % window_size
	pad_w := (window_size - (w % window_size)) % window_size
	padded_h := h + pad_h
	padded_w := w + pad_w
	n_h := padded_h / window_size
	n_w := padded_w / window_size
	n_windows := b * n_h * n_w

	// Single allocation for windowed data
	windows := tensor.zeros(
		T,
		[]uint{n_windows, window_size * window_size, c},
		context.temp_allocator,
	)

	// Extract windows with inline padding (no intermediate tensor)
	#no_bounds_check {
		window_idx := uint(0)
		for batch in 0 ..< b {
			batch_offset := batch * h * w * c
			for h_win in 0 ..< n_h {
				for w_win in 0 ..< n_w {
					win_h_start := h_win * window_size
					win_w_start := w_win * window_size

					// Extract window
					for local_h in 0 ..< window_size {
						src_h := win_h_start + local_h
						if src_h >= h {continue} 	// Padding region

						for local_w in 0 ..< window_size {
							src_w := win_w_start + local_w
							if src_w >= w {continue} 	// Padding region

							src_idx := batch_offset + (src_h * w + src_w) * c
							dst_idx :=
								(window_idx * window_size * window_size +
									local_h * window_size +
									local_w) *
								c

							// Copy channels - unroll for better performance
							i := uint(0)
							for ; i + 8 <= c; i += 8 {
								#unroll for j in 0 ..< 8 {
									windows.data[dst_idx + i + uint(j)] =
										x.data[src_idx + i + uint(j)]
								}
							}
							for ; i < c; i += 1 {
								windows.data[dst_idx + i] = x.data[src_idx + i]
							}
						}
					}
					window_idx += 1
				}
			}
		}
	}

	// Apply attention
	attn_windows := forward_attention(block.attn, windows, context.temp_allocator)

	// Merge windows back AND add residual in one pass
	result_3d := tensor.zeros(T, []uint{b, l, c}, context.temp_allocator)

	#no_bounds_check {
		window_idx := uint(0)
		for batch in 0 ..< b {
			batch_offset := batch * h * w * c
			for h_win in 0 ..< n_h {
				for w_win in 0 ..< n_w {
					win_h_start := h_win * window_size
					win_w_start := w_win * window_size

					// Merge window back
					for local_h in 0 ..< window_size {
						src_h := win_h_start + local_h
						if src_h >= h {continue} 	// Skip padding

						for local_w in 0 ..< window_size {
							src_w := win_w_start + local_w
							if src_w >= w {continue} 	// Skip padding

							src_idx :=
								(window_idx * window_size * window_size +
									local_h * window_size +
									local_w) *
								c
							dst_idx := batch_offset + (src_h * w + src_w) * c

							// Merge + residual in one pass
							i := uint(0)
							for ; i + 8 <= c; i += 8 {
								#unroll for j in 0 ..< 8 {
									idx := dst_idx + i + uint(j)
									result_3d.data[idx] =
										attn_windows.data[src_idx + i + uint(j)] + x.data[idx]
								}
							}
							for ; i < c; i += 1 {
								idx := dst_idx + i
								result_3d.data[idx] = attn_windows.data[src_idx + i] + x.data[idx]
							}
						}
					}
					window_idx += 1
				}
			}
		}
	}

	trace.end_scoped_trace(win_attention_trace)

	// Local conv
	xs_4d := tensor.reshape(result_3d, []uint{b, h, w, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)
	conv_out := forward_conv_2d_bn(block.local_conv, xs_conv, context.temp_allocator)
	conv_flat := tensor.reshape(conv_out, []uint{b, c, l}, context.temp_allocator)
	conv_final := tensor.transpose(conv_flat, 1, 2, context.temp_allocator)

	// MLP with residual
	mlp_out := forward_mlp(block.mlp, conv_final, context.temp_allocator)

	// Final output - allocate and compute in one pass
	final_result := tensor.zeros(T, []uint{b, l, c}, allocator, loc)
	#no_bounds_check for i in 0 ..< b * l * c {
		final_result.data[i] = conv_final.data[i] + mlp_out.data[i]
	}

	return final_result
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
	init := true,
	allocator := context.allocator,
) -> ^Basic_Layer(T) {
	// Create blocks
	blocks := make([]^Tiny_ViT_Block(T), depth, allocator)
	for i in 0 ..< depth {
		blocks[i] = new_tiny_vit_block(
			T,
			dim,
			input_resolution,
			num_heads,
			window_size,
			init,
			allocator,
		)
	}

	// Create downsample if needed
	downsample_layer: Maybe(^Patch_Merging(T)) = nil
	if downsample {
		downsample_layer = new_patch_merging(T, input_resolution, dim, out, init, allocator)
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
	init := true,
	allocator := context.allocator,
) -> ^Tiny_ViT_5m(T) {
	embed_dims := []uint{64, 128, 160, 320}
	depths := []uint{2, 2, 6, 2}
	num_heads := []uint{2, 4, 5, 10}
	window_sizes := []uint{7, 7, 14, 7}

	patch_embed := new_patch_embed(T, IN_CHANNELS, embed_dims[0], init, allocator)
	patches_resolution := uint(input_size / 4) // After patch embedding

	// Layer 0 (ConvLayer) -
	layer0 := new_conv_layer(
		T,
		embed_dims[0],
		embed_dims[1],
		[2]uint{patches_resolution, patches_resolution},
		depths[0],
		true, // downsample
		MBCONV_EXPAND_RATIO,
		init,
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
			init,
			allocator,
		)
		layers[i_layer - 1] = layer
	}

	// Neck layers
	last_embed_dim := embed_dims[len(embed_dims) - 1]
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
		init = init,
		allocator = allocator,
	)
	// LayerNorm2d expects spatial dimensions based on final output
	// Final spatial dimension after all downsampling: input_size / 4 / (1 << min(3,2)) = input_size / 16
	final_spatial_dim := uint(input_size / 16)
	neck_ln1 := nn.new_layer_norm_2d(T, {256}, allocator)
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
		init,
		allocator,
	)
	neck_ln2 := nn.new_layer_norm_2d(T, []uint{256}, allocator)

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

Tiny_Vit_Result :: struct($T: typeid) {
	patch_embedding: ^tensor.Tensor(T),
	output_final:    ^tensor.Tensor(T),
}

// Argument return_intermediary_tensors has no effect for arena allocators
forward_tiny_vit_5m :: proc(
	model: ^Tiny_ViT_5m($T),
	x: ^tensor.Tensor(T),
	return_intermediary_tensors: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> Tiny_Vit_Result(T) {
	tiny_vit_trace := trace.TRACE_FUNCTION("tiny_vit_5m_forward")
	defer trace.end_scoped_trace(tiny_vit_trace)

	// Patch embedding
	patch_embedding := forward_patch_embed(model.patch_embed, x, allocator)

	// Layer 0
	layer0_trace := trace.TRACE_SECTION("layer0_conv")
	xs := forward_conv_layer(model.layer0, patch_embedding, context.temp_allocator)
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
	sequence_length := xs.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	xs_4d := tensor.reshape(xs, []uint{b, spatial_dim, spatial_dim, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)

	// Apply neck convolutions with layer norms
	conv1_out := nn.forward_conv2d(model.neck_conv1, xs_conv, context.temp_allocator)
	ln1_out := nn.forward_layer_norm_2d(model.neck_ln1, conv1_out, context.temp_allocator)

	fmt.println(model.neck_conv2)
	fmt.println(ln1_out.shape)
	conv2_out := nn.forward_conv2d(model.neck_conv2, ln1_out, context.temp_allocator)

	result := nn.forward_layer_norm_2d(model.neck_ln2, conv2_out, allocator, loc)
	trace.end_scoped_trace(neck_trace)

	if !return_intermediary_tensors {
		tensor.free_tensor(patch_embedding, allocator)
	}

	return {patch_embedding = patch_embedding, output_final = result}
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
