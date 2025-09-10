package vit

import "../../../nn"
import st "../../../safetensors"
import "../../../tensor"
import "../../../trace"
import "../var_builder"
import vb "../var_builder"
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
	vb_root: ^vb.Var_Builder(T),
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
	vb.assign(vb_root, "c.weight", conv.w)

	bn := nn.new_batch_norm_2d(T, out_channels, allocator)
	vb.assign(vb_root, "bn.weight", bn.weight)
	vb.assign(vb_root, "bn.bias", bn.bias)
	vb.assign(vb_root, "bn.running_mean", bn.running_mean)
	vb.assign(vb_root, "bn.running_var", bn.running_var)

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
	nn.free_conv2d(layer.conv, allocator)
	nn.free_batch_norm_2d(layer.bn, allocator)
	free(layer, allocator)
}

// PatchEmbed - Initial patch embedding with two convolutions
Patch_Embed :: struct($T: typeid) {
	conv1, conv2: ^Conv_2d_BN(T),
}

new_patch_embed :: proc(
	$T: typeid,
	vb_patch_embed: ^vb.Var_Builder(T),
	in_channels, embed_dim: uint,
	init := true,
	allocator := context.allocator,
) -> ^Patch_Embed(T) {
	vb_conv1 := vb.vb_make(T, "seq.0", vb_patch_embed)
	vb_conv2 := vb.vb_make(T, "seq.2", vb_patch_embed)
	conv1 := new_conv_2d_bn(T, &vb_conv1, in_channels, embed_dim / 2, 3, 2, 1, 1, init, allocator)
	conv2 := new_conv_2d_bn(T, &vb_conv2, embed_dim / 2, embed_dim, 3, 2, 1, 1, init, allocator)

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
	vb_root: ^vb.Var_Builder(T),
	in_channels, out_channels: uint,
	expand_ratio: uint,
	init := true,
	allocator := context.allocator,
) -> ^MB_Conv(T) {
	hidden := in_channels * expand_ratio

	vb_conv1 := vb.vb_make(T, "conv1", vb_root)
	vb_conv2 := vb.vb_make(T, "conv2", vb_root)
	vb_conv3 := vb.vb_make(T, "conv3", vb_root)
	conv1 := new_conv_2d_bn(T, &vb_conv1, in_channels, hidden, 1, 1, 0, 1, init, allocator)
	conv2 := new_conv_2d_bn(T, &vb_conv2, hidden, hidden, 3, 1, 1, hidden, init, allocator)
	conv3 := new_conv_2d_bn(T, &vb_conv3, hidden, out_channels, 1, 1, 0, 1, init, allocator)

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
	vb_root: ^vb.Var_Builder(T),
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

	vb_conv1 := vb.vb_make(T, "conv1", vb_root)
	vb_conv2 := vb.vb_make(T, "conv2", vb_root)
	vb_conv3 := vb.vb_make(T, "conv3", vb_root)
	conv1 := new_conv_2d_bn(T, &vb_conv1, dim, out, 1, 1, 0, 1, init, allocator)
	conv2 := new_conv_2d_bn(T, &vb_conv2, out, out, 3, stride, 1, out, init, allocator) // groups=out (depthwise)
	conv3 := new_conv_2d_bn(T, &vb_conv3, out, out, 1, 1, 0, 1, init, allocator)

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
	vb_root: ^vb.Var_Builder(T),
	dim, out: uint,
	input_resolution: [2]uint,
	depth: uint,
	downsample: bool,
	conv_expand_ratio: uint,
	init := true,
	allocator := context.allocator,
) -> ^Conv_Layer(T) {
	blocks := make([]^MB_Conv(T), depth, allocator)
	vb_blocks := vb.vb_make(T, "blocks", vb_root)
	for i in 0 ..< depth {
		vb_blocks_i := vb.vb_make(T, fmt.tprintf("%d", i), &vb_blocks)
		blocks[i] = new_mb_conv(T, &vb_blocks_i, dim, dim, conv_expand_ratio, init, allocator)
	}

	// Create downsample if needed
	downsample_layer: Maybe(^Patch_Merging(T)) = nil
	vb_downsample := vb.vb_make(T, "downsample", vb_root)
	if downsample {
		downsample_layer = new_patch_merging(
			T,
			&vb_downsample,
			input_resolution,
			dim,
			out,
			init,
			allocator,
		)
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
	vb_root: ^vb.Var_Builder(T),
	dim, key_dim, num_heads, attn_ratio: uint,
	resolution: [2]uint,
	init := true,
	allocator := context.allocator,
) -> ^Attention(T) {
	d := attn_ratio * key_dim
	dh := d * num_heads
	nh_kd := key_dim * num_heads
	h := dh + nh_kd * 2 // query + key + value

	norm := nn.new_layer_norm_1d(T, dim, 1e-5, allocator)
	vb.assign(vb_root, "norm.weight", norm.weight)
	vb.assign(vb_root, "norm.bias", norm.bias)

	qkv := nn.new_linear(T, dim, h, true, init, allocator)
	vb.assign(vb_root, "qkv.weight", qkv.w, true)
	vb.assign(vb_root, "qkv.bias", qkv.b.?)

	proj := nn.new_linear(T, dh, dim, true, init, allocator)
	vb.assign(vb_root, "proj.weight", proj.w, true)
	vb.assign(vb_root, "proj.bias", proj.b.?)

	// Build relative position bias indices
	num_points := resolution[0] * resolution[1]
	// Create points grid
	points := make([][2]int, num_points, context.temp_allocator)
	idx := 0
	for x in 0 ..< resolution[0] {
		for y in 0 ..< resolution[1] {
			points[idx] = {int(x), int(y)}
			idx += 1
		}
	}
	// Map relative offsets to unique indices
	offset_map := make(map[[2]uint]uint, context.temp_allocator)
	idxs := make([]uint, num_points * num_points, allocator)

	idx = 0
	for p1 in points {
		for p2 in points {
			offset := [2]uint{uint(abs(p2[0] - p1[0])), uint(abs(p2[1] - p1[1]))}

			if existing_idx, ok := offset_map[offset]; ok {
				idxs[idx] = existing_idx
			} else {
				new_idx := uint(len(offset_map))
				offset_map[offset] = new_idx
				idxs[idx] = new_idx
			}
			idx += 1
		}
	}

	num_unique_offsets := uint(len(offset_map))
	attention_biases := tensor.zeros(T, []uint{num_heads, num_unique_offsets})
	defer tensor.free_tensor(attention_biases)
	vb.assign(vb_root, "attention_biases", attention_biases)


	ab := tensor.zeros(T, []uint{num_heads, num_points, num_points}, allocator)
	// ab[head, i] = attention_biases[head, idxs[i]]
	for head in 0 ..< num_heads {
		for i in 0 ..< (num_points * num_points) {
			src_idx := idxs[i]
			// Copy from attention_biases[head, src_idx] to ab[head, i//num_points, i%num_points]
			row := i / num_points
			col := i % num_points
			ab.data[head * num_points * num_points + row * num_points + col] =
				attention_biases.data[head * num_unique_offsets + src_idx]
		}
	}

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

	// Add attention biases - broadcast across batch dimension
	#no_bounds_check for batch in 0 ..< b {
		for head in 0 ..< h {
			for i in 0 ..< n {
				for j in 0 ..< n {
					scores_idx := batch * h * n * n + head * n * n + i * n + j
					bias_idx := head * n * n + i * n + j
					attn_scores.data[scores_idx] += attn.ab.data[bias_idx]
				}
			}
		}
	}

	// Softmax - do it in-place to avoid allocation
	// Process each (batch, head) separately
	attention_softmax_trace := trace.TRACE_FUNCTION("attention_softmax")
	tensor.softmax_last_dim_inplace(attn_scores)
	trace.end_scoped_trace(attention_softmax_trace)

	// Apply attention to values
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
	vb_root: ^vb.Var_Builder(T),
	in_features, hidden_features: uint,
	init := true,
	allocator := context.allocator,
) -> ^Mlp(T) {
	norm := nn.new_layer_norm_1d(T, in_features, 1e-5, allocator)
	vb.assign(vb_root, "norm.weight", norm.weight)
	vb.assign(vb_root, "norm.bias", norm.bias)

	fc1 := nn.new_linear(T, in_features, hidden_features, true, init, allocator)
	fc2 := nn.new_linear(T, hidden_features, in_features, true, init, allocator)
	// Should transpose since pytorch's way of fc is tranposed matmul
	vb.assign(vb_root, "fc1.weight", fc1.w, should_transpose = true)
	vb.assign(vb_root, "fc2.weight", fc2.w, should_transpose = true)
	vb.assign(vb_root, "fc1.bias", fc1.b.?)
	vb.assign(vb_root, "fc2.bias", fc2.b.?)

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
	vb_root: ^vb.Var_Builder(T),
	dim: uint,
	input_resolution: [2]uint,
	num_heads, window_size: uint,
	init := true,
	allocator := context.allocator,
) -> ^Tiny_ViT_Block(T) {
	vb_attn := vb.vb_make(T, "attn", vb_root)
	head_dim := dim / num_heads
	attn := new_attention(
		T,
		&vb_attn,
		dim,
		head_dim,
		num_heads,
		1,
		[2]uint{window_size, window_size},
		init,
		allocator,
	)

	vb_mlp := vb.vb_make(T, "mlp", vb_root)
	mlp := new_mlp(T, &vb_mlp, dim, dim * MLP_RATIO, init, allocator)

	vb_local_conv := vb.vb_make(T, "local_conv", vb_root)
	local_conv := new_conv_2d_bn(
		T,
		&vb_local_conv,
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

		// xs_4d := tensor.reshape(xs, []uint{b, actual_h, actual_w, c}, context.temp_allocator)
		// xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)
		xs_transposed := tensor.transpose(xs, 1, 2, context.temp_allocator) // [b, l, c] → [b, c, l]
		xs_conv := tensor.reshape(xs_transposed, []uint{b, c, h, w}, context.temp_allocator) // [b, c, l] → [b, c, h, w]
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
	vb_root: ^vb.Var_Builder(T),
	dim, out: uint,
	input_resolution: [2]uint,
	depth, num_heads, window_size: uint,
	downsample: bool,
	init := true,
	allocator := context.allocator,
) -> ^Basic_Layer(T) {
	// Create blocks
	blocks := make([]^Tiny_ViT_Block(T), depth, allocator)
	vb_blocks := vb.vb_make(T, "blocks", vb_root)
	for i in 0 ..< depth {
		vb_blocks_i := vb.vb_make(T, fmt.tprintf("%d", i), &vb_blocks)
		blocks[i] = new_tiny_vit_block(
			T,
			&vb_blocks_i,
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
		vb_downsample := vb.vb_make(T, "downsample", vb_root)
		downsample_layer = new_patch_merging(
			T,
			&vb_downsample,
			input_resolution,
			dim,
			out,
			init,
			allocator,
		)
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
	neck_ln1, neck_ln2:     ^nn.Channel_Layer_Norm(T),
}


new_tiny_vit_5m :: proc(
	$T: typeid,
	safetensors: ^st.Safe_Tensors(T),
	input_size: uint = IMG_SIZE,
	init := true,
	allocator := context.allocator,
) -> ^Tiny_ViT_5m(T) {
	embed_dims := []uint{64, 128, 160, 320}
	depths := []uint{2, 2, 6, 2}
	num_heads := []uint{2, 4, 5, 10}
	window_sizes := []uint{7, 7, 14, 7}

	vb_root := vb.Var_Builder(T){"image_encoder", safetensors, nil}
	vb_patch_embed := vb.vb_make(T, "patch_embed", &vb_root)

	patch_embed := new_patch_embed(T, &vb_patch_embed, IN_CHANNELS, embed_dims[0], init, allocator)
	patches_resolution := uint(input_size / 4) // After patch embedding

	vb_layers := vb.vb_make(T, "layers", &vb_root)

	// Layer 0 (ConvLayer) -
	vb_layer0 := vb.vb_make(T, "0", &vb_layers)
	layer0 := new_conv_layer(
		T,
		&vb_layer0,
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

		vb_layer_i := vb.vb_make(T, fmt.tprintf("%d", i_layer), &vb_layers)
		layer := new_basic_layer(
			T,
			&vb_layer_i,
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
	vb.assign(&vb_root, "neck.0.weight", neck_conv1.w)

	// LayerNorm2d expects spatial dimensions based on final output
	// Final spatial dimension after all downsampling: input_size / 4 / (1 << min(3,2)) = input_size / 16
	final_spatial_dim := uint(input_size / 16)
	neck_ln1 := nn.new_channel_layer_norm(T, 256, 1e-5, allocator)
	vb.assign(&vb_root, "neck.1.weight", neck_ln1.weight)
	vb.assign(&vb_root, "neck.1.bias", neck_ln1.bias)

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
	vb.assign(&vb_root, "neck.2.weight", neck_conv2.w)

	neck_ln2 := nn.new_channel_layer_norm(T, 256, 1e-5, allocator)
	vb.assign(&vb_root, "neck.3.weight", neck_ln2.weight)
	vb.assign(&vb_root, "neck.3.bias", neck_ln2.bias)

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


// Argument return_intermediary_tensors has no effect for arena allocators
forward_tiny_vit_5m :: proc(
	model: ^Tiny_ViT_5m($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	tiny_vit_trace := trace.TRACE_FUNCTION("tiny_vit_5m_forward")
	defer trace.end_scoped_trace(tiny_vit_trace)

	// Patch embedding
	patch_embedding := forward_patch_embed(model.patch_embed, x, context.temp_allocator)

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

	sequence_length := xs.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	xs_4d := tensor.reshape(xs, []uint{b, spatial_dim, spatial_dim, c}, context.temp_allocator)
	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)

	// Apply neck convolutions with layer norms
	conv1_out := nn.forward_conv2d(model.neck_conv1, xs_conv, context.temp_allocator)
	ln1_out := nn.forward_channel_layer_norm(model.neck_ln1, conv1_out, context.temp_allocator)

	conv2_out := nn.forward_conv2d(model.neck_conv2, ln1_out, context.temp_allocator)

	result := nn.forward_channel_layer_norm(model.neck_ln2, conv2_out, allocator, loc)
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

	nn.free_conv2d(model.neck_conv1, allocator)
	nn.free_channel_layer_norm(model.neck_ln1, allocator)
	nn.free_conv2d(model.neck_conv2, allocator)
	nn.free_channel_layer_norm(model.neck_ln2, allocator)
	free(model, allocator)
}
