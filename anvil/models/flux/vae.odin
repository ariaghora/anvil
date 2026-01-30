// FLUX VAE Implementation
//
// AutoencoderKLFlux2 - Variational Autoencoder for FLUX.2
// - 32 latent channels (128 after patchification)
// - 16x spatial compression
// - Channel multipliers: [1, 2, 4, 4] -> [128, 256, 512, 512]
// - GroupNorm (32 groups) + Swish activation

package flux

import "../../nn"
import "../../tensor"
import "../../trace"
import st "../../safetensors"
import "core:fmt"
import "core:math"
import "core:mem"

// ResBlock weights
Res_Block :: struct($T: typeid) {
	norm1_weight, norm1_bias: ^tensor.Tensor(T),
	conv1_weight, conv1_bias: ^tensor.Tensor(T),
	norm2_weight, norm2_bias: ^tensor.Tensor(T),
	conv2_weight, conv2_bias: ^tensor.Tensor(T),
	skip_weight, skip_bias:   ^tensor.Tensor(T), // nil if in_ch == out_ch
	in_channels:              uint,
	out_channels:             uint,
}

// Self-attention block weights
Attn_Block :: struct($T: typeid) {
	norm_weight, norm_bias: ^tensor.Tensor(T),
	q_weight, q_bias:       ^tensor.Tensor(T),
	k_weight, k_bias:       ^tensor.Tensor(T),
	v_weight, v_bias:       ^tensor.Tensor(T),
	out_weight, out_bias:   ^tensor.Tensor(T),
	channels:               uint,
}

// Mid block: ResBlock + Attention + ResBlock
Mid_Block :: struct($T: typeid) {
	block1: Res_Block(T),
	attn:   Attn_Block(T),
	block2: Res_Block(T),
}

// Downsample: stride-2 conv
Downsample :: struct($T: typeid) {
	conv_weight, conv_bias: ^tensor.Tensor(T),
	channels:               uint,
}

// Upsample: nearest + conv
Upsample :: struct($T: typeid) {
	conv_weight, conv_bias: ^tensor.Tensor(T),
	channels:               uint,
}

// VAE context
VAE :: struct($T: typeid) {
	// Configuration
	z_channels:     uint,
	base_channels:  uint,
	ch_mult:        [4]uint,
	num_res_blocks: uint,
	num_groups:     uint,
	eps:            T,

	// Keep safetensors alive (tensors point into mmap)
	_safetensors:   ^st.Safe_Tensors(T),

	// Encoder
	enc_conv_in_weight, enc_conv_in_bias: ^tensor.Tensor(T),
	enc_down_blocks:                      []Res_Block(T), // 4 levels * 2 blocks = 8
	enc_downsample:                       []Downsample(T), // 3 (not at last level)
	enc_mid:                              Mid_Block(T),
	enc_norm_out_weight, enc_norm_out_bias: ^tensor.Tensor(T),
	enc_conv_out_weight, enc_conv_out_bias: ^tensor.Tensor(T),

	// Decoder
	dec_conv_in_weight, dec_conv_in_bias: ^tensor.Tensor(T),
	dec_mid:                              Mid_Block(T),
	dec_up_blocks:                        []Res_Block(T), // 4 levels * 3 blocks = 12
	dec_upsample:                         []Upsample(T), // 3
	dec_norm_out_weight, dec_norm_out_bias: ^tensor.Tensor(T),
	dec_conv_out_weight, dec_conv_out_bias: ^tensor.Tensor(T),

	// Quantization convs
	quant_conv_weight, quant_conv_bias:           ^tensor.Tensor(T),
	post_quant_conv_weight, post_quant_conv_bias: ^tensor.Tensor(T),
	// Batch norm for latent denormalization
	bn_running_mean, bn_running_var: ^tensor.Tensor(T),
}

// Load VAE from safetensors file
load_vae :: proc(
	$T: typeid,
	path: string,
	allocator := context.allocator,
) -> (vae: ^VAE(T), err: string) {
	sf, sf_err := st.read_from_file(T, path, allocator)
	if sf_err != nil {
		return nil, fmt.tprintf("Failed to read safetensors: %v", sf_err)
	}
	// Don't free - tensors point into mmap, stored in VAE

	vae = new(VAE(T), allocator)
	vae._safetensors = sf
	vae.z_channels = VAE_Z_CHANNELS
	vae.base_channels = VAE_BASE_CHANNELS
	vae.ch_mult = {1, 2, 4, 4}
	vae.num_res_blocks = VAE_NUM_RES_BLOCKS
	vae.num_groups = VAE_NUM_GROUPS
	vae.eps = T(1e-6)

	// Allocate blocks
	vae.enc_down_blocks = make([]Res_Block(T), 8, allocator) // 4 levels * 2
	vae.enc_downsample = make([]Downsample(T), 3, allocator)
	vae.dec_up_blocks = make([]Res_Block(T), 12, allocator) // 4 levels * 3
	vae.dec_upsample = make([]Upsample(T), 3, allocator)

	// Load encoder weights
	load_vae_encoder_weights(vae, sf, allocator)

	// Load decoder weights
	load_vae_decoder_weights(vae, sf, allocator)

	return vae, ""
}

// Free VAE
free_vae :: proc(vae: ^VAE($T), allocator := context.allocator) {
	if vae == nil do return

	// Free safetensors (this frees all weight tensors via mmap)
	if vae._safetensors != nil {
		st.free_safe_tensors(vae._safetensors, allocator)
	}

	// Free block arrays (structs only, tensors already freed above)
	delete(vae.enc_down_blocks, allocator)
	delete(vae.enc_downsample, allocator)
	delete(vae.dec_up_blocks, allocator)
	delete(vae.dec_upsample, allocator)

	free(vae, allocator)
}

// Load encoder weights from safetensors
@(private = "file")
load_vae_encoder_weights :: proc(vae: ^VAE($T), sf: ^st.Safe_Tensors(T), allocator := context.allocator) {
	// Conv in: [128, 3, 3, 3]
	vae.enc_conv_in_weight = get_tensor(sf, "encoder.conv_in.weight", allocator)
	vae.enc_conv_in_bias = get_tensor(sf, "encoder.conv_in.bias", allocator)

	// Down blocks
	block_idx := 0
	for level in 0 ..< 4 {
		ch := vae.base_channels * vae.ch_mult[level]
		in_ch := level == 0 ? vae.base_channels : vae.base_channels * vae.ch_mult[level - 1]

		for i in 0 ..< int(vae.num_res_blocks) {
			prefix := fmt.tprintf("encoder.down_blocks.%d.resnets.%d", level, i)
			load_res_block(&vae.enc_down_blocks[block_idx], sf, prefix, in_ch if i == 0 else ch, ch, allocator)
			block_idx += 1
		}

		// Downsample (not at last level)
		if level < 3 {
			ds_prefix := fmt.tprintf("encoder.down_blocks.%d.downsamplers.0.conv", level)
			vae.enc_downsample[level].conv_weight = get_tensor(sf, fmt.tprintf("%s.weight", ds_prefix), allocator)
			vae.enc_downsample[level].conv_bias = get_tensor(sf, fmt.tprintf("%s.bias", ds_prefix), allocator)
			vae.enc_downsample[level].channels = ch
		}
	}

	// Mid block
	mid_ch := vae.base_channels * vae.ch_mult[3] // 512
	load_res_block(&vae.enc_mid.block1, sf, "encoder.mid_block.resnets.0", mid_ch, mid_ch, allocator)
	load_attn_block(&vae.enc_mid.attn, sf, "encoder.mid_block.attentions.0", mid_ch, allocator)
	load_res_block(&vae.enc_mid.block2, sf, "encoder.mid_block.resnets.1", mid_ch, mid_ch, allocator)

	// Output
	vae.enc_norm_out_weight = get_tensor(sf, "encoder.conv_norm_out.weight", allocator)
	vae.enc_norm_out_bias = get_tensor(sf, "encoder.conv_norm_out.bias", allocator)
	vae.enc_conv_out_weight = get_tensor(sf, "encoder.conv_out.weight", allocator)
	vae.enc_conv_out_bias = get_tensor(sf, "encoder.conv_out.bias", allocator)

	// Quant conv
	vae.quant_conv_weight = get_tensor(sf, "quant_conv.weight", allocator)
	vae.quant_conv_bias = get_tensor(sf, "quant_conv.bias", allocator)
}

// Load decoder weights from safetensors
@(private = "file")
load_vae_decoder_weights :: proc(vae: ^VAE($T), sf: ^st.Safe_Tensors(T), allocator := context.allocator) {
	// Post quant conv
	vae.post_quant_conv_weight = get_tensor(sf, "post_quant_conv.weight", allocator)
	vae.post_quant_conv_bias = get_tensor(sf, "post_quant_conv.bias", allocator)
	// Batch norm stats for latent denormalization (FLUX.2-klein specific)
	vae.bn_running_mean = get_tensor(sf, "bn.running_mean", allocator)
	vae.bn_running_var = get_tensor(sf, "bn.running_var", allocator)

	// Conv in
	vae.dec_conv_in_weight = get_tensor(sf, "decoder.conv_in.weight", allocator)
	vae.dec_conv_in_bias = get_tensor(sf, "decoder.conv_in.bias", allocator)

	// Mid block
	mid_ch := vae.base_channels * vae.ch_mult[3] // 512
	load_res_block(&vae.dec_mid.block1, sf, "decoder.mid_block.resnets.0", mid_ch, mid_ch, allocator)
	load_attn_block(&vae.dec_mid.attn, sf, "decoder.mid_block.attentions.0", mid_ch, allocator)
	load_res_block(&vae.dec_mid.block2, sf, "decoder.mid_block.resnets.1", mid_ch, mid_ch, allocator)

	// Up blocks (reverse order of channels)
	// ch_mult = [1,2,4,4] -> channels = [128, 256, 512, 512]
	// up_blocks go: 512->512, 512->512, 512->256, 256->128
	block_idx := 0
	dec_channels := [5]uint{512, 512, 512, 256, 128} // in channels for each level + final
	for level in 0 ..< 4 {
		in_ch := dec_channels[level]
		out_ch := dec_channels[level + 1]

		for i in 0 ..< int(vae.num_res_blocks) + 1 { // decoder has one extra resblock (3 total)
			prefix := fmt.tprintf("decoder.up_blocks.%d.resnets.%d", level, i)
			// First resnet may have channel change, rest are out_ch->out_ch
			res_in := in_ch if i == 0 else out_ch
			load_res_block(&vae.dec_up_blocks[block_idx], sf, prefix, res_in, out_ch, allocator)
			block_idx += 1
		}

		// Upsample (not at last level)
		if level < 3 {
			us_prefix := fmt.tprintf("decoder.up_blocks.%d.upsamplers.0.conv", level)
			vae.dec_upsample[level].conv_weight = get_tensor(sf, fmt.tprintf("%s.weight", us_prefix), allocator)
			vae.dec_upsample[level].conv_bias = get_tensor(sf, fmt.tprintf("%s.bias", us_prefix), allocator)
			vae.dec_upsample[level].channels = out_ch
		}
	}

	// Output
	vae.dec_norm_out_weight = get_tensor(sf, "decoder.conv_norm_out.weight", allocator)
	vae.dec_norm_out_bias = get_tensor(sf, "decoder.conv_norm_out.bias", allocator)
	vae.dec_conv_out_weight = get_tensor(sf, "decoder.conv_out.weight", allocator)
	vae.dec_conv_out_bias = get_tensor(sf, "decoder.conv_out.bias", allocator)
}

@(private = "file")
load_res_block :: proc(
	block: ^Res_Block($T),
	sf: ^st.Safe_Tensors(T),
	prefix: string,
	in_ch, out_ch: uint,
	allocator := context.allocator,
) {
	block.in_channels = in_ch
	block.out_channels = out_ch

	block.norm1_weight = get_tensor(sf, fmt.tprintf("%s.norm1.weight", prefix), allocator)
	block.norm1_bias = get_tensor(sf, fmt.tprintf("%s.norm1.bias", prefix), allocator)
	block.conv1_weight = get_tensor(sf, fmt.tprintf("%s.conv1.weight", prefix), allocator)
	block.conv1_bias = get_tensor(sf, fmt.tprintf("%s.conv1.bias", prefix), allocator)
	block.norm2_weight = get_tensor(sf, fmt.tprintf("%s.norm2.weight", prefix), allocator)
	block.norm2_bias = get_tensor(sf, fmt.tprintf("%s.norm2.bias", prefix), allocator)
	block.conv2_weight = get_tensor(sf, fmt.tprintf("%s.conv2.weight", prefix), allocator)
	block.conv2_bias = get_tensor(sf, fmt.tprintf("%s.conv2.bias", prefix), allocator)

	if in_ch != out_ch {
		block.skip_weight = get_tensor(sf, fmt.tprintf("%s.conv_shortcut.weight", prefix), allocator)
		block.skip_bias = get_tensor(sf, fmt.tprintf("%s.conv_shortcut.bias", prefix), allocator)
	}
}

@(private = "file")
load_attn_block :: proc(
	block: ^Attn_Block($T),
	sf: ^st.Safe_Tensors(T),
	prefix: string,
	channels: uint,
	allocator := context.allocator,
) {
	block.channels = channels
	block.norm_weight = get_tensor(sf, fmt.tprintf("%s.group_norm.weight", prefix), allocator)
	block.norm_bias = get_tensor(sf, fmt.tprintf("%s.group_norm.bias", prefix), allocator)
	// PyTorch convention: Linear weights are [out, in], no transpose needed
	block.q_weight = get_tensor(sf, fmt.tprintf("%s.to_q.weight", prefix), allocator)
	block.q_bias = get_tensor(sf, fmt.tprintf("%s.to_q.bias", prefix), allocator)
	block.k_weight = get_tensor(sf, fmt.tprintf("%s.to_k.weight", prefix), allocator)
	block.k_bias = get_tensor(sf, fmt.tprintf("%s.to_k.bias", prefix), allocator)
	block.v_weight = get_tensor(sf, fmt.tprintf("%s.to_v.weight", prefix), allocator)
	block.v_bias = get_tensor(sf, fmt.tprintf("%s.to_v.bias", prefix), allocator)
	block.out_weight = get_tensor(sf, fmt.tprintf("%s.to_out.0.weight", prefix), allocator)
	block.out_bias = get_tensor(sf, fmt.tprintf("%s.to_out.0.bias", prefix), allocator)
}

@(private = "file")
get_tensor :: proc(sf: ^st.Safe_Tensors($T), name: string, allocator := context.allocator) -> ^tensor.Tensor(T) {
	t, exists := sf.tensors[name]
	if !exists {
		fmt.panicf("Tensor not found: %s", name)
	}
	return t
}

// Swish/SiLU activation in-place
@(private = "file")
swish_inplace :: proc(x: ^tensor.Tensor($T)) {
	for i in 0 ..< len(x.data) {
		x.data[i] = x.data[i] / (T(1.0) + math.exp(-x.data[i]))
	}
}

// Group normalization
@(private = "file")
group_norm :: proc(
	x: ^tensor.Tensor($T),
	weight, bias: ^tensor.Tensor(T),
	num_groups: uint,
	eps: T,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// x: [B, C, H, W]
	b := x.shape[0]
	c := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	channels_per_group := c / num_groups

	result := tensor.tensor_alloc(T, x.shape, true, allocator)

	for batch in 0 ..< b {
		for g in 0 ..< num_groups {
			// Compute mean and variance for this group
			sum: T = 0
			sq_sum: T = 0
			count := channels_per_group * h * w

			for ch in 0 ..< channels_per_group {
				c_idx := g * channels_per_group + ch
				for y in 0 ..< h {
					for x_pos in 0 ..< w {
						idx := batch * c * h * w + c_idx * h * w + y * w + x_pos
						val := x.data[idx]
						sum += val
						sq_sum += val * val
					}
				}
			}

			mean := sum / T(count)
			variance := sq_sum / T(count) - mean * mean
			std := math.sqrt(variance + eps)

			// Normalize and apply affine
			for ch in 0 ..< channels_per_group {
				c_idx := g * channels_per_group + ch
				gamma := weight.data[c_idx]
				beta := bias.data[c_idx]

				for y in 0 ..< h {
					for x_pos in 0 ..< w {
						idx := batch * c * h * w + c_idx * h * w + y * w + x_pos
						result.data[idx] = gamma * (x.data[idx] - mean) / std + beta
					}
				}
			}
		}
	}

	return result
}

// Forward pass for ResBlock
@(private = "file")
res_block_forward :: proc(
	block: ^Res_Block($T),
	x: ^tensor.Tensor(T),
	num_groups: uint,
	eps: T,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Shortcut
	shortcut: ^tensor.Tensor(T)
	if block.in_channels != block.out_channels {
		shortcut = tensor.conv2d_xwb(x, block.skip_weight, block.skip_bias, 1, 1, 0, 1, allocator)
	} else {
		shortcut = tensor.clone(x, allocator)
	}

	// Main path: norm1 -> swish -> conv1 -> norm2 -> swish -> conv2
	h := group_norm(x, block.norm1_weight, block.norm1_bias, num_groups, eps, allocator)
	swish_inplace(h)
	h2 := tensor.conv2d_xwb(h, block.conv1_weight, block.conv1_bias, 1, 1, 1, 1, allocator)
	tensor.free_tensor(h, allocator)

	h3 := group_norm(h2, block.norm2_weight, block.norm2_bias, num_groups, eps, allocator)
	tensor.free_tensor(h2, allocator)
	swish_inplace(h3)

	h4 := tensor.conv2d_xwb(h3, block.conv2_weight, block.conv2_bias, 1, 1, 1, 1, allocator)
	tensor.free_tensor(h3, allocator)

	// Add residual
	result := tensor.add(shortcut, h4, allocator)
	tensor.free_tensor(shortcut, h4, allocator = allocator)

	return result
}

// Forward pass for attention block
@(private = "file")
attn_block_forward :: proc(
	block: ^Attn_Block($T),
	x: ^tensor.Tensor(T),
	num_groups: uint,
	eps: T,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	b := x.shape[0]
	c := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	spatial := h * w

	// GroupNorm
	normed := group_norm(x, block.norm_weight, block.norm_bias, num_groups, eps, allocator)

	// Reshape [B, C, H, W] -> [B, HW, C] for linear projections
	normed_flat := tensor.reshape(normed, []uint{b, c, spatial}, allocator)
	tensor.free_tensor(normed, allocator)
	normed_t := tensor.transpose(normed_flat, 1, 2, allocator) // [B, HW, C]
	tensor.free_tensor(normed_flat, allocator)

	// Project to Q, K, V using nn.forward_linear (weights already transposed at load time)
	q_linear := nn.Linear(T){w = block.q_weight, b = block.q_bias}
	k_linear := nn.Linear(T){w = block.k_weight, b = block.k_bias}
	v_linear := nn.Linear(T){w = block.v_weight, b = block.v_bias}

	q_t := nn.forward_linear(&q_linear, normed_t, allocator) // [B, HW, C]
	k_t := nn.forward_linear(&k_linear, normed_t, allocator)
	v_t := nn.forward_linear(&v_linear, normed_t, allocator)
	tensor.free_tensor(normed_t, allocator)

	scale := T(1.0) / math.sqrt(T(c))

	// Scale Q
	for i in 0 ..< len(q_t.data) {
		q_t.data[i] *= scale
	}

	// Attention: Q @ K^T
	k_transposed := tensor.transpose(k_t, 1, 2, allocator)
	scores := tensor.matmul(q_t, k_transposed, allocator)
	tensor.free_tensor(q_t, k_transposed, allocator = allocator)

	// Softmax
	tensor.softmax_last_dim_inplace(scores)

	// scores @ V
	attn_out := tensor.matmul(scores, v_t, allocator) // [B, HW, C]
	tensor.free_tensor(scores, v_t, k_t, allocator = allocator)

	// Project output using nn.forward_linear
	out_linear := nn.Linear(T){w = block.out_weight, b = block.out_bias}
	proj_flat := nn.forward_linear(&out_linear, attn_out, allocator) // [B, HW, C]
	tensor.free_tensor(attn_out, allocator)

	// Transpose back [B, HW, C] -> [B, C, HW]
	proj_t := tensor.transpose(proj_flat, 1, 2, allocator)
	tensor.free_tensor(proj_flat, allocator)

	// Reshape back to [B, C, H, W]
	proj := tensor.reshape(proj_t, []uint{b, c, h, w}, allocator)
	tensor.free_tensor(proj_t, allocator)

	// Add residual
	result := tensor.add(x, proj, allocator)
	tensor.free_tensor(proj, allocator)

	return result
}

// VAE encode: image -> latent
vae_encode :: proc(
	vae: ^VAE($T),
	img: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Conv in
	h := tensor.conv2d_xwb(img, vae.enc_conv_in_weight, vae.enc_conv_in_bias, 1, 1, 1, 1, allocator)

	// Down blocks
	block_idx := 0
	for level in 0 ..< 4 {
		for i in 0 ..< int(vae.num_res_blocks) {
			h2 := res_block_forward(&vae.enc_down_blocks[block_idx], h, vae.num_groups, vae.eps, allocator)
			tensor.free_tensor(h, allocator)
			h = h2
			block_idx += 1
		}

		// Downsample
		if level < 3 {
			ds := &vae.enc_downsample[level]
			h2 := tensor.conv2d_xwb(h, ds.conv_weight, ds.conv_bias, 2, 1, 1, 1, allocator)
			tensor.free_tensor(h, allocator)
			h = h2
		}
	}

	// Mid block
	h2 := res_block_forward(&vae.enc_mid.block1, h, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	h = attn_block_forward(&vae.enc_mid.attn, h2, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h2, allocator)
	h2 = res_block_forward(&vae.enc_mid.block2, h, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	h = h2

	// Output
	h2 = group_norm(h, vae.enc_norm_out_weight, vae.enc_norm_out_bias, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	swish_inplace(h2)
	h = tensor.conv2d_xwb(h2, vae.enc_conv_out_weight, vae.enc_conv_out_bias, 1, 1, 1, 1, allocator)
	tensor.free_tensor(h2, allocator)

	// Quant conv
	latent := tensor.conv2d_xwb(h, vae.quant_conv_weight, vae.quant_conv_bias, 1, 1, 0, 1, allocator)
	tensor.free_tensor(h, allocator)

	return latent
}

// Denormalize latent using batch norm statistics
// x = x * sqrt(var + eps) + mean
@(private = "file")
denormalize_latent :: proc(x: ^tensor.Tensor($T), mean, var_: ^tensor.Tensor(T), allocator := context.allocator) -> ^tensor.Tensor(T) {
	batch := x.shape[0]
	channels := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	eps := T(1e-5)

	result := tensor.tensor_alloc(T, x.shape, true, allocator)

	for b in 0 ..< int(batch) {
		for c in 0 ..< int(channels) {
			std := math.sqrt(var_.data[c] + eps)
			m := mean.data[c]
			for y in 0 ..< int(h) {
				for x_pos in 0 ..< int(w) {
					idx := b * int(channels * h * w) + c * int(h * w) + y * int(w) + x_pos
					result.data[idx] = x.data[idx] * std + m
				}
			}
		}
	}
	return result
}

// Unpatchify: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
// Reverses the 2x2 patchify operation
@(private = "file")
vae_unpatchify :: proc(x: ^tensor.Tensor($T), z_channels: uint, allocator := context.allocator) -> ^tensor.Tensor(T) {
	batch := x.shape[0]
	in_channels := x.shape[1] // 128
	in_h := x.shape[2]
	in_w := x.shape[3]
	patch_size : uint = 2
	out_channels := z_channels // 32
	out_h := in_h * patch_size
	out_w := in_w * patch_size

	result := tensor.tensor_alloc(T, []uint{batch, out_channels, out_h, out_w}, true, allocator)

	// in_channels = out_channels * patch_size * patch_size = 32 * 4 = 128
	for b in 0 ..< int(batch) {
		for c in 0 ..< int(out_channels) {
			for y in 0 ..< int(in_h) {
				for x_pos in 0 ..< int(in_w) {
					for py in 0 ..< int(patch_size) {
						for px in 0 ..< int(patch_size) {
							// Source channel = c * patch_size^2 + py * patch_size + px
							src_c := c * int(patch_size * patch_size) + py * int(patch_size) + px
							src_idx := b * int(in_channels * in_h * in_w) + src_c * int(in_h * in_w) + y * int(in_w) + x_pos

							// Destination position
							dst_y := y * int(patch_size) + py
							dst_x := x_pos * int(patch_size) + px
							dst_idx := b * int(out_channels * out_h * out_w) + c * int(out_h * out_w) + dst_y * int(out_w) + dst_x

							result.data[dst_idx] = x.data[src_idx]
						}
					}
				}
			}
		}
	}
	return result
}

// VAE decode: latent -> image tensor
vae_decode :: proc(
	vae: ^VAE($T),
	latent: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("vae_decode", "vae")
	defer trace.global_end_scoped(_t)

	// Denormalize: x = x * sqrt(var + eps) + mean
	_t_denorm := trace.global_scoped("denormalize", "vae")
	denorm := denormalize_latent(latent, vae.bn_running_mean, vae.bn_running_var, allocator)
	trace.global_end_scoped(_t_denorm)
	defer tensor.free_tensor(denorm, allocator)

	// Unpatchify: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
	_t_unpatch := trace.global_scoped("unpatchify", "vae")
	h := vae_unpatchify(denorm, vae.z_channels, allocator)
	trace.global_end_scoped(_t_unpatch)

	// Post-quant conv (1x1): 32 -> 32
	_t_pqc := trace.global_scoped("post_quant_conv", "vae")
	h2 := tensor.conv2d_xwb(h, vae.post_quant_conv_weight, vae.post_quant_conv_bias, 1, 0, 1, 1, allocator)
	tensor.free_tensor(h, allocator)
	h = h2
	trace.global_end_scoped(_t_pqc)

	// Conv in
	_t_cin := trace.global_scoped("conv_in", "vae")
	h2 = tensor.conv2d_xwb(h, vae.dec_conv_in_weight, vae.dec_conv_in_bias, 1, 1, 1, 1, allocator)
	tensor.free_tensor(h, allocator)
	h = h2
	trace.global_end_scoped(_t_cin)

	// Mid block
	_t_mid := trace.global_scoped("mid_block", "vae")
	h2 = res_block_forward(&vae.dec_mid.block1, h, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	h = attn_block_forward(&vae.dec_mid.attn, h2, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h2, allocator)
	h2 = res_block_forward(&vae.dec_mid.block2, h, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	h = h2
	trace.global_end_scoped(_t_mid)

	// Up blocks
	_t_up := trace.global_scoped("up_blocks", "vae")
	block_idx := 0
	for level in 0 ..< 4 {
		for i in 0 ..< int(vae.num_res_blocks) + 1 {
			h2 = res_block_forward(&vae.dec_up_blocks[block_idx], h, vae.num_groups, vae.eps, allocator)
			tensor.free_tensor(h, allocator)
			h = h2
			block_idx += 1
		}

		// Upsample
		if level < 3 {
			// Nearest neighbor 2x upsample
			cur_h := h.shape[2]
			cur_w := h.shape[3]
			h2 = tensor.upsample_nearest_2d(h, cur_h * 2, cur_w * 2, allocator)
			tensor.free_tensor(h, allocator)
			us := &vae.dec_upsample[level]
			h = tensor.conv2d_xwb(h2, us.conv_weight, us.conv_bias, 1, 1, 1, 1, allocator)
			tensor.free_tensor(h2, allocator)
		}
	}
	trace.global_end_scoped(_t_up)

	// Output
	_t_out := trace.global_scoped("conv_out", "vae")
	h2 = group_norm(h, vae.dec_norm_out_weight, vae.dec_norm_out_bias, vae.num_groups, vae.eps, allocator)
	tensor.free_tensor(h, allocator)
	swish_inplace(h2)
	img := tensor.conv2d_xwb(h2, vae.dec_conv_out_weight, vae.dec_conv_out_bias, 1, 1, 1, 1, allocator)
	tensor.free_tensor(h2, allocator)
	trace.global_end_scoped(_t_out)

	return img
}

// VAE decode to Image struct (u8 RGB)
vae_decode_to_image :: proc(
	vae: ^VAE($T),
	latent: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^Image {
	img_tensor := vae_decode(vae, latent, allocator)
	defer tensor.free_tensor(img_tensor, allocator)

	// Convert [1, 3, H, W] float [-1, 1] to [H, W, 3] u8 [0, 255]
	h := img_tensor.shape[2]
	w := img_tensor.shape[3]
	c := img_tensor.shape[1]

	img := new(Image, allocator)
	img.width = int(w)
	img.height = int(h)
	img.channels = int(c)
	img.data = make([]u8, h * w * c, allocator)

	for y in 0 ..< h {
		for x in 0 ..< w {
			for ch in 0 ..< c {
				src_idx := ch * h * w + y * w + x
				dst_idx := (y * w + x) * c + ch
				// Denormalize from [-1, 1] to [0, 255]
				val := (img_tensor.data[src_idx] + T(1.0)) * T(127.5)
				val = max(T(0), min(T(255), val))
				img.data[dst_idx] = u8(val)
			}
		}
	}

	return img
}
