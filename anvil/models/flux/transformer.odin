// FLUX Transformer (DiT) Implementation
//
// MM-DiT architecture with:
// - 5 double-stream blocks (separate img/txt paths, joint attention)
// - 20 single-stream blocks (parallel attention + FFN)
// - 24 attention heads, 128 dim per head (3072 hidden)
// - AdaLN-Zero modulation
// - SwiGLU MLP
// - QK-Norm (RMSNorm)
//
// True lazy loading: block weights loaded on-demand, freed after use.

package flux

import "../../tensor"
import "../../trace"
import "../../nn"
import st "../../safetensors"
import "core:fmt"
import "core:math"

// Time embedding MLP
Time_Embed :: struct($T: typeid) {
	fc1: ^nn.Linear(T),
	fc2: ^nn.Linear(T),
}

// Double block weights (loaded/freed per forward pass)
Double_Block_Weights :: struct($T: typeid) {
	// Image stream
	img_to_q, img_to_k, img_to_v: ^nn.Linear(T),
	img_norm_q, img_norm_k:       ^tensor.Tensor(T),
	img_proj:                     ^nn.Linear(T),
	img_mlp_fc1, img_mlp_fc2:     ^nn.Linear(T),

	// Text stream
	txt_to_q, txt_to_k, txt_to_v: ^nn.Linear(T),
	txt_norm_q, txt_norm_k:       ^tensor.Tensor(T),
	txt_proj:                     ^nn.Linear(T),
	txt_mlp_fc1, txt_mlp_fc2:     ^nn.Linear(T),
}

// Single block weights (loaded/freed per forward pass)
Single_Block_Weights :: struct($T: typeid) {
	qkv_mlp:        ^nn.Linear(T),
	norm_q, norm_k: ^tensor.Tensor(T),
	proj:           ^nn.Linear(T),
}

// Transformer context
Transformer :: struct($T: typeid) {
	config: Transformer_Config,

	// Safetensors file (kept open for lazy loading)
	_sf: ^st.Safe_Tensors(T),

	// Shared weights (small, kept in memory)
	time_embed: Time_Embed(T),
	img_in:     ^nn.Linear(T),
	txt_in:     ^nn.Linear(T),
	final_mod:  ^nn.Linear(T),
	final_proj: ^nn.Linear(T),
	mod_img:    ^nn.Linear(T),
	mod_txt:    ^nn.Linear(T),
	mod_single: ^nn.Linear(T),
	_loaded:    bool,
}

Transformer_Config :: struct {
	hidden_size:       uint,
	num_heads:         uint,
	head_dim:          uint,
	mlp_hidden:        uint,
	num_double_layers: uint,
	num_single_layers: uint,
	rope_theta:        f32,
	rope_dim:          uint,
}

// Load transformer - only metadata, weights loaded lazily
load_transformer :: proc(
	$T: typeid,
	path: string,
	config: Flux_Config,
	allocator := context.allocator,
) -> (tf: ^Transformer(T), err: string) {
	sf, sf_err := st.read_from_file_lazy(T, path, allocator)
	if sf_err != nil {
		return nil, fmt.tprintf("Failed to read safetensors: %v", sf_err)
	}

	tf = new(Transformer(T), allocator)
	tf.config = Transformer_Config {
		hidden_size       = config.hidden_size,
		num_heads         = config.num_heads,
		head_dim          = config.head_dim,
		mlp_hidden        = config.mlp_hidden,
		num_double_layers = config.num_double_layers,
		num_single_layers = config.num_single_layers,
		rope_theta        = config.rope_theta,
		rope_dim          = config.rope_dim,
	}

	tf._sf = sf

	return tf, ""
}

// Free transformer
free_transformer :: proc(tf: ^Transformer($T), allocator := context.allocator) {
	if tf == nil do return

	// Shared weights
	if tf.img_in != nil do nn.free_linear(tf.img_in, allocator)
	if tf.txt_in != nil do nn.free_linear(tf.txt_in, allocator)
	if tf.time_embed.fc1 != nil do nn.free_linear(tf.time_embed.fc1, allocator)
	if tf.time_embed.fc2 != nil do nn.free_linear(tf.time_embed.fc2, allocator)
	if tf.final_proj != nil do nn.free_linear(tf.final_proj, allocator)
	if tf.mod_img != nil do nn.free_linear(tf.mod_img, allocator)
	if tf.mod_txt != nil do nn.free_linear(tf.mod_txt, allocator)
	if tf.mod_single != nil do nn.free_linear(tf.mod_single, allocator)

	if tf._sf != nil {
		st.free_safe_tensors(tf._sf, allocator)
	}

	free(tf, allocator)
}

// ============================================================================
// Weight Loading Helpers
// ============================================================================

@(private = "file")
get_tensor :: proc(sf: ^st.Safe_Tensors($T), name: string, expected_shape: []uint = nil) -> ^tensor.Tensor(T) {
	t, ok := st.get_tensor_lazy(sf, name)
	if !ok {
		fmt.panicf("Tensor not found: %s", name)
	}
	if expected_shape != nil {
		if len(t.shape) != len(expected_shape) {
			fmt.panicf("%s: expected rank %d, got %d", name, len(expected_shape), len(t.shape))
		}
		for i in 0 ..< len(expected_shape) {
			if expected_shape[i] != 0 && t.shape[i] != expected_shape[i] {
				fmt.panicf("%s: expected shape[%d]=%d, got %d", name, i, expected_shape[i], t.shape[i])
			}
		}
	}
	return t
}

@(private = "file")
load_linear :: proc(sf: ^st.Safe_Tensors($T), weight_name: string, in_features, out_features: uint, allocator := context.allocator) -> ^nn.Linear(T) {
	// PyTorch convention: weight is [out_features, in_features], no transpose needed
	// Clone tensor so Linear owns it (original is in safetensors cache)
	cached := get_tensor(sf, weight_name, []uint{out_features, in_features})
	w := tensor.clone(cached, allocator)
	return new_clone(nn.Linear(T){w = w, b = nil}, allocator)
}

// Load shared weights on first forward
@(private = "file")
ensure_shared_loaded :: proc(tf: ^Transformer($T), allocator := context.allocator) {
	if tf._loaded do return

	sf := tf._sf
	h := tf.config.hidden_size
	latent_ch : uint = 128
	text_dim : uint = 7680

	tf.img_in = load_linear(sf, "x_embedder.weight", latent_ch, h, allocator)
	tf.txt_in = load_linear(sf, "context_embedder.weight", text_dim, h, allocator)
	tf.time_embed.fc1 = load_linear(sf, "time_guidance_embed.timestep_embedder.linear_1.weight", 256, h, allocator)
	tf.time_embed.fc2 = load_linear(sf, "time_guidance_embed.timestep_embedder.linear_2.weight", h, h, allocator)
	tf.final_mod = load_linear(sf, "norm_out.linear.weight", h, h * 2, allocator)
	tf.final_proj = load_linear(sf, "proj_out.weight", h, latent_ch, allocator)
	tf.mod_img = load_linear(sf, "double_stream_modulation_img.linear.weight", h, h * 6, allocator)
	tf.mod_txt = load_linear(sf, "double_stream_modulation_txt.linear.weight", h, h * 6, allocator)
	tf.mod_single = load_linear(sf, "single_stream_modulation.linear.weight", h, h * 3, allocator)

	tf._loaded = true
}

// Load double block weights
@(private = "file")
load_double_block :: proc(sf: ^st.Safe_Tensors($T), idx: int, h, mlp_h, head_dim: uint, allocator := context.allocator) -> Double_Block_Weights(T) {
	_t := trace.global_scoped("load_double_block", "weight_loading")
	defer trace.global_end_scoped(_t)

	prefix := fmt.tprintf("transformer_blocks.%d", idx)
	return Double_Block_Weights(T){
		img_to_q    = load_linear(sf, fmt.tprintf("%s.attn.to_q.weight", prefix), h, h, allocator),
		img_to_k    = load_linear(sf, fmt.tprintf("%s.attn.to_k.weight", prefix), h, h, allocator),
		img_to_v    = load_linear(sf, fmt.tprintf("%s.attn.to_v.weight", prefix), h, h, allocator),
		img_norm_q  = get_tensor(sf, fmt.tprintf("%s.attn.norm_q.weight", prefix), []uint{head_dim}),
		img_norm_k  = get_tensor(sf, fmt.tprintf("%s.attn.norm_k.weight", prefix), []uint{head_dim}),
		img_proj    = load_linear(sf, fmt.tprintf("%s.attn.to_out.0.weight", prefix), h, h, allocator),
		img_mlp_fc1 = load_linear(sf, fmt.tprintf("%s.ff.linear_in.weight", prefix), h, mlp_h * 2, allocator),
		img_mlp_fc2 = load_linear(sf, fmt.tprintf("%s.ff.linear_out.weight", prefix), mlp_h, h, allocator),
		txt_to_q    = load_linear(sf, fmt.tprintf("%s.attn.add_q_proj.weight", prefix), h, h, allocator),
		txt_to_k    = load_linear(sf, fmt.tprintf("%s.attn.add_k_proj.weight", prefix), h, h, allocator),
		txt_to_v    = load_linear(sf, fmt.tprintf("%s.attn.add_v_proj.weight", prefix), h, h, allocator),
		txt_norm_q  = get_tensor(sf, fmt.tprintf("%s.attn.norm_added_q.weight", prefix), []uint{head_dim}),
		txt_norm_k  = get_tensor(sf, fmt.tprintf("%s.attn.norm_added_k.weight", prefix), []uint{head_dim}),
		txt_proj    = load_linear(sf, fmt.tprintf("%s.attn.to_add_out.weight", prefix), h, h, allocator),
		txt_mlp_fc1 = load_linear(sf, fmt.tprintf("%s.ff_context.linear_in.weight", prefix), h, mlp_h * 2, allocator),
		txt_mlp_fc2 = load_linear(sf, fmt.tprintf("%s.ff_context.linear_out.weight", prefix), mlp_h, h, allocator),
	}
}

// Free double block weights
@(private = "file")
free_double_block :: proc(b: ^Double_Block_Weights($T), allocator := context.allocator) {
	_t := trace.global_scoped("free_double_block", "weight_loading")
	defer trace.global_end_scoped(_t)

	nn.free_linear(b.img_to_q, allocator)
	nn.free_linear(b.img_to_k, allocator)
	nn.free_linear(b.img_to_v, allocator)
	nn.free_linear(b.img_proj, allocator)
	nn.free_linear(b.img_mlp_fc1, allocator)
	nn.free_linear(b.img_mlp_fc2, allocator)
	nn.free_linear(b.txt_to_q, allocator)
	nn.free_linear(b.txt_to_k, allocator)
	nn.free_linear(b.txt_to_v, allocator)
	nn.free_linear(b.txt_proj, allocator)
	nn.free_linear(b.txt_mlp_fc1, allocator)
	nn.free_linear(b.txt_mlp_fc2, allocator)
	// norm tensors point into mmap, don't free
}

// Load single block weights
// NOTE(Aria): This is slow, but intentional design to make things run in limited memory.
@(private = "file")
load_single_block :: proc(sf: ^st.Safe_Tensors($T), idx: int, h, mlp_h, head_dim: uint, allocator := context.allocator) -> Single_Block_Weights(T) {
	_t := trace.global_scoped("load_single_block", "weight_loading")
	defer trace.global_end_scoped(_t)

	prefix := fmt.tprintf("single_transformer_blocks.%d", idx)
	return Single_Block_Weights(T){
		qkv_mlp = load_linear(sf, fmt.tprintf("%s.attn.to_qkv_mlp_proj.weight", prefix), h, h * 3 + mlp_h * 2, allocator),
		norm_q  = get_tensor(sf, fmt.tprintf("%s.attn.norm_q.weight", prefix), []uint{head_dim}),
		norm_k  = get_tensor(sf, fmt.tprintf("%s.attn.norm_k.weight", prefix), []uint{head_dim}),
		proj    = load_linear(sf, fmt.tprintf("%s.attn.to_out.weight", prefix), h + mlp_h, h, allocator),
	}
}

// Free single block weights
@(private = "file")
free_single_block :: proc(b: ^Single_Block_Weights($T), allocator := context.allocator) {
	_t := trace.global_scoped("free_single_block", "weight_loading")
	defer trace.global_end_scoped(_t)

	nn.free_linear(b.qkv_mlp, allocator)
	nn.free_linear(b.proj, allocator)
	// norm tensors point into mmap, don't free
}

// ============================================================================
// Operations
// ============================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
rms_norm :: proc(x: ^tensor.Tensor($T), weight: ^tensor.Tensor(T), eps: T, allocator := context.allocator) -> ^tensor.Tensor(T) {
	result := tensor.tensor_alloc(T, x.shape, true, allocator)
	last_dim := x.shape[len(x.shape) - 1]
	num_vectors := 1
	for i in 0 ..< len(x.shape) - 1 {
		num_vectors *= int(x.shape[i])
	}

	for v in 0 ..< num_vectors {
		offset := v * int(last_dim)
		sq_sum: T = 0
		for i in 0 ..< int(last_dim) {
			val := x.data[offset + i]
			sq_sum += val * val
		}
		inv_rms := T(1.0) / math.sqrt(sq_sum / T(last_dim) + eps)
		for i in 0 ..< int(last_dim) {
			result.data[offset + i] = x.data[offset + i] * inv_rms * weight.data[i]
		}
	}
	return result
}

silu :: proc(x: $T) -> T {
	return x / (T(1.0) + math.exp(-x))
}

// SwiGLU MLP
// x: [batch, seq, hidden] -> [batch, seq, hidden]
swiglu_mlp :: proc(x: ^tensor.Tensor($T), fc1, fc2: ^nn.Linear(T), allocator := context.allocator) -> ^tensor.Tensor(T) {
	// fc1: [batch, seq, hidden] -> [batch, seq, mlp_h*2]
	hidden := nn.forward_linear(fc1, x, allocator)
	defer tensor.free_tensor(hidden, allocator)

	// Preserve original shape structure
	mlp_hidden := hidden.shape[len(hidden.shape) - 1] / 2

	// Allocate gated with same shape as hidden but half the last dim
	gated_shape := make([]uint, len(hidden.shape), context.temp_allocator)
	for i in 0 ..< len(hidden.shape) - 1 {
		gated_shape[i] = hidden.shape[i]
	}
	gated_shape[len(hidden.shape) - 1] = mlp_hidden

	gated := tensor.tensor_alloc(T, gated_shape, true, allocator)

	// Compute SiLU(gate) * up for each position
	num_positions := 1
	for i in 0 ..< len(hidden.shape) - 1 {
		num_positions *= int(hidden.shape[i])
	}

	for p in 0 ..< num_positions {
		for i in 0 ..< int(mlp_hidden) {
			gate_val := hidden.data[p * int(mlp_hidden * 2) + i]
			up_val := hidden.data[p * int(mlp_hidden * 2) + int(mlp_hidden) + i]
			gated.data[p * int(mlp_hidden) + i] = silu(gate_val) * up_val
		}
	}

	// fc2: [batch, seq, mlp_h] -> [batch, seq, hidden]
	result := nn.forward_linear(fc2, gated, allocator)
	tensor.free_tensor(gated, allocator)
	return result
}

// Timestep embedding (sinusoidal, cos first like antirez)
timestep_embedding :: proc($T: typeid, timestep: T, dim: uint, allocator := context.allocator) -> ^tensor.Tensor(T) {
	half := dim / 2
	emb := tensor.tensor_alloc(T, []uint{1, dim}, true, allocator)
	log_max := math.ln(T(10000.0))

	for i in 0 ..< int(half) {
		freq := math.exp(-log_max * T(i) / T(half))
		angle := timestep * T(1000.0) * freq
		emb.data[i] = math.cos(angle)
		emb.data[int(half) + i] = math.sin(angle)
	}
	return emb
}

// Time embedding forward
time_embed_forward :: proc(te: ^Time_Embed($T), t: T, allocator := context.allocator) -> ^tensor.Tensor(T) {
	t_sincos := timestep_embedding(T, t, 256, allocator)
	defer tensor.free_tensor(t_sincos, allocator)

	h := nn.forward_linear(te.fc1, t_sincos, allocator)
	defer tensor.free_tensor(h, allocator)

	for i in 0 ..< len(h.data) {
		h.data[i] = silu(h.data[i])
	}

	return nn.forward_linear(te.fc2, h, allocator)
}

// Slice along last dimension
slice_last_dim :: proc(t: ^tensor.Tensor($T), start, end: uint, allocator := context.allocator) -> ^tensor.Tensor(T) {
	rank := len(t.shape)
	slices := make([]tensor.Slice, rank, context.temp_allocator)
	for i in 0 ..< rank - 1 {
		slices[i] = tensor.Range{}
	}
	slices[rank - 1] = tensor.Range{int(start), int(end), 1}
	return tensor.slice(t, slices, false, allocator)
}

// Slice along specific dimension
slice_dim :: proc(t: ^tensor.Tensor($T), dim: int, start, end: uint, allocator := context.allocator) -> ^tensor.Tensor(T) {
	rank := len(t.shape)
	slices := make([]tensor.Slice, rank, context.temp_allocator)
	for i in 0 ..< rank {
		slices[i] = dim == i ? tensor.Range{int(start), int(end), 1} : tensor.Range{}
	}
	return tensor.slice(t, slices, false, allocator)
}

// QK-Norm: RMSNorm applied per-head
// x: [batch, seq, num_heads * head_dim], weight: [head_dim]
qk_norm :: proc(x: ^tensor.Tensor($T), weight: ^tensor.Tensor(T), num_heads, head_dim: uint, eps: T, allocator := context.allocator) -> ^tensor.Tensor(T) {
	result := tensor.tensor_alloc(T, x.shape, true, allocator)
	batch := x.shape[0]
	seq_len := x.shape[1]

	for b in 0 ..< int(batch) {
		for s in 0 ..< int(seq_len) {
			for h in 0 ..< int(num_heads) {
				base := b * int(seq_len * num_heads * head_dim) + s * int(num_heads * head_dim) + h * int(head_dim)
				// Compute RMS for this head
				sq_sum: T = 0
				for d in 0 ..< int(head_dim) {
					val := x.data[base + d]
					sq_sum += val * val
				}
				inv_rms := T(1.0) / math.sqrt(sq_sum / T(head_dim) + eps)
				// Apply norm with weight
				for d in 0 ..< int(head_dim) {
					result.data[base + d] = x.data[base + d] * inv_rms * weight.data[d]
				}
			}
		}
	}
	return result
}

// Attention with QK-Norm and RoPE
attention_with_rope :: proc(
	q, k, v: ^tensor.Tensor($T),
	q_norm, k_norm: ^tensor.Tensor(T),
	freqs: ^tensor.Tensor(T),
	num_heads, head_dim: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("attention_with_rope", "attention")
	defer trace.global_end_scoped(_t)

	batch := q.shape[0]
	seq_len := q.shape[1]
	eps := T(1e-6)

	q_normed := qk_norm(q, q_norm, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(q_normed, allocator)
	k_normed := qk_norm(k, k_norm, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(k_normed, allocator)

	apply_rope_flux_inplace(q_normed, freqs, num_heads, head_dim)
	apply_rope_flux_inplace(k_normed, freqs, num_heads, head_dim)

	q_heads := tensor.reshape(q_normed, []uint{batch, seq_len, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(q_heads, allocator)
	k_heads := tensor.reshape(k_normed, []uint{batch, seq_len, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(k_heads, allocator)
	v_heads := tensor.reshape(v, []uint{batch, seq_len, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(v_heads, allocator)

	q_t := tensor.transpose(q_heads, 1, 2, allocator)
	defer tensor.free_tensor(q_t, allocator)
	k_t := tensor.transpose(k_heads, 1, 2, allocator)
	defer tensor.free_tensor(k_t, allocator)
	v_t := tensor.transpose(v_heads, 1, 2, allocator)
	defer tensor.free_tensor(v_t, allocator)

	scale := T(1.0) / math.sqrt(T(head_dim))

	k_transposed := tensor.transpose(k_t, 2, 3, allocator)
	defer tensor.free_tensor(k_transposed, allocator)

	scores := tensor.matmul(q_t, k_transposed, allocator)
	defer tensor.free_tensor(scores, allocator)

	for i in 0 ..< len(scores.data) {
		scores.data[i] *= scale
	}
	tensor.softmax_last_dim_inplace(scores)

	attn_out := tensor.matmul(scores, v_t, allocator)
	attn_t := tensor.transpose(attn_out, 1, 2, allocator)
	tensor.free_tensor(attn_out, allocator)

	result := tensor.reshape(attn_t, []uint{batch, seq_len, num_heads * head_dim}, allocator)
	tensor.free_tensor(attn_t, allocator)

	return result
}

// Attention without QK-Norm or RoPE (already applied externally)
// q: [batch, q_seq, hidden], k/v: [batch, kv_seq, hidden]
// Supports different sequence lengths for Q vs K/V (cross-attention style)
attention_no_rope :: proc(
	q, k, v: ^tensor.Tensor($T),
	num_heads, head_dim: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("attention_no_rope", "attention")
	defer trace.global_end_scoped(_t)

	batch := q.shape[0]
	q_seq := q.shape[1]
	kv_seq := k.shape[1]

	// Reshape to [batch, seq, num_heads, head_dim]
	q_heads := tensor.reshape(q, []uint{batch, q_seq, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(q_heads, allocator)
	k_heads := tensor.reshape(k, []uint{batch, kv_seq, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(k_heads, allocator)
	v_heads := tensor.reshape(v, []uint{batch, kv_seq, num_heads, head_dim}, allocator)
	defer tensor.free_tensor(v_heads, allocator)

	// Transpose to [batch, num_heads, seq, head_dim]
	q_t := tensor.transpose(q_heads, 1, 2, allocator)
	defer tensor.free_tensor(q_t, allocator)
	k_t := tensor.transpose(k_heads, 1, 2, allocator)
	defer tensor.free_tensor(k_t, allocator)
	v_t := tensor.transpose(v_heads, 1, 2, allocator)
	defer tensor.free_tensor(v_t, allocator)

	scale := T(1.0) / math.sqrt(T(head_dim))

	// K^T: [batch, num_heads, head_dim, kv_seq]
	k_transposed := tensor.transpose(k_t, 2, 3, allocator)
	defer tensor.free_tensor(k_transposed, allocator)

	// scores: [batch, num_heads, q_seq, kv_seq]
	scores := tensor.matmul(q_t, k_transposed, allocator)
	defer tensor.free_tensor(scores, allocator)

	for i in 0 ..< len(scores.data) {
		scores.data[i] *= scale
	}
	tensor.softmax_last_dim_inplace(scores)

	// attn_out: [batch, num_heads, q_seq, head_dim]
	attn_out := tensor.matmul(scores, v_t, allocator)
	// Transpose back: [batch, q_seq, num_heads, head_dim]
	attn_t := tensor.transpose(attn_out, 1, 2, allocator)
	tensor.free_tensor(attn_out, allocator)

	// Reshape to [batch, q_seq, hidden]
	result := tensor.reshape(attn_t, []uint{batch, q_seq, num_heads * head_dim}, allocator)
	tensor.free_tensor(attn_t, allocator)

	return result
}

// ============================================================================
// Block Forward Passes
// ============================================================================

// Double block forward (MM-DiT: separate img/txt streams, joint attention)
// Per antirez: Q stays separate, only K/V concatenated, RoPE applied BEFORE concat
double_block_forward :: proc(
	b: ^Double_Block_Weights($T),
	img, txt: ^tensor.Tensor(T),
	t_emb: ^tensor.Tensor(T),
	mod_img, mod_txt: ^nn.Linear(T),
	img_rope, txt_rope: ^tensor.Tensor(T),  // Separate RoPE for each stream
	num_heads, head_dim: uint,
	allocator := context.allocator,
) -> (img_out, txt_out: ^tensor.Tensor(T)) {
	_t := trace.global_scoped("double_block", "transformer")
	defer trace.global_end_scoped(_t)

	hidden := num_heads * head_dim
	eps := T(1e-6)

	// Compute modulation (SiLU then linear)
	t_silu := tensor.tensor_alloc(T, t_emb.shape, true, allocator)
	defer tensor.free_tensor(t_silu, allocator)
	for i in 0 ..< len(t_emb.data) {
		t_silu.data[i] = silu(t_emb.data[i])
	}

	img_mod := nn.forward_linear(mod_img, t_silu, allocator)
	defer tensor.free_tensor(img_mod, allocator)
	txt_mod := nn.forward_linear(mod_txt, t_silu, allocator)
	defer tensor.free_tensor(txt_mod, allocator)

	// Split modulation: shift1, scale1, gate1, shift2, scale2, gate2
	mod_dim := hidden
	img_shift1 := slice_last_dim(img_mod, 0, mod_dim, allocator)
	defer tensor.free_tensor(img_shift1, allocator)
	img_scale1 := slice_last_dim(img_mod, mod_dim, 2*mod_dim, allocator)
	defer tensor.free_tensor(img_scale1, allocator)
	img_gate1 := slice_last_dim(img_mod, 2*mod_dim, 3*mod_dim, allocator)
	defer tensor.free_tensor(img_gate1, allocator)
	img_shift2 := slice_last_dim(img_mod, 3*mod_dim, 4*mod_dim, allocator)
	defer tensor.free_tensor(img_shift2, allocator)
	img_scale2 := slice_last_dim(img_mod, 4*mod_dim, 5*mod_dim, allocator)
	defer tensor.free_tensor(img_scale2, allocator)
	img_gate2 := slice_last_dim(img_mod, 5*mod_dim, 6*mod_dim, allocator)
	defer tensor.free_tensor(img_gate2, allocator)

	txt_shift1 := slice_last_dim(txt_mod, 0, mod_dim, allocator)
	defer tensor.free_tensor(txt_shift1, allocator)
	txt_scale1 := slice_last_dim(txt_mod, mod_dim, 2*mod_dim, allocator)
	defer tensor.free_tensor(txt_scale1, allocator)
	txt_gate1 := slice_last_dim(txt_mod, 2*mod_dim, 3*mod_dim, allocator)
	defer tensor.free_tensor(txt_gate1, allocator)
	txt_shift2 := slice_last_dim(txt_mod, 3*mod_dim, 4*mod_dim, allocator)
	defer tensor.free_tensor(txt_shift2, allocator)
	txt_scale2 := slice_last_dim(txt_mod, 4*mod_dim, 5*mod_dim, allocator)
	defer tensor.free_tensor(txt_scale2, allocator)
	txt_gate2 := slice_last_dim(txt_mod, 5*mod_dim, 6*mod_dim, allocator)
	defer tensor.free_tensor(txt_gate2, allocator)

	// === Image stream: AdaLN + QKV projection ===
	img_normed := layer_norm_adaln(img, img_shift1, img_scale1, eps, allocator)
	defer tensor.free_tensor(img_normed, allocator)

	img_q := nn.forward_linear(b.img_to_q, img_normed, allocator)
	defer tensor.free_tensor(img_q, allocator)
	img_k := nn.forward_linear(b.img_to_k, img_normed, allocator)
	defer tensor.free_tensor(img_k, allocator)
	img_v := nn.forward_linear(b.img_to_v, img_normed, allocator)
	defer tensor.free_tensor(img_v, allocator)

	// === Text stream: AdaLN + QKV projection ===
	txt_normed := layer_norm_adaln(txt, txt_shift1, txt_scale1, eps, allocator)
	defer tensor.free_tensor(txt_normed, allocator)

	txt_q := nn.forward_linear(b.txt_to_q, txt_normed, allocator)
	defer tensor.free_tensor(txt_q, allocator)
	txt_k := nn.forward_linear(b.txt_to_k, txt_normed, allocator)
	defer tensor.free_tensor(txt_k, allocator)
	txt_v := nn.forward_linear(b.txt_to_v, txt_normed, allocator)
	defer tensor.free_tensor(txt_v, allocator)

	// === Apply QK-norm SEPARATELY with correct weights per stream ===
	img_q_normed := qk_norm(img_q, b.img_norm_q, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(img_q_normed, allocator)
	img_k_normed := qk_norm(img_k, b.img_norm_k, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(img_k_normed, allocator)

	txt_q_normed := qk_norm(txt_q, b.txt_norm_q, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(txt_q_normed, allocator)
	txt_k_normed := qk_norm(txt_k, b.txt_norm_k, num_heads, head_dim, eps, allocator)
	defer tensor.free_tensor(txt_k_normed, allocator)

	// === Apply RoPE SEPARATELY before concatenation ===
	// img uses 2D spatial (axes 1,2), txt uses 1D sequential (axis 3)
	apply_rope_flux_inplace(img_q_normed, img_rope, num_heads, head_dim)
	apply_rope_flux_inplace(img_k_normed, img_rope, num_heads, head_dim)
	apply_rope_flux_inplace(txt_q_normed, txt_rope, num_heads, head_dim)
	apply_rope_flux_inplace(txt_k_normed, txt_rope, num_heads, head_dim)

	// === Joint attention: concatenate ONLY K and V (Q stays separate!) ===
	// Order: [txt, img] per antirez
	cat_k := tensor.cat([]^tensor.Tensor(T){txt_k_normed, img_k_normed}, 1, allocator)
	defer tensor.free_tensor(cat_k, allocator)
	cat_v := tensor.cat([]^tensor.Tensor(T){txt_v, img_v}, 1, allocator)
	defer tensor.free_tensor(cat_v, allocator)

	// === Two SEPARATE attention calls ===
	img_attn := attention_no_rope(img_q_normed, cat_k, cat_v, num_heads, head_dim, allocator)
	defer tensor.free_tensor(img_attn, allocator)
	txt_attn := attention_no_rope(txt_q_normed, cat_k, cat_v, num_heads, head_dim, allocator)
	defer tensor.free_tensor(txt_attn, allocator)

	// Project and gate
	img_attn_proj := nn.forward_linear(b.img_proj, img_attn, allocator)
	defer tensor.free_tensor(img_attn_proj, allocator)
	apply_gate_inplace(img_attn_proj, img_gate1)

	txt_attn_proj := nn.forward_linear(b.txt_proj, txt_attn, allocator)
	defer tensor.free_tensor(txt_attn_proj, allocator)
	apply_gate_inplace(txt_attn_proj, txt_gate1)

	// Residual
	img_res := tensor.add(img, img_attn_proj, allocator)
	txt_res := tensor.add(txt, txt_attn_proj, allocator)

	// === MLP ===
	img_normed2 := layer_norm_adaln(img_res, img_shift2, img_scale2, eps, allocator)
	defer tensor.free_tensor(img_normed2, allocator)
	img_mlp := swiglu_mlp(img_normed2, b.img_mlp_fc1, b.img_mlp_fc2, allocator)
	defer tensor.free_tensor(img_mlp, allocator)
	apply_gate_inplace(img_mlp, img_gate2)
	img_out = tensor.add(img_res, img_mlp, allocator)
	tensor.free_tensor(img_res, allocator)

	txt_normed2 := layer_norm_adaln(txt_res, txt_shift2, txt_scale2, eps, allocator)
	defer tensor.free_tensor(txt_normed2, allocator)
	txt_mlp := swiglu_mlp(txt_normed2, b.txt_mlp_fc1, b.txt_mlp_fc2, allocator)
	defer tensor.free_tensor(txt_mlp, allocator)
	apply_gate_inplace(txt_mlp, txt_gate2)
	txt_out = tensor.add(txt_res, txt_mlp, allocator)
	tensor.free_tensor(txt_res, allocator)

	return
}

// Single block forward (parallel attention + MLP)
single_block_forward :: proc(
	b: ^Single_Block_Weights($T),
	x: ^tensor.Tensor(T),
	t_emb: ^tensor.Tensor(T),
	mod_single: ^nn.Linear(T),
	freqs: ^tensor.Tensor(T),
	num_heads, head_dim, mlp_hidden: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("single_block", "transformer")
	defer trace.global_end_scoped(_t)

	hidden := num_heads * head_dim
	eps := T(1e-6)

	// Modulation
	t_silu := tensor.tensor_alloc(T, t_emb.shape, true, allocator)
	defer tensor.free_tensor(t_silu, allocator)
	for i in 0 ..< len(t_emb.data) {
		t_silu.data[i] = silu(t_emb.data[i])
	}

	mod := nn.forward_linear(mod_single, t_silu, allocator)
	defer tensor.free_tensor(mod, allocator)

	shift := slice_last_dim(mod, 0, hidden, allocator)
	defer tensor.free_tensor(shift, allocator)
	scale := slice_last_dim(mod, hidden, 2*hidden, allocator)
	defer tensor.free_tensor(scale, allocator)
	gate := slice_last_dim(mod, 2*hidden, 3*hidden, allocator)
	defer tensor.free_tensor(gate, allocator)

	// AdaLN + fused projection
	normed := layer_norm_adaln(x, shift, scale, eps, allocator)
	defer tensor.free_tensor(normed, allocator)

	qkv_mlp := nn.forward_linear(b.qkv_mlp, normed, allocator)
	defer tensor.free_tensor(qkv_mlp, allocator)

	// Split: Q, K, V, gate, up
	q := slice_last_dim(qkv_mlp, 0, hidden, allocator)
	defer tensor.free_tensor(q, allocator)
	k := slice_last_dim(qkv_mlp, hidden, 2*hidden, allocator)
	defer tensor.free_tensor(k, allocator)
	v := slice_last_dim(qkv_mlp, 2*hidden, 3*hidden, allocator)
	defer tensor.free_tensor(v, allocator)
	mlp_gate := slice_last_dim(qkv_mlp, 3*hidden, 3*hidden + mlp_hidden, allocator)
	defer tensor.free_tensor(mlp_gate, allocator)
	mlp_up := slice_last_dim(qkv_mlp, 3*hidden + mlp_hidden, 3*hidden + 2*mlp_hidden, allocator)
	defer tensor.free_tensor(mlp_up, allocator)

	// Attention
	attn_out := attention_with_rope(q, k, v, b.norm_q, b.norm_k, freqs, num_heads, head_dim, allocator)
	defer tensor.free_tensor(attn_out, allocator)

	// SwiGLU for MLP part
	gated := tensor.tensor_alloc(T, mlp_gate.shape, true, allocator)
	defer tensor.free_tensor(gated, allocator)
	for i in 0 ..< len(mlp_gate.data) {
		gated.data[i] = silu(mlp_gate.data[i]) * mlp_up.data[i]
	}

	// Fused projection: [attn_out, gated] -> proj
	// proj.weight is [3072, 12288] = [hidden, hidden + mlp_hidden]
	// So we concat attn_out and gated, then project
	concat := tensor.cat([]^tensor.Tensor(T){attn_out, gated}, 2, allocator)
	defer tensor.free_tensor(concat, allocator)

	proj_out := nn.forward_linear(b.proj, concat, allocator)
	defer tensor.free_tensor(proj_out, allocator)

	apply_gate_inplace(proj_out, gate)

	return tensor.add(x, proj_out, allocator)
}

// LayerNorm with AdaLN modulation: (1 + scale) * LayerNorm(x) + shift
layer_norm_adaln :: proc(x: ^tensor.Tensor($T), shift, scale: ^tensor.Tensor(T), eps: T, allocator := context.allocator) -> ^tensor.Tensor(T) {
	result := tensor.tensor_alloc(T, x.shape, true, allocator)
	last_dim := int(x.shape[len(x.shape) - 1])
	num_vectors := 1
	for i in 0 ..< len(x.shape) - 1 {
		num_vectors *= int(x.shape[i])
	}

	for v in 0 ..< num_vectors {
		offset := v * last_dim

		// Mean
		mean: T = 0
		for i in 0 ..< last_dim {
			mean += x.data[offset + i]
		}
		mean /= T(last_dim)

		// Variance
		var: T = 0
		for i in 0 ..< last_dim {
			d := x.data[offset + i] - mean
			var += d * d
		}
		var /= T(last_dim)
		inv_std := T(1.0) / math.sqrt(var + eps)

		// Normalize and modulate
		for i in 0 ..< last_dim {
			normed := (x.data[offset + i] - mean) * inv_std
			result.data[offset + i] = (T(1.0) + scale.data[i]) * normed + shift.data[i]
		}
	}
	return result
}

// Apply gate inplace
apply_gate_inplace :: proc(x: ^tensor.Tensor($T), gate: ^tensor.Tensor(T)) {
	last_dim := int(x.shape[len(x.shape) - 1])
	for i in 0 ..< len(x.data) {
		x.data[i] *= gate.data[i % last_dim]
	}
}

// ============================================================================
// Transformer Forward
// ============================================================================

transformer_forward :: proc(
	tf: ^Transformer($T),
	img_latent: ^tensor.Tensor(T),
	txt_emb: ^tensor.Tensor(T),
	timestep: T,
	img_height, img_width: uint,  // Needed for 4-axis RoPE
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("transformer_forward", "transformer")
	defer trace.global_end_scoped(_t)

	// Load shared weights on first call
	ensure_shared_loaded(tf, allocator)

	txt_len := txt_emb.shape[1]

	// Compute separate RoPE for double blocks (applied before K/V concat)
	img_rope := compute_rope_freqs_img(T, img_height, img_width, tf.config.head_dim, tf.config.rope_theta, allocator)
	defer tensor.free_tensor(img_rope, allocator)
	txt_rope := compute_rope_freqs_txt(T, txt_len, tf.config.head_dim, tf.config.rope_theta, allocator)
	defer tensor.free_tensor(txt_rope, allocator)

	// Compute unified RoPE for single blocks (txt first, then img)
	rope_freqs := compute_rope_freqs_flux(T, txt_len, img_height, img_width, tf.config.head_dim, tf.config.rope_theta, allocator)
	defer tensor.free_tensor(rope_freqs, allocator)

	// Time embedding (used only for modulation, NOT added directly to img)
	t_emb := time_embed_forward(&tf.time_embed, timestep, allocator)
	defer tensor.free_tensor(t_emb, allocator)

	// Project inputs
	img := nn.forward_linear(tf.img_in, img_latent, allocator)
	txt := nn.forward_linear(tf.txt_in, txt_emb, allocator)

	// Double blocks (load -> forward -> free)
	for i in 0 ..< int(tf.config.num_double_layers) {
		block := load_double_block(tf._sf, i, tf.config.hidden_size, tf.config.mlp_hidden, tf.config.head_dim, allocator)
		img_new, txt_new := double_block_forward(&block, img, txt, t_emb, tf.mod_img, tf.mod_txt, img_rope, txt_rope, tf.config.num_heads, tf.config.head_dim, allocator)
		free_double_block(&block, allocator)

		tensor.free_tensor(img, allocator)
		tensor.free_tensor(txt, allocator)
		img = img_new
		txt = txt_new
	}

	// Concatenate for single blocks - [txt, img] order to match double blocks
	txt_seq_len := txt_emb.shape[1]
	combined := tensor.cat([]^tensor.Tensor(T){txt, img}, 1, allocator)
	tensor.free_tensor(img, allocator)
	tensor.free_tensor(txt, allocator)

	// Single blocks (load -> forward -> free)
	for i in 0 ..< int(tf.config.num_single_layers) {
		block := load_single_block(tf._sf, i, tf.config.hidden_size, tf.config.mlp_hidden, tf.config.head_dim, allocator)
		combined_new := single_block_forward(&block, combined, t_emb, tf.mod_single, rope_freqs, tf.config.num_heads, tf.config.head_dim, tf.config.mlp_hidden, allocator)
		free_single_block(&block, allocator)

		tensor.free_tensor(combined, allocator)
		combined = combined_new
	}

	// Extract image portion (comes after txt in [txt, img] order)
	img_out := slice_dim(combined, 1, txt_seq_len, combined.shape[1], allocator)
	tensor.free_tensor(combined, allocator)

	// Final modulation
	hidden := tf.config.num_heads * tf.config.head_dim
	eps := T(1e-6)

	t_silu := tensor.tensor_alloc(T, t_emb.shape, true, allocator)
	defer tensor.free_tensor(t_silu, allocator)
	for i in 0 ..< len(t_emb.data) {
		t_silu.data[i] = silu(t_emb.data[i])
	}

	final_mod := nn.forward_linear(tf.final_mod, t_silu, allocator)
	defer tensor.free_tensor(final_mod, allocator)

	final_scale := slice_last_dim(final_mod, 0, hidden, allocator)
	defer tensor.free_tensor(final_scale, allocator)
	final_shift := slice_last_dim(final_mod, hidden, 2*hidden, allocator)
	defer tensor.free_tensor(final_shift, allocator)

	normed := layer_norm_adaln(img_out, final_shift, final_scale, eps, allocator)
	tensor.free_tensor(img_out, allocator)

	result := nn.forward_linear(tf.final_proj, normed, allocator)
	tensor.free_tensor(normed, allocator)

	return result
}
