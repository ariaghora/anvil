// Qwen3 Text Encoder for FLUX
//
// Qwen3-4B architecture:
// - 36 layers
// - 2560 hidden dim
// - 512 max sequence length
// - 7680 dim output embeddings
// - RoPE positional encoding
// - RMSNorm
// - SwiGLU MLP

package flux

import "../../nn"
import "../../tensor"
import vb "../sam/var_builder"
import st "../../safetensors"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strings"

// Qwen3 configuration
QWEN3_VOCAB_SIZE :: 151936
QWEN3_HIDDEN :: 2560
QWEN3_INTERMEDIATE :: 6912 // MLP hidden
QWEN3_NUM_HEADS :: 32
QWEN3_NUM_KV_HEADS :: 4 // Grouped query attention
QWEN3_HEAD_DIM :: 80
QWEN3_LAYERS :: 36
QWEN3_MAX_SEQ :: 512
QWEN3_OUTPUT_DIM :: 7680
QWEN3_ROPE_THETA :: 1000000.0

// Attention weights for Qwen3
Qwen3_Attention :: struct($T: typeid) {
	q_proj: ^nn.Linear(T),
	k_proj: ^nn.Linear(T),
	v_proj: ^nn.Linear(T),
	o_proj: ^nn.Linear(T),
	q_norm: ^tensor.Tensor(T), // QK-Norm
	k_norm: ^tensor.Tensor(T),
}

// MLP weights for Qwen3 (SwiGLU)
Qwen3_MLP :: struct($T: typeid) {
	gate_proj: ^nn.Linear(T), // Gate
	up_proj:   ^nn.Linear(T), // Up
	down_proj: ^nn.Linear(T), // Down
}

// Single Qwen3 layer
Qwen3_Layer :: struct($T: typeid) {
	self_attn:                 Qwen3_Attention(T),
	mlp:                       Qwen3_MLP(T),
	input_layernorm:           ^tensor.Tensor(T), // RMSNorm weight
	post_attention_layernorm:  ^tensor.Tensor(T),
}

// Qwen3 encoder context
Qwen3 :: struct($T: typeid) {
	embed_tokens:   ^tensor.Tensor(T), // [vocab_size, hidden]
	layers:         []Qwen3_Layer(T),
	norm:           ^tensor.Tensor(T), // Final RMSNorm
	output_proj:    ^nn.Linear(T), // Project to FLUX text dim
	rope_freqs:     ^tensor.Tensor(T), // Precomputed RoPE
	config:         Qwen3_Config,
}

Qwen3_Config :: struct {
	vocab_size:       uint,
	hidden_size:      uint,
	intermediate_size: uint,
	num_heads:        uint,
	num_kv_heads:     uint,
	head_dim:         uint,
	num_layers:       uint,
	max_seq_len:      uint,
	output_dim:       uint,
	rope_theta:       f32,
	rms_norm_eps:     f32,
}

qwen3_config_default :: proc() -> Qwen3_Config {
	return Qwen3_Config {
		vocab_size        = QWEN3_VOCAB_SIZE,
		hidden_size       = QWEN3_HIDDEN,
		intermediate_size = QWEN3_INTERMEDIATE,
		num_heads         = QWEN3_NUM_HEADS,
		num_kv_heads      = QWEN3_NUM_KV_HEADS,
		head_dim          = QWEN3_HEAD_DIM,
		num_layers        = QWEN3_LAYERS,
		max_seq_len       = QWEN3_MAX_SEQ,
		output_dim        = QWEN3_OUTPUT_DIM,
		rope_theta        = QWEN3_ROPE_THETA,
		rms_norm_eps      = 1e-6,
	}
}

// Load Qwen3 encoder from sharded safetensors
load_qwen3 :: proc(
	$T: typeid,
	model_dir: string,
	allocator := context.allocator,
) -> (enc: ^Qwen3(T), err: string) {
	enc = new(Qwen3(T), allocator)
	enc.config = qwen3_config_default()
	cfg := enc.config

	// Allocate tensors and layers
	enc.embed_tokens = tensor.tensor_alloc(T, []uint{cfg.vocab_size, cfg.hidden_size}, true, allocator)
	enc.norm = tensor.tensor_alloc(T, []uint{cfg.hidden_size}, true, allocator)
	enc.output_proj = nn.new_linear(T, cfg.hidden_size, cfg.output_dim, true, false, allocator)
	enc.layers = make([]Qwen3_Layer(T), cfg.num_layers, allocator)

	// Allocate each layer
	for &layer in enc.layers {
		alloc_qwen3_layer(T, &layer, cfg, allocator)
	}

	// Find and load all safetensor shards
	// Typically: model-00001-of-00002.safetensors, model-00002-of-00002.safetensors
	shard_pattern := strings.concatenate({model_dir, "/model-"}, context.temp_allocator)

	// Try loading shards
	shard_idx := 1
	for {
		shard_path := fmt.tprintf("%s%05d-of-00002.safetensors", shard_pattern, shard_idx)
		if !os.exists(shard_path) {
			// Try single file fallback
			if shard_idx == 1 {
				single_path := strings.concatenate({model_dir, "/model.safetensors"}, context.temp_allocator)
				if os.exists(single_path) {
					load_qwen3_shard(enc, single_path, allocator)
					break
				}
			}
			break
		}

		load_qwen3_shard(enc, shard_path, allocator)
		shard_idx += 1
	}

	// Precompute RoPE frequencies
	enc.rope_freqs = compute_rope_freqs(T, enc.config.max_seq_len, enc.config.head_dim, enc.config.rope_theta, allocator)

	return enc, ""
}

// Allocate a single Qwen3 layer
@(private = "file")
alloc_qwen3_layer :: proc($T: typeid, layer: ^Qwen3_Layer(T), cfg: Qwen3_Config, allocator := context.allocator) {
	hidden := cfg.hidden_size
	kv_dim := cfg.num_kv_heads * cfg.head_dim
	q_dim := cfg.num_heads * cfg.head_dim

	// Attention
	layer.self_attn.q_proj = nn.new_linear(T, hidden, q_dim, true, false, allocator)
	layer.self_attn.k_proj = nn.new_linear(T, hidden, kv_dim, true, false, allocator)
	layer.self_attn.v_proj = nn.new_linear(T, hidden, kv_dim, true, false, allocator)
	layer.self_attn.o_proj = nn.new_linear(T, q_dim, hidden, false, false, allocator)
	layer.self_attn.q_norm = tensor.tensor_alloc(T, []uint{cfg.head_dim}, true, allocator)
	layer.self_attn.k_norm = tensor.tensor_alloc(T, []uint{cfg.head_dim}, true, allocator)

	// MLP
	layer.mlp.gate_proj = nn.new_linear(T, hidden, cfg.intermediate_size, false, false, allocator)
	layer.mlp.up_proj = nn.new_linear(T, hidden, cfg.intermediate_size, false, false, allocator)
	layer.mlp.down_proj = nn.new_linear(T, cfg.intermediate_size, hidden, false, false, allocator)

	// Norms
	layer.input_layernorm = tensor.tensor_alloc(T, []uint{hidden}, true, allocator)
	layer.post_attention_layernorm = tensor.tensor_alloc(T, []uint{hidden}, true, allocator)
}

// Load weights from a single shard
@(private = "file")
load_qwen3_shard :: proc(enc: ^Qwen3($T), path: string, allocator := context.allocator) {
	sf, sf_err := st.read_from_file(T, path, allocator)
	if sf_err != nil {
		fmt.panicf("Failed to read Qwen3 shard %s: %v", path, sf_err)
	}
	defer st.free_safe_tensors(sf, allocator)

	// Try to load each tensor if present in this shard
	st.tensor_assign_from_safe_tensors(enc.embed_tokens, "model.embed_tokens.weight", sf, false)
	st.tensor_assign_from_safe_tensors(enc.norm, "model.norm.weight", sf, false)

	// Output projection
	try_assign_linear(enc.output_proj, sf, "text_projection")

	// Load layers
	for i in 0 ..< int(enc.config.num_layers) {
		load_qwen3_layer_from_shard(&enc.layers[i], sf, i)
	}
}

// Helper to assign linear weights from safetensors (with transpose)
@(private = "file")
try_assign_linear :: proc(lin: ^nn.Linear($T), sf: ^st.Safe_Tensors(T), name: string) {
	if lin == nil do return
	st.tensor_assign_from_safe_tensors(lin.w, fmt.tprintf("%s.weight", name), sf, true)
	if b, ok := lin.b.?; ok {
		st.tensor_assign_from_safe_tensors(b, fmt.tprintf("%s.bias", name), sf, false)
	}
}

@(private = "file")
load_qwen3_layer_from_shard :: proc(
	layer: ^Qwen3_Layer($T),
	sf: ^st.Safe_Tensors(T),
	idx: int,
) {
	prefix := fmt.tprintf("model.layers.%d", idx)

	// Attention
	try_assign_linear(layer.self_attn.q_proj, sf, fmt.tprintf("%s.self_attn.q_proj", prefix))
	try_assign_linear(layer.self_attn.k_proj, sf, fmt.tprintf("%s.self_attn.k_proj", prefix))
	try_assign_linear(layer.self_attn.v_proj, sf, fmt.tprintf("%s.self_attn.v_proj", prefix))
	try_assign_linear(layer.self_attn.o_proj, sf, fmt.tprintf("%s.self_attn.o_proj", prefix))

	// QK-Norm
	if layer.self_attn.q_norm != nil {
		st.tensor_assign_from_safe_tensors(layer.self_attn.q_norm, fmt.tprintf("%s.self_attn.q_norm.weight", prefix), sf, false)
	}
	if layer.self_attn.k_norm != nil {
		st.tensor_assign_from_safe_tensors(layer.self_attn.k_norm, fmt.tprintf("%s.self_attn.k_norm.weight", prefix), sf, false)
	}

	// MLP
	try_assign_linear(layer.mlp.gate_proj, sf, fmt.tprintf("%s.mlp.gate_proj", prefix))
	try_assign_linear(layer.mlp.up_proj, sf, fmt.tprintf("%s.mlp.up_proj", prefix))
	try_assign_linear(layer.mlp.down_proj, sf, fmt.tprintf("%s.mlp.down_proj", prefix))

	// Norms
	if layer.input_layernorm != nil {
		st.tensor_assign_from_safe_tensors(layer.input_layernorm, fmt.tprintf("%s.input_layernorm.weight", prefix), sf, false)
	}
	if layer.post_attention_layernorm != nil {
		st.tensor_assign_from_safe_tensors(layer.post_attention_layernorm, fmt.tprintf("%s.post_attention_layernorm.weight", prefix), sf, false)
	}
}

// Free Qwen3 encoder
free_qwen3 :: proc(enc: ^Qwen3($T), allocator := context.allocator) {
	if enc == nil do return

	tensor.free_tensor(enc.embed_tokens, allocator)
	tensor.free_tensor(enc.norm, allocator)
	if enc.output_proj != nil {
		nn.free_linear(enc.output_proj, allocator)
	}
	tensor.free_tensor(enc.rope_freqs, allocator)

	for &layer in enc.layers {
		free_qwen3_layer(&layer, allocator)
	}
	delete(enc.layers, allocator)

	free(enc, allocator)
}

@(private = "file")
free_qwen3_layer :: proc(layer: ^Qwen3_Layer($T), allocator := context.allocator) {
	// Attention
	if layer.self_attn.q_proj != nil do nn.free_linear(layer.self_attn.q_proj, allocator)
	if layer.self_attn.k_proj != nil do nn.free_linear(layer.self_attn.k_proj, allocator)
	if layer.self_attn.v_proj != nil do nn.free_linear(layer.self_attn.v_proj, allocator)
	if layer.self_attn.o_proj != nil do nn.free_linear(layer.self_attn.o_proj, allocator)
	tensor.free_tensor(layer.self_attn.q_norm, layer.self_attn.k_norm, allocator = allocator)

	// MLP
	if layer.mlp.gate_proj != nil do nn.free_linear(layer.mlp.gate_proj, allocator)
	if layer.mlp.up_proj != nil do nn.free_linear(layer.mlp.up_proj, allocator)
	if layer.mlp.down_proj != nil do nn.free_linear(layer.mlp.down_proj, allocator)

	// Norms
	tensor.free_tensor(layer.input_layernorm, layer.post_attention_layernorm, allocator = allocator)
}

// Embedding lookup
@(private = "file")
embed_tokens :: proc(
	embed_table: ^tensor.Tensor($T),
	tokens: []i32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	seq_len := uint(len(tokens))
	hidden := embed_table.shape[1]

	result := tensor.tensor_alloc(T, []uint{1, seq_len, hidden}, true, allocator)

	for i, tok in tokens {
		src_offset := int(tok) * int(hidden)
		dst_offset := int(i) * int(hidden)
		for j in 0 ..< int(hidden) {
			result.data[dst_offset + j] = embed_table.data[src_offset + j]
		}
	}

	return result
}

// Qwen3 attention with grouped query attention and RoPE
@(private = "file")
qwen3_attention :: proc(
	attn: ^Qwen3_Attention($T),
	x: ^tensor.Tensor(T),
	freqs: ^tensor.Tensor(T),
	config: Qwen3_Config,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	batch := x.shape[0]
	seq_len := x.shape[1]
	eps := T(config.rms_norm_eps)

	// Project Q, K, V
	q := nn.forward_linear(attn.q_proj, x, allocator)
	defer tensor.free_tensor(q, allocator)
	k := nn.forward_linear(attn.k_proj, x, allocator)
	defer tensor.free_tensor(k, allocator)
	v := nn.forward_linear(attn.v_proj, x, allocator)
	defer tensor.free_tensor(v, allocator)

	// QK-Norm
	if attn.q_norm != nil {
		q_normed := rms_norm(q, attn.q_norm, eps, allocator)
		tensor.free_tensor(q, allocator)
		q = q_normed
	}
	if attn.k_norm != nil {
		k_normed := rms_norm(k, attn.k_norm, eps, allocator)
		tensor.free_tensor(k, allocator)
		k = k_normed
	}

	// Apply RoPE
	apply_rope_inplace(q, freqs, config.num_heads, config.head_dim)
	apply_rope_inplace(k, freqs, config.num_kv_heads, config.head_dim)

	// Reshape for multi-head attention
	// Q: [B, S, num_heads * head_dim] -> [B, num_heads, S, head_dim]
	// K, V: [B, S, num_kv_heads * head_dim] -> [B, num_kv_heads, S, head_dim]
	q_heads := tensor.reshape(q, []uint{batch, seq_len, config.num_heads, config.head_dim}, allocator)
	defer tensor.free_tensor(q_heads, allocator)
	k_heads := tensor.reshape(k, []uint{batch, seq_len, config.num_kv_heads, config.head_dim}, allocator)
	defer tensor.free_tensor(k_heads, allocator)
	v_heads := tensor.reshape(v, []uint{batch, seq_len, config.num_kv_heads, config.head_dim}, allocator)
	defer tensor.free_tensor(v_heads, allocator)

	q_t := tensor.transpose(q_heads, 1, 2, allocator) // [B, H, S, D]
	defer tensor.free_tensor(q_t, allocator)
	k_t := tensor.transpose(k_heads, 1, 2, allocator)
	defer tensor.free_tensor(k_t, allocator)
	v_t := tensor.transpose(v_heads, 1, 2, allocator)
	defer tensor.free_tensor(v_t, allocator)

	// Expand K, V for grouped query attention
	// num_heads / num_kv_heads copies
	kv_groups := config.num_heads / config.num_kv_heads
	k_expanded := repeat_kv(k_t, kv_groups, allocator)
	defer tensor.free_tensor(k_expanded, allocator)
	v_expanded := repeat_kv(v_t, kv_groups, allocator)
	defer tensor.free_tensor(v_expanded, allocator)

	// Scaled dot-product attention
	scale := T(1.0) / math.sqrt(T(config.head_dim))

	k_transposed := tensor.transpose(k_expanded, 2, 3, allocator)
	defer tensor.free_tensor(k_transposed, allocator)

	scores := tensor.matmul(q_t, k_transposed, allocator)
	defer tensor.free_tensor(scores, allocator)

	// Scale
	for i in 0 ..< len(scores.data) {
		scores.data[i] *= scale
	}

	// Causal mask
	apply_causal_mask(scores, seq_len)

	// Softmax
	tensor.softmax_last_dim_inplace(scores)

	// Attention @ V
	attn_out := tensor.matmul(scores, v_expanded, allocator)

	// Reshape back: [B, H, S, D] -> [B, S, H*D]
	attn_t := tensor.transpose(attn_out, 1, 2, allocator)
	tensor.free_tensor(attn_out, allocator)

	result_flat := tensor.reshape(attn_t, []uint{batch, seq_len, config.num_heads * config.head_dim}, allocator)
	tensor.free_tensor(attn_t, allocator)

	// Output projection
	result := nn.forward_linear(attn.o_proj, result_flat, allocator)
	tensor.free_tensor(result_flat, allocator)

	return result
}

// Repeat K/V for grouped query attention
@(private = "file")
repeat_kv :: proc(
	x: ^tensor.Tensor($T), // [B, kv_heads, S, D]
	n_rep: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	if n_rep == 1 {
		return tensor.clone(x, allocator)
	}

	batch := x.shape[0]
	kv_heads := x.shape[1]
	seq_len := x.shape[2]
	head_dim := x.shape[3]

	result := tensor.tensor_alloc(T, []uint{batch, kv_heads * n_rep, seq_len, head_dim}, true, allocator)

	for b in 0 ..< int(batch) {
		for h in 0 ..< int(kv_heads) {
			for rep in 0 ..< int(n_rep) {
				dst_h := h * int(n_rep) + rep
				for s in 0 ..< int(seq_len) {
					for d in 0 ..< int(head_dim) {
						src_idx := b * int(kv_heads * seq_len * head_dim) + h * int(seq_len * head_dim) + s * int(head_dim) + d
						dst_idx := b * int(kv_heads * n_rep * seq_len * head_dim) + dst_h * int(seq_len * head_dim) + s * int(head_dim) + d
						result.data[dst_idx] = x.data[src_idx]
					}
				}
			}
		}
	}

	return result
}

// Apply causal mask (set future positions to -inf)
@(private = "file")
apply_causal_mask :: proc(scores: ^tensor.Tensor($T), seq_len: uint) {
	// scores: [B, H, S, S]
	batch := scores.shape[0]
	heads := scores.shape[1]
	neg_inf := T(-1e9)

	for b in 0 ..< int(batch) {
		for h in 0 ..< int(heads) {
			for i in 0 ..< int(seq_len) {
				for j in i + 1 ..< int(seq_len) {
					idx := b * int(heads * seq_len * seq_len) + h * int(seq_len * seq_len) + i * int(seq_len) + j
					scores.data[idx] = neg_inf
				}
			}
		}
	}
}

// Qwen3 MLP (SwiGLU)
@(private = "file")
qwen3_mlp :: proc(
	mlp: ^Qwen3_MLP($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// gate = gate_proj(x)
	// up = up_proj(x)
	// out = down_proj(silu(gate) * up)
	gate := nn.forward_linear(mlp.gate_proj, x, allocator)
	defer tensor.free_tensor(gate, allocator)

	up := nn.forward_linear(mlp.up_proj, x, allocator)
	defer tensor.free_tensor(up, allocator)

	// SiLU(gate) * up
	for i in 0 ..< len(gate.data) {
		gate.data[i] = silu(gate.data[i]) * up.data[i]
	}

	result := nn.forward_linear(mlp.down_proj, gate, allocator)
	return result
}

// Qwen3 layer forward
@(private = "file")
qwen3_layer_forward :: proc(
	layer: ^Qwen3_Layer($T),
	x: ^tensor.Tensor(T),
	freqs: ^tensor.Tensor(T),
	config: Qwen3_Config,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	eps := T(config.rms_norm_eps)

	// Pre-attention norm
	normed := rms_norm(x, layer.input_layernorm, eps, allocator)
	defer tensor.free_tensor(normed, allocator)

	// Self-attention
	attn_out := qwen3_attention(&layer.self_attn, normed, freqs, config, allocator)
	defer tensor.free_tensor(attn_out, allocator)

	// Residual
	h := tensor.add(x, attn_out, allocator)

	// Post-attention norm
	normed2 := rms_norm(h, layer.post_attention_layernorm, eps, allocator)
	defer tensor.free_tensor(normed2, allocator)

	// MLP
	mlp_out := qwen3_mlp(&layer.mlp, normed2, allocator)
	defer tensor.free_tensor(mlp_out, allocator)

	// Residual
	result := tensor.add(h, mlp_out, allocator)
	tensor.free_tensor(h, allocator)

	return result
}

// Encode text tokens to embeddings
qwen3_encode :: proc(
	enc: ^Qwen3($T),
	tokens: []i32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Embedding lookup
	h := embed_tokens(enc.embed_tokens, tokens, allocator)

	// Run through all layers
	for &layer in enc.layers {
		h_new := qwen3_layer_forward(&layer, h, enc.rope_freqs, enc.config, allocator)
		tensor.free_tensor(h, allocator)
		h = h_new
	}

	// Final norm
	eps := T(enc.config.rms_norm_eps)
	normed := rms_norm(h, enc.norm, eps, allocator)
	tensor.free_tensor(h, allocator)

	// Project to FLUX text dimension if output projection exists
	if enc.output_proj != nil {
		result := nn.forward_linear(enc.output_proj, normed, allocator)
		tensor.free_tensor(normed, allocator)
		return result
	}

	return normed
}
