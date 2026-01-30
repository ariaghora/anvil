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
import "../../trace"
import st "../../safetensors"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strings"

// Qwen3 configuration
QWEN3_VOCAB_SIZE :: 151936
QWEN3_HIDDEN :: 2560
QWEN3_INTERMEDIATE :: 9728 // MLP hidden
QWEN3_NUM_HEADS :: 32
QWEN3_NUM_KV_HEADS :: 8 // Grouped query attention
QWEN3_HEAD_DIM :: 128
QWEN3_LAYERS :: 36
QWEN3_MAX_SEQ :: 512
QWEN3_OUTPUT_DIM :: 7680
QWEN3_ROPE_THETA :: 1000000.0
// Layers to extract for text embeddings (0-indexed: layers 9, 18, 27 in 1-indexed)
QWEN3_OUTPUT_LAYERS :: [3]int{8, 17, 26}

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

// Single Qwen3 layer weights - for lazy loading
Qwen3_Layer_Weights :: struct($T: typeid) {
	self_attn:                 Qwen3_Attention(T),
	mlp:                       Qwen3_MLP(T),
	input_layernorm:           ^tensor.Tensor(T), // RMSNorm weight
	post_attention_layernorm:  ^tensor.Tensor(T),
}

// Qwen3 encoder context - lazy loading version
Qwen3 :: struct($T: typeid) {
	embed_tokens:   ^tensor.Tensor(T), // [vocab_size, hidden]
	norm:           ^tensor.Tensor(T), // Final RMSNorm
	rope_freqs:     ^tensor.Tensor(T), // Precomputed RoPE
	config:         Qwen3_Config,
	// Lazy loading - keep safetensor files open
	_sf_shards:     [dynamic]^st.Safe_Tensors(T),
	_model_dir:     string,
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

// Load Qwen3 encoder - lazy loading version
// Only loads embeddings, final norm, output projection, and rope freqs
// Layers are loaded on-demand during forward pass
load_qwen3 :: proc(
	$T: typeid,
	model_dir: string,
	allocator := context.allocator,
) -> (enc: ^Qwen3(T), err: string) {
	enc = new(Qwen3(T), allocator)
	enc.config = qwen3_config_default()
	enc._model_dir = strings.clone(model_dir, allocator)
	enc._sf_shards = make([dynamic]^st.Safe_Tensors(T), allocator)

	// Open all safetensor shards lazily
	shard_pattern := strings.concatenate({model_dir, "/model-"}, context.temp_allocator)
	shard_idx := 1
	for {
		shard_path := fmt.tprintf("%s%05d-of-00002.safetensors", shard_pattern, shard_idx)
		if !os.exists(shard_path) {
			if shard_idx == 1 {
				single_path := strings.concatenate({model_dir, "/model.safetensors"}, context.temp_allocator)
				if os.exists(single_path) {
					sf, sf_err := st.read_from_file_lazy(T, single_path, allocator)
					if sf_err != nil {
						return nil, fmt.tprintf("Failed to open Qwen3 model: %v", sf_err)
					}
					append(&enc._sf_shards, sf)
				}
			}
			break
		}

		sf, sf_err := st.read_from_file_lazy(T, shard_path, allocator)
		if sf_err != nil {
			return nil, fmt.tprintf("Failed to open Qwen3 shard %s: %v", shard_path, sf_err)
		}
		append(&enc._sf_shards, sf)
		shard_idx += 1
	}

	if len(enc._sf_shards) == 0 {
		return nil, "No Qwen3 safetensor files found"
	}

	// Load only the shared weights (embeddings, final norm, output projection)
	load_qwen3_shared_weights(enc, allocator)

	// Precompute RoPE frequencies
	enc.rope_freqs = compute_rope_freqs(T, enc.config.max_seq_len, enc.config.head_dim, enc.config.rope_theta, allocator)

	return enc, ""
}

// Load shared weights that are always needed
@(private = "file")
load_qwen3_shared_weights :: proc(enc: ^Qwen3($T), allocator := context.allocator) {
	cfg := enc.config

	// Load embeddings
	enc.embed_tokens = get_tensor_from_shards(enc._sf_shards[:], "model.embed_tokens.weight", []uint{cfg.vocab_size, cfg.hidden_size}, allocator)

	// Load final norm
	enc.norm = get_tensor_from_shards(enc._sf_shards[:], "model.norm.weight", []uint{cfg.hidden_size}, allocator)
}

// Get tensor from shards - searches all shards for the tensor
@(private = "file")
get_tensor_from_shards :: proc(
	shards: []^st.Safe_Tensors($T),
	name: string,
	expected_shape: []uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	for sf in shards {
		t, ok := st.get_tensor_lazy(sf, name, allocator)
		if !ok do continue

		// Validate shape
		if len(t.shape) != len(expected_shape) {
			continue
		}
		shape_ok := true
		for i in 0 ..< len(expected_shape) {
			if t.shape[i] != expected_shape[i] {
				shape_ok = false
				break
			}
		}
		if !shape_ok {
			continue
		}

		return tensor.clone(t, allocator)
	}
	return nil
}

// Load linear layer from shards
@(private = "file")
load_linear_from_shards :: proc(
	shards: []^st.Safe_Tensors($T),
	name: string,
	in_features, out_features: uint,
	has_bias: bool,
	allocator := context.allocator,
) -> ^nn.Linear(T) {
	weight_name := fmt.tprintf("%s.weight", name)
	// PyTorch convention: weight is [out_features, in_features], no transpose needed
	// get_tensor_from_shards already clones, so Linear owns the tensor
	w := get_tensor_from_shards(shards, weight_name, []uint{out_features, in_features}, allocator)
	if w == nil do return nil

	lin := new(nn.Linear(T), allocator)
	lin.w = w

	if has_bias {
		bias_name := fmt.tprintf("%s.bias", name)
		lin.b = get_tensor_from_shards(shards, bias_name, []uint{out_features}, allocator)
	}

	return lin
}

// Load a single Qwen3 layer on-demand
@(private = "file")
load_qwen3_layer :: proc(
	shards: []^st.Safe_Tensors($T),
	idx: int,
	cfg: Qwen3_Config,
	allocator := context.allocator,
) -> Qwen3_Layer_Weights(T) {
	layer: Qwen3_Layer_Weights(T)
	prefix := fmt.tprintf("model.layers.%d", idx)

	hidden := cfg.hidden_size
	kv_dim := cfg.num_kv_heads * cfg.head_dim
	q_dim := cfg.num_heads * cfg.head_dim

	// Attention projections
	layer.self_attn.q_proj = load_linear_from_shards(shards, fmt.tprintf("%s.self_attn.q_proj", prefix), hidden, q_dim, true, allocator)
	layer.self_attn.k_proj = load_linear_from_shards(shards, fmt.tprintf("%s.self_attn.k_proj", prefix), hidden, kv_dim, true, allocator)
	layer.self_attn.v_proj = load_linear_from_shards(shards, fmt.tprintf("%s.self_attn.v_proj", prefix), hidden, kv_dim, true, allocator)
	layer.self_attn.o_proj = load_linear_from_shards(shards, fmt.tprintf("%s.self_attn.o_proj", prefix), q_dim, hidden, false, allocator)

	// QK-Norm
	layer.self_attn.q_norm = get_tensor_from_shards(shards, fmt.tprintf("%s.self_attn.q_norm.weight", prefix), []uint{cfg.head_dim}, allocator)
	layer.self_attn.k_norm = get_tensor_from_shards(shards, fmt.tprintf("%s.self_attn.k_norm.weight", prefix), []uint{cfg.head_dim}, allocator)

	// MLP
	layer.mlp.gate_proj = load_linear_from_shards(shards, fmt.tprintf("%s.mlp.gate_proj", prefix), hidden, cfg.intermediate_size, false, allocator)
	layer.mlp.up_proj = load_linear_from_shards(shards, fmt.tprintf("%s.mlp.up_proj", prefix), hidden, cfg.intermediate_size, false, allocator)
	layer.mlp.down_proj = load_linear_from_shards(shards, fmt.tprintf("%s.mlp.down_proj", prefix), cfg.intermediate_size, hidden, false, allocator)

	// Norms
	layer.input_layernorm = get_tensor_from_shards(shards, fmt.tprintf("%s.input_layernorm.weight", prefix), []uint{hidden}, allocator)
	layer.post_attention_layernorm = get_tensor_from_shards(shards, fmt.tprintf("%s.post_attention_layernorm.weight", prefix), []uint{hidden}, allocator)

	return layer
}

// Free a single layer's weights
@(private = "file")
free_qwen3_layer_weights :: proc(layer: ^Qwen3_Layer_Weights($T), allocator := context.allocator) {
	// Free Linear layers (they own cloned weights)
	if layer.self_attn.q_proj != nil do nn.free_linear(layer.self_attn.q_proj, allocator)
	if layer.self_attn.k_proj != nil do nn.free_linear(layer.self_attn.k_proj, allocator)
	if layer.self_attn.v_proj != nil do nn.free_linear(layer.self_attn.v_proj, allocator)
	if layer.self_attn.o_proj != nil do nn.free_linear(layer.self_attn.o_proj, allocator)
	// q_norm, k_norm point to cache - don't free

	if layer.mlp.gate_proj != nil do nn.free_linear(layer.mlp.gate_proj, allocator)
	if layer.mlp.up_proj != nil do nn.free_linear(layer.mlp.up_proj, allocator)
	if layer.mlp.down_proj != nil do nn.free_linear(layer.mlp.down_proj, allocator)

	// input_layernorm, post_attention_layernorm point to cache - don't free
}

// Free Qwen3 encoder
free_qwen3 :: proc(enc: ^Qwen3($T), allocator := context.allocator) {
	if enc == nil do return

	tensor.free_tensor(enc.embed_tokens, allocator)
	tensor.free_tensor(enc.norm, allocator)
	tensor.free_tensor(enc.rope_freqs, allocator)

	// Close all safetensor shards
	for sf in enc._sf_shards {
		st.free_safe_tensors(sf, allocator)
	}
	delete(enc._sf_shards)

	delete(enc._model_dir, allocator)
	free(enc, allocator)
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

	for idx in 0 ..< len(tokens) {
		tok := tokens[idx]
		src_offset := int(tok) * int(hidden)
		dst_offset := idx * int(hidden)
		for j in 0 ..< int(hidden) {
			result.data[dst_offset + j] = embed_table.data[src_offset + j]
		}
	}

	return result
}

// QK-Norm: RMSNorm applied per-head
// x: [batch, seq, num_heads * head_dim], weight: [head_dim]
@(private = "file")
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

	// QK-Norm (per-head)
	if attn.q_norm != nil {
		q_normed := qk_norm(q, attn.q_norm, config.num_heads, config.head_dim, eps, allocator)
		tensor.free_tensor(q, allocator)
		q = q_normed
	}
	if attn.k_norm != nil {
		k_normed := qk_norm(k, attn.k_norm, config.num_kv_heads, config.head_dim, eps, allocator)
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

// Qwen3 layer forward - takes layer weights directly
@(private = "file")
qwen3_layer_forward :: proc(
	layer: ^Qwen3_Layer_Weights($T),
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

// Encode text tokens to embeddings - lazy loading version
// Extracts hidden states from layers 9, 18, 27 (1-indexed) and concatenates them
qwen3_encode :: proc(
	enc: ^Qwen3($T),
	tokens: []i32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	_t := trace.global_scoped("qwen3_encode", "text_encoder")
	defer trace.global_end_scoped(_t)

	// Embedding lookup
	_t_emb := trace.global_scoped("embed_tokens", "text_encoder")
	h := embed_tokens(enc.embed_tokens, tokens, allocator)
	trace.global_end_scoped(_t_emb)

	// Storage for layer outputs to concatenate (layers 9, 18, 27 = indices 8, 17, 26)
	layer_outputs: [3]^tensor.Tensor(T)
	output_layers := QWEN3_OUTPUT_LAYERS

	// Run through all layers - load, forward, free each layer
	_t_layers := trace.global_scoped("qwen3_layers", "text_encoder")
	for i in 0 ..< int(enc.config.num_layers) {
		_t_layer := trace.global_scoped("qwen3_layer", "text_encoder")
		layer := load_qwen3_layer(enc._sf_shards[:], i, enc.config, allocator)
		h_new := qwen3_layer_forward(&layer, h, enc.rope_freqs, enc.config, allocator)
		free_qwen3_layer_weights(&layer, allocator)
		tensor.free_tensor(h, allocator)
		h = h_new
		trace.global_end_scoped(_t_layer)

		// Save outputs at designated layers
		if i == output_layers[0] {
			layer_outputs[0] = tensor.clone(h, allocator)
		} else if i == output_layers[1] {
			layer_outputs[1] = tensor.clone(h, allocator)
		} else if i == output_layers[2] {
			layer_outputs[2] = tensor.clone(h, allocator)
		}
	}
	trace.global_end_scoped(_t_layers)

	// Free final hidden state (not needed)
	tensor.free_tensor(h, allocator)

	// Concatenate layer outputs along hidden dimension: [1, seq, hidden] x 3 -> [1, seq, 3*hidden]
	result := tensor.cat(layer_outputs[:], 2, allocator)

	// Free layer outputs
	tensor.free_tensor(layer_outputs[0], layer_outputs[1], layer_outputs[2], allocator = allocator)

	return result
}
