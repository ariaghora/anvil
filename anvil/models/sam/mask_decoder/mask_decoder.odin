package mask_decoder

import "../../../nn"
import st "../../../safetensors"
import "../../../tensor"
import vb "../var_builder"
import "core:fmt"
import "core:math"
import "core:terminal"

Attention_Mask_Decoder :: struct($T: typeid) {
	q_proj:    ^nn.Linear(T),
	k_proj:    ^nn.Linear(T),
	v_proj:    ^nn.Linear(T),
	out_proj:  ^nn.Linear(T),
	num_heads: uint,
}

new_attention :: proc(
	$T: typeid,
	vb_parent: ^vb.Var_Builder(T),
	embedding_dim: uint,
	num_heads: uint,
	downsample_rate: uint,
	allocator := context.allocator,
) -> ^Attention_Mask_Decoder(T) {
	internal_dim := embedding_dim / downsample_rate
	q_proj := nn.new_linear(T, embedding_dim, internal_dim, true, false, allocator)
	k_proj := nn.new_linear(T, embedding_dim, internal_dim, true, false, allocator)
	v_proj := nn.new_linear(T, embedding_dim, internal_dim, true, false, allocator)
	out_proj := nn.new_linear(T, internal_dim, embedding_dim, true, false, allocator)

	vb.assign(vb_parent, "q_proj.weight", q_proj.w, true)
	vb.assign(vb_parent, "q_proj.bias", q_proj.b.?)
	vb.assign(vb_parent, "k_proj.weight", k_proj.w, true)
	vb.assign(vb_parent, "k_proj.bias", k_proj.b.?)
	vb.assign(vb_parent, "v_proj.weight", v_proj.w, true)
	vb.assign(vb_parent, "v_proj.bias", v_proj.b.?)
	vb.assign(vb_parent, "out_proj.weight", out_proj.w, true)
	vb.assign(vb_parent, "out_proj.bias", out_proj.b.?)

	return new_clone(
		Attention_Mask_Decoder(T) {
			q_proj = q_proj,
			k_proj = k_proj,
			v_proj = v_proj,
			out_proj = out_proj,
			num_heads = num_heads,
		},
		allocator,
	)
}

@(private = "file")
separate_heads :: proc(
	x: ^tensor.Tensor($T),
	num_heads: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	out: ^tensor.Tensor(T),
) {
	b, n, c := x.shape[0], x.shape[1], x.shape[2]
	out = tensor.reshape(x, {b, n, num_heads, c / num_heads}, allocator, loc)
	out = tensor.transpose(out, 1, 2, allocator)
	return
}

@(private = "file")
recombine_heads :: proc(
	x: ^tensor.Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	b, n_heads, n_tokens, c_per_head := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	return tensor.reshape(
		tensor.transpose(x, 1, 2, context.temp_allocator, loc),
		{b, n_tokens, n_heads * c_per_head},
		allocator,
	)
}

forward_attention :: proc(
	attn: ^Attention_Mask_Decoder($T),
	q, k, v: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	talloc := context.temp_allocator
	q := nn.forward_linear(attn.q_proj, q, talloc)
	k := nn.forward_linear(attn.k_proj, k, talloc)
	v := nn.forward_linear(attn.v_proj, v, talloc)

	q = separate_heads(q, attn.num_heads, talloc)
	k = separate_heads(k, attn.num_heads, talloc)
	v = separate_heads(v, attn.num_heads, talloc)

	c_per_head := q.shape[3]
	kt := tensor.transpose(k, 2, 3, talloc)
	numerator := tensor.matmul(q, kt, talloc)
	for v, i in numerator.data do numerator.data[i] /= math.sqrt(T(c_per_head))
	attn_t := tensor.softmax_last_dim(numerator, talloc)
	out := tensor.matmul(attn_t, v, talloc)

	out = nn.forward_linear(attn.out_proj, recombine_heads(out, talloc), allocator)
	return out
}

free_attention :: proc(attn: ^Attention_Mask_Decoder($T)) {
	nn.free_linear(attn.q_proj)
	nn.free_linear(attn.k_proj)
	nn.free_linear(attn.v_proj)
	nn.free_linear(attn.out_proj)
	free(attn)
}

MLP_Mask_Decoder :: struct($T: typeid) {
	layers:         [dynamic]^nn.Linear(T),
	sigmoid_output: bool,
}


new_mlp_mask_decoder :: proc(
	$T: typeid,
	vb_parent: ^vb.Var_Builder(T),
	input_dim: uint,
	hidden_dim: uint,
	output_dim: uint,
	num_layers: uint,
	sigmoid_output: bool,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^MLP_Mask_Decoder(T) {
	layers := make([dynamic]^nn.Linear(T), allocator)
	for i in 0 ..< num_layers {
		_in_dim := i == 0 ? input_dim : hidden_dim
		_out_dim := i + 1 == num_layers ? output_dim : hidden_dim
		layer := nn.new_linear(T, _in_dim, _out_dim, true, false, allocator)
		vb.assign(vb_parent, fmt.tprintf("layers.%d.weight", i), layer.w, true, loc)
		vb.assign(vb_parent, fmt.tprintf("layers.%d.bias", i), layer.b.?, false, loc)
		append(&layers, layer)
	}
	return new_clone(
		MLP_Mask_Decoder(T){layers = layers, sigmoid_output = sigmoid_output},
		allocator,
	)
}

forward_mlp_mask_decoder :: proc(
	mlp: ^MLP_Mask_Decoder($T),
	xs: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	xs := xs
	for l, i in mlp.layers {
		xs = nn.forward_linear(l, xs, allocator)
		if i + 1 < len(mlp.layers) do xs = tensor.relu(xs, allocator)
	}
	if mlp.sigmoid_output {
		fmt.panicf("sigmoid is not implemented yet")
	}
	return xs
}

free_mlp_mask_decoder :: proc(mlp: ^MLP_Mask_Decoder($T), allocator := context.allocator) {
	for l in mlp.layers {
		nn.free_linear(l, allocator)
	}
	delete(mlp.layers)
}

MLP_Block :: struct($T: typeid) {
	lin1: ^nn.Linear(T),
	lin2: ^nn.Linear(T),
}

forward_mlp_block :: proc(
	mlp: ^MLP_Block($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> (
	out: ^tensor.Tensor(T),
) {
	out = nn.forward_linear(mlp.lin1, x, context.temp_allocator)
	out = tensor.relu(out, context.temp_allocator)
	out = nn.forward_linear(mlp.lin2, out, allocator)
	return
}

free_mlp_block :: proc(mlp: ^MLP_Block($T)) -> ^tensor.Tensor(T) {
	nn.free_linear(mlp.lin1)
	nn.free_linear(mlp.lin2)
}


Two_Way_Attention_Block :: struct($T: typeid) {
	self_attn:                 ^Attention_Mask_Decoder(T),
	norm1:                     ^nn.Layer_Norm(T),
	cross_attn_token_to_image: ^Attention_Mask_Decoder(T),
	norm2:                     ^nn.Layer_Norm(T),
	mlp:                       ^MLP_Block(T),
	norm3:                     ^nn.Layer_Norm(T),
	norm4:                     ^nn.Layer_Norm(T),
	cross_attn_image_to_token: ^Attention_Mask_Decoder(T),
	skip_first_layer_pe:       bool,
}

new_two_way_attention_block :: proc(
	$T: typeid,
	vb_parent: ^vb.Var_Builder(T),
	embedding_dim: uint,
	num_heads: uint,
	mlp_dim: uint,
	skip_first_layer_pe: bool,
	allocator := context.allocator,
) -> ^Two_Way_Attention_Block(T) {
	norm1 := nn.new_layer_norm_1d(T, embedding_dim, 1e-5, allocator)
	norm2 := nn.new_layer_norm_1d(T, embedding_dim, 1e-5, allocator)
	norm3 := nn.new_layer_norm_1d(T, embedding_dim, 1e-5, allocator)
	norm4 := nn.new_layer_norm_1d(T, embedding_dim, 1e-5, allocator)
	vb.assign(vb_parent, "norm1.weight", norm1.weight)
	vb.assign(vb_parent, "norm1.bias", norm1.bias)
	vb.assign(vb_parent, "norm2.weight", norm2.weight)
	vb.assign(vb_parent, "norm2.bias", norm2.bias)
	vb.assign(vb_parent, "norm3.weight", norm3.weight)
	vb.assign(vb_parent, "norm3.bias", norm3.bias)
	vb.assign(vb_parent, "norm4.weight", norm4.weight)
	vb.assign(vb_parent, "norm4.bias", norm4.bias)

	vb_self_attn := vb.vb_make(T, "self_attn", vb_parent)
	self_attn := new_attention(
		T,
		&vb_self_attn,
		embedding_dim,
		num_heads,
		1,
		allocator = allocator,
	)

	vb_cross_attn_token_to_image := vb.vb_make(T, "cross_attn_token_to_image", vb_parent)
	cross_attn_token_to_image := new_attention(
		T,
		&vb_cross_attn_token_to_image,
		embedding_dim,
		num_heads,
		2,
		allocator = allocator,
	)

	vb_cross_attn_image_to_token := vb.vb_make(T, "cross_attn_image_to_token", vb_parent)
	cross_attn_image_to_token := new_attention(
		T,
		&vb_cross_attn_image_to_token,
		embedding_dim,
		num_heads,
		2,
		allocator = allocator,
	)

	mlp := new_clone(
		MLP_Block(T) {
			lin1 = nn.new_linear(T, embedding_dim, mlp_dim, true, false, allocator),
			lin2 = nn.new_linear(T, mlp_dim, embedding_dim, true, false, allocator),
		},
		allocator,
	)
	vb.assign(vb_parent, "mlp.lin1.weight", mlp.lin1.w, true)
	vb.assign(vb_parent, "mlp.lin1.bias", mlp.lin1.b.?)
	vb.assign(vb_parent, "mlp.lin2.weight", mlp.lin2.w, true)
	vb.assign(vb_parent, "mlp.lin2.bias", mlp.lin2.b.?)

	return new_clone(
		Two_Way_Attention_Block(T) {
			self_attn = self_attn,
			norm1 = norm1,
			cross_attn_image_to_token = cross_attn_image_to_token,
			norm2 = norm2,
			mlp = mlp,
			norm3 = norm3,
			norm4 = norm4,
			cross_attn_token_to_image = cross_attn_token_to_image,
			skip_first_layer_pe = skip_first_layer_pe,
		},
		allocator,
	)
}

forward_two_way_attention_block :: proc(
	tt: ^Two_Way_Attention_Block($T),
	queries: ^tensor.Tensor(T),
	keys: ^tensor.Tensor(T),
	query_pe: ^tensor.Tensor(T),
	key_pe: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	talloc := context.temp_allocator
	queries := queries
	if tt.skip_first_layer_pe {
		queries = forward_attention(tt.self_attn, queries, queries, queries, talloc)
	} else {
		q := tensor.add(queries, query_pe, allocator)
		attn_out := forward_attention(tt.self_attn, q, q, queries, talloc)
		queries = tensor.add(queries, attn_out, talloc)
	}
	queries = nn.forward_layer_norm_1d(tt.norm1, queries, talloc)

	// Cross attention block, tokens attending to image embedding
	q := tensor.add(queries, query_pe, talloc)
	k := tensor.add(keys, key_pe, talloc)
	attn_out := forward_attention(tt.cross_attn_token_to_image, q, k, keys, talloc)
	queries = tensor.add(queries, attn_out, talloc)
	queries = nn.forward_layer_norm_1d(tt.norm2, queries, talloc)

	// MLP block
	mlp_out := forward_mlp_block(tt.mlp, queries, talloc)
	queries = tensor.add(queries, mlp_out, talloc)
	queries = nn.forward_layer_norm_1d(tt.norm3, queries, allocator)

	// Cross attention block, image embedding attending to tokens
	q = tensor.add(queries, query_pe, talloc)
	k = tensor.add(keys, key_pe, talloc)
	attn_out = forward_attention(tt.cross_attn_image_to_token, k, q, queries, talloc)
	keys := tensor.add(keys, attn_out, talloc)
	keys = nn.forward_layer_norm_1d(tt.norm4, keys, allocator)

	return queries, keys
}

free_two_way_attention_block :: proc(
	tt: ^Two_Way_Attention_Block($T),
	allocator := context.allocator,
) {
	nn.free_layer_norm(tt.norm1, allocator)
	nn.free_layer_norm(tt.norm2, allocator)
	nn.free_layer_norm(tt.norm3, allocator)
	nn.free_layer_norm(tt.norm4, allocator)
	nn.free_linear(tt.mlp.lin1)
	nn.free_linear(tt.mlp.lin2)
	free_attention(tt.self_attn, allocator)
	free_attention(tt.cross_attn_image_to_token, allocator)
	free_attention(tt.cross_attn_token_to_image, allocator)
}

Two_Way_Transformer :: struct($T: typeid) {
	layers:                    [dynamic]^Two_Way_Attention_Block(T),
	final_attn_token_to_image: ^Attention_Mask_Decoder(T),
	norm_final_attn:           ^nn.Layer_Norm(T),
}

new_two_way_transformer :: proc(
	$T: typeid,
	vb_parent: ^vb.Var_Builder(T),
	depth: uint,
	embedding_dim: uint,
	num_heads: uint,
	mlp_dim: uint,
	allocator := context.allocator,
) -> ^Two_Way_Transformer(T) {
	vb_layers := vb.vb_make(T, "layers", vb_parent)
	layers := make([dynamic]^Two_Way_Attention_Block(T), allocator)
	for i in 0 ..< depth {
		vb_layer_i := vb.vb_make(T, fmt.tprintf("%d", i), &vb_layers)
		l := new_two_way_attention_block(
			T,
			&vb_layer_i,
			embedding_dim,
			num_heads,
			mlp_dim,
			i == 0,
			allocator,
		)
		append(&layers, l)
	}

	vb_final_attn_token_to_image := vb.vb_make(T, "final_attn_token_to_image", vb_parent)
	final_attn_token_to_image := new_attention(
		T,
		&vb_final_attn_token_to_image,
		embedding_dim,
		num_heads,
		2,
		allocator,
	)

	norm_final_attn := nn.new_layer_norm_1d(T, embedding_dim, T(1e-5), allocator)
	vb.assign(vb_parent, "norm_final_attn.weight", norm_final_attn.weight)
	vb.assign(vb_parent, "norm_final_attn.bias", norm_final_attn.bias)

	return new_clone(
		Two_Way_Transformer(T) {
			layers = layers,
			final_attn_token_to_image = final_attn_token_to_image,
			norm_final_attn = norm_final_attn,
		},
		allocator,
	)
}

forward_two_way_transformer :: proc(
	tt: ^Two_Way_Transformer($T),
	image_embedding: ^tensor.Tensor(T),
	image_pe: ^tensor.Tensor(T),
	point_embedding: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	talloc := context.temp_allocator
	image_embedding := tensor.permute(
		tensor.flatten(image_embedding, 2, talloc),
		{0, 2, 1},
		talloc,
	)
	image_pe := tensor.permute(tensor.flatten(image_pe, 2, talloc), {0, 2, 1}, talloc)

	queries, keys := point_embedding, image_embedding
	for layer in tt.layers {
		queries, keys = forward_two_way_attention_block(
			layer,
			queries,
			keys,
			point_embedding,
			image_pe,
			talloc,
		)
	}

	q := tensor.add(queries, point_embedding, talloc)
	k := tensor.add(keys, image_pe, talloc)

	attn_out := forward_attention(tt.final_attn_token_to_image, q, k, keys, talloc)
	queries = nn.forward_layer_norm_1d(
		tt.norm_final_attn,
		tensor.add(queries, attn_out, talloc),
		talloc,
	)
	return queries, keys
}

free_two_way_transformer :: proc(tt: ^Two_Way_Transformer($T)) {
	for l in tt.layers do free_two_way_attention_block(l)
	delete(tt.layers)
	nn.free_layer_norm(tt.norm_final_attn)
	free_attention(tt.final_attn_token_to_image)
}

Mask_Decoder :: struct($T: typeid) {
	iou_token:                 ^nn.Embedding(T),
	mask_tokens:               ^nn.Embedding(T),
	iou_prediction_head:       ^MLP_Mask_Decoder(T),
	output_upscaling_conv1:    ^nn.Conv_Transpose_2d(T),
	output_upscaling_ln:       ^nn.Channel_Layer_Norm(T),
	output_upscaling_conv2:    ^nn.Conv_Transpose_2d(T),
	num_mask_tokens:           uint,
	output_hypernetworks_mlps: [dynamic]^MLP_Mask_Decoder(T),
	transformer:               ^Two_Way_Transformer(T),
}


new_mask_decoder :: proc(
	$T: typeid,
	safetensors: ^st.Safe_Tensors(T),
	transformer_dim: uint,
	num_multimask_outputs: uint,
	iou_head_depth: uint,
	iou_head_hidden_dim: uint,
	allocator := context.allocator,
) -> ^Mask_Decoder(T) {
	num_mask_tokens := num_multimask_outputs + 1

	vb_mask_decoder := vb.Var_Builder(T) {
		parent      = nil,
		name        = "mask_decoder",
		safetensors = safetensors,
	}

	vb_iou_prediction_head := vb.vb_make(T, "iou_prediction_head", &vb_mask_decoder)
	iou_prediction_head := new_mlp_mask_decoder(
		T,
		&vb_iou_prediction_head,
		transformer_dim,
		iou_head_hidden_dim,
		num_mask_tokens,
		iou_head_depth,
		false,
		allocator,
	)

	iou_token := nn.new_embedding(T, 1, transformer_dim, true, allocator)
	vb.assign(&vb_mask_decoder, "iou_token.weight", iou_token.weight)

	mask_tokens := nn.new_embedding(T, num_mask_tokens, transformer_dim, true, allocator)
	vb.assign(&vb_mask_decoder, "mask_tokens.weight", mask_tokens.weight)

	output_upscaling_conv1 := nn.new_conv_transpose_2d(
		T,
		transformer_dim,
		transformer_dim / 4,
		{2, 2},
		stride = 2,
		allocator = allocator,
	)
	vb.assign(&vb_mask_decoder, "output_upscaling.0.weight", output_upscaling_conv1.w)
	vb.assign(&vb_mask_decoder, "output_upscaling.0.bias", output_upscaling_conv1.b.?)

	output_upscaling_ln := nn.new_channel_layer_norm(T, transformer_dim / 4, 1e-6, allocator)
	vb.assign(&vb_mask_decoder, "output_upscaling.1.weight", output_upscaling_ln.weight)
	vb.assign(&vb_mask_decoder, "output_upscaling.1.bias", output_upscaling_ln.bias)

	output_upscaling_conv2 := nn.new_conv_transpose_2d(
		T,
		transformer_dim / 4,
		transformer_dim / 8,
		{2, 2},
		stride = 2,
		allocator = allocator,
	)
	vb.assign(&vb_mask_decoder, "output_upscaling.3.weight", output_upscaling_conv2.w)
	vb.assign(&vb_mask_decoder, "output_upscaling.3.bias", output_upscaling_conv2.b.?)

	vb_transformer := vb.vb_make(T, "transformer", &vb_mask_decoder)
	transformer := new_two_way_transformer(
		T,
		&vb_transformer,
		depth = 2,
		embedding_dim = transformer_dim,
		num_heads = 8,
		mlp_dim = 2048,
		allocator = allocator,
	)

	output_hypernetworks_mlps := make([dynamic]^MLP_Mask_Decoder(T), allocator)
	vb_hypernets := vb.vb_make(T, "output_hypernetworks_mlps", &vb_mask_decoder)
	for i in 0 ..< num_mask_tokens {
		vb_i := vb.vb_make(T, fmt.tprintf("%d", i), &vb_hypernets)
		mlp := new_mlp_mask_decoder(
			T,
			&vb_i,
			transformer_dim,
			transformer_dim,
			transformer_dim / 8,
			3,
			false,
			allocator,
		)
		append(&output_hypernetworks_mlps, mlp)
	}

	return new_clone(
		Mask_Decoder(T) {
			iou_token = iou_token,
			mask_tokens = mask_tokens,
			iou_prediction_head = iou_prediction_head,
			num_mask_tokens = num_mask_tokens,
			output_upscaling_conv1 = output_upscaling_conv1,
			output_upscaling_conv2 = output_upscaling_conv2,
			output_upscaling_ln = output_upscaling_ln,
			transformer = transformer,
			output_hypernetworks_mlps = output_hypernetworks_mlps,
		},
		allocator,
	)
}

import "../vit"

// @(private = "file")
predict_mask :: proc(
	md: ^Mask_Decoder($T),
	image_embeddings: ^tensor.Tensor(T),
	image_pe: ^tensor.Tensor(T),
	sparse_prompt_embeddings: ^tensor.Tensor(T),
	dense_prompt_embeddings: ^tensor.Tensor(T),
	allocator := context.temp_allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	talloc := context.temp_allocator

	// Concatenate output tokens...
	output_tokens := tensor.cat(
		[]^tensor.Tensor(T){md.iou_token.weight, md.mask_tokens.weight},
		0,
		talloc,
	)
	d1, d2 := output_tokens.shape[0], output_tokens.shape[1]
	output_tokens = tensor.unsqueeze(output_tokens, 0, talloc) // [1, d1, d2]
	batch_size := sparse_prompt_embeddings.shape[0]
	output_tokens = tensor.broadcast_as(output_tokens, []uint{batch_size, d1, d2}, talloc)
	tokens := tensor.cat([]^tensor.Tensor(T){output_tokens, sparse_prompt_embeddings}, 1, talloc)
	src := tensor.repeat_interleave(image_embeddings, tokens.shape[0], 0, talloc)
	src = tensor.add(src, dense_prompt_embeddings, talloc)
	pos_src := tensor.repeat_interleave(image_pe, tokens.shape[0], 0, talloc)

	// Run the transformer
	b, c, h, w := src.shape[0], src.shape[1], src.shape[2], src.shape[3]
	hs: ^tensor.Tensor(T)
	hs, src = forward_two_way_transformer(md.transformer, src, pos_src, tokens, talloc)
	iou_token_out := tensor.flatten(tensor.slice(hs, {{}, {0, 1, 1}, {}}, talloc), 1, talloc)
	mask_tokens_out := tensor.slice(hs, {{}, {1, 1 + int(md.num_mask_tokens), 1}, {}}, talloc)

	// Upscale mask embeddings
	src = tensor.reshape(tensor.transpose(src, 1, 2, talloc), {b, c, h, w}, talloc)
	out := nn.forward_conv_transpose_2d(md.output_upscaling_conv1, src, talloc)
	out = nn.forward_channel_layer_norm(md.output_upscaling_ln, out, talloc)
	out = vit.gelu_fast(out, talloc)
	out = nn.forward_conv_transpose_2d(md.output_upscaling_conv2, out, talloc)
	upscaled_embedding := vit.gelu_fast(out, talloc)

	h_list := make([dynamic]^tensor.Tensor(T), talloc)
	hyper_in_list := make([dynamic]^tensor.Tensor(T), talloc)
	for mlp, i in md.output_hypernetworks_mlps {
		h := tensor.slice(mask_tokens_out, {{}, {i, i + 1, 1}, {}}, talloc)
		h = tensor.flatten(h, 1, talloc)
		append(&h_list, h)
		h = forward_mlp_mask_decoder(mlp, h, talloc)
		append(&hyper_in_list, h)
	}
	hyper_in := tensor.stack(hyper_in_list[:], 1, talloc)
	b, c, h, w =
		upscaled_embedding.shape[0],
		upscaled_embedding.shape[1],
		upscaled_embedding.shape[2],
		upscaled_embedding.shape[3]
	masks := tensor.matmul(
		hyper_in,
		tensor.reshape(upscaled_embedding, {b, c, h * w}, talloc),
		talloc,
	)

	// Now output channel is something different, thus, gotta recalc according to
	// current length and divided by product of the original b, h, and w
	c_out := tensor.shape_to_size(masks.shape) / (b * h * w)

	// These last two should be promoted to caller's allocator!
	masks = tensor.reshape(masks, {b, c_out, h, w}, allocator)
	iou_pred := forward_mlp_mask_decoder(md.iou_prediction_head, iou_token_out, allocator)

	return masks, iou_pred
}

forward_mask_decoder :: proc(
	md: ^Mask_Decoder($T),
	image_embeddings: ^tensor.Tensor(T),
	image_pe: ^tensor.Tensor(T),
	sparse_prompt_embeddings: ^tensor.Tensor(T),
	dense_prompt_embeddings: ^tensor.Tensor(T),
	// multimask_output: bool, // TODO(Aria): allow this
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	masks, iou_pred := predict_mask(
		md,
		image_embeddings,
		image_pe,
		sparse_prompt_embeddings,
		dense_prompt_embeddings,
		context.temp_allocator,
	)

	// When multimask is false, just take the first channel from masks and iou_pred
	// TODO(Aria): implement multimask output
	masks = tensor.slice(masks, {{}, {0, 1, 1}, {}, {}}, allocator)
	iou_pred = tensor.slice(iou_pred, {{}, {0, 1, 1}}, allocator)
	return masks, iou_pred
}

free_mask_decoder :: proc(md: ^Mask_Decoder($T), allocator := context.allocator) {
	nn.free_embedding(md.iou_token, allocator)
	nn.free_embedding(md.mask_tokens, allocator)
	nn.free_channel_layer_norm(md.output_upscaling_ln, allocator)
	nn.free_conv_transpose_2d(md.output_upscaling_conv1, allocator)
	nn.free_conv_transpose_2d(md.output_upscaling_conv2, allocator)
	free_mlp_mask_decoder(md.iou_prediction_head, allocator)

	for l in md.output_hypernetworks_mlps do free_mlp_mask_decoder(l, allocator)
	delete(md.output_hypernetworks_mlps)

	free(md)
}
