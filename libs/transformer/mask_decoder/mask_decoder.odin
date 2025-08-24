package mask_decoder

import "../../nn"
import st "../../safetensors"
import "../../tensor"
import vb "../var_builder"
import "core:fmt"
import "core:math"
import "core:terminal"

MLP_Mask_Decoder :: struct($T: typeid) {
	layers:         [dynamic]^nn.Linear(T),
	sigmoid_output: bool,
}

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
	attn_t := tensor.softmax(numerator, talloc)
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


new_mlp_mask_decoder :: proc(
	$T: typeid,
	vb_parent: ^vb.Var_Builder(T),
	input_dim: uint,
	hidden_dim: uint,
	output_dim: uint,
	num_layers: uint,
	sigmoid_output: bool,
) -> MLP_Mask_Decoder(T) {
	return {sigmoid_output = sigmoid_output}
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

forward_two_way_transformer :: proc(tt: ^Two_Way_Transformer($T)) {
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
	iou_prediction_head:       MLP_Mask_Decoder(T),
	output_upscaling_conv1:    ^nn.Conv_Transpose_2d(T),
	output_upscaling_ln:       ^nn.Channel_Layer_Norm(T),
	output_upscaling_conv2:    ^nn.Conv_Transpose_2d(T),
	num_mask_tokens:           uint,
	output_hypernetworks_mlps: [dynamic]MLP_Mask_Decoder(T),
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

	iou_prediction_head := new_mlp_mask_decoder(
		T,
		&vb_mask_decoder,
		transformer_dim,
		iou_head_hidden_dim,
		num_mask_tokens,
		iou_head_depth,
		false,
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
		},
		allocator,
	)
}

forward_mask_decoder :: proc(md: ^Mask_Decoder($T)) -> ^tensor.Tensor(T) {
	return nil
}

free_mask_decoder :: proc(md: ^Mask_Decoder($T), allocator := context.allocator) {
	nn.free_embedding(md.iou_token, allocator)
	nn.free_embedding(md.mask_tokens, allocator)
	nn.free_channel_layer_norm(md.output_upscaling_ln, allocator)
	nn.free_conv_transpose_2d(md.output_upscaling_conv1, allocator)
	nn.free_conv_transpose_2d(md.output_upscaling_conv2, allocator)
	free(md)
}
