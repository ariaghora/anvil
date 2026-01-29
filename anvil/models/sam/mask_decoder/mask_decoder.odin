package mask_decoder

import "../../../nn"
import st "../../../safetensors"
import "../../../tensor"
import vb "../var_builder"
import "core:fmt"
import "core:math"
import "core:terminal"

R :: tensor.R

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
	reshaped := tensor.reshape(x, {b, n, num_heads, c / num_heads}, allocator, loc)
	defer tensor.free_tensor(reshaped, allocator = allocator)
	out = tensor.transpose(reshaped, 1, 2, allocator)
	return
}

@(private = "file")
recombine_heads :: proc(
	x: ^tensor.Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	b, n_heads, n_tokens, c_per_head := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	transposed := tensor.transpose(x, 1, 2, allocator, loc)
	defer tensor.free_tensor(transposed, allocator = allocator)
	reshaped := tensor.reshape(transposed, {b, n_tokens, n_heads * c_per_head}, allocator)
	defer tensor.free_tensor(reshaped, allocator = allocator)
	return tensor.clone(reshaped, allocator)
}

forward_attention :: proc(
	attn: ^Attention_Mask_Decoder($T),
	q, k, v: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	q_proj := nn.forward_linear(attn.q_proj, q, allocator)
	defer tensor.free_tensor(q_proj, allocator = allocator)
	k_proj := nn.forward_linear(attn.k_proj, k, allocator)
	defer tensor.free_tensor(k_proj, allocator = allocator)
	v_proj := nn.forward_linear(attn.v_proj, v, allocator)
	defer tensor.free_tensor(v_proj, allocator = allocator)

	q_heads := separate_heads(q_proj, attn.num_heads, allocator)
	defer tensor.free_tensor(q_heads, allocator = allocator)
	k_heads := separate_heads(k_proj, attn.num_heads, allocator)
	defer tensor.free_tensor(k_heads, allocator = allocator)
	v_heads := separate_heads(v_proj, attn.num_heads, allocator)
	defer tensor.free_tensor(v_heads, allocator = allocator)

	c_per_head := q_heads.shape[3]
	kt := tensor.transpose(k_heads, 2, 3, allocator)
	defer tensor.free_tensor(kt, allocator = allocator)
	numerator := tensor.matmul(q_heads, kt, allocator)
	defer tensor.free_tensor(numerator, allocator = allocator)
	for val, i in numerator.data do numerator.data[i] /= math.sqrt(T(c_per_head))
	attn_t := tensor.softmax_last_dim(numerator, allocator)
	defer tensor.free_tensor(attn_t, allocator = allocator)
	attn_out := tensor.matmul(attn_t, v_heads, allocator)
	defer tensor.free_tensor(attn_out, allocator = allocator)

	recombined := recombine_heads(attn_out, allocator)
	defer tensor.free_tensor(recombined, allocator = allocator)
	out := nn.forward_linear(attn.out_proj, recombined, allocator)
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
	current := xs
	prev: ^tensor.Tensor(T) = nil
	for l, i in mlp.layers {
		new_xs := nn.forward_linear(l, current, allocator)
		if prev != nil {
			tensor.free_tensor(prev, allocator = allocator)
		}
		if i + 1 < len(mlp.layers) {
			relu_out := tensor.relu(new_xs, allocator)
			tensor.free_tensor(new_xs, allocator = allocator)
			prev = relu_out
			current = relu_out
		} else {
			prev = new_xs
			current = new_xs
		}
	}
	if mlp.sigmoid_output {
		fmt.panicf("sigmoid is not implemented yet")
	}
	return current
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
	lin1_out := nn.forward_linear(mlp.lin1, x, allocator)
	defer tensor.free_tensor(lin1_out, allocator = allocator)
	relu_out := tensor.relu(lin1_out, allocator)
	defer tensor.free_tensor(relu_out, allocator = allocator)
	out = nn.forward_linear(mlp.lin2, relu_out, allocator)
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
	queries_in: ^tensor.Tensor(T),
	keys_in: ^tensor.Tensor(T),
	query_pe: ^tensor.Tensor(T),
	key_pe: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	queries: ^tensor.Tensor(T)
	if tt.skip_first_layer_pe {
		queries = forward_attention(tt.self_attn, queries_in, queries_in, queries_in, allocator)
	} else {
		q := tensor.add(queries_in, query_pe, allocator)
		defer tensor.free_tensor(q, allocator = allocator)
		attn_out := forward_attention(tt.self_attn, q, q, queries_in, allocator)
		defer tensor.free_tensor(attn_out, allocator = allocator)
		queries = tensor.add(queries_in, attn_out, allocator)
	}
	queries_norm1 := nn.forward_layer_norm_1d(tt.norm1, queries, allocator)
	tensor.free_tensor(queries, allocator = allocator)

	// Cross attention block, tokens attending to image embedding
	q1 := tensor.add(queries_norm1, query_pe, allocator)
	defer tensor.free_tensor(q1, allocator = allocator)
	k1 := tensor.add(keys_in, key_pe, allocator)
	defer tensor.free_tensor(k1, allocator = allocator)
	attn_out1 := forward_attention(tt.cross_attn_token_to_image, q1, k1, keys_in, allocator)
	defer tensor.free_tensor(attn_out1, allocator = allocator)
	queries_add1 := tensor.add(queries_norm1, attn_out1, allocator)
	tensor.free_tensor(queries_norm1, allocator = allocator)
	queries_norm2 := nn.forward_layer_norm_1d(tt.norm2, queries_add1, allocator)
	tensor.free_tensor(queries_add1, allocator = allocator)

	// MLP block
	mlp_out := forward_mlp_block(tt.mlp, queries_norm2, allocator)
	defer tensor.free_tensor(mlp_out, allocator = allocator)
	queries_add2 := tensor.add(queries_norm2, mlp_out, allocator)
	tensor.free_tensor(queries_norm2, allocator = allocator)
	queries_out := nn.forward_layer_norm_1d(tt.norm3, queries_add2, allocator)
	tensor.free_tensor(queries_add2, allocator = allocator)

	// Cross attention block, image embedding attending to tokens
	q2 := tensor.add(queries_out, query_pe, allocator)
	defer tensor.free_tensor(q2, allocator = allocator)
	k2 := tensor.add(keys_in, key_pe, allocator)
	defer tensor.free_tensor(k2, allocator = allocator)
	attn_out2 := forward_attention(tt.cross_attn_image_to_token, k2, q2, queries_out, allocator)
	defer tensor.free_tensor(attn_out2, allocator = allocator)
	keys_add := tensor.add(keys_in, attn_out2, allocator)
	defer tensor.free_tensor(keys_add, allocator = allocator)
	keys_out := nn.forward_layer_norm_1d(tt.norm4, keys_add, allocator)

	return queries_out, keys_out
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
	img_flat := tensor.flatten(image_embedding, 2, allocator)
	defer tensor.free_tensor(img_flat, allocator = allocator)
	img_emb := tensor.permute(img_flat, {0, 2, 1}, allocator)

	pe_flat := tensor.flatten(image_pe, 2, allocator)
	defer tensor.free_tensor(pe_flat, allocator = allocator)
	img_pe := tensor.permute(pe_flat, {0, 2, 1}, allocator)

	queries, keys := point_embedding, img_emb
	prev_queries: ^tensor.Tensor(T) = nil
	prev_keys: ^tensor.Tensor(T) = nil
	for layer in tt.layers {
		new_queries, new_keys := forward_two_way_attention_block(
			layer,
			queries,
			keys,
			point_embedding,
			img_pe,
			allocator,
		)
		if prev_queries != nil {
			tensor.free_tensor(prev_queries, allocator = allocator)
		}
		if prev_keys != nil {
			tensor.free_tensor(prev_keys, allocator = allocator)
		}
		prev_queries = new_queries
		prev_keys = new_keys
		queries = new_queries
		keys = new_keys
	}

	q := tensor.add(queries, point_embedding, allocator)
	defer tensor.free_tensor(q, allocator = allocator)
	k := tensor.add(keys, img_pe, allocator)
	defer tensor.free_tensor(k, allocator = allocator)

	attn_out := forward_attention(tt.final_attn_token_to_image, q, k, keys, allocator)
	defer tensor.free_tensor(attn_out, allocator = allocator)
	queries_add := tensor.add(queries, attn_out, allocator)
	if prev_queries != nil {
		tensor.free_tensor(prev_queries, allocator = allocator)
	}
	defer tensor.free_tensor(queries_add, allocator = allocator)
	queries_out := nn.forward_layer_norm_1d(tt.norm_final_attn, queries_add, allocator)

	tensor.free_tensor(img_pe, allocator = allocator)
	tensor.free_tensor(img_emb, allocator = allocator)

	return queries_out, keys
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

predict_mask :: proc(
	md: ^Mask_Decoder($T),
	image_embeddings: ^tensor.Tensor(T),
	image_pe: ^tensor.Tensor(T),
	sparse_prompt_embeddings: ^tensor.Tensor(T),
	dense_prompt_embeddings: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	// Concatenate output tokens...
	output_tokens_cat := tensor.cat(
		[]^tensor.Tensor(T){md.iou_token.weight, md.mask_tokens.weight},
		0,
		allocator,
	)
	defer tensor.free_tensor(output_tokens_cat, allocator = allocator)
	d1, d2 := output_tokens_cat.shape[0], output_tokens_cat.shape[1]
	output_tokens_unsq := tensor.unsqueeze(output_tokens_cat, 0, allocator) // [1, d1, d2]
	defer tensor.free_tensor(output_tokens_unsq, allocator = allocator)
	batch_size := sparse_prompt_embeddings.shape[0]
	output_tokens := tensor.broadcast_as(output_tokens_unsq, []uint{batch_size, d1, d2}, allocator)
	defer tensor.free_tensor(output_tokens, allocator = allocator)
	tokens := tensor.cat([]^tensor.Tensor(T){output_tokens, sparse_prompt_embeddings}, 1, allocator)
	defer tensor.free_tensor(tokens, allocator = allocator)
	src_repeat := tensor.repeat_interleave(image_embeddings, tokens.shape[0], 0, allocator)
	defer tensor.free_tensor(src_repeat, allocator = allocator)
	src := tensor.add(src_repeat, dense_prompt_embeddings, allocator)
	defer tensor.free_tensor(src, allocator = allocator)
	pos_src := tensor.repeat_interleave(image_pe, tokens.shape[0], 0, allocator)
	defer tensor.free_tensor(pos_src, allocator = allocator)

	// Run the transformer
	b, c, h, w := src.shape[0], src.shape[1], src.shape[2], src.shape[3]
	hs, src_out := forward_two_way_transformer(md.transformer, src, pos_src, tokens, allocator)
	defer tensor.free_tensor(hs, allocator = allocator)
	defer tensor.free_tensor(src_out, allocator = allocator)

	hs_slice := tensor.slice(hs, {{}, R(1)}, allocator = allocator)
	defer tensor.free_tensor(hs_slice, allocator = allocator)
	iou_token_out := tensor.flatten(hs_slice, 1, allocator)
	defer tensor.free_tensor(iou_token_out, allocator = allocator)
	mask_tokens_out := tensor.slice(hs, {{}, R(1, 1 + int(md.num_mask_tokens))}, allocator = allocator)
	defer tensor.free_tensor(mask_tokens_out, allocator = allocator)

	// Upscale mask embeddings
	src_transposed := tensor.transpose(src_out, 1, 2, allocator)
	defer tensor.free_tensor(src_transposed, allocator = allocator)
	src_reshaped := tensor.reshape(src_transposed, {b, c, h, w}, allocator)
	defer tensor.free_tensor(src_reshaped, allocator = allocator)
	conv1_out := nn.forward_conv_transpose_2d(md.output_upscaling_conv1, src_reshaped, allocator)
	defer tensor.free_tensor(conv1_out, allocator = allocator)
	ln_out := nn.forward_channel_layer_norm(md.output_upscaling_ln, conv1_out, allocator)
	defer tensor.free_tensor(ln_out, allocator = allocator)
	gelu1_out := vit.gelu_fast(ln_out, allocator)
	defer tensor.free_tensor(gelu1_out, allocator = allocator)
	conv2_out := nn.forward_conv_transpose_2d(md.output_upscaling_conv2, gelu1_out, allocator)
	defer tensor.free_tensor(conv2_out, allocator = allocator)
	upscaled_embedding := vit.gelu_fast(conv2_out, allocator)
	defer tensor.free_tensor(upscaled_embedding, allocator = allocator)

	hyper_in_list := make([dynamic]^tensor.Tensor(T), allocator)
	defer {
		for h in hyper_in_list {
			tensor.free_tensor(h, allocator = allocator)
		}
		delete(hyper_in_list)
	}
	for mlp, i in md.output_hypernetworks_mlps {
		h_slice := tensor.slice(mask_tokens_out, {{}, i, {}}, keepdims = false, allocator = allocator)
		defer tensor.free_tensor(h_slice, allocator = allocator)
		h_mlp := forward_mlp_mask_decoder(mlp, h_slice, allocator)
		append(&hyper_in_list, h_mlp)
	}
	hyper_in := tensor.stack(hyper_in_list[:], 1, allocator)
	defer tensor.free_tensor(hyper_in, allocator = allocator)

	b, c, h, w =
		upscaled_embedding.shape[0],
		upscaled_embedding.shape[1],
		upscaled_embedding.shape[2],
		upscaled_embedding.shape[3]
	upscaled_reshaped := tensor.reshape(upscaled_embedding, {b, c, h * w}, allocator)
	defer tensor.free_tensor(upscaled_reshaped, allocator = allocator)
	masks_flat := tensor.matmul(hyper_in, upscaled_reshaped, allocator)
	defer tensor.free_tensor(masks_flat, allocator = allocator)

	// Now output channel is something different, thus, gotta recalc according to
	// current length and divided by product of the original b, h, and w
	c_out := tensor.shape_to_size(masks_flat.shape) / (b * h * w)

	masks_reshaped := tensor.reshape(masks_flat, {b, c_out, h, w}, allocator)
	defer tensor.free_tensor(masks_reshaped, allocator = allocator)
	masks := tensor.clone(masks_reshaped, allocator)
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
	masks_full, iou_pred_full := predict_mask(
		md,
		image_embeddings,
		image_pe,
		sparse_prompt_embeddings,
		dense_prompt_embeddings,
		allocator,
	)
	defer tensor.free_tensor(masks_full, allocator = allocator)
	defer tensor.free_tensor(iou_pred_full, allocator = allocator)

	// When multimask is false, just take the first channel from masks and iou_pred
	// TODO(Aria): implement multimask output
	masks := tensor.slice(masks_full, {{}, R(1)}, allocator = allocator)
	iou_pred := tensor.slice(iou_pred_full, {{}, R(1)}, allocator = allocator)
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
