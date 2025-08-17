package prompt_encoder

import "../../nn"
import st "../../safetensors"
import "../../tensor"
import vb "../var_builder"
import "core:fmt"
import "core:math"

NUM_POINTS_EMBEDDINGS :: 4

Position_Embedding_Random :: struct($T: typeid) {
	positional_encoding_gaussian_matrix: ^tensor.Tensor(T),
}


pe_encoding :: proc(
	pe: ^Position_Embedding_Random($T),
	coords: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	talloc := context.temp_allocator

	// coords = coords * 2 - 1 (affine transformation)
	coords_transformed := tensor.clone(coords, talloc)
	for v, i in coords_transformed.data {
		coords_transformed.data[i] = coords_transformed.data[i] * 2 - 1
	}

	// Matmul with gaussian matrix
	coords_encoded := tensor.matmul(
		coords_transformed,
		pe.positional_encoding_gaussian_matrix,
		talloc,
	)

	// Multiply by 2π
	two_pi := tensor.new_with_init([]T{T(2 * math.PI)}, []uint{}, talloc)
	coords_scaled_pi := tensor.mul(coords_encoded, two_pi, talloc)

	// Get sin and cos
	sin_coords := tensor.sin(coords_scaled_pi, talloc)
	cos_coords := tensor.cos(coords_scaled_pi, talloc)

	// Concatenate along last dimension
	result := tensor.cat(
		[]^tensor.Tensor(T){sin_coords, cos_coords},
		uint(len(coords.shape) - 1),
		allocator,
	)

	return result
}

forward_position_embedding :: proc(
	pe: ^Position_Embedding_Random($T),
	h, w: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	talloc := context.temp_allocator

	// Create x coordinates: [0, 1, ..., w-1] + 0.5
	x_embed_norm := tensor.arange(T, w, talloc)
	for v, i in x_embed_norm.data do x_embed_norm.data[i] = (v + T(0.5)) / T(w)

	x_embed_reshaped := tensor.reshape(x_embed_norm, []uint{1, w}, allocator)
	x_embed_broadcast := tensor.broadcast_as(x_embed_reshaped, []uint{h, w}, talloc)

	// Create y coordinates similarly
	y_embed_norm := tensor.arange(T, h, talloc)
	for v, i in y_embed_norm.data do y_embed_norm.data[i] = (v + T(0.5)) / T(h)
	y_embed_reshaped := tensor.reshape(y_embed_norm, []uint{h, 1}, talloc)
	y_embed_broadcast := tensor.broadcast_as(y_embed_reshaped, []uint{h, w}, talloc)
	coords := tensor.stack(
		[]^tensor.Tensor(T){x_embed_broadcast, y_embed_broadcast},
		2, // last axis
		talloc,
	)

	encoded := pe_encoding(pe, coords, talloc)
	result := tensor.permute(encoded, []uint{2, 0, 1}, allocator)

	return result
}

forward_position_embedding_with_coords :: proc(
	pe: ^Position_Embedding_Random($T),
	coords_input: ^tensor.Tensor(T),
	image_w, image_h: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	coords0 := tensor.slice(coords_input, {{}, {}, {0, 1, 1}}, context.temp_allocator)
	for v, i in coords0.data do coords0.data[i] /= T(image_h)
	coords1 := tensor.slice(coords_input, {{}, {}, {1, 2, 1}}, context.temp_allocator)
	for v, i in coords1.data do coords1.data[i] /= T(image_w)
	c := len(coords_input.shape) - 1
	coords := tensor.cat([]^tensor.Tensor(T){coords0, coords1}, uint(c))
	result := pe_encoding(pe, coords, allocator)
	return result
}

Prompt_Encoder :: struct($T: typeid) {
	pe_layer:               ^Position_Embedding_Random(T),
	point_embeddings:       [dynamic]^nn.Embedding(T),
	not_a_point_embed:      ^nn.Embedding(T),
	mask_downscaling_conv1: ^nn.Conv_2d(T),
	mask_downscaling_conv2: ^nn.Conv_2d(T),
	mask_downscaling_conv3: ^nn.Conv_2d(T),
	mask_downscaling_ln1:   ^nn.Channel_Layer_Norm(T),
	mask_downscaling_ln2:   ^nn.Channel_Layer_Norm(T),
	no_mask_embed:          ^nn.Embedding(T),
	image_embedding_size:   [2]u64,
	input_image_size:       [2]u64,
	embed_dim:              u64,
}

new_prompt_encoder :: proc(
	$T: typeid,
	safetensors: ^st.Safe_Tensors(T),
	embed_dim: u64,
	image_embedding_size: [2]u64,
	input_image_size: [2]u64,
	mask_in_chans: u64,
	allocator := context.allocator,
) -> ^Prompt_Encoder(T) {
	vb_prompt_encoder := vb.Var_Builder(T){"prompt_encoder", safetensors, nil}

	vb_pe_layer := vb.vb_make(T, "pe_layer", &vb_prompt_encoder)
	pe_layer := new_clone(
		Position_Embedding_Random(T) {
			positional_encoding_gaussian_matrix = tensor.zeros(
				T,
				{2, uint(embed_dim / 2)},
				allocator,
			),
		},
		allocator,
	)
	vb.assign_to_tensor(
		&vb_pe_layer,
		"positional_encoding_gaussian_matrix",
		pe_layer.positional_encoding_gaussian_matrix,
	)

	not_a_point_embed := nn.new_embedding(T, 1, uint(embed_dim), true, allocator)
	vb.assign_to_tensor(&vb_prompt_encoder, "not_a_point_embed.weight", not_a_point_embed.weight)

	no_mask_embed := nn.new_embedding(T, 1, uint(embed_dim), true, allocator)
	vb.assign_to_tensor(&vb_prompt_encoder, "no_mask_embed.weight", no_mask_embed.weight)

	mask_downscaling_conv1 := nn.new_conv2d(
		T,
		1,
		uint(mask_in_chans / 4),
		{2, 2},
		stride = 2,
		padding = 0,
		dilation = 1,
		groups = 1,
		use_bias = true,
		init = false,
		allocator = allocator,
	)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.0.weight", mask_downscaling_conv1.w)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.0.bias", mask_downscaling_conv1.b.?)

	mask_downscaling_conv2 := nn.new_conv2d(
		T,
		uint(mask_in_chans / 4),
		uint(mask_in_chans),
		{2, 2},
		stride = 2,
		init = false,
		allocator = allocator,
	)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.3.weight", mask_downscaling_conv2.w)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.3.bias", mask_downscaling_conv2.b.?)

	mask_downscaling_conv3 := nn.new_conv2d(
		T,
		uint(mask_in_chans),
		uint(embed_dim),
		{1, 1},
		init = false,
		allocator = allocator,
	)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.6.weight", mask_downscaling_conv3.w)
	vb.assign_to_tensor(&vb_prompt_encoder, "mask_downscaling.6.bias", mask_downscaling_conv3.b.?)

	mask_downscaling_ln1 := nn.new_channel_layer_norm(T, uint(mask_in_chans / 4), 1e-6, allocator)
	vb.assign_to_tensor(
		&vb_prompt_encoder,
		"mask_downscaling.1.weight",
		mask_downscaling_ln1.weight,
	)

	mask_downscaling_ln2 := nn.new_channel_layer_norm(T, uint(mask_in_chans), 1e-6, allocator)
	vb.assign_to_tensor(
		&vb_prompt_encoder,
		"mask_downscaling.4.weight",
		mask_downscaling_ln2.weight,
	)

	point_embeddings: [dynamic]^nn.Embedding(T)
	for i in 0 ..< NUM_POINTS_EMBEDDINGS {
		emb := nn.new_embedding(T, 1, uint(embed_dim), true, allocator)
		vb.assign_to_tensor(
			&vb_prompt_encoder,
			fmt.tprintf("point_embeddings.%d.weight", i),
			emb.weight,
		)
		append(&point_embeddings, emb)
	}

	pe := new_clone(
		Prompt_Encoder(T) {
			pe_layer = pe_layer,
			point_embeddings = point_embeddings,
			not_a_point_embed = not_a_point_embed,
			mask_downscaling_conv1 = mask_downscaling_conv1,
			mask_downscaling_ln1 = mask_downscaling_ln1,
			mask_downscaling_conv2 = mask_downscaling_conv2,
			mask_downscaling_ln2 = mask_downscaling_ln2,
			mask_downscaling_conv3 = mask_downscaling_conv3,
			no_mask_embed = no_mask_embed,
			image_embedding_size = image_embedding_size,
			input_image_size = input_image_size,
			embed_dim = embed_dim,
		},
		allocator,
	)
	return pe
}

prompt_encoder_embed_points :: proc(
	pe: ^Prompt_Encoder($T),
	points, labels: ^tensor.Tensor(T),
	pad: bool,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	talloc := context.temp_allocator

	// Add 0.5 to points
	points := tensor.clone(points, talloc)
	for i in 0 ..< len(points.data) {
		points.data[i] += T(0.5)
	}

	// Handle padding
	labels := labels
	if pad {
		points_padded := tensor.zeros(T, {points.shape[0], 1, 2}, talloc)
		labels_padded := tensor.new_with_init([]T{-1}, {labels.shape[0], 1}, talloc)
		points = tensor.cat([]^tensor.Tensor(T){points, points_padded}, 1, talloc)
		labels = tensor.cat([]^tensor.Tensor(T){labels, labels_padded}, 1, talloc)
	} else {
		labels = tensor.clone(labels, talloc)
	}

	// Get point embeddings
	point_embedding := forward_position_embedding_with_coords(
		pe.pe_layer,
		points,
		uint(pe.input_image_size.x),
		uint(pe.input_image_size.y),
		talloc,
	)

	n_points := points.shape[1]
	embed_dim := pe.point_embeddings[0].embedding_dim
	result := tensor.clone(point_embedding, allocator)
	for i in 0 ..< n_points {
		label := labels.data[i]
		base_idx := i * embed_dim

		// If label < 0, replace with not_a_point_embed
		if label < 0 {
			for d in 0 ..< embed_dim {
				result.data[base_idx + d] = pe.not_a_point_embed.weight.data[d]
			}
		} else if label == 0 {
			// If label == 0, ADD negative embedding
			for d in 0 ..< embed_dim {
				result.data[base_idx + d] += pe.point_embeddings[0].weight.data[d]
			}
		} else if label == 1 {
			// If label == 1, ADD positive embedding
			for d in 0 ..< embed_dim {
				result.data[base_idx + d] += pe.point_embeddings[1].weight.data[d]
			}
		}
	}

	return result
}

forward_prompt_encoder :: proc(
	pe: ^Prompt_Encoder($T),
	points, labels: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> (
	sparse_embeddings, dense_embeddings: ^tensor.Tensor(T),
) {
	// Prompt encoder forward
	//// Embed points (se_points)
	se_points := prompt_encoder_embed_points(pe, points, labels, true, allocator)

	//// Sparse embeddings (which is se_points since se_boxes is nil)
	sparse_embeddings = se_points

	//// Dense embedding (should generate because masks is assumed to be nonexistent)
	// Assuming no masks case:
	emb := pe.no_mask_embed.weight
	embed_dim := emb.shape[1] // assuming shape is [1, embed_dim]
	reshaped := tensor.reshape(emb, []uint{1, embed_dim, 1, 1}, context.temp_allocator)
	// Broadcast/expand to [1, embed_dim, H, W]
	target_shape := []uint {
		1,
		embed_dim,
		uint(pe.image_embedding_size[0]),
		uint(pe.image_embedding_size[1]),
	}
	dense_embeddings = tensor.broadcast_as(reshaped, target_shape, allocator)
	return
}

free_prompt_encoder :: proc(pe: ^Prompt_Encoder($T), allocator := context.allocator) {
	tensor.free_tensor(pe.pe_layer.positional_encoding_gaussian_matrix, allocator)
	nn.free_embedding(pe.not_a_point_embed, allocator)
	nn.free_embedding(pe.no_mask_embed, allocator)
	nn.free_conv2d(pe.mask_downscaling_conv1, allocator)
	nn.free_conv2d(pe.mask_downscaling_conv2, allocator)
	nn.free_conv2d(pe.mask_downscaling_conv3, allocator)
	nn.free_channel_layer_norm(pe.mask_downscaling_ln1, allocator)
	nn.free_channel_layer_norm(pe.mask_downscaling_ln2, allocator)
	for e in pe.point_embeddings {
		nn.free_embedding(e, allocator)
	}
	delete(pe.point_embeddings)

	free(pe, allocator)
}
