package transformer

import "../nn"
import "../tensor"

Position_Embedding_Random :: struct {}

Prompt_Encoder :: struct {
	pe_layer:               ^Position_Embedding_Random,
	point_embeddings:       []^nn.Embedding,
	not_a_point_embed:      ^nn.Embedding,
	mask_downscaling_conv1: ^nn.Conv_2d,
	mask_downscaling_conv2: ^nn.Conv_2d,
	mask_downscaling_conv3: ^nn.Conv_2d,
	mask_downscaling_ln1:   ^nn.Layer_Norm_2d,
	mask_downscaling_ln2:   ^nn.Layer_Norm_2d,
	no_mask_embed:          ^nn.Embedding,
	image_embedding_size:   [2]u64,
	input_image_size:       [2]u64,
	embed_dim:              u64,
}

new_prompt_encoder :: proc(
	$T: typeid,
	embed_dim: u64,
	image_embedding_size: [2]u64,
	input_image_size: [2]u64,
	mask_in_chans: u64,
	allocator := context.allocator,
) -> ^Prompt_Encoder {
	pe := new_clone(
		Prompt_Encoder {
			pe_layer = nil,
			point_embeddings = nil,
			not_a_point_embed = nil,
			mask_downscaling_conv1 = nil,
			mask_downscaling_ln1 = nil,
			mask_downscaling_conv2 = nil,
			mask_downscaling_ln2 = nil,
			mask_downscaling_conv3 = nil,
			no_mask_embed = nil,
			image_embedding_size = image_embedding_size,
			input_image_size = input_image_size,
			embed_dim = embed_dim,
		},
		allocator,
	)
	return pe
}

free_prompt_encoder :: proc(pe: ^Prompt_Encoder, allocator := context.allocator) {
	free(pe, allocator)
}
