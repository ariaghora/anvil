package sam

import st "../../safetensors"
import "../../tensor"
import "core:fmt"
import md "mask_decoder"
import pe "prompt_encoder"
import "vit"

VIT_PATCH_SIZE :: 16
IMAGE_SIZE :: 1024
PROMPT_EMBED_DIM :: 256

Sam :: struct($T: typeid) {
	image_encoder:         Image_Encoder(T),
	prompt_encoder:        ^pe.Prompt_Encoder(T),
	mask_decoder:          ^md.Mask_Decoder(T),
	pixel_mean, pixel_std: ^tensor.Tensor(T),
}


new_tiny :: proc(
	$T: typeid,
	safetensors: ^st.Safe_Tensors(T),
	allocator := context.allocator,
) -> ^Sam(T) {
	image_embedding_size := u64(IMAGE_SIZE / VIT_PATCH_SIZE)
	image_encoder := vit.new_tiny_vit_5m(T, safetensors, IMAGE_SIZE, false, allocator)
	promt_encoder := pe.new_prompt_encoder(
		T,
		safetensors,
		PROMPT_EMBED_DIM,
		[2]u64{image_embedding_size, image_embedding_size},
		[2]u64{IMAGE_SIZE, IMAGE_SIZE},
		16,
		allocator,
	)
	mask_decoder := md.new_mask_decoder(T, safetensors, PROMPT_EMBED_DIM, 3, 3, 256)

	return new_clone(
		Sam(T) {
			prompt_encoder = promt_encoder,
			image_encoder = image_encoder,
			mask_decoder = mask_decoder,
			pixel_mean = nil,
			pixel_std = nil,
		},
		allocator,
	)
}

Point :: struct($T: typeid) {
	x, y:        T,
	is_positive: bool,
}

forward_sam_for_embedding :: proc(
	sam: ^Sam($T),
	img_embeddings: ^tensor.Tensor(T),
	original_h, original_w: uint,
	points: []Point(T),
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	talloc := context.temp_allocator
	image_pe := pe.forward_position_embedding(
		sam.prompt_encoder.pe_layer,
		uint(sam.prompt_encoder.image_embedding_size[0]),
		uint(sam.prompt_encoder.image_embedding_size[1]),
		talloc,
	)
	image_pe = tensor.unsqueeze(image_pe, 0, talloc)

	// Build flat array of scaled coordinates
	n_points: uint = len(points)
	xys := make([]f32, n_points * 2, context.temp_allocator)
	labels := make([]f32, n_points, context.temp_allocator)
	// original_w, original_h := input.shape[2], input.shape[3]
	for point, i in points {
		xys[i * 2] = f32(point.x) * f32(original_w)
		xys[i * 2 + 1] = f32(point.y) * f32(original_h)
		labels[i] = point.is_positive ? 1.0 : 0.0
	}
	points_tensor := tensor.new_with_init(xys, []uint{1, n_points, 2}, talloc)
	labels_tensor := tensor.new_with_init(labels, []uint{1, n_points}, talloc)

	// Prompt encoder forward
	sparse_prompt_embeddings, dense_prompt_embeddings := pe.forward_prompt_encoder(
		sam.prompt_encoder,
		points_tensor,
		labels_tensor,
		talloc,
	)

	return md.forward_mask_decoder(
		sam.mask_decoder,
		img_embeddings,
		image_pe,
		sparse_prompt_embeddings,
		dense_prompt_embeddings,
		allocator,
	)
}

free_tiny :: proc(sam: ^Sam($T), allocator := context.allocator) {

	// free image encoder
	free_image_encoder(sam.image_encoder, allocator)
	pe.free_prompt_encoder(sam.prompt_encoder, allocator)

	// free mask decoder
	md.free_mask_decoder(sam.mask_decoder, allocator)

	// free tensors
	// TODO tensor.tensor_free(sam.pixel_mean, sam.pixel_std)

	free(sam, allocator)
}
