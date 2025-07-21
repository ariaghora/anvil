package transformer

import "../tensor"

VIT_PATCH_SIZE :: 16
IMAGE_SIZE :: 1024
PROMPT_EMBED_DIM :: 256

Sam :: struct($T: typeid) {
	image_encoder:         Image_Encoder,
	prompt_encoder:        ^Prompt_Encoder,
	mask_decoder:          ^Mask_Decoder,
	pixel_mean, pixel_std: ^tensor.Tensor(T),
}


new_tiny :: proc($T: typeid, allocator := context.allocator) -> ^Sam(T) {
	image_embedding_size := u64(IMAGE_SIZE / VIT_PATCH_SIZE)
	image_encoder := new_vit_5m(T, allocator)
	promt_encoder := new_prompt_encoder(
		T,
		PROMPT_EMBED_DIM,
		[2]u64{image_embedding_size, image_embedding_size},
		[2]u64{IMAGE_SIZE, IMAGE_SIZE},
		16,
		allocator,
	)

	return new_clone(
		Sam(T) {
			prompt_encoder = promt_encoder,
			image_encoder = image_encoder,
			mask_decoder = nil,
			pixel_mean = nil,
			pixel_std = nil,
		},
		allocator,
	)
}

free_tiny :: proc(sam: ^Sam($T), allocator := context.allocator) {
	free_prompt_encoder(sam.prompt_encoder)
	free_image_encoder(sam.image_encoder)

	// free image encoder
	// TODO

	// free mask decoder
	// TODO

	// free tensors
	// tensor.tensor_free(sam.pixel_mean, sam.pixel_std)

	free(sam, allocator)
}
