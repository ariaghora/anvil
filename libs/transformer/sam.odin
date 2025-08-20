package transformer

import st "../safetensors"
import "../tensor"
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
) {
	image_pe := forward_position_embedding(sam.prompt_encoder.pe_layer, allocator)
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
