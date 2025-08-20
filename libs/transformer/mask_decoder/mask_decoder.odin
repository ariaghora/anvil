package mask_decoder

import "../../nn"
import st "../../safetensors"

MLP_Mask_Decoder :: struct($T: typeid) {
	layers:         ^nn.Linear(T),
	sigmoid_output: bool,
}

new_mlp_mask_decoder :: proc(
	$T: typeid,
	input_dim: uint,
	hidden_dim: uint,
	output_dim: uint,
	num_layers: uint,
	sigmoid_output: bool,
) -> MLP_Mask_Decoder(T) {
	return {sigmoid_output = sigmoid_output}
}


Two_Way_Transformer :: struct($T: typeid) {}

Mask_Decoder :: struct($T: typeid) {
	iou_token:                 ^nn.Embedding(T),
	mask_tokens:               ^nn.Embedding(T),
	iou_prediction_head:       MLP_Mask_Decoder(T),
	// output_upscaling_conv1: candle_nn::ConvTranspose2d,
	output_upscaling_ln:       nn.Channel_Layer_Norm(T),
	// output_upscaling_conv2: candle_nn::ConvTranspose2d,
	num_mask_tokens:           uint,
	output_hypernetworks_mlps: [dynamic]MLP_Mask_Decoder(T),
	transformer:               Two_Way_Transformer(T),
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

	iou_prediction_head := new_mlp_mask_decoder(
		T,
		transformer_dim,
		iou_head_hidden_dim,
		num_mask_tokens,
		iou_head_depth,
		false,
	)

	iou_token := nn.new_embedding(T, 1, transformer_dim, true, allocator)
	// TODO: vb_load "iou_token"

	mask_tokens := nn.new_embedding(T, num_mask_tokens, transformer_dim, true, allocator)
	// TODO: vb_load "mask_tokens"

	return new_clone(
		Mask_Decoder(T) {
			iou_token = iou_token,
			mask_tokens = mask_tokens,
			iou_prediction_head = iou_prediction_head,
			num_mask_tokens = num_mask_tokens,
		},
		allocator,
	)
}

free_mask_decoder :: proc(md: ^Mask_Decoder($T), allocator := context.allocator) {
	nn.free_embedding(md.iou_token)
	nn.free_embedding(md.mask_tokens)
	free(md)
}
