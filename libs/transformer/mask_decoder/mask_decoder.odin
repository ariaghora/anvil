package mask_decoder

import "../../nn"
import st "../../safetensors"
import "../../tensor"

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
) -> MLP_Mask_Decoder(T) {
	return {sigmoid_output = sigmoid_output}
}


Two_Way_Transformer :: struct($T: typeid) {}

Mask_Decoder :: struct($T: typeid) {
	iou_token:                 ^nn.Embedding(T),
	mask_tokens:               ^nn.Embedding(T),
	iou_prediction_head:       MLP_Mask_Decoder(T),
	output_upscaling_conv1:    ^nn.Conv_Transpose_2d(T),
	output_upscaling_ln:       ^nn.Channel_Layer_Norm(T),
	output_upscaling_conv2:    ^nn.Conv_Transpose_2d(T),
	num_mask_tokens:           uint,
	output_hypernetworks_mlps: [dynamic]MLP_Mask_Decoder(T),
	transformer:               Two_Way_Transformer(T),
}

import vb "../var_builder"

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

	return new_clone(
		Mask_Decoder(T) {
			iou_token = iou_token,
			mask_tokens = mask_tokens,
			iou_prediction_head = iou_prediction_head,
			num_mask_tokens = num_mask_tokens,
			output_upscaling_conv1 = output_upscaling_conv1,
			output_upscaling_conv2 = output_upscaling_conv2,
			output_upscaling_ln = output_upscaling_ln,
			transformer = {},
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
