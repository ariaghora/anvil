package transformer

import "../nn"
import "../tensor"
import "core:container/intrusive/list"
import "core:mem"
import "core:terminal"

Attention :: struct($T: typeid) {
	norm:               ^nn.Layer_Norm(T),
	qkv, proj:          ^nn.Linear(T),
	ab:                 ^tensor.Tensor(T),
	key_dim, num_heads: uint,
	d:                  uint,
	dh:                 uint,
	scale:              T,
}

Mlp :: struct($T: typeid) {
	norm:     ^nn.Layer_Norm(T),
	fc1, fc2: ^nn.Linear(T),
}

// norm -> fc1 -> gelu -> fc2
forward_mlp :: proc(
	mlp: ^Mlp($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// TODO(Aria): layer norm
	// ...

	fc1_out := nn.forward_linear(mlp.fc1, x, allocator, loc)
	defer tensor.free_tensor(fc1_out, allocator)

	gelu_out = tensor.gelu(fc1_out, allocator, loc)
	defer tensor.free_tensor(gelu_out, allocator)

	fc2_out = nn.forward_linear(gelu_out, out, allocator)
	return fc2_out
}

Conv_2d_BN :: struct($T: typeid) {
	conv: ^nn.Conv_2d(T),
	bn:   ^nn.Batch_Norm_2d(T),
}

Patch_Embed :: struct($T: typeid) {
	conv1, conv_2: Conv_2d_BN(T),
}

Tiny_ViT_Block :: struct($T: typeid) {
	attn:             Attention(T),
	local_conv:       Conv_2d_BN(T),
	mlp:              Mlp(T),
	window_size:      uint,
	input_resolution: [2]uint,
}

Patch_Merging :: struct($T: typeid) {
	conv1, conv2, conv3: Conv_2d_BN(T),
	input_resolution:    [2]uint,
}

Basic_Layer :: struct($T: typeid) {
	blocks:     []Tiny_ViT_Block(T),
	downsample: Maybe(Patch_Merging(T)),
}

MB_Conv :: struct($T: typeid) {
	conv1, conv2, conv3: Conv_2d_BN(T),
}

Conv_Layer :: struct($T: typeid) {
	blocks:     []MB_Conv(T),
	downsample: Maybe(Patch_Merging(T)),
}

Tiny_ViT_5m :: struct($T: typeid) {
	patch_embed:            Patch_Embed(T),
	layer0:                 Conv_Layer(T),
	layers:                 []Basic_Layer(T),
	neck_conv1, neck_conv2: ^nn.Conv_2d(T),
	neck_ln1, neck_ln2:     ^nn.Layer_Norm(T),
}

new_vit_5m :: proc($T: typeid, allocator := context.allocator) -> ^Tiny_ViT_5m(T) {
	embed_dims := []u64{64, 128, 160, 320}
	depths := []u64{2, 2, 6, 2}
	num_heads := []u64{2, 4, 5, 10}
	window_sizes := []u64{7, 7, 14, 7}

	return new_clone(
		Tiny_ViT_5m(T) {
			// embed_dims = embed_dims,
			// depths = depths,
			// num_heads = num_heads,
			// window_sizes = window_sizes,
			// num_classes = 1000,
		},
		allocator,
	)
}

free_vit_5m :: proc(vit: ^Tiny_ViT_5m($T), allocator := context.allocator) {
	free(vit, allocator)
}
