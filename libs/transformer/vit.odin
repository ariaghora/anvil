package transformer

import "../nn"
import "../tensor"
import "core:container/intrusive/list"
import "core:mem"
import "core:terminal"

Attention :: struct {}

Mlp :: struct($T: typeid) {
	norm:     ^nn.Layer_Norm,
	fc1, fc2: ^nn.Linear(T),
}

// norm -> fc1 -> gelu -> fc2
forward_mlp :: proc(
	mlp: ^Mlp($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	out := nn.forward_linear(mlp.fc1, x, context.temp_allocator)
	out = nn.forward_linear(mlp.fc2, out, context.temp_allocator)
	return out
}

Patch_Embed :: struct {}

Basic_Layer :: struct {}

Conv_Layer :: struct {}

Tiny_ViT_5m :: struct {
	patch_embed:            ^Patch_Embed,
	layer0:                 ^Conv_Layer,
	layers:                 []^Basic_Layer,
	neck_conv1, neck_conv2: ^nn.Conv_2d,
	neck_ln1, neck_ln2:     ^nn.Layer_Norm_2d,
}

new_vit_5m :: proc($T: typeid, allocator := context.allocator) -> ^Tiny_ViT_5m {
	embed_dims := []u64{64, 128, 160, 320}
	depths := []u64{2, 2, 6, 2}
	num_heads := []u64{2, 4, 5, 10}
	window_sizes := []u64{7, 7, 14, 7}

	return new_clone(
		Tiny_ViT_5m {
			// embed_dims = embed_dims,
			// depths = depths,
			// num_heads = num_heads,
			// window_sizes = window_sizes,
			// num_classes = 1000,
		},
		allocator,
	)
}

free_vit_5m :: proc(vit: ^Tiny_ViT_5m, allocator := context.allocator) {
	free(vit, allocator)
}
