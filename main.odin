package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import vmem "core:mem/virtual"
import "core:time"
import "libs/nn"
import st "libs/safetensors"
import "libs/tensor"
import "libs/trace"
import tf "libs/transformer"

IMAGE_SIZE :: uint(1024)

main :: proc() {
	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	ensure(arena_err == nil)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	trace.init_trace()
	defer trace.finish_trace()

	main_trace := trace.TRACE_FUNCTION("main")
	defer trace.end_scoped_trace(main_trace)

	model_file := "models/mobile_sam-tiny-vitt.safetensors"
	safetensors, err_st_load := st.read_from_file(f32, model_file, arena_alloc)
	assert(err_st_load == nil)

	model_init_trace := trace.TRACE_SECTION("model_initialization")
	sam := tf.new_tiny(f32, safetensors, arena_alloc)
	defer tf.free_tiny(sam, arena_alloc)
	vit := sam.image_encoder.(^tf.Tiny_ViT_5m(f32))
	trace.end_scoped_trace(model_init_trace)

	input_st, err_in_st := st.read_from_file(
		f32,
		"models/image.safetensors",
		context.temp_allocator,
	)
	if err_in_st != nil {
		fmt.panicf("cannot set tensors: %v\n", err_in_st)
	}
	input := input_st.tensors["image"]

	// Patch embedding
	t := time.now()
	talloc := context.temp_allocator
	patch_embedding_conv1 := tf.forward_conv_2d_bn(vit.patch_embed.conv1, input, talloc)
	patch_embedding_conv1_gelu := tf.gelu_fast(patch_embedding_conv1, talloc)
	patch_embedding_conv2 := tf.forward_conv_2d_bn(
		vit.patch_embed.conv2,
		patch_embedding_conv1_gelu,
		talloc,
	)

	// layer0
	layer0 := tf.forward_conv_layer(vit.layer0, patch_embedding_conv2, talloc)
	// layer1 to n
	layers := layer0
	for i in 0 ..< len(vit.layers) {
		layer := vit.layers[i]
		layers = tf.forward_basic_layer(layer, layers, talloc)
	}

	// neck_conv1
	b := layers.shape[0]
	c := layers.shape[2]
	sequence_length := layers.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	layers_4d := tensor.reshape(layers, []uint{b, spatial_dim, spatial_dim, c}, talloc)
	layers_conv := tensor.permute(layers_4d, []uint{0, 3, 1, 2}, talloc)
	neck_conv1 := nn.forward_conv2d(vit.neck_conv1, layers_conv, talloc)
	neck_ln1 := nn.forward_channel_layer_norm(vit.neck_ln1, neck_conv1, talloc)
	neck_conv2 := nn.forward_conv2d(vit.neck_conv2, neck_ln1, talloc)
	neck_ln2 := nn.forward_channel_layer_norm(vit.neck_ln2, neck_conv2, talloc)
	fmt.println("inference time:", time.since(t))

	image_pe := tf.forward_position_embedding(
		sam.prompt_encoder.pe_layer,
		uint(sam.prompt_encoder.image_embedding_size[0]),
		uint(sam.prompt_encoder.image_embedding_size[1]),
		talloc,
	)
	// pe coords
	w, h := sam.prompt_encoder.input_image_size[0], sam.prompt_encoder.input_image_size[1]
	x_embed_norm := tensor.arange(f32, uint(w), talloc)
	for v, i in x_embed_norm.data do x_embed_norm.data[i] = (v + 0.5) / f32(w)
	x_embed_reshaped := tensor.reshape(x_embed_norm, {1, uint(w)}, talloc)
	x_embed_broadcast := tensor.broadcast_as(x_embed_reshaped, {uint(h), uint(w)}, talloc)
	y_embed_norm := tensor.arange(f32, uint(h), talloc)
	for v, i in y_embed_norm.data do y_embed_norm.data[i] = (v + 0.5) / f32(h)
	y_embed_reshaped := tensor.reshape(y_embed_norm, []uint{uint(h), 1}, talloc)
	y_embed_broadcast := tensor.broadcast_as(y_embed_reshaped, {uint(h), uint(w)}, talloc)
	coords := tensor.stack([]^tensor.Tensor(f32){x_embed_broadcast, y_embed_broadcast}, 2, talloc)
	// coords = tensor.permute(coords, {2, 0, 1}, talloc)
	// pe encoding

	output_tensors := make(map[string]^tensor.Tensor(f32), context.temp_allocator)
	map_insert(&output_tensors, "1-input", input)
	map_insert(&output_tensors, "2-patch_embedding_conv1", patch_embedding_conv1)
	map_insert(&output_tensors, "3-patch_embedding_conv1_gelu", patch_embedding_conv1_gelu)
	map_insert(&output_tensors, "4-patch_embedding_conv2", patch_embedding_conv2)
	map_insert(&output_tensors, "5-layer0", layer0)
	map_insert(&output_tensors, "6-layers", layers)
	map_insert(&output_tensors, "7-neck_conv1", neck_conv1)
	map_insert(&output_tensors, "8-neck_ln1", neck_ln1)
	map_insert(&output_tensors, "9-neck_conv2", neck_conv2)
	map_insert(&output_tensors, "10-neck_ln2", neck_ln2)
	map_insert(&output_tensors, "pr_en_1-pe_x_embed", x_embed_broadcast)
	map_insert(&output_tensors, "pr_en_2-pe_y_embed", y_embed_broadcast)
	map_insert(&output_tensors, "pr_en_3-coords", coords)

	err_st_wr := st.write_tensors_to_file(
		&st.Safe_Tensors(f32){tensors = output_tensors},
		"models/patch_embedding_odin.safetensors",
	)
	assert(err_st_wr == nil)

}
