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
import pe "libs/transformer/prompt_encoder"
import "libs/transformer/vit"

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
	image_encoder := sam.image_encoder.(^vit.Tiny_ViT_5m(f32))
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
	patch_embedding_conv1 := vit.forward_conv_2d_bn(image_encoder.patch_embed.conv1, input, talloc)
	patch_embedding_conv1_gelu := vit.gelu_fast(patch_embedding_conv1, talloc)
	patch_embedding_conv2 := vit.forward_conv_2d_bn(
		image_encoder.patch_embed.conv2,
		patch_embedding_conv1_gelu,
		talloc,
	)

	// layer0
	layer0 := vit.forward_conv_layer(image_encoder.layer0, patch_embedding_conv2, talloc)
	// layer1 to n
	layers := layer0
	for i in 0 ..< len(image_encoder.layers) {
		layer := image_encoder.layers[i]
		layers = vit.forward_basic_layer(layer, layers, talloc)
	}

	// neck_conv1
	b := layers.shape[0]
	c := layers.shape[2]
	sequence_length := layers.shape[1]
	spatial_dim := uint(math.sqrt(f64(sequence_length)))
	layers_4d := tensor.reshape(layers, []uint{b, spatial_dim, spatial_dim, c}, talloc)
	layers_conv := tensor.permute(layers_4d, []uint{0, 3, 1, 2}, talloc)
	neck_conv1 := nn.forward_conv2d(image_encoder.neck_conv1, layers_conv, talloc)
	neck_ln1 := nn.forward_channel_layer_norm(image_encoder.neck_ln1, neck_conv1, talloc)
	neck_conv2 := nn.forward_conv2d(image_encoder.neck_conv2, neck_ln1, talloc)
	neck_ln2 := nn.forward_channel_layer_norm(image_encoder.neck_ln2, neck_conv2, talloc)
	fmt.println("inference time phase 1:", time.since(t))

	t = time.now()
	pe_final := pe.forward_position_embedding(
		sam.prompt_encoder.pe_layer,
		uint(sam.prompt_encoder.image_embedding_size[0]),
		uint(sam.prompt_encoder.image_embedding_size[1]),
		talloc,
	)
	points := []tf.Point(f32){{0.5, 0.55, true}}
	n_points := uint(len(points))

	// Build flat array of scaled coordinates
	xys := make([]f32, n_points * 2, context.temp_allocator)
	labels := make([]f32, n_points, context.temp_allocator)

	original_w, original_h := input.shape[2], input.shape[3]
	for point, i in points {
		xys[i * 2] = f32(point.x) * f32(original_w)
		xys[i * 2 + 1] = f32(point.y) * f32(original_h)
		labels[i] = point.is_positive ? 1.0 : 0.0
	}
	points_tensor := tensor.new_with_init(xys, []uint{1, n_points, 2}, talloc)
	labels_tensor := tensor.new_with_init(labels, []uint{1, n_points}, talloc)

	// Prompt encoder forward
	//// Embed points (se_points)
	se_points := pe.prompt_encoder_embed_points(
		sam.prompt_encoder,
		points_tensor,
		labels_tensor,
		true,
		talloc,
	)

	//// Sparse embeddings (which is se_points since se_boxes is nil)

	//// Dense embedding (should generate because masks is nil)

	// Mask decoder forward


	fmt.println("inference time phase 2:", time.since(t))

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
	map_insert(&output_tensors, "pr_en_7-pe_final", pe_final)
	map_insert(&output_tensors, "pr_en_se_points", se_points)

	err_st_wr := st.write_tensors_to_file(
		&st.Safe_Tensors(f32){tensors = output_tensors},
		"models/patch_embedding_odin.safetensors",
	)
	assert(err_st_wr == nil)

}
