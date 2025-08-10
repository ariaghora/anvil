package main

import "core:fmt"
import "core:math/rand"
import vmem "core:mem/virtual"
import st "libs/safetensors"
import "libs/tensor"
import "libs/trace"
import tf "libs/transformer"

vit_load_weights :: proc(
	vit: ^tf.Tiny_ViT_5m($T),
	safetensors: ^st.Safe_Tensors(T),
) -> st.Safe_Tensors_Error {
	// Inference flow:
	//   Patch embedding > layer0 > layer_seqs > neck_conv1+ln1 > neck_conv2+ln2

	// Patch embedding
	st.tensor_assign_from_safe_tensors(
		[]^tensor.Tensor(T) {
			vit.patch_embed.conv1.conv.w,
			vit.patch_embed.conv1.bn.weight,
			vit.patch_embed.conv1.bn.bias,
			vit.patch_embed.conv1.bn.running_mean,
			vit.patch_embed.conv1.bn.running_var,
			vit.patch_embed.conv2.conv.w,
			vit.patch_embed.conv2.bn.weight,
			vit.patch_embed.conv2.bn.bias,
			vit.patch_embed.conv2.bn.running_mean,
			vit.patch_embed.conv2.bn.running_var,
		},
		[]string {
			"image_encoder.patch_embed.seq.0.c.weight",
			"image_encoder.patch_embed.seq.0.bn.weight",
			"image_encoder.patch_embed.seq.0.bn.bias",
			"image_encoder.patch_embed.seq.0.bn.running_mean",
			"image_encoder.patch_embed.seq.0.bn.running_var",
			"image_encoder.patch_embed.seq.2.c.weight",
			"image_encoder.patch_embed.seq.2.bn.weight",
			"image_encoder.patch_embed.seq.2.bn.bias",
			"image_encoder.patch_embed.seq.2.bn.running_mean",
			"image_encoder.patch_embed.seq.2.bn.running_var",
		},
		safetensors,
	) or_return


	// layer0
	l0: ^tf.Conv_Layer(f32) = vit.layer0
	l0_blocks: []^tf.MB_Conv(f32) = l0.blocks
	for b in &l0_blocks {
		// TODO
	}
	foo := [4]f32{1, 2, 3, 4}

	l0_downsample_: Maybe(^tf.Patch_Merging) = l0.downsample
	if l0_downsample := l0_downsample_; l0_downsample != nil {
		// load l0_downsample
	}

	return nil
}

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


	model_init_trace := trace.TRACE_SECTION("model_initialization")
	vit := tf.new_tiny_vit_5m(f32, IMAGE_SIZE, false, arena_alloc)
	trace.end_scoped_trace(model_init_trace)

	model_file := #load("../yolo-studio/mobile_sam-tiny-vitt.safetensors")
	safetensors, err_st_load := st.read_from_bytes(f32, model_file, arena_alloc)
	assert(err_st_load == nil)
	err_st_assign := vit_load_weights(vit, safetensors)
	if err_st_assign != nil do fmt.println(err_st_assign)
	assert(err_st_assign == nil)

	input_st, err_in_st := st.read_from_file(
		f32,
		"tensorgen/safetensors/image.safetensors",
		context.temp_allocator,
	)
	assert(err_in_st == nil)
	input := input_st.tensors["image"]

	// Patch embedding
	talloc := context.temp_allocator
	patch_embedding_conv1 := tf.forward_conv_2d_bn(vit.patch_embed.conv1, input, talloc)
	patch_embedding_conv1_gelu := tf.gelu_fast(patch_embedding_conv1, talloc)
	patch_embedding_conv2 := tf.forward_conv_2d_bn(
		vit.patch_embed.conv2,
		patch_embedding_conv1_gelu,
		talloc,
	)


	// trace.trace_instant("starting_forward_pass")
	// forward_trace := trace.TRACE_SECTION("tiny_vit_forward_pass")
	// output := tf.forward_tiny_vit_5m(vit, input, true, arena_alloc)
	// trace.end_scoped_trace(forward_trace)
	// trace.trace_instant("forward_pass_completed")

	output_tensors := make(map[string]^tensor.Tensor(f32), context.temp_allocator)
	map_insert(&output_tensors, "input", input)
	// map_insert(&output_tensors, "patch_embedding", output.patch_embedding)
	map_insert(&output_tensors, "patch_embedding_conv1", patch_embedding_conv1)
	map_insert(&output_tensors, "patch_embedding_conv1_gelu", patch_embedding_conv1_gelu)
	map_insert(&output_tensors, "patch_embedding_conv2", patch_embedding_conv2)

	err_st_wr := st.write_tensors_to_file(
		&st.Safe_Tensors(f32){tensors = output_tensors},
		"tensorgen/safetensors/patch_embedding_odin.safetensors",
	)
	assert(err_st_wr == nil)

}
