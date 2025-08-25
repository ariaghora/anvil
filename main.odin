package main

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:math"
import "core:math/rand"
import vmem "core:mem/virtual"
import "core:os"
import "core:os/os2"
import "core:slice"
import "core:testing"
import "core:time"
import "libs/nn"
import st "libs/safetensors"
import "libs/tensor"
import "libs/trace"
import tf "libs/transformer"
import md "libs/transformer/mask_decoder"
import pe "libs/transformer/prompt_encoder"
import "libs/transformer/vit"
import rl "vendor:raylib"
// import "vendor:stb/image"

IMAGE_SIZE :: uint(1024)

preprocess :: proc(
	$T: typeid,
	image: ^rl.Image,
	target_size: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	width := uint(image.width)
	height := uint(image.height)

	means := []f32{123.675, 116.28, 103.53}
	std := []f32{58.395, 57.12, 57.375}

	w_out, h_out: uint
	if width > height {
		w_out = target_size
		h_out = uint(f32(target_size * height) / f32(width))
	} else {
		h_out = target_size
		w_out = uint(f32(target_size * width) / f32(height))
	}

	image_resized := rl.ImageCopy(image^)
	defer rl.UnloadImage(image_resized)
	rl.ImageResize(&image_resized, i32(w_out), i32(h_out))

	image_data := cast([^]byte)image_resized.data

	image_chw := make([]f32, 3 * w_out * h_out, context.temp_allocator)
	for row in 0 ..< h_out {
		for col in 0 ..< w_out {
			src_idx := (row * w_out + col) * 4

			r_idx := 0 * h_out * w_out + row * w_out + col
			g_idx := 1 * h_out * w_out + row * w_out + col
			b_idx := 2 * h_out * w_out + row * w_out + col

			image_chw[r_idx] = (f32(image_data[src_idx + 0]) - means[0]) / std[0]
			image_chw[g_idx] = (f32(image_data[src_idx + 1]) - means[1]) / std[1]
			image_chw[b_idx] = (f32(image_data[src_idx + 2]) - means[2]) / std[2]
		}
	}

	image_chw_padded := make([]f32, 3 * target_size * target_size, context.temp_allocator)

	for c in 0 ..< 3 {
		for row in 0 ..< h_out {
			for col in 0 ..< w_out {
				src_idx := uint(c) * h_out * w_out + row * w_out + col
				dst_idx := uint(c) * target_size * target_size + row * target_size + col
				image_chw_padded[dst_idx] = image_chw[src_idx]
			}
		}
	}

	image_out := tensor.new_with_init(
		image_chw_padded,
		{1, 3, target_size, target_size},
		allocator,
	)

	return image_out
}

main :: proc() {
	rl.InitWindow(1024, 1024, "Segment Anything")


	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	ensure(arena_err == nil)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	trace.init_trace()
	defer trace.finish_trace()


	main_trace := trace.TRACE_FUNCTION("main")
	defer trace.end_scoped_trace(main_trace)

	// Inputs
	image := rl.LoadImage("../candle-tinyvit-comp/car.jpg")
	defer rl.UnloadImage(image)
	x := image.width
	y := image.height
	chan := i32(4)

	// Convert to RGB if needed (SAM expects RGB, not RGBA)
	rl.ImageFormat(&image, rl.PixelFormat.UNCOMPRESSED_R8G8B8)

	//// Image as tensor
	input := preprocess(f32, &image, 1024, arena_alloc)

	//// mouse clicks
	points := []tf.Point(f32){{0.5, 0.85, true}, {0.4, 0.2, false}}
	n_points := uint(len(points))

	// Model loading
	model_init_trace := trace.TRACE_SECTION("model_initialization")
	model_file := "models/mobile_sam-tiny-vitt.safetensors"
	safetensors, err_st_load := st.read_from_file(f32, model_file, arena_alloc)
	assert(err_st_load == nil)
	sam := tf.new_tiny(f32, safetensors, arena_alloc)
	defer tf.free_tiny(sam, arena_alloc)
	image_encoder := sam.image_encoder.(^vit.Tiny_ViT_5m(f32))
	trace.end_scoped_trace(model_init_trace)

	// 1) Get image embedding
	t := time.now()

	image_embedding := vit.forward_tiny_vit_5m(image_encoder, input, arena_alloc)
	fmt.println("inference time phase 1:", time.since(t))

	// 2) Do mask decoding (LOOP, nest the allocator)
	masks, masks_bin, iou_pred: ^tensor.Tensor(f32)
	for !rl.WindowShouldClose() {
		t = time.now()
		masks, iou_pred = tf.forward_sam_for_embedding(
			sam,
			image_embedding,
			input.shape[2],
			input.shape[3],
			points,
			context.temp_allocator,
		)
		fmt.println("inference time phase 2:", time.since(t))

		// Postprocessing
		//// Low-res binary mask
		masks_bin = tensor.clone(masks, context.temp_allocator)
		for v, i in masks_bin.data do masks_bin.data[i] = v > 0 ? 1.0 : 0.0

		free_all(context.temp_allocator)
	}

}
