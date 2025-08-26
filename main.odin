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
import "core:strings"
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

	rl.ImageFormat(&image_resized, rl.PixelFormat.UNCOMPRESSED_R8G8B8A8)
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

tensor_to_image :: proc(tensor: ^tensor.Tensor(f32), allocator := context.allocator) -> rl.Image {
	means := []f32{123.675, 116.28, 103.53}
	std := []f32{58.395, 57.12, 57.375}

	// tensor shape is [1, 3, H, W], we want H and W
	height := tensor.shape[2]
	width := tensor.shape[3]

	// Need persistent memory for the image data
	pixels := make([]byte, width * height * 4, allocator)

	// Convert CHW to HWC and denormalize
	for row in 0 ..< height {
		for col in 0 ..< width {
			// Account for batch dimension (index 0) in tensor layout
			// Tensor is [1, 3, H, W], so skip first dimension
			r_idx := 0 * height * width + row * width + col
			g_idx := 1 * height * width + row * width + col
			b_idx := 2 * height * width + row * width + col

			// Denormalize: pixel = value * std + mean
			r := tensor.data[r_idx] * std[0] + means[0]
			g := tensor.data[g_idx] * std[1] + means[1]
			b := tensor.data[b_idx] * std[2] + means[2]

			// RGBA pixel layout
			pixel_idx := (row * width + col) * 4
			pixels[pixel_idx + 0] = byte(math.clamp(r, 0, 255))
			pixels[pixel_idx + 1] = byte(math.clamp(g, 0, 255))
			pixels[pixel_idx + 2] = byte(math.clamp(b, 0, 255))
			pixels[pixel_idx + 3] = 255
		}
	}

	img: rl.Image
	img.data = raw_data(pixels)
	img.width = i32(width)
	img.height = i32(height)
	img.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8A8
	img.mipmaps = 1

	return img
}

main :: proc() {
	args := os.args
	if len(args) != 2 {
		fmt.println("Image path as the first argument is required")
		os.exit(1)
	}
	image_path_c, err_cstr := strings.clone_to_cstring(args[1], context.temp_allocator)
	if err_cstr != nil {
		fmt.printfln("cannot open file %s", args[1])
		os.exit(1)
	}

	rl.SetTraceLogLevel(.ERROR)
	rl.InitWindow(1024, 1024, "Segment Anything")
	defer rl.CloseWindow()

	rl.SetTargetFPS(60)

	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	trace.init_trace()
	defer trace.finish_trace()

	main_trace := trace.TRACE_FUNCTION("main")
	defer trace.end_scoped_trace(main_trace)

	image := rl.LoadImage(image_path_c)
	defer rl.UnloadImage(image)

	input := preprocess(f32, &image, 1024, arena_alloc)

	model_init_trace := trace.TRACE_SECTION("model_initialization")
	model_file := "models/mobile_sam-tiny-vitt.safetensors"
	safetensors, err_st_load := st.read_from_file(f32, model_file, arena_alloc)
	assert(err_st_load == nil)
	sam := tf.new_tiny(f32, safetensors, arena_alloc)
	defer tf.free_tiny(sam, arena_alloc)
	image_encoder := sam.image_encoder.(^vit.Tiny_ViT_5m(f32))
	trace.end_scoped_trace(model_init_trace)

	t := time.now()
	image_embedding := vit.forward_tiny_vit_5m(image_encoder, input, arena_alloc)
	fmt.println("Image embedding inference time:", time.since(t))

	display_image := tensor_to_image(input, arena_alloc)
	texture := rl.LoadTextureFromImage(display_image)
	defer rl.UnloadTexture(texture)

	masks, masks_bin, iou_pred: ^tensor.Tensor(f32)
	for !rl.WindowShouldClose() {
		// Get normalized mouse position (0-1)
		mouse_x := f32(rl.GetMouseX()) / 1024.0
		mouse_y := f32(rl.GetMouseY()) / 1024.0

		// Single positive point at mouse position
		points := []tf.Point(f32){{mouse_x, mouse_y, true}}

		t = time.now()
		masks, iou_pred = tf.forward_sam_for_embedding(
			sam,
			image_embedding,
			input.shape[2],
			input.shape[3],
			points,
			context.temp_allocator,
		)

		// Create mask overlay texture
		mask_h := masks.shape[2]
		mask_w := masks.shape[3]
		mask_pixels := make([]byte, mask_w * mask_h * 4, context.temp_allocator)

		for row in 0 ..< mask_h {
			for col in 0 ..< mask_w {
				mask_idx := row * mask_w + col
				mask_value := masks.data[mask_idx]

				pixel_idx := mask_idx * 4
				if mask_value > 0 {
					// Red overlay for positive mask
					mask_pixels[pixel_idx + 0] = 255 // R
					mask_pixels[pixel_idx + 1] = 20 // G
					mask_pixels[pixel_idx + 2] = 20 // B
					mask_pixels[pixel_idx + 3] = 128 // A
				} else {
					// Transparent for negative mask
					mask_pixels[pixel_idx + 0] = 0
					mask_pixels[pixel_idx + 1] = 0
					mask_pixels[pixel_idx + 2] = 0
					mask_pixels[pixel_idx + 3] = 0
				}
			}
		}

		mask_image: rl.Image
		mask_image.data = raw_data(mask_pixels)
		mask_image.width = i32(mask_w)
		mask_image.height = i32(mask_h)
		mask_image.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8A8
		mask_image.mipmaps = 1

		mask_texture := rl.LoadTextureFromImage(mask_image)
		defer rl.UnloadTexture(mask_texture)

		rl.BeginDrawing()
		rl.ClearBackground(rl.BLACK)

		rl.DrawTexture(texture, 0, 0, rl.WHITE)

		// Draw the mask overlay, scaled up to 1024x1024
		scale := 1024.0 / f32(mask_w)
		rl.DrawTextureEx(mask_texture, {0, 0}, 0, scale, rl.WHITE)

		// Draw the point indicator
		screen_x := i32(mouse_x * 1024)
		screen_y := i32(mouse_y * 1024)
		rl.DrawCircle(screen_x, screen_y, 5, rl.GREEN)
		rl.DrawCircleLines(screen_x, screen_y, 5, rl.WHITE)

		// Show FPS and IoU score
		rl.DrawFPS(10, 10)
		rl.DrawText(rl.TextFormat("IoU: %.3f", iou_pred.data[0]), 10, 35, 20, rl.RED)

		rl.EndDrawing()

		free_all(context.temp_allocator)
	}
}
