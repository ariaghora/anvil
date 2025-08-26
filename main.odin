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
	rl.SetTraceLogLevel(.ERROR)
	rl.InitWindow(800, 600, "Segment Anything - Drop an image to start")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	// Load model once at startup
	t := time.now()
	model_file := "models/mobile_sam-tiny-vitt.safetensors"
	safetensors, err_st_load := st.read_from_file(f32, model_file, arena_alloc)
	assert(err_st_load == nil)
	sam := tf.new_tiny(f32, safetensors, arena_alloc)
	defer tf.free_tiny(sam, arena_alloc)
	image_encoder := sam.image_encoder.(^vit.Tiny_ViT_5m(f32))
	fmt.println("Model loading time:", time.since(t))

	// Image-related state
	current_image: rl.Image
	texture: rl.Texture2D
	input: ^tensor.Tensor(f32)
	image_embedding: ^tensor.Tensor(f32)
	embedding_time: time.Duration
	has_image := false
	mask_texture: rl.Texture2D

	for !rl.WindowShouldClose() {
		if rl.IsFileDropped() {
			dropped_files := rl.LoadDroppedFiles()
			defer rl.UnloadDroppedFiles(dropped_files)

			if dropped_files.count > 0 {
				// Clean up previous image if exists
				if has_image {
					rl.UnloadTexture(texture)
					rl.UnloadImage(current_image)
					tensor.free_tensor(input, arena_alloc)
					tensor.free_tensor(image_embedding, arena_alloc)

				}

				// Then load new image
				current_image = rl.LoadImage(dropped_files.paths[0])

				// Let's make it nice by resizing window to the content, but make either side
				// not exceeding 1024
				window_w, window_h: i32
				if current_image.width > current_image.height {
					window_w = min(current_image.width, 1024)
					window_h = i32(
						f32(window_w) * f32(current_image.height) / f32(current_image.width),
					)
				} else {
					window_h = min(current_image.height, 1024)
					window_w = i32(
						f32(window_h) * f32(current_image.width) / f32(current_image.height),
					)
				}
				rl.SetWindowSize(window_w, window_h)

				// Center the window
				monitor := rl.GetCurrentMonitor()
				monitor_width := rl.GetMonitorWidth(monitor)
				monitor_height := rl.GetMonitorHeight(monitor)
				window_x := (monitor_width - window_w) / 2
				window_y := (monitor_height - window_h) / 2
				rl.SetWindowPosition(window_x, window_y)

				// Preprocess new image
				input = preprocess(f32, &current_image, 1024, arena_alloc)

				t = time.now()
				image_embedding = vit.forward_tiny_vit_5m(image_encoder, input, arena_alloc)
				embedding_time = time.since(t)

				texture = rl.LoadTextureFromImage(current_image)
				has_image = true
			}
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.DARKGRAY)

		if !has_image {
			// Show drop zone
			text := strings.clone_to_cstring("Drop an image here", context.temp_allocator)
			text_width := rl.MeasureText(text, 30)
			rl.DrawText(
				text,
				rl.GetScreenWidth() / 2 - text_width / 2,
				rl.GetScreenHeight() / 2 - 15,
				30,
				rl.LIGHTGRAY,
			)
		} else {
			// Run segmentation and display
			window_w := rl.GetScreenWidth()
			window_h := rl.GetScreenHeight()

			mouse_window_x := f32(rl.GetMouseX()) / f32(window_w)
			mouse_window_y := f32(rl.GetMouseY()) / f32(window_h)

			mouse_x, mouse_y: f32
			if current_image.width > current_image.height {
				mouse_x = mouse_window_x
				mouse_y = mouse_window_y * (f32(current_image.height) / f32(current_image.width))
			} else {
				mouse_x = mouse_window_x * (f32(current_image.width) / f32(current_image.height))
				mouse_y = mouse_window_y
			}

			points := []tf.Point(f32){{mouse_x, mouse_y, true}}

			t = time.now()
			masks, iou_pred := tf.forward_sam_for_embedding(
				sam,
				image_embedding,
				input.shape[2],
				input.shape[3],
				points,
				context.temp_allocator,
			)
			mask_time := time.since(t)

			positive_count := 0
			for v in masks.data {
				if v > 0 do positive_count += 1
			}


			// Create mask overlay
			mask_h := masks.shape[2]
			mask_w := masks.shape[3]
			mask_pixels := make([]byte, mask_w * mask_h * 4, context.temp_allocator)

			for row in 0 ..< mask_h {
				for col in 0 ..< mask_w {
					mask_idx := row * mask_w + col
					mask_value := masks.data[mask_idx]

					pixel_idx := mask_idx * 4
					if mask_value > 0 {
						mask_pixels[pixel_idx + 0] = 255
						mask_pixels[pixel_idx + 1] = 20
						mask_pixels[pixel_idx + 2] = 20
						mask_pixels[pixel_idx + 3] = 128
					} else {
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

			mask_texture = rl.LoadTextureFromImage(mask_image)

			rl.DrawTexture(texture, 0, 0, rl.WHITE)
			scale := f32(window_w) / f32(mask_w)
			rl.DrawTextureEx(mask_texture, {0, 0}, 0, scale, rl.WHITE)


			screen_x := i32(mouse_window_x * f32(window_w))
			screen_y := i32(mouse_window_y * f32(window_h))
			rl.DrawCircle(screen_x, screen_y, 5, rl.GREEN)
			rl.DrawCircleLines(screen_x, screen_y, 5, rl.WHITE)


			panel_x := i32(10)
			panel_y := i32(10)
			panel_width := i32(140)
			panel_height := i32(85)
			rl.DrawRectangle(panel_x, panel_y, panel_width, panel_height, {0, 0, 0, 180})

			// Draw text in white
			rl.DrawFPS(panel_x + 5, panel_y + 5)
			rl.DrawText(
				rl.TextFormat("IoU: %.3f", iou_pred.data[0]),
				panel_x + 5,
				panel_y + 25,
				12,
				rl.WHITE,
			)
			rl.DrawText(
				rl.TextFormat("Embedding: %.0fms", time.duration_milliseconds(embedding_time)),
				panel_x + 5,
				panel_y + 45,
				12,
				rl.WHITE,
			)
			rl.DrawText(
				rl.TextFormat("Mask: %.0fms", time.duration_milliseconds(time.since(t))),
				panel_x + 5,
				panel_y + 65,
				12,
				rl.WHITE,
			)

			free_all(context.temp_allocator)
		}

		rl.EndDrawing()
		if has_image && mask_texture.id != 0 {
			rl.UnloadTexture(mask_texture)
		}
	}
}
