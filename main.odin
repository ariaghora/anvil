package main

import tf "anvil/models/sam"
import md "anvil/models/sam/mask_decoder"
import pe "anvil/models/sam/prompt_encoder"
import "anvil/models/sam/vit"
import "anvil/nn"
import st "anvil/safetensors"
import "anvil/tensor"
import "anvil/trace"
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
import rl "vendor:raylib"

IMAGE_SIZE :: uint(1024)
MODEL_PATH :: "weights/mobile_sam-tiny-vitt.safetensors"

App_State :: struct {
	sam:             ^tf.Sam(f32),
	image_encoder:   ^vit.Tiny_ViT_5m(f32),
	current_image:   rl.Image,
	texture:         rl.Texture2D,
	input:           ^tensor.Tensor(f32),
	image_embedding: ^tensor.Tensor(f32),
	embedding_time:  time.Duration,
	has_image:       bool,
	mask_texture:    rl.Texture2D,
}

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

load_model :: proc(
	allocator := context.allocator,
) -> (
	sam: ^tf.Sam(f32),
	encoder: ^vit.Tiny_ViT_5m(f32),
) {
	t := time.now()
	safetensors, err := st.read_from_file(f32, MODEL_PATH, allocator)
	assert(err == nil)
	sam = tf.new_tiny(f32, safetensors, allocator)
	encoder = sam.image_encoder.(^vit.Tiny_ViT_5m(f32))
	fmt.println("Model loading time:", time.since(t))
	return
}

handle_dropped_image :: proc(state: ^App_State, path: cstring, allocator := context.allocator) {
	if state.has_image {
		rl.UnloadTexture(state.texture)
		rl.UnloadImage(state.current_image)
		tensor.free_tensor(state.input, allocator)
		tensor.free_tensor(state.image_embedding, allocator)
	}

	state.current_image = rl.LoadImage(path)

	window_w, window_h := calculate_window_size(state.current_image)
	rl.SetWindowSize(window_w, window_h)
	center_window(window_w, window_h)

	state.input = preprocess(f32, &state.current_image, 1024, allocator)

	t := time.now()
	state.image_embedding = vit.forward_tiny_vit_5m(state.image_encoder, state.input, allocator)
	state.embedding_time = time.since(t)

	state.texture = rl.LoadTextureFromImage(state.current_image)
	state.has_image = true
}

calculate_window_size :: proc(image: rl.Image) -> (i32, i32) {
	if image.width > image.height {
		w := min(image.width, 1024)
		h := i32(f32(w) * f32(image.height) / f32(image.width))
		return w, h
	} else {
		h := min(image.height, 1024)
		w := i32(f32(h) * f32(image.width) / f32(image.height))
		return w, h
	}
}

center_window :: proc(window_w, window_h: i32) {
	monitor := rl.GetCurrentMonitor()
	monitor_width := rl.GetMonitorWidth(monitor)
	monitor_height := rl.GetMonitorHeight(monitor)
	window_x := (monitor_width - window_w) / 2
	window_y := (monitor_height - window_h) / 2
	rl.SetWindowPosition(window_x, window_y)
}

get_normalized_mouse :: proc(image: rl.Image) -> (f32, f32) {
	window_w := f32(rl.GetScreenWidth())
	window_h := f32(rl.GetScreenHeight())

	mouse_window_x := f32(rl.GetMouseX()) / window_w
	mouse_window_y := f32(rl.GetMouseY()) / window_h

	if image.width > image.height {
		return mouse_window_x, mouse_window_y * (f32(image.height) / f32(image.width))
	} else {
		return mouse_window_x * (f32(image.width) / f32(image.height)), mouse_window_y
	}
}

create_mask_texture :: proc(masks: ^tensor.Tensor(f32)) -> rl.Texture2D {
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
			}
		}
	}

	mask_image: rl.Image
	mask_image.data = raw_data(mask_pixels)
	mask_image.width = i32(mask_w)
	mask_image.height = i32(mask_h)
	mask_image.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8A8
	mask_image.mipmaps = 1

	return rl.LoadTextureFromImage(mask_image)
}

draw_info_panel :: proc(iou: f32, embedding_time: time.Duration, mask_time: time.Duration) {
	panel_x := i32(10)
	panel_y := i32(10)
	panel_width := i32(140)
	panel_height := i32(85)

	rl.DrawRectangle(panel_x, panel_y, panel_width, panel_height, {0, 0, 0, 180})

	rl.DrawFPS(panel_x + 5, panel_y + 5)
	rl.DrawText(rl.TextFormat("IoU: %.3f", iou), panel_x + 5, panel_y + 25, 12, rl.WHITE)
	rl.DrawText(
		rl.TextFormat("Embedding: %.0fms", time.duration_milliseconds(embedding_time)),
		panel_x + 5,
		panel_y + 45,
		12,
		rl.WHITE,
	)
	rl.DrawText(
		rl.TextFormat("Mask: %.0fms", time.duration_milliseconds(mask_time)),
		panel_x + 5,
		panel_y + 65,
		12,
		rl.WHITE,
	)
}

draw_drop_zone :: proc() {
	text := strings.clone_to_cstring("Drop an image here", context.temp_allocator)
	text_width := rl.MeasureText(text, 30)
	rl.DrawText(
		text,
		rl.GetScreenWidth() / 2 - text_width / 2,
		rl.GetScreenHeight() / 2 - 15,
		30,
		rl.LIGHTGRAY,
	)
}

main :: proc() {
	rl.SetTraceLogLevel(.ERROR)
	rl.InitWindow(800, 600, "Segment Anything - Drop an image to start")
	defer rl.CloseWindow()
	rl.SetTargetFPS(60)

	arena: vmem.Arena
	assert(vmem.arena_init_growing(&arena) == nil)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	state: App_State
	state.sam, state.image_encoder = load_model(arena_alloc)
	defer tf.free_tiny(state.sam, arena_alloc)

	for !rl.WindowShouldClose() {
		if rl.IsFileDropped() {
			dropped_files := rl.LoadDroppedFiles()
			defer rl.UnloadDroppedFiles(dropped_files)

			if dropped_files.count > 0 {
				handle_dropped_image(&state, dropped_files.paths[0], arena_alloc)
			}
		}

		rl.BeginDrawing()
		rl.ClearBackground(rl.DARKGRAY)

		if !state.has_image {
			draw_drop_zone()
		} else {
			mouse_x, mouse_y := get_normalized_mouse(state.current_image)
			points := []tf.Point(f32){{mouse_x, mouse_y, true}}

			t := time.now()
			masks, iou_pred := tf.forward_sam_for_embedding(
				state.sam,
				state.image_embedding,
				state.input.shape[2],
				state.input.shape[3],
				points,
				context.temp_allocator,
			)
			mask_time := time.since(t)

			state.mask_texture = create_mask_texture(masks)

			rl.DrawTexture(state.texture, 0, 0, rl.WHITE)

			scale := f32(rl.GetScreenWidth()) / f32(masks.shape[3])
			rl.DrawTextureEx(state.mask_texture, {0, 0}, 0, scale, rl.WHITE)

			screen_x := i32(f32(rl.GetMouseX()))
			screen_y := i32(f32(rl.GetMouseY()))
			rl.DrawCircle(screen_x, screen_y, 5, rl.GREEN)
			rl.DrawCircleLines(screen_x, screen_y, 5, rl.WHITE)

			draw_info_panel(iou_pred.data[0], state.embedding_time, mask_time)

			free_all(context.temp_allocator)
		}

		rl.EndDrawing()

		if state.has_image && state.mask_texture.id != 0 {
			rl.UnloadTexture(state.mask_texture)
		}
	}
}
