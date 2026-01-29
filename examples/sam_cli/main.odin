package main

import sam "../../anvil/models/sam"
import vit "../../anvil/models/sam/vit"
import st "../../anvil/safetensors"
import "../../anvil/tensor"
import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import stbi "vendor:stb/image"

IMAGE_SIZE :: uint(1024)
DEFAULT_MODEL_PATH :: "weights/mobile_sam-tiny-vitt.safetensors"

Point :: struct {
	x, y:        f32,
	is_positive: bool,
}

Args :: struct {
	image_path:  string,
	output_path: string,
	model_path:  string,
	points:      [dynamic]Point,
}

parse_point :: proc(s: string) -> (x: f32, y: f32, ok: bool) {
	parts := strings.split(s, ",")
	defer delete(parts)
	if len(parts) != 2 do return 0, 0, false

	x_int := strconv.parse_int(strings.trim_space(parts[0])) or_return
	y_int := strconv.parse_int(strings.trim_space(parts[1])) or_return

	return f32(x_int), f32(y_int), true
}

parse_args :: proc() -> (args: Args, ok: bool) {
	os_args := os.args
	if len(os_args) < 2 {
		print_usage()
		return args, false
	}

	args.image_path = os_args[1]
	args.output_path = "mask.png"
	args.model_path = DEFAULT_MODEL_PATH
	args.points = make([dynamic]Point)

	i := 2
	for i < len(os_args) {
		arg := os_args[i]
		switch arg {
		case "--point":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: --point requires an argument")
				return args, false
			}
			x, y, point_ok := parse_point(os_args[i + 1])
			if !point_ok {
				fmt.eprintln("Error: invalid point format, expected x,y")
				return args, false
			}
			append(&args.points, Point{x, y, true})
			i += 2
		case "--neg-point":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: --neg-point requires an argument")
				return args, false
			}
			x, y, point_ok := parse_point(os_args[i + 1])
			if !point_ok {
				fmt.eprintln("Error: invalid point format, expected x,y")
				return args, false
			}
			append(&args.points, Point{x, y, false})
			i += 2
		case "--output":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: --output requires an argument")
				return args, false
			}
			args.output_path = os_args[i + 1]
			i += 2
		case "--model":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: --model requires an argument")
				return args, false
			}
			args.model_path = os_args[i + 1]
			i += 2
		case:
			fmt.eprintfln("Error: unknown argument '%s'", arg)
			return args, false
		}
	}

	if len(args.points) == 0 {
		fmt.eprintln("Error: at least one --point or --neg-point is required")
		return args, false
	}

	return args, true
}

print_usage :: proc() {
	fmt.eprintln("Usage: sam_cli <image_path> [options]")
	fmt.eprintln("")
	fmt.eprintln("Options:")
	fmt.eprintln("  --point <x,y>       Add a foreground point (can repeat)")
	fmt.eprintln("  --neg-point <x,y>   Add a background point (can repeat)")
	fmt.eprintln("  --output <path>     Output mask path (default: mask.png)")
	fmt.eprintln("  --model <path>      Model path (default: weights/mobile_sam-tiny-vitt.safetensors)")
	fmt.eprintln("")
	fmt.eprintln("Examples:")
	fmt.eprintln("  sam_cli photo.jpg --point 512,384")
	fmt.eprintln("  sam_cli photo.jpg --point 100,200 --point 150,250 --point 200,300")
	fmt.eprintln("  sam_cli photo.jpg --point 100,200 --neg-point 300,300")
	fmt.eprintln("  sam_cli photo.jpg --point 256,256 --output result.png")
}

load_image :: proc(path: string) -> (data: []u8, width: int, height: int, ok: bool) {
	path_cstr := strings.clone_to_cstring(path, context.temp_allocator)
	w, h, channels: i32
	img_data := stbi.load(path_cstr, &w, &h, &channels, 4) // Force RGBA
	if img_data == nil {
		fmt.eprintfln("Error: failed to load image '%s'", path)
		return nil, 0, 0, false
	}
	width = int(w)
	height = int(h)
	data = img_data[:width * height * 4]
	return data, width, height, true
}

preprocess :: proc(
	image_data: []u8,
	width, height: int,
	target_size: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(f32) {
	means := [3]f32{123.675, 116.28, 103.53}
	std := [3]f32{58.395, 57.12, 57.375}

	w_out, h_out: uint
	if width > height {
		w_out = target_size
		h_out = uint(f32(target_size) * f32(height) / f32(width))
	} else {
		h_out = target_size
		w_out = uint(f32(target_size) * f32(width) / f32(height))
	}

	// Resize using simple bilinear-ish sampling
	image_chw := make([]f32, 3 * w_out * h_out, context.temp_allocator)
	for row in 0 ..< h_out {
		for col in 0 ..< w_out {
			// Map to source coordinates
			src_x := f32(col) * f32(width) / f32(w_out)
			src_y := f32(row) * f32(height) / f32(h_out)
			src_col := min(uint(src_x), uint(width - 1))
			src_row := min(uint(src_y), uint(height - 1))

			src_idx := (src_row * uint(width) + src_col) * 4

			r_idx := 0 * h_out * w_out + row * w_out + col
			g_idx := 1 * h_out * w_out + row * w_out + col
			b_idx := 2 * h_out * w_out + row * w_out + col

			image_chw[r_idx] = (f32(image_data[src_idx + 0]) - means[0]) / std[0]
			image_chw[g_idx] = (f32(image_data[src_idx + 1]) - means[1]) / std[1]
			image_chw[b_idx] = (f32(image_data[src_idx + 2]) - means[2]) / std[2]
		}
	}

	// Pad to target_size x target_size
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

	return tensor.new_with_init(image_chw_padded, {1, 3, target_size, target_size}, allocator)
}

save_mask_png :: proc(mask: ^tensor.Tensor(f32), path: string, allocator := context.allocator) -> bool {
	height := mask.shape[2]
	width := mask.shape[3]

	// Create grayscale image (white = foreground, black = background)
	pixels := make([]u8, width * height, context.temp_allocator)
	for row in 0 ..< height {
		for col in 0 ..< width {
			idx := row * width + col
			if mask.data[idx] > 0 {
				pixels[idx] = 255
			} else {
				pixels[idx] = 0
			}
		}
	}

	path_cstr := strings.clone_to_cstring(path, context.temp_allocator)
	result := stbi.write_png(path_cstr, i32(width), i32(height), 1, raw_data(pixels), i32(width))
	return result != 0
}

main :: proc() {
	args, args_ok := parse_args()
	if !args_ok {
		os.exit(1)
	}
	defer delete(args.points)

	// Setup arena allocator
	arena: vmem.Arena
	assert(vmem.arena_init_growing(&arena) == nil)
	allocator := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	// Load image
	fmt.printfln("Loading image: %s", args.image_path)
	image_data, width, height, img_ok := load_image(args.image_path)
	if !img_ok {
		os.exit(1)
	}
	defer stbi.image_free(raw_data(image_data))

	fmt.printfln("Image size: %dx%d", width, height)

	// Load model
	fmt.printfln("Loading model: %s", args.model_path)
	t_load := time.now()
	safetensors, st_err := st.read_from_file(f32, args.model_path, allocator)
	if st_err != nil {
		fmt.eprintfln("Error: failed to load model: %v", st_err)
		os.exit(1)
	}
	sam_model := sam.new_tiny(f32, safetensors, allocator)
	encoder := sam_model.image_encoder.(^vit.Tiny_ViT_5m(f32))
	fmt.printfln("Model loaded in %.2fs", time.duration_seconds(time.since(t_load)))

	// Preprocess image
	input := preprocess(image_data, width, height, IMAGE_SIZE, allocator)

	// Compute image embedding
	fmt.println("Computing image embedding...")
	t_embed := time.now()
	embedding := vit.forward_tiny_vit_5m(encoder, input, allocator)
	fmt.printfln("Embedding computed in %.2fs", time.duration_seconds(time.since(t_embed)))

	// Convert points to normalized coordinates and SAM format
	sam_points := make([]sam.Point(f32), len(args.points), context.temp_allocator)
	for p, i in args.points {
		sam_points[i] = sam.Point(f32){
			p.x / f32(width),
			p.y / f32(height),
			p.is_positive,
		}
		label := "foreground" if p.is_positive else "background"
		fmt.printfln("Point %d: (%v, %v) [%s]", i + 1, int(p.x), int(p.y), label)
	}

	// Run mask decoder
	fmt.println("Generating mask...")
	t_mask := time.now()
	masks, iou_pred := sam.forward_sam_for_embedding(
		sam_model,
		embedding,
		input.shape[2],
		input.shape[3],
		sam_points,
		context.temp_allocator,
	)
	fmt.printfln("Mask generated in %.3fs", time.duration_seconds(time.since(t_mask)))

	// Get best mask (highest IoU)
	best_idx := 0
	best_iou := iou_pred.data[0]
	for i in 1 ..< len(iou_pred.data) {
		if iou_pred.data[i] > best_iou {
			best_iou = iou_pred.data[i]
			best_idx = i
		}
	}

	// Extract best mask (masks shape is [1, num_masks, H, W])
	mask_h := masks.shape[2]
	mask_w := masks.shape[3]
	best_mask_data := make([]f32, mask_h * mask_w, context.temp_allocator)
	mask_offset := uint(best_idx) * mask_h * mask_w
	copy(best_mask_data, masks.data[mask_offset:mask_offset + mask_h * mask_w])
	best_mask := tensor.new_with_init(best_mask_data, {1, 1, mask_h, mask_w}, context.temp_allocator)

	// Save mask
	if save_mask_png(best_mask, args.output_path) {
		fmt.printfln("Mask saved to: %s", args.output_path)
	} else {
		fmt.eprintfln("Error: failed to save mask to '%s'", args.output_path)
		os.exit(1)
	}

	// Print IoU score
	fmt.printfln("IoU score: %.4f", best_iou)
}
