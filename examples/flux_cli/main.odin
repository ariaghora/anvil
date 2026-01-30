// FLUX.2-klein CLI - Text-to-Image and Image-to-Image Generation
//
// Port of antirez's flux2.c to Odin/anvil.
//
// Usage:
//   flux_cli -d <model_dir> -p "prompt" [options]
//
// Options:
//   -d, --model-dir DIR   Model directory (required)
//   -p, --prompt TEXT     Prompt text (required)
//   -o, --output PATH     Output image path (default: output.png)
//   -W, --width N         Output width (default: 256)
//   -H, --height N        Output height (default: 256)
//   -s, --steps N         Number of sampling steps (default: 4)
//   -S, --seed N          Random seed (-1 for random, default: -1)
//   -i, --input PATH      Input image for img2img (can repeat for multi-ref)
//   -g, --guidance N      Guidance scale for CFG (default: 1.0, disabled)
//   -h, --help            Show this help

package main

import flux "../../anvil/models/flux"
import "../../anvil/tensor"
import "../../anvil/trace"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import stbi "vendor:stb/image"

DEFAULT_WIDTH :: 256
DEFAULT_HEIGHT :: 256
DEFAULT_STEPS :: 4
DEFAULT_SEED :: -1
DEFAULT_OUTPUT :: "output.png"
DEFAULT_GUIDANCE :: 1.0

Args :: struct {
	model_dir:      string,
	prompt:         string,
	output_path:    string,
	width:          int,
	height:         int,
	steps:          int,
	seed:           i64,
	input_images:   [dynamic]string,
	guidance_scale: f32,
	show_help:      bool,
}

print_usage :: proc() {
	fmt.println("FLUX.2-klein CLI - Text-to-Image Generation")
	fmt.println("")
	fmt.println("Usage:")
	fmt.println("  flux_cli -d <model_dir> -p \"prompt\" [options]")
	fmt.println("")
	fmt.println("Required:")
	fmt.println("  -d, --model-dir DIR   Model directory containing weights")
	fmt.println("  -p, --prompt TEXT     Text prompt for generation")
	fmt.println("")
	fmt.println("Options:")
	fmt.println("  -o, --output PATH     Output image path (default: output.png)")
	fmt.println("  -W, --width N         Output width in pixels (default: 256)")
	fmt.println("  -H, --height N        Output height in pixels (default: 256)")
	fmt.println("  -s, --steps N         Number of sampling steps (default: 4)")
	fmt.println("  -S, --seed N          Random seed, -1 for random (default: -1)")
	fmt.println("  -i, --input PATH      Input image for img2img (can repeat)")
	fmt.println("  -g, --guidance N      CFG guidance scale (default: 1.0)")
	fmt.println("  -h, --help            Show this help message")
	fmt.println("")
	fmt.println("Examples:")
	fmt.println("  # Text-to-image")
	fmt.println("  flux_cli -d flux-klein-model -p \"A fluffy cat sitting on a windowsill\"")
	fmt.println("")
	fmt.println("  # Higher resolution")
	fmt.println("  flux_cli -d flux-klein-model -p \"Mountain landscape\" -W 512 -H 512 -s 8")
	fmt.println("")
	fmt.println("  # Image-to-image")
	fmt.println("  flux_cli -d flux-klein-model -p \"oil painting style\" -i photo.png")
	fmt.println("")
	fmt.println("  # With seed for reproducibility")
	fmt.println("  flux_cli -d flux-klein-model -p \"A robot\" -S 42 -o robot.png")
}

parse_args :: proc() -> (args: Args, ok: bool) {
	args.output_path = DEFAULT_OUTPUT
	args.width = DEFAULT_WIDTH
	args.height = DEFAULT_HEIGHT
	args.steps = DEFAULT_STEPS
	args.seed = DEFAULT_SEED
	args.guidance_scale = DEFAULT_GUIDANCE
	args.input_images = make([dynamic]string)

	os_args := os.args
	if len(os_args) < 2 {
		print_usage()
		return args, false
	}

	i := 1
	for i < len(os_args) {
		arg := os_args[i]

		switch arg {
		case "-h", "--help":
			args.show_help = true
			return args, true

		case "-d", "--model-dir":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -d/--model-dir requires an argument")
				return args, false
			}
			args.model_dir = os_args[i + 1]
			i += 2

		case "-p", "--prompt":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -p/--prompt requires an argument")
				return args, false
			}
			args.prompt = os_args[i + 1]
			i += 2

		case "-o", "--output":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -o/--output requires an argument")
				return args, false
			}
			args.output_path = os_args[i + 1]
			i += 2

		case "-W", "--width":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -W/--width requires an argument")
				return args, false
			}
			val, parse_ok := strconv.parse_int(os_args[i + 1])
			if !parse_ok {
				fmt.eprintfln("Error: invalid width value '%s'", os_args[i + 1])
				return args, false
			}
			args.width = val
			i += 2

		case "-H", "--height":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -H/--height requires an argument")
				return args, false
			}
			val, parse_ok := strconv.parse_int(os_args[i + 1])
			if !parse_ok {
				fmt.eprintfln("Error: invalid height value '%s'", os_args[i + 1])
				return args, false
			}
			args.height = val
			i += 2

		case "-s", "--steps":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -s/--steps requires an argument")
				return args, false
			}
			val, parse_ok := strconv.parse_int(os_args[i + 1])
			if !parse_ok {
				fmt.eprintfln("Error: invalid steps value '%s'", os_args[i + 1])
				return args, false
			}
			args.steps = val
			i += 2

		case "-S", "--seed":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -S/--seed requires an argument")
				return args, false
			}
			val, parse_ok := strconv.parse_i64(os_args[i + 1])
			if !parse_ok {
				fmt.eprintfln("Error: invalid seed value '%s'", os_args[i + 1])
				return args, false
			}
			args.seed = val
			i += 2

		case "-i", "--input":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -i/--input requires an argument")
				return args, false
			}
			append(&args.input_images, os_args[i + 1])
			i += 2

		case "-g", "--guidance":
			if i + 1 >= len(os_args) {
				fmt.eprintln("Error: -g/--guidance requires an argument")
				return args, false
			}
			val, parse_ok := strconv.parse_f32(os_args[i + 1])
			if !parse_ok {
				fmt.eprintfln("Error: invalid guidance value '%s'", os_args[i + 1])
				return args, false
			}
			args.guidance_scale = val
			i += 2

		case:
			fmt.eprintfln("Error: unknown argument '%s'", arg)
			fmt.eprintln("Use -h or --help for usage information")
			return args, false
		}
	}

	// Validate required arguments
	if args.model_dir == "" {
		fmt.eprintln("Error: -d/--model-dir is required")
		return args, false
	}
	if args.prompt == "" {
		fmt.eprintln("Error: -p/--prompt is required")
		return args, false
	}

	// Validate dimensions (must be multiples of 16 for VAE)
	if args.width % 16 != 0 {
		fmt.eprintfln("Warning: width %d is not a multiple of 16, rounding down", args.width)
		args.width = (args.width / 16) * 16
	}
	if args.height % 16 != 0 {
		fmt.eprintfln("Warning: height %d is not a multiple of 16, rounding down", args.height)
		args.height = (args.height / 16) * 16
	}
	if args.width < 64 || args.height < 64 {
		fmt.eprintln("Error: minimum dimension is 64 pixels")
		return args, false
	}

	return args, true
}

// Load image using stb_image
load_image :: proc(path: string, allocator := context.allocator) -> (img: ^flux.Image, ok: bool) {
	path_cstr := strings.clone_to_cstring(path, context.temp_allocator)
	w, h, channels: i32
	img_data := stbi.load(path_cstr, &w, &h, &channels, 3) // Force RGB
	if img_data == nil {
		fmt.eprintfln("Error: failed to load image '%s'", path)
		return nil, false
	}

	img = new(flux.Image, allocator)
	img.width = int(w)
	img.height = int(h)
	img.channels = 3
	img.data = make([]u8, int(w * h * 3), allocator)
	copy(img.data, img_data[:w * h * 3])

	stbi.image_free(img_data)
	return img, true
}

// Save image as PNG using stb_image_write
save_image_png :: proc(img: ^flux.Image, path: string) -> bool {
	path_cstr := strings.clone_to_cstring(path, context.temp_allocator)
	result := stbi.write_png(
		path_cstr,
		i32(img.width),
		i32(img.height),
		i32(img.channels),
		raw_data(img.data),
		i32(img.width * img.channels),
	)
	return result != 0
}

main :: proc() {
	// Initialize tracer
	trace.global_init()
	defer {
		trace_json := trace.global_finish()
		if len(trace_json) > 0 {
			os.write_entire_file("trace.json", transmute([]u8)trace_json)
			fmt.println("Trace saved to: trace.json")
		}
		trace.global_destroy()
	}

	// Parse arguments
	args, args_ok := parse_args()
	if !args_ok {
		os.exit(1)
	}
	defer delete(args.input_images)

	if args.show_help {
		print_usage()
		return
	}

	allocator := context.allocator

	// Print configuration
	fmt.println("=== FLUX.2-klein Configuration ===")
	fmt.printfln("Model directory: %s", args.model_dir)
	fmt.printfln("Prompt: \"%s\"", args.prompt)
	fmt.printfln("Output: %s", args.output_path)
	fmt.printfln("Dimensions: %dx%d", args.width, args.height)
	fmt.printfln("Steps: %d", args.steps)
	fmt.printfln("Seed: %d", args.seed)
	if len(args.input_images) > 0 {
		fmt.printfln("Input images: %d", len(args.input_images))
		for path in args.input_images {
			fmt.printfln("  - %s", path)
		}
	}
	fmt.println("")

	// Load model
	fmt.println("Loading FLUX model...")
	t_load := time.now()

	load_trace := trace.global_scoped("model_loading", "init")
	flux_model, load_err := flux.new_flux(f32, args.model_dir, allocator)
	trace.global_end_scoped(load_trace)

	if load_err != "" {
		fmt.eprintfln("Error loading model: %s", load_err)
		os.exit(1)
	}
	defer flux.free_flux(flux_model, allocator)

	fmt.printfln("Model loaded in %.2fs", time.duration_seconds(time.since(t_load)))

	// Set up generation parameters
	params := flux.Gen_Params {
		width     = args.width,
		height    = args.height,
		num_steps = args.steps,
		seed      = args.seed,
	}

	// Generate image
	fmt.println("Generating image...")
	t_gen := time.now()

	result_img: ^flux.Image
	gen_err: string

	if len(args.input_images) == 0 {
		// Text-to-image
		gen_trace := trace.global_scoped("text_to_image", "inference")
		result_img, gen_err = flux.generate(flux_model, args.prompt, params, allocator)
		trace.global_end_scoped(gen_trace)
	} else {
		// Image-to-image
		input_img, img_ok := load_image(args.input_images[0], allocator)
		if !img_ok {
			os.exit(1)
		}
		defer flux.free_image(input_img, allocator)

		gen_trace := trace.global_scoped("image_to_image", "inference")
		result_img, gen_err = flux.img2img(flux_model, args.prompt, input_img, params, allocator)
		trace.global_end_scoped(gen_trace)
	}

	if gen_err != "" {
		fmt.eprintfln("Error generating image: %s", gen_err)
		os.exit(1)
	}
	defer flux.free_image(result_img, allocator)

	fmt.printfln("Image generated in %.2fs", time.duration_seconds(time.since(t_gen)))

	// Save output
	fmt.printfln("Saving to %s...", args.output_path)
	if !save_image_png(result_img, args.output_path) {
		fmt.eprintfln("Error: failed to save image to '%s'", args.output_path)
		os.exit(1)
	}

	fmt.println("Done!")
}
