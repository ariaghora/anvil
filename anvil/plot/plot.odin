package plot
import "../tensor"
import "core:fmt"
import "core:math"
import "core:slice"
import "core:strings"
import rl "vendor:raylib"

// Input tensor should be in image-like format. It can be either 2D or 3D tensors.
// If 3D, assume tensor is in channel x height x width format. If the tensor is 3D,
// then it is expected to have either 1 channel (grayscale) or 3 channels (RGB).
// No matter what value is, the pixel values will be normalized to 0-255 at render time.
visualize_tensor :: proc(
	tensor_chw: ^tensor.Tensor($T),
	window_title: string,
	max_size: uint = 400,
) {
	ensure(
		len(tensor_chw.shape) == 3 || len(tensor_chw.shape) == 2,
		fmt.tprintf("input tensor must have 3 or 2 dimensions, got %d", len(tensor_chw.shape)),
	)

	C, H, W: uint
	if len(tensor_chw.shape) == 3 {
		C, H, W = tensor_chw.shape[0], tensor_chw.shape[1], tensor_chw.shape[2]
		ensure(
			C == 1 || C == 3,
			fmt.tprintf("3D input tensor must have 1 or 3 color channels, got %d", C),
		)
	} else {
		C, H, W = 1, tensor_chw.shape[0], tensor_chw.shape[1]
	}

	pixels_raw: rawptr
	pixels_gray: []u8
	pixels_rgb: [][3]u8
	if C == 1 {
		pixels_gray = make([]u8, H * W, context.temp_allocator)
	} else {
		pixels_rgb = make([][3]u8, H * W, context.temp_allocator)
	}

	vmin, vmax, _ := slice.min_max(tensor_chw.data)
	vrange := vmax - vmin

	i := 0
	for h in 0 ..< H {
		for w in 0 ..< W {
			if C == 1 {
				pixels_gray[i] = u8(
					((tensor.tensor_get(tensor_chw, 0, h, w) - vmin) / vrange) * 255,
				)
			} else {
				r := u8(((tensor.tensor_get(tensor_chw, 0, h, w) - vmin) / vrange) * 255)
				g := u8(((tensor.tensor_get(tensor_chw, 1, h, w) - vmin) / vrange) * 255)
				b := u8(((tensor.tensor_get(tensor_chw, 2, h, w) - vmin) / vrange) * 255)
				pixels_rgb[i] = {r, g, b}
			}
			i += 1
		}
	}
	image := rl.Image {
		data    = C == 1 ? raw_data(pixels_gray) : raw_data(pixels_rgb),
		width   = i32(tensor_chw.shape[2]),
		height  = i32(tensor_chw.shape[1]),
		mipmaps = 1,
		format  = C == 1 ? .UNCOMPRESSED_GRAYSCALE : .UNCOMPRESSED_R8G8B8,
	}

	window_width := f32(W) / f32(W + H) * f32(max_size) * 2
	window_height := f32(H) / f32(W + H) * f32(max_size) * 2

	rl.SetTraceLogLevel(.NONE)
	rl.InitWindow(
		i32(window_width),
		i32(window_height),
		strings.clone_to_cstring(window_title, context.temp_allocator),
	)
	defer rl.CloseWindow()

	tex := rl.LoadTextureFromImage(image)

	rl.SetTargetFPS(30)
	for !rl.WindowShouldClose() {
		rl.BeginDrawing()
		rl.ClearBackground(rl.BLACK)
		rl.DrawTexturePro(
			tex,
			{0, 0, f32(tex.width), f32(tex.height)},
			{0, 0, window_width, window_height},
			{0, 0},
			0,
			rl.WHITE,
		)
		rl.EndDrawing()
	}
}
