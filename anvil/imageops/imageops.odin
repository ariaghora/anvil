package imageops

import "../io"
import "../tensor"
import "core:fmt"
import "core:math"
import "core:testing"

Resize_Method :: enum {
	Bilinear,
}

// TODO(Aria): this might be slow
resize :: proc(
	image_hwc: ^tensor.Tensor($T),
	target_height, target_width: uint,
	resize_method: Resize_Method,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	using tensor

	h, w, c := image_hwc.shape[0], image_hwc.shape[1], image_hwc.shape[2]

	x_ratio := f32(w - 1) / f32(target_width - 1) if target_width > 1 else 0
	y_ratio := f32(h - 1) / f32(target_height - 1) if target_height > 1 else 0

	out_resized := tensor_alloc(T, {target_height, target_width, c}, true, allocator)

	for out_y in 0 ..< target_height {
		for out_x in 0 ..< target_width {
			// Source coordinates
			x_f := f32(out_x) * x_ratio
			y_f := f32(out_y) * y_ratio

			x_l := int(math.floor(x_f))
			y_l := int(math.floor(y_f))
			x_h := min(int(math.ceil(x_f)), int(w - 1))
			y_h := min(int(math.ceil(y_f)), int(h - 1))

			x_weight := x_f - f32(x_l)
			y_weight := y_f - f32(y_l)

			for ch in 0 ..< c {
				// Get the 4 corner pixels for this channel
				a := image_hwc.data[y_l * int(w * c) + x_l * int(c) + int(ch)]
				b := image_hwc.data[y_l * int(w * c) + x_h * int(c) + int(ch)]
				c_val := image_hwc.data[y_h * int(w * c) + x_l * int(c) + int(ch)]
				d := image_hwc.data[y_h * int(w * c) + x_h * int(c) + int(ch)]

				// Bilinear interpolation
				out_idx := int(out_y * target_width * c + out_x * c) + int(ch)

				when T == f16 || T == f32 || T == f64 {
					out_resized.data[out_idx] =
						a * T(1 - x_weight) * T(1 - y_weight) +
						b * T(x_weight) * T(1 - y_weight) +
						c_val * T(y_weight) * T(1 - x_weight) +
						d * T(x_weight * y_weight)
				} else {
					fmt.panicf("resize does not support %v", typeid_of(T))
				}
			}
		}
	}

	return out_resized
}

// Draw a rectangle outline on an image tensor in-place.
// Expects image in HWC format with 3 channels (RGB).
// Rectangle specified as [x, y, width, height] in pixel coordinates.
// Color values should be in range [0, 1].
draw_rectangle_line :: proc(
	image_hwc: ^tensor.Tensor($T),
	x, y, w, h: uint,
	color: [3]f32,
	line_width: uint,
) {
	ensure(len(image_hwc.shape) == 3, "Image tensor must be 3D (HWC format)")
	ensure(image_hwc.shape[2] == 3, "Image must have 3 channels (RGB)")

	height := image_hwc.shape[0]
	width := image_hwc.shape[1]
	channels := image_hwc.shape[2]

	// Clamp bounds
	x_end := min(x + w, width)
	y_end := min(y + h, height)

	// Top and bottom horizontal lines (with thickness)
	for i in x ..< x_end {
		// Top edge
		for t in 0 ..< line_width {
			if y + t < height {
				offset := ((y + t) * width + i) * channels
				for c in 0 ..< 3 {
					image_hwc.data[int(offset) + c] = T(color[c])
				}
			}
		}
		// Bottom edge
		for t in 0 ..< line_width {
			if y_end > t && y_end - t - 1 < height {
				offset := ((y_end - t - 1) * width + i) * channels
				for c in 0 ..< 3 {
					image_hwc.data[int(offset) + c] = T(color[c])
				}
			}
		}
	}

	// Left and right vertical lines (with thickness)
	for j in y ..< y_end {
		// Left edge
		for t in 0 ..< line_width {
			if x + t < width {
				offset := (j * width + x + t) * channels
				for c in 0 ..< 3 {
					image_hwc.data[int(offset) + c] = T(color[c])
				}
			}
		}
		// Right edge
		for t in 0 ..< line_width {
			if x_end > t && x_end - t - 1 < width {
				offset := (j * width + (x_end - t - 1)) * channels
				for c in 0 ..< 3 {
					image_hwc.data[int(offset) + c] = T(color[c])
				}
			}
		}
	}
}
