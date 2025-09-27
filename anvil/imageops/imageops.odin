package imageops

import "../io"
import "../tensor"
import "core:fmt"
import "core:math"
import "core:testing"

Resize_Method :: enum {
	Nearest,
	Bilinear,
}

resize :: proc(
	image_hwc: ^tensor.Tensor($T),
	target_height, target_width: uint,
	resize_method: Resize_Method,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) where (T == f16 || T == f32 || T == f64) {
	using tensor

	out_resized := tensor_alloc(
		T,
		{target_height, target_width, image_hwc.shape[2]},
		true,
		allocator,
	)

	if resize_method == .Bilinear {
		resize_bilinear(image_hwc, out_resized, target_height, target_width)
	} else if resize_method == .Nearest {
		resize_nearest(image_hwc, out_resized, target_height, target_width)
	}

	return out_resized
}

resize_nearest :: proc(
	image_in, image_out: ^tensor.Tensor($T),
	target_height, target_width: uint,
) {
	h, w, c := image_in.shape[0], image_in.shape[1], image_in.shape[2]
	x_ratio := f32(w) / f32(target_width)
	y_ratio := f32(h) / f32(target_height)

	for out_y in 0 ..< target_height {
		for out_x in 0 ..< target_width {
			// Find nearest source pixel
			src_x := min(int(f32(out_x) * x_ratio + 0.5), int(w - 1))
			src_y := min(int(f32(out_y) * y_ratio + 0.5), int(h - 1))

			// Copy all channels
			for ch in 0 ..< c {
				src_idx := src_y * int(w * c) + src_x * int(c) + int(ch)
				out_idx := int(out_y * target_width * c + out_x * c) + int(ch)
				image_out.data[out_idx] = image_in.data[src_idx]
			}
		}
	}
}

// TODO(Aria): this might be slow
resize_bilinear :: proc(
	image_in, image_out: ^tensor.Tensor($T),
	target_height, target_width: uint,
) {
	h, w, c := image_in.shape[0], image_in.shape[1], image_in.shape[2]
	x_ratio := f32(w - 1) / f32(target_width - 1) if target_width > 1 else 0
	y_ratio := f32(h - 1) / f32(target_height - 1) if target_height > 1 else 0

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
				a := image_in.data[y_l * int(w * c) + x_l * int(c) + int(ch)]
				b := image_in.data[y_l * int(w * c) + x_h * int(c) + int(ch)]
				c_val := image_in.data[y_h * int(w * c) + x_l * int(c) + int(ch)]
				d := image_in.data[y_h * int(w * c) + x_h * int(c) + int(ch)]

				// Bilinear interpolation
				out_idx := int(out_y * target_width * c + out_x * c) + int(ch)

				image_out.data[out_idx] =
					a * T(1 - x_weight) * T(1 - y_weight) +
					b * T(x_weight) * T(1 - y_weight) +
					c_val * T(y_weight) * T(1 - x_weight) +
					d * T(x_weight * y_weight)
			}
		}
	}
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
