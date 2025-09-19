package file_io

import "../tensor"
import "core:image"
import "core:image/bmp"
import "core:image/netpbm"
import "core:image/png"
import "core:image/qoi"

// Read image from file as a 3D tensor. The output shape will be [height, width, channel].
// The purpose of channel dimension as the innermost is for convenience, e.g., normalization.
// Resulting tensor will be floating number familty ranging from 0 (dark) to 1 (light).
read_image_from_file :: proc(
	$T: typeid,
	file_name: string,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^tensor.Tensor(T),
	err: IO_Error,
) where (T == f16 || T == f32 || T == f64) {
	img := image.load_from_file(file_name, {}, allocator) or_return
	defer image.destroy(img, allocator)

	height, width, channels := uint(img.height), uint(img.width), uint(img.channels)

	res = tensor.tensor_alloc(T, {height, width, channels}, true, allocator, loc)
	for h in 0 ..< height {
		for w in 0 ..< width {
			for c in 0 ..< channels {
				pixel_idx := (h * width + w) * channels + c
				tensor_idx := h * width * channels + w * channels + c
				res.data[tensor_idx] = T(img.pixels.buf[pixel_idx]) / 255.0
			}
		}
	}

	return res, nil
}

import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
read_image_from_file_test :: proc(t: ^testing.T) {
	img, err := read_image_from_file(f32, "assets/test_images/mandrill.png")
	testing.expect(t, err == nil, fmt.tprint(err))
	defer tensor.free_tensor(img)
	testing.expect(t, slice.equal(img.shape, []uint{288, 288, 3}))
}
