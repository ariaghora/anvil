package file_io

import "../tensor"
import "core:os"
import "vendor:stb/image"

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
	data, ok := os.read_entire_file(file_name, allocator)
	if !ok do return nil, Cannot_Read_File{}
	defer delete(data, allocator)

	width, height, channels_in_file: i32
	desired_channels: i32 = 0 // 0 = use image's native channel count

	pixels := image.load_from_memory(
		raw_data(data),
		i32(len(data)),
		&width,
		&height,
		&channels_in_file,
		desired_channels,
	)
	if pixels == nil do return nil, Invalid_Image_Format{}
	defer image.image_free(pixels)

	// Actual channels we got (if desired_channels was 0, this equals channels_in_file)
	channels := channels_in_file if desired_channels == 0 else desired_channels
	res = tensor.tensor_alloc(T, {uint(height), uint(width), uint(channels)}, true, allocator, loc)

	// Copy and normalize pixels
	// stb returns data in row-major order: [height][width][channels]
	for i in 0 ..< int(height * width * channels) {
		res.data[i] = T(pixels[i]) / 255.0
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
