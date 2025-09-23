package file_io

import "../tensor"
import "core:os"
import "core:strings"
import "vendor:stb/image"

// Read image from file as a 3D tensor. The output shape will be [height, width, channel].
// The purpose of channel dimension as the innermost is for convenience, e.g., normalization.
// Resulting tensor will be floating number familty ranging from 0 (dark) to 1 (light).
read_image_from_bytes :: proc(
	$T: typeid,
	data: []byte,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^tensor.Tensor(T),
	err: IO_Error,
) where (T == f16 || T == f32 || T == f64) {
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
	if pixels == nil do return nil, Image_Invalid_Format{}
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
	if !ok do return nil, Image_Cannot_Read_File{}
	defer delete(data, allocator)
	return read_image_from_bytes(T, data, allocator, loc)
}

write_image :: proc(
	image_hwc: ^tensor.Tensor($T),
	out_path: string,
	loc := #caller_location,
) -> IO_Error {
	ensure(len(image_hwc.shape) == 3, "expects image tensor to be 3D", loc = loc)
	out_path_norm := strings.to_lower(out_path, context.temp_allocator)

	out_type: string
	if strings.ends_with(out_path_norm, ".png") {
		out_type = "png"
	} else if (strings.ends_with(out_path_norm, ".jpg") ||
		   strings.ends_with(out_path_norm, ".jpeg")) {
		out_type = "jpg"
	} else {
		return Image_Unsupported_Output_Extension{}
	}

	height := i32(image_hwc.shape[0])
	width := i32(image_hwc.shape[1])
	channels := i32(image_hwc.shape[2])

	num_pixels := int(height * width * channels)
	pixels := make([]u8, num_pixels, context.temp_allocator)
	for i in 0 ..< num_pixels {
		val := clamp(image_hwc.data[i], 0, 1) * 255.0
		pixels[i] = u8(val)
	}

	out_path_c := strings.clone_to_cstring(out_path, context.temp_allocator)
	success: i32
	if out_type == "png" {
		success = image.write_png(out_path_c, width, height, channels, raw_data(pixels), 0)
	} else if out_type == "jpg" {
		// For JPEG, quality parameter (1-100, higher = better)
		jpeg_quality: i32 = 95 // TODO(Aria): Parameterize this
		success = image.write_jpg(
			out_path_c,
			width,
			height,
			channels,
			raw_data(pixels),
			jpeg_quality,
		)
	} else do panic("should be unreachable")

	if success == 0 do return Image_Write_Failed{}

	return nil
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
