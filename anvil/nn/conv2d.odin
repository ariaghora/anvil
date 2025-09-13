package nn

import "../tensor"
import "../trace"
import "core:fmt"
import "core:simd"

Conv_2d :: struct($T: typeid) {
	w:            ^tensor.Tensor(T),
	b:            Maybe(^tensor.Tensor(T)),
	in_channels:  uint,
	out_channels: uint,
	kernel_size:  [2]uint, // [height, width]
	stride:       uint,
	padding:      uint,
	dilation:     uint,
	groups:       uint,
}

new_conv2d :: proc(
	$T: typeid,
	in_channels, out_channels: uint,
	kernel_size: [2]uint,
	stride := uint(1),
	padding := uint(0),
	dilation := uint(1),
	groups := uint(1),
	use_bias := true,
	init := true,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Conv_2d(T) {
	if in_channels % groups != 0 {
		panic("in_channels must be divisible by groups")
	}
	if out_channels % groups != 0 {
		panic("out_channels must be divisible by groups")
	}

	in_channels_per_group := in_channels / groups
	w_shape := []uint{out_channels, in_channels_per_group, kernel_size[0], kernel_size[1]}

	w: ^tensor.Tensor(T)
	if init {
		// Xavier/Glorot normal
		fan_in := T(in_channels_per_group * kernel_size[0] * kernel_size[1])
		fan_out := T(out_channels * kernel_size[0] * kernel_size[1])
		std := T(2.0 / (fan_in + fan_out))
		w = tensor.randn(T, w_shape, T(0), std, allocator, loc)
	} else {
		w = tensor.tensor_alloc(T, w_shape, true, allocator, loc)
	}

	// Create bias tensor if requested
	b: Maybe(^tensor.Tensor(T)) = nil
	if use_bias {
		b = tensor.zeros(T, []uint{out_channels}, allocator)
	}

	return new_clone(
		Conv_2d(T) {
			w = w,
			b = b,
			in_channels = in_channels,
			out_channels = out_channels,
			kernel_size = kernel_size,
			stride = stride,
			padding = padding,
			dilation = dilation,
			groups = groups,
		},
		allocator,
	)
}

free_conv2d :: proc(conv: ^Conv_2d($T), allocator := context.allocator) {
	tensor.free_tensor(conv.w, allocator)
	if bias, has_bias := conv.b.?; has_bias {
		tensor.free_tensor(bias, allocator)
	}
	free(conv, allocator)
}

forward_conv2d :: proc(
	conv: ^Conv_2d($T),
	x: ^tensor.Tensor(T), // Input: (batch_size, in_channels, height, width)
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Validate input shape
	if len(x.shape) != 4 {
		panic("Input must be 4D tensor (batch_size, in_channels, height, width)")
	}

	if x.shape[1] != conv.in_channels {
		fmt.panicf(
			"Tensor channels (%d) and weight input channels (%d) mismatch",
			x.shape[1],
			conv.in_channels,
		)
	}

	out := tensor.conv2d_xwb(
		x,
		conv.w,
		conv.b,
		stride = conv.stride,
		dilation = conv.dilation,
		padding = conv.padding,
		groups = conv.groups,
		allocator = allocator,
		loc = loc,
	)

	return out
}

import "core:testing"

@(test)
test_new_conv2d :: proc(t: ^testing.T) {
	// Test basic conv2d creation
	conv := new_conv2d(
		f32,
		3,
		16,
		[2]uint{3, 3},
		stride = 1,
		padding = 1,
		allocator = context.temp_allocator,
	)

	// Check weight shape: (16, 3, 3, 3)
	expected_weight_shape := []uint{16, 3, 3, 3}
	testing.expect(t, len(conv.w.shape) == 4, "Weight should be 4D")
	for i in 0 ..< len(expected_weight_shape) {
		testing.expect(t, conv.w.shape[i] == expected_weight_shape[i], "Weight shape mismatch")
	}

	// Check bias exists
	bias, has_bias := conv.b.?
	testing.expect(t, has_bias, "Should have bias by default")
	if has_bias {
		testing.expect(t, len(bias.shape) == 1, "Bias should be 1D")
		testing.expect(t, bias.shape[0] == 16, "Bias should have out_channels elements")
	}

	// Test without bias
	conv_no_bias := new_conv2d(
		f32,
		3,
		16,
		[2]uint{3, 3},
		use_bias = false,
		allocator = context.temp_allocator,
	)
	_, has_bias_no := conv_no_bias.b.?
	testing.expect(t, !has_bias_no, "Should not have bias when use_bias=false")
}

@(test)
test_forward_conv2d :: proc(t: ^testing.T) {
	// Create a small conv2d layer
	conv := new_conv2d(
		f32,
		2,
		4,
		[2]uint{3, 3},
		stride = 1,
		padding = 1,
		allocator = context.temp_allocator,
	)

	// Create input tensor: (1, 2, 5, 5)
	input_data := make([]f32, 50, context.temp_allocator) // 1*2*5*5 = 50
	for i in 0 ..< 50 {
		input_data[i] = f32(i + 1)
	}
	input := tensor.new_with_init(input_data, []uint{1, 2, 5, 5}, context.temp_allocator)

	// Forward pass
	output := forward_conv2d(conv, input, context.temp_allocator)

	// Check output shape: should be (1, 4, 5, 5) with stride=1, padding=1
	expected_output_shape := []uint{1, 4, 5, 5}
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	for i in 0 ..< len(expected_output_shape) {
		testing.expect(t, output.shape[i] == expected_output_shape[i], "Output shape mismatch")
	}
}

@(test)
test_conv2d_grouped :: proc(t: ^testing.T) {
	// Test grouped convolution
	conv := new_conv2d(
		f32,
		4,
		8,
		[2]uint{3, 3},
		stride = 1,
		padding = 1,
		groups = 2,
		allocator = context.temp_allocator,
	)

	// Check weight shape for grouped conv: (8, 2, 3, 3) - in_channels/groups = 4/2 = 2
	expected_weight_shape := []uint{8, 2, 3, 3}
	for i in 0 ..< len(expected_weight_shape) {
		testing.expect(
			t,
			conv.w.shape[i] == expected_weight_shape[i],
			"Grouped conv weight shape mismatch",
		)
	}

	// Create input: (1, 4, 4, 4)
	input_data := make([]f32, 64, context.temp_allocator)
	for i in 0 ..< 64 {
		input_data[i] = f32(i + 1)
	}
	input := tensor.new_with_init(input_data, []uint{1, 4, 4, 4}, context.temp_allocator)

	// Forward pass
	output := forward_conv2d(conv, input, context.temp_allocator)

	// Check output shape: (1, 8, 4, 4)
	expected_output_shape := []uint{1, 8, 4, 4}
	for i in 0 ..< len(expected_output_shape) {
		testing.expect(
			t,
			output.shape[i] == expected_output_shape[i],
			"Grouped conv output shape mismatch",
		)
	}
}
