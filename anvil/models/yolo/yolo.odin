package yolo

import "../../nn"
import st "../../safetensors"
import "../../tensor"
import "../../trace"
import vb "../sam/var_builder"

Conv_Block :: struct($T: typeid) {
	conv: ^nn.Conv_2d(T),
	bn:   ^nn.Batch_Norm_2d(T),
}

new_conv_block :: proc(
	$T: typeid,
	vb_root: ^vb.Var_Builder(T),
	in_channels, out_channels: uint,
	kernel_size: uint,
	stride: uint = 1,
	padding: uint = 0,
	groups: uint = 1,
	init := true,
	allocator := context.allocator,
) -> ^Conv_Block(T) {
	conv := nn.new_conv2d(
		T,
		in_channels,
		out_channels,
		[2]uint{kernel_size, kernel_size},
		stride,
		padding,
		1,
		groups,
		use_bias = false,
		init = init,
		allocator = allocator,
	)
	vb.assign(vb_root, "c.weight", conv.w)

	bn := nn.new_batch_norm_2d(T, out_channels, allocator)
	vb.assign(vb_root, "bn.weight", bn.weight)
	vb.assign(vb_root, "bn.bias", bn.bias)
	vb.assign(vb_root, "bn.running_mean", bn.running_mean)
	vb.assign(vb_root, "bn.running_var", bn.running_var)

	return new_clone(Conv_Block(T){conv = conv, bn = bn}, allocator)
}

forward_conv_block :: proc(
	layer: ^Conv_Block($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	conv_bn_trace := trace.TRACE_FUNCTION("conv2d_bn")
	defer trace.end_scoped_trace(conv_bn_trace)

	conv_trace := trace.TRACE_SECTION("conv2d")
	conv_out := nn.forward_conv2d(layer.conv, x, context.temp_allocator)
	trace.end_scoped_trace(conv_trace)

	bn_trace := trace.TRACE_SECTION("batch_norm")
	bn_out := nn.forward_batch_norm_2d(layer.bn, conv_out, allocator, loc)
	trace.end_scoped_trace(bn_trace)

	return bn_out
}

free_conv_block :: proc(layer: ^Conv_Block($T), allocator := context.allocator) {
	nn.free_conv2d(layer.conv, allocator)
	nn.free_batch_norm_2d(layer.bn, allocator)
	free(layer, allocator)
}


Multiples :: struct {
	depth, width, ratio: f32,
}

filters_by_size :: proc(m: Multiples) -> (uint, uint, uint) {
	f1 := uint(256 * m.width)
	f2 := uint(512 * m.width)
	f3 := uint(512 * m.width * m.ratio)
	return f1, f2, f3
}

YOLO_Size :: enum {
	Nano,
	Small,
	Medium,
	Large,
	Extra_Large,
}

Bottleneck :: struct($T: typeid) {
	cv1, cv2: ^Conv_Block(T),
	residual: bool,
}

C2f :: struct($T: typeid) {
	cv1, cv2:   ^Conv_Block(T),
	bottleneck: [dynamic]^Bottleneck(T),
}

Sppf :: struct($T: typeid) {
	cv1, cv2: ^Conv_Block(T),
	k:        uint,
}

Dark_Net :: struct($T: typeid) {
	b1_0: ^Conv_Block(T),
	b1_1: ^Conv_Block(T),
	b2_0: ^C2f(T),
	b2_1: ^Conv_Block(T),
	b2_2: ^C2f(T),
	b3_0: ^Conv_Block(T),
	b3_1: ^C2f(T),
	b4_0: ^Conv_Block(T),
	b4_1: ^C2f(T),
	b5:   ^Sppf(T),
}


Yolo_V8_Neck :: struct($T: typeid) {
	upsample_factor: uint,
	n1:              ^C2f(T),
	n2:              ^C2f(T),
	n3:              ^Conv_Block(T),
	n4:              ^C2f(T),
	n5:              ^Conv_Block(T),
	n6:              ^C2f(T),
}

Dfl :: struct($T: typeid) {
	conv:        nn.Conv_2d(T),
	num_classes: uint,
}

Detection_Head :: struct($T: typeid) {
	dfl:      Dfl(T),
	cv2, cv3: struct {
		cb1, cb2: ^Conv_Block(T),
		c:        ^nn.Conv_2d(T),
	},
	ch, no:   uint,
}

YOLO_V8 :: struct($T: typeid) {
	net:  ^Dark_Net(T),
	fpn:  ^Yolo_V8_Neck(T),
	head: ^Detection_Head(T),
}

new_yolo :: proc(
	$T: typeid,
	safetensors: ^st.Safe_Tensors(T),
	allocator := context.allocator,
) -> ^YOLO_V8(T) {
	return new_clone(YOLO_V8(T){net = nil, fpn = nil, head = nil}, allocator)
}

free_yolo :: proc(model: ^YOLO_V8($T), allocator := context.allocator) {
	free(model, allocator)
}
