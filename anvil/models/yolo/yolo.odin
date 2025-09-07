package yolo

import "../../nn"
import "../../plot"
import st "../../safetensors"
import "../../tensor"
import "../../trace"
import vb "../sam/var_builder"
import "core:fmt"
import "core:math"
import "core:slice"

make_anchors :: proc(
	xs0, xs1, xs2: ^tensor.Tensor($T),
	strides: [3]uint,
	grid_cell_offset: T,
	allocator := context.allocator,
) -> (
	anchor_points: ^tensor.Tensor(T),
	stride_tensor: ^tensor.Tensor(T),
) {
	s0, s1, s2 := strides[0], strides[1], strides[2]

	anchor_points_list := make([dynamic]^tensor.Tensor(T), 0, 3, context.temp_allocator)
	stride_tensor_list := make([dynamic]^tensor.Tensor(T), 0, 3, context.temp_allocator)

	xs_list := [3]^tensor.Tensor(T){xs0, xs1, xs2}
	stride_list := [3]uint{s0, s1, s2}

	for i in 0 ..< 3 {
		xs := xs_list[i]
		stride := stride_list[i]
		h := xs.shape[2]
		w := xs.shape[3]

		// Create coordinate grids
		sx_range := tensor.arange(T, w, context.temp_allocator)
		sy_range := tensor.arange(T, h, context.temp_allocator)

		// Add grid_cell_offset to each coordinate
		offset_tensor := tensor.tensor_alloc(T, []uint{}, true, context.temp_allocator)
		offset_tensor.data[0] = grid_cell_offset

		sx := tensor.add(sx_range, offset_tensor, context.temp_allocator)
		sy := tensor.add(sy_range, offset_tensor, context.temp_allocator)

		// Create grid by repeating
		// sx: (w,) -> (1, w) -> repeat (h, 1) -> (h, w) -> flatten
		sx_reshaped := tensor.reshape(sx, []uint{1, w}, context.temp_allocator)
		sx_repeated := tensor.repeat(sx_reshaped, []uint{h, 1}, context.temp_allocator)
		sx_flat := tensor.flatten_all(sx_repeated, context.temp_allocator)

		// sy: (h,) -> (h, 1) -> repeat (1, w) -> (h, w) -> flatten
		sy_reshaped := tensor.reshape(sy, []uint{h, 1}, context.temp_allocator)
		sy_repeated := tensor.repeat(sy_reshaped, []uint{1, w}, context.temp_allocator)
		sy_flat := tensor.flatten_all(sy_repeated, context.temp_allocator)

		// Reshape to (h*w, 1) before stacking
		sx_flat_2d := tensor.reshape(sx_flat, []uint{h * w, 1}, context.temp_allocator)
		sy_flat_2d := tensor.reshape(sy_flat, []uint{h * w, 1}, context.temp_allocator)

		// Concatenate along dimension 1 to get (h*w, 2)
		points := tensor.cat(
			[]^tensor.Tensor(T){sx_flat_2d, sy_flat_2d},
			1,
			context.temp_allocator,
		)
		append(&anchor_points_list, points)

		// Create stride tensor with shape (h*w,) filled with stride value
		stride_val := tensor.ones(T, []uint{h * w}, context.temp_allocator)
		stride_scalar := tensor.tensor_alloc(T, []uint{}, true, context.temp_allocator)
		stride_scalar.data[0] = T(stride)

		stride_scaled := tensor.mul(stride_val, stride_scalar, context.temp_allocator)
		append(&stride_tensor_list, stride_scaled)
	}

	// Concatenate all anchor points and stride tensors
	anchor_points = tensor.cat(anchor_points_list[:], 0, allocator)
	stride_tensor_concat := tensor.cat(stride_tensor_list[:], 0, context.temp_allocator)
	stride_tensor = tensor.unsqueeze(stride_tensor_concat, 1, allocator)

	return anchor_points, stride_tensor
}

Conv_Block :: struct($T: typeid) {
	conv: ^nn.Conv_2d(T),
	bn:   ^nn.Batch_Norm_2d(T),
}

load_conv_block :: proc(
	vb_root: ^vb.Var_Builder($T),
	in_channels, out_channels: uint,
	kernel_size: uint,
	stride: uint = 1,
	padding: Maybe(uint) = nil,
	groups: uint = 1,
	init := true,
	allocator := context.allocator,
) -> ^Conv_Block(T) {
	padding := padding.? or_else kernel_size / 2
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
	vb.assign(vb_root, "conv.weight", conv.w)

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
	conv_out := nn.forward_conv2d(layer.conv, x, context.temp_allocator, loc)
	trace.end_scoped_trace(conv_trace)

	bn_trace := trace.TRACE_SECTION("batch_norm")
	bn_out := tensor.silu(nn.forward_batch_norm_2d(layer.bn, conv_out, allocator, loc), allocator)
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

load_bottleneck :: proc(
	vb_root: ^vb.Var_Builder($T),
	c1, c2: uint,
	shortcut: bool,
	allocator := context.allocator,
) -> ^Bottleneck(T) {
	channel_factor := T(1.)
	c_ := uint(T(c2) * channel_factor)
	vb_cv1 := vb.vb_make(T, "cv1", vb_root)
	cv1 := load_conv_block(&vb_cv1, c1, c_, 3, 1, allocator = allocator)
	vb_cv2 := vb.vb_make(T, "cv2", vb_root)
	cv2 := load_conv_block(&vb_cv2, c_, c2, 3, 1, allocator = allocator)
	residual := c1 == c2 && shortcut
	return new_clone(Bottleneck(T){cv1 = cv1, cv2 = cv2, residual = residual}, allocator)
}

forward_bottleneck :: proc(
	bottleneck: ^Bottleneck($T),
	xs: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	if bottleneck.residual {
		ys := forward_conv_block(
			bottleneck.cv2,
			forward_conv_block(bottleneck.cv1, xs, context.temp_allocator, loc),
			context.temp_allocator,
			loc,
		)
		return tensor.add(xs, ys, allocator)
	} else {
		return forward_conv_block(
			bottleneck.cv2,
			forward_conv_block(bottleneck.cv1, xs, context.temp_allocator, loc),
			allocator,
			loc,
		)
	}
}

free_bottleneck :: proc(bottleneck: ^Bottleneck($T), allocator := context.allocator) {
	free_conv_block(bottleneck.cv1, allocator)
	free_conv_block(bottleneck.cv2, allocator)
	free(bottleneck, allocator)
}

C2f :: struct($T: typeid) {
	cv1, cv2:   ^Conv_Block(T),
	bottleneck: [dynamic]^Bottleneck(T),
}

load_c2f :: proc(
	vb_root: ^vb.Var_Builder($T),
	c1, c2: uint,
	n: uint,
	shortcut: bool,
	allocator := context.allocator,
) -> ^C2f(T) {
	c := uint(T(c2) * 0.5)
	vb_cv1 := vb.vb_make(T, "cv1", vb_root)
	cv1 := load_conv_block(&vb_cv1, c1, 2 * c, 1, 1, allocator = allocator)
	vb_cv2 := vb.vb_make(T, "cv2", vb_root)
	cv2 := load_conv_block(&vb_cv2, (2 + n) * c, c2, 1, 1, allocator = allocator)

	bottleneck := make([dynamic]^Bottleneck(T), allocator)
	for idx in 0 ..< n {
		vb_i := vb.vb_make(T, fmt.tprintf("bottleneck.%d", idx), vb_root)
		b := load_bottleneck(&vb_i, c, c, shortcut, allocator)
		append(&bottleneck, b)
	}

	return new_clone(C2f(T){cv1 = cv1, cv2 = cv2, bottleneck = bottleneck}, allocator)
}

forward_c2f :: proc(
	c2f: ^C2f($T),
	xs: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	ys := forward_conv_block(c2f.cv1, xs, context.temp_allocator)
	ys_list := slice.to_dynamic(
		tensor.chunk(ys, 2, 1, context.temp_allocator),
		context.temp_allocator,
	)
	for m in c2f.bottleneck {
		last := ys_list[len(ys_list) - 1]
		b := forward_bottleneck(m, last, context.temp_allocator, loc)
		append(&ys_list, b)
	}
	zs := tensor.cat(ys_list[:], 1, context.temp_allocator, loc)
	return forward_conv_block(c2f.cv2, zs, allocator, loc)
}

free_c2f :: proc(c2f: ^C2f($T), allocator := context.allocator) {
	free_conv_block(c2f.cv1, allocator)
	free_conv_block(c2f.cv2, allocator)

	for l in c2f.bottleneck do free_bottleneck(l, allocator)
	delete(c2f.bottleneck)

	free(c2f, allocator)
}

Sppf :: struct($T: typeid) {
	cv1, cv2: ^Conv_Block(T),
	k:        uint,
}

load_sppf :: proc(
	vb_root: ^vb.Var_Builder($T),
	c1, c2, k: uint,
	allocator := context.allocator,
) -> ^Sppf(T) {
	c_ := c1 / 2
	vb_cv1 := vb.vb_make(T, "cv1", vb_root)
	cv1 := load_conv_block(&vb_cv1, c1, c_, 1, 1)
	vb_cv2 := vb.vb_make(T, "cv2", vb_root)
	cv2 := load_conv_block(&vb_cv2, c_ * 4, c2, 1, 1)

	return new_clone(Sppf(T){cv1 = cv1, cv2 = cv2, k = k}, allocator)
}

forward_sppf :: proc(
	sppf: ^Sppf($T),
	xs: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	xs := forward_conv_block(sppf.cv1, xs, context.temp_allocator)
	xs2 := tensor.pad_with_zero(xs, 2, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs2 = tensor.pad_with_zero(xs2, 3, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs2 = tensor.max_pool_2d(xs2, sppf.k, 1, context.temp_allocator)

	xs3 := tensor.pad_with_zero(xs2, 2, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs3 = tensor.pad_with_zero(xs3, 3, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs3 = tensor.max_pool_2d(xs3, sppf.k, 1, context.temp_allocator)

	xs4 := tensor.pad_with_zero(xs3, 2, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs4 = tensor.pad_with_zero(xs4, 3, sppf.k / 2, sppf.k / 2, context.temp_allocator)
	xs4 = tensor.max_pool_2d(xs4, sppf.k, 1, context.temp_allocator)

	xs_cat := tensor.cat([]^tensor.Tensor(T){xs, xs2, xs3, xs4}, 1, context.temp_allocator)
	return forward_conv_block(sppf.cv2, xs_cat, allocator)
}

free_sppf :: proc(sppf: ^Sppf($T), allocator := context.allocator) {
	free_conv_block(sppf.cv1, allocator)
	free_conv_block(sppf.cv2, allocator)
	free(sppf, allocator)
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

load_dark_net :: proc(
	vb_root: ^vb.Var_Builder($T),
	m: Multiples,
	allocator := context.allocator,
) -> ^Dark_Net(T) {
	w, r, d := m.width, m.ratio, m.depth
	vb_b1_0 := vb.vb_make(T, "b1.0", vb_root)
	b1_0 := load_conv_block(
		&vb_b1_0,
		3,
		uint(64 * w),
		kernel_size = 3,
		stride = 2,
		padding = 1,
		init = false,
		allocator = allocator,
	)
	vb_b1_1 := vb.vb_make(T, "b1.1", vb_root)
	b1_1 := load_conv_block(
		&vb_b1_1,
		uint(64 * w),
		uint(128 * w),
		kernel_size = 3,
		stride = 2,
		padding = 1,
		init = false,
		allocator = allocator,
	)
	vb_b2_0 := vb.vb_make(T, "b2.0", vb_root)
	b2_0 := load_c2f(
		&vb_b2_0,
		uint(128 * w),
		uint(128 * w),
		uint(math.round(3 * d)),
		true,
		allocator,
	)
	vb_b2_1 := vb.vb_make(T, "b2.1", vb_root)
	b2_1 := load_conv_block(
		&vb_b2_1,
		uint(128 * w),
		uint(256 * w),
		kernel_size = 3,
		stride = 2,
		padding = 1,
		init = false,
		allocator = allocator,
	)
	vb_b2_2 := vb.vb_make(T, "b2.2", vb_root)
	b2_2 := load_c2f(
		&vb_b2_2,
		uint(256 * w),
		uint(256 * w),
		uint(math.round(6 * d)),
		true,
		allocator,
	)
	vb_b3_0 := vb.vb_make(T, "b3.0", vb_root)
	b3_0 := load_conv_block(
		&vb_b3_0,
		uint(256 * w),
		uint(512 * w),
		kernel_size = 3,
		stride = 2,
		padding = 1,
		init = false,
		allocator = allocator,
	)
	vb_b3_1 := vb.vb_make(T, "b3.1", vb_root)
	b3_1 := load_c2f(
		&vb_b3_1,
		uint(512 * w),
		uint(512 * w),
		uint(math.round(6 * d)),
		true,
		allocator,
	)
	vb_b4_0 := vb.vb_make(T, "b4.0", vb_root)
	b4_0 := load_conv_block(
		&vb_b4_0,
		uint(512 * w),
		uint(512 * w * r),
		kernel_size = 3,
		stride = 2,
		padding = 1,
		init = false,
		allocator = allocator,
	)
	vb_b4_1 := vb.vb_make(T, "b4.1", vb_root)
	b4_1 := load_c2f(
		&vb_b4_1,
		uint(512 * w * r),
		uint(512 * w * r),
		uint(math.round(3 * d)),
		true,
		allocator,
	)

	vb_b5 := vb.vb_make(T, "b5.0", vb_root)
	b5 := load_sppf(&vb_b5, uint(512 * w * r), uint(512 * w * r), 5, allocator)


	return new_clone(
		Dark_Net(T) {
			b1_0 = b1_0,
			b1_1 = b1_1,
			b2_0 = b2_0,
			b2_1 = b2_1,
			b2_2 = b2_2,
			b3_0 = b3_0,
			b3_1 = b3_1,
			b4_0 = b4_0,
			b4_1 = b4_1,
			b5 = b5,
		},
		allocator,
	)
}

free_dark_net :: proc(net: ^Dark_Net($T), allocator := context.allocator) {
	free_conv_block(net.b1_0, allocator)
	free_conv_block(net.b1_1, allocator)
	free_c2f(net.b2_0, allocator)
	free_conv_block(net.b2_1, allocator)
	free_c2f(net.b2_2, allocator)
	free_conv_block(net.b3_0, allocator)
	free_c2f(net.b3_1, allocator)
	free_conv_block(net.b4_0, allocator)
	free_c2f(net.b4_1, allocator)
	free_sppf(net.b5, allocator)
	free(net, allocator)
}

forward_dark_net :: proc(
	dn: ^Dark_Net($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	x1 := forward_conv_block(
		dn.b1_1,
		forward_conv_block(dn.b1_0, x, context.temp_allocator),
		context.temp_allocator,
	)

	// Returned, use parent allocator
	x2 := forward_c2f(
		dn.b2_2,
		forward_conv_block(
			dn.b2_1,
			forward_c2f(dn.b2_0, x1, context.temp_allocator),
			context.temp_allocator,
		),
		allocator,
	)

	// Returned, use parent allocator
	x3 := forward_c2f(dn.b3_1, forward_conv_block(dn.b3_0, x2, context.temp_allocator), allocator)

	x4 := forward_c2f(
		dn.b4_1,
		forward_conv_block(dn.b4_0, x3, context.temp_allocator),
		context.temp_allocator,
	)

	x5 := forward_sppf(dn.b5, x4, allocator)

	return x2, x3, x5
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

load_yolo_v8_neck :: proc(
	vb_root: ^vb.Var_Builder($T),
	m: Multiples,
	allocator := context.allocator,
) -> ^Yolo_V8_Neck(T) {
	w, r, d := m.width, m.ratio, m.depth
	n := uint(math.round(3 * d))
	vb_n1 := vb.vb_make(T, "n1", vb_root)
	n1 := load_c2f(&vb_n1, uint(512 * w * (1 + r)), uint(512 * w), n, false, allocator)
	vb_n2 := vb.vb_make(T, "n2", vb_root)
	n2 := load_c2f(&vb_n2, uint(768 * w), uint(256 * w), n, false, allocator)
	vb_n3 := vb.vb_make(T, "n3", vb_root)
	n3 := load_conv_block(
		&vb_n3,
		uint(256 * w),
		uint(256 * w),
		3,
		2,
		1,
		init = false,
		allocator = allocator,
	)
	vb_n4 := vb.vb_make(T, "n4", vb_root)
	n4 := load_c2f(&vb_n4, uint(768 * w), uint(512 * w), n, false, allocator)
	vb_n5 := vb.vb_make(T, "n5", vb_root)
	n5 := load_conv_block(
		&vb_n5,
		uint(512 * w),
		uint(512 * w),
		3,
		2,
		1,
		init = false,
		allocator = allocator,
	)
	vb_n6 := vb.vb_make(T, "n6", vb_root)
	n6 := load_c2f(&vb_n6, uint(512 * w * (1 + r)), uint(512 * w * r), n, false, allocator)

	return new_clone(
		Yolo_V8_Neck(T){upsample_factor = 2, n1 = n1, n2 = n2, n3 = n3, n4 = n4, n5 = n5, n6 = n6},
		allocator,
	)
}

free_yolo_v8_neck :: proc(fpn: ^Yolo_V8_Neck($T), allocator := context.allocator) {
	free_c2f(fpn.n1, allocator)
	free_c2f(fpn.n2, allocator)
	free_conv_block(fpn.n3, allocator)
	free_c2f(fpn.n4, allocator)
	free_conv_block(fpn.n5, allocator)
	free_c2f(fpn.n6, allocator)
	free(fpn)
}

forward_neck :: proc(
	dn: ^Yolo_V8_Neck($T),
	p3, p4, p5: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	p5_up := tensor.upsample_nearest_2d(
		p5,
		dn.upsample_factor * p5.shape[2],
		dn.upsample_factor * p5.shape[3],
		context.temp_allocator,
		loc,
	)
	x := forward_c2f(
		dn.n1,
		tensor.cat([]^tensor.Tensor(T){p5_up, p4}, 1, context.temp_allocator, loc),
		context.temp_allocator,
		loc,
	)

	x_up := tensor.upsample_nearest_2d(
		x,
		dn.upsample_factor * x.shape[2],
		dn.upsample_factor * x.shape[3],
		context.temp_allocator,
	)
	head_1 := forward_c2f(
		dn.n2,
		tensor.cat([]^tensor.Tensor(T){x_up, p3}, 1, context.temp_allocator, loc),
		allocator,
		loc,
	)
	head_2 := forward_c2f(
		dn.n4,
		tensor.cat(
			[]^tensor.Tensor(T){forward_conv_block(dn.n3, head_1, context.temp_allocator, loc), x},
			1,
			context.temp_allocator,
			loc,
		),
		allocator,
		loc,
	)
	head_3 := forward_c2f(
		dn.n6,
		tensor.cat(
			[]^tensor.Tensor(T) {
				forward_conv_block(dn.n5, head_2, context.temp_allocator, loc),
				p5,
			},
			1,
			context.temp_allocator,
			loc,
		),
		allocator,
		loc,
	)

	return head_1, head_2, head_3
}

Dfl :: struct($T: typeid) {
	conv:        ^nn.Conv_2d(T),
	num_classes: uint,
}

load_dfl :: proc(
	vb_root: ^vb.Var_Builder($T),
	num_classes: uint,
	allocator := context.allocator,
) -> ^Dfl(T) {
	conv := nn.new_conv2d(
		T,
		num_classes,
		1,
		{1, 1},
		use_bias = false,
		init = false,
		allocator = allocator,
	)
	vb.assign(vb_root, "conv.weight", conv.w)
	return new_clone(Dfl(T){conv = conv, num_classes = num_classes}, allocator)
}

free_dfl :: proc(dfl: ^Dfl($T), allocator := context.allocator) {
	nn.free_conv2d(dfl.conv, allocator)
	free(dfl, allocator)
}

Detection_Output :: struct($T: typeid) {
	pred, anchors, strides: ^tensor.Tensor(T),
}

CV_Head :: struct($T: typeid) {
	cb0, cb1: ^Conv_Block(T),
	conv:     ^nn.Conv_2d(T),
}

Detection_Head :: struct($T: typeid) {
	dfl:      ^Dfl(T),
	cv2, cv3: [3]^CV_Head(T),
	ch, no:   uint,
}


load_cv_head_3 :: proc(
	vb_root: ^vb.Var_Builder($T),
	c1, nc, filter: uint,
	allocator := context.allocator,
) -> ^CV_Head(T) {
	vb_0 := vb.vb_make(T, "0", vb_root)
	cb0 := load_conv_block(&vb_0, filter, c1, 3, 1, init = false, allocator = allocator)
	vb_1 := vb.vb_make(T, "1", vb_root)
	cb1 := load_conv_block(&vb_1, c1, c1, 3, 1, init = false, allocator = allocator)

	conv := nn.new_conv2d(T, c1, nc, {1, 1}, init = false, allocator = allocator)
	vb.assign(vb_root, "2.weight", conv.w)
	vb.assign(vb_root, "2.bias", conv.b.?)

	return new_clone(CV_Head(T){cb0 = cb0, cb1 = cb1, conv = conv}, allocator)
}

load_cv_head_2 :: proc(
	vb_root: ^vb.Var_Builder($T),
	c2, nc, filter: uint,
	allocator := context.allocator,
) -> ^CV_Head(T) {
	vb_0 := vb.vb_make(T, "0", vb_root)
	cb0 := load_conv_block(&vb_0, filter, c2, 3, 1, init = false, allocator = allocator)
	vb_1 := vb.vb_make(T, "1", vb_root)
	cb1 := load_conv_block(&vb_1, c2, c2, 3, 1, init = false, allocator = allocator)

	conv := nn.new_conv2d(T, c2, 4 * nc, {1, 1}, init = false, allocator = allocator)
	vb.assign(vb_root, "2.weight", conv.w)
	vb.assign(vb_root, "2.bias", conv.b.?)

	return new_clone(CV_Head(T){cb0 = cb0, cb1 = cb1, conv = conv}, allocator)
}

free_cv_head :: proc(cv: ^CV_Head($T), allocator := context.allocator) {
	free_conv_block(cv.cb0, allocator)
	free_conv_block(cv.cb1, allocator)
	nn.free_conv2d(cv.conv, allocator)
	free(cv, allocator)
}

load_detection_head :: proc(
	vb_root: ^vb.Var_Builder($T),
	nc: uint,
	f1, f2, f3: uint,
	allocator := context.allocator,
) -> ^Detection_Head(T) {
	ch: uint = 16

	vb_dfl := vb.vb_make(T, "dfl", vb_root)
	dfl := load_dfl(&vb_dfl, ch, allocator)
	c1 := max(nc, f1)
	c2 := max(ch * 4, f1 / 4)

	vb_cv3_0 := vb.vb_make(T, "cv3.0", vb_root)
	vb_cv3_1 := vb.vb_make(T, "cv3.1", vb_root)
	vb_cv3_2 := vb.vb_make(T, "cv3.2", vb_root)
	cv3 := [3]^CV_Head(T) {
		load_cv_head_3(&vb_cv3_0, c1, nc, f1, allocator),
		load_cv_head_3(&vb_cv3_1, c1, nc, f2, allocator),
		load_cv_head_3(&vb_cv3_2, c1, nc, f3, allocator),
	}

	vb_cv2_0 := vb.vb_make(T, "cv2.0", vb_root)
	vb_cv2_1 := vb.vb_make(T, "cv2.1", vb_root)
	vb_cv2_2 := vb.vb_make(T, "cv2.2", vb_root)
	cv2 := [3]^CV_Head(T) {
		load_cv_head_2(&vb_cv2_0, c2, ch, f1, allocator),
		load_cv_head_2(&vb_cv2_1, c2, ch, f2, allocator),
		load_cv_head_2(&vb_cv2_2, c2, ch, f3, allocator),
	}

	no := nc + ch * 4
	return new_clone(
		Detection_Head(T){dfl = dfl, cv3 = cv3, cv2 = cv2, ch = ch, no = no},
		allocator,
	)
}

free_detection_head :: proc(head: ^Detection_Head($T), allocator := context.allocator) {
	free_dfl(head.dfl, allocator)
	for cv in head.cv3 do free_cv_head(cv, allocator)
	for cv in head.cv2 do free_cv_head(cv, allocator)
	free(head, allocator)
}


forward_head :: proc(
	dn: ^Detection_Head($T),
	xs0, xs1, xs2: ^tensor.Tensor(T),
	allocator := context.allocator,
	// ) -> ^Detection_Output(T) {
) -> (
	^tensor.Tensor(T),
	^tensor.Tensor(T),
	^tensor.Tensor(T),
) {
	forward_cv := proc(
		cv2, cv3: [3]^CV_Head(T),
		xs: ^tensor.Tensor(T),
		i: uint,
		allocator := context.allocator,
	) -> ^tensor.Tensor(T) {
		xs_2 := forward_conv_block(cv2[i].cb0, xs, context.temp_allocator)
		xs_2 = forward_conv_block(cv2[i].cb1, xs_2, context.temp_allocator)
		xs_2 = nn.forward_conv2d(cv2[i].conv, xs_2, context.temp_allocator)

		xs_3 := forward_conv_block(cv3[i].cb0, xs, context.temp_allocator)
		xs_3 = forward_conv_block(cv3[i].cb1, xs_3, context.temp_allocator)
		xs_3 = nn.forward_conv2d(cv3[i].conv, xs_3, context.temp_allocator)

		return tensor.cat([]^tensor.Tensor(T){xs_2, xs_3}, 1, allocator)
	}

	xs0 := forward_cv(dn.cv2, dn.cv3, xs0, 0, allocator = context.temp_allocator)
	xs1 := forward_cv(dn.cv2, dn.cv3, xs1, 1, allocator = context.temp_allocator)
	xs2 := forward_cv(dn.cv2, dn.cv3, xs2, 2, allocator = context.temp_allocator)

	// let (anchors, strides) = make_anchors(&xs0, &xs1, &xs2, (8, 16, 32), 0.5)?;
	anchors, strides := make_anchors(xs0, xs1, xs2, {8, 16, 32}, 0.5, context.temp_allocator)
	return xs0, xs1, xs2
}

YOLO_V8 :: struct($T: typeid) {
	net:  ^Dark_Net(T),
	fpn:  ^Yolo_V8_Neck(T),
	head: ^Detection_Head(T),
}

load_yolo :: proc(
	safetensors: ^st.Safe_Tensors($T),
	m: Multiples,
	num_classes: uint,
	allocator := context.allocator,
) -> ^YOLO_V8(T) {
	vb_net := vb.Var_Builder(T) {
		name        = "net",
		safetensors = safetensors,
		parent      = nil,
	}
	net := load_dark_net(&vb_net, m, allocator)

	vb_fpn := vb.Var_Builder(T) {
		name        = "fpn",
		safetensors = safetensors,
		parent      = nil,
	}
	fpn := load_yolo_v8_neck(&vb_fpn, m, allocator)

	vb_head := vb.Var_Builder(T) {
		name        = "head",
		safetensors = safetensors,
		parent      = nil,
	}
	f1, f2, f3 := filters_by_size(m)
	head := load_detection_head(&vb_head, num_classes, f1, f2, f3)
	return new_clone(YOLO_V8(T){net = net, fpn = fpn, head = head}, allocator)
}

forward_yolo :: proc(
	yolo: ^YOLO_V8($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
) -> ^Detection_Output(T) {
	x1, x2, x3 := forward_dark_net(yolo.net, x, context.temp_allocator)
	x1, x2, x3 = forward_neck(yolo.fpn, x1, x2, x3, context.temp_allocator)
	return forward_head(yolo.head, x1, x2, x3, allocator)
}

free_yolo :: proc(model: ^YOLO_V8($T), allocator := context.allocator) {
	free_dark_net(model.net, allocator)
	free_yolo_v8_neck(model.fpn, allocator)
	free_detection_head(model.head, allocator)
	free(model, allocator)
}
