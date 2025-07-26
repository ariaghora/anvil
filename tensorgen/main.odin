package main

import "../libs/nn"
import "../libs/tensor"
import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strconv"
import "core:strings"
import "core:time"

atot :: proc($T: typeid, s: string) -> T {
	s := strings.trim_space(s)
	res: T
	when T == uint {
		v, ok := strconv.parse_uint(s)
	} else when T == f32 {
		v, ok := strconv.parse_f32(s)
	} else when T == bool {
		v_i, ok := strconv.parse_int(s)
		v := v_i != 0
	} else {
		#panic("unsupported")
	}
	assert(ok)
	res = T(v)
	return res
}

write_tensor :: proc(t: ^tensor.Tensor($T), file_name: string) {
	n_dims := u64(len(t.shape))
	shape_u64 := mem.slice_data_cast([]u64, t.shape)
	header_size := size_of(u64) + len(shape_u64) * size_of(u64)
	data_size := int(tensor.shape_to_size(t.shape)) * size_of(T)
	total_size := header_size + data_size

	buf := make([]byte, total_size)
	mem.copy(&buf[0], &n_dims, size_of(u64))
	mem.copy(&buf[size_of(u64)], raw_data(shape_u64), len(shape_u64) * size_of(u64))
	mem.copy(&buf[header_size], raw_data(t.data), data_size)
	assert(os.write_entire_file(file_name, buf))
}

main :: proc() {
	data, ok := os.read_entire_file_from_filename("config_conv2d.csv", context.temp_allocator)
	assert(ok)
	str := transmute(string)data
	res, err := strings.split(str, "\n", context.temp_allocator)
	assert(err == nil)

	for i in 1 ..< len(res) {
		if len(strings.trim_space(res[i])) == 0 do continue
		file_name, _ := strings.replace(res[i], ",", "_", -1, allocator = context.temp_allocator)
		file_name = fmt.tprintf("conv2d/%s.tensor", strings.trim_space(file_name))

		l, err := strings.split(res[i], ",", context.temp_allocator)
		assert(err == nil)
		ic, ih, iw, oc, ch, cw, groups, strides, dilation, padding, use_bias :=
			atot(uint, l[0]),
			atot(uint, l[1]),
			atot(uint, l[2]),
			atot(uint, l[3]),
			atot(uint, l[4]),
			atot(uint, l[5]),
			atot(uint, l[6]),
			atot(uint, l[7]),
			atot(uint, l[8]),
			atot(uint, l[9]),
			atot(bool, l[10])

		conv2d := nn.new_conv2d(
			f32,
			ic,
			oc,
			{ch, cw},
			strides,
			padding,
			dilation,
			groups,
			use_bias = use_bias,
			allocator = context.temp_allocator,
		)

		filter_data_len := tensor.shape_to_size(conv2d.w.shape)
		for i in 0 ..< filter_data_len do conv2d.w.data[i] = f32(i) / f32(filter_data_len)

		if bias, ok := conv2d.b.?; ok {
			bias_data_len := tensor.shape_to_size(bias.shape)
			for i in 0 ..< bias_data_len do bias.data[i] = 1.0
		}

		x := tensor.tensor_alloc(f32, {1, ic, ih, iw}, true, allocator = context.temp_allocator)
		numel := int(tensor.shape_to_size(x.shape))
		for a in 0 ..< ic {
			for b in 0 ..< ih {
				for c in 0 ..< iw {
					idx := a * x.strides[1] + b * x.strides[2] + c * x.strides[3]


					h_norm := f32(b) / f32(ih)
					w_norm := f32(c) / f32(iw)

					pattern_type := a % 13
					switch pattern_type {
					case 0:
						x.data[idx] = h_norm * w_norm // Multiplication
					case 1:
						x.data[idx] = h_norm - w_norm // Subtraction
					case 2:
						x.data[idx] = max(h_norm, w_norm) // Max
					case 3:
						x.data[idx] = min(h_norm, w_norm) // Min
					case 4:
						x.data[idx] = abs(h_norm - w_norm) // Absolute difference
					case 5:
						x.data[idx] = h_norm + w_norm // Addition
					case 6:
						x.data[idx] = h_norm * h_norm - w_norm * w_norm // Quadratic
					case 7:
						x.data[idx] = f32(b ~ c) / f32(ih * iw) // XOR pattern
					case 8:
						x.data[idx] = f32((b + c) % 64) / 64.0 // Modular stripes
					case 9:
						x.data[idx] = f32(b % 32) * f32(c % 32) / (32.0 * 32.0) // Block pattern
					case 10:
						x.data[idx] = h_norm * h_norm * w_norm // Cubic mix
					case 11:
						x.data[idx] = f32(int(h_norm * 8.0) % 2) * f32(int(w_norm * 8.0) % 2) // Checkerboard
					case 12:
						x.data[idx] = (h_norm + w_norm > 1.0) ? 1.0 : 0.0 // Triangle split
					}

					// Add channel-specific scaling
					scale := f32(a / 13 + 1) * 0.3
					x.data[idx] = x.data[idx] * scale

				}
			}
		}

		start_time := time.now()
		res := nn.forward_conv2d(conv2d, x, allocator = context.temp_allocator)
		fmt.println(time.since(start_time), "\n")

		write_tensor(res, file_name)
	}
}
