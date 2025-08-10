package nn

import "../tensor"

Linear :: struct($T: typeid) {
	w: ^tensor.Tensor(T),
	b: Maybe(^tensor.Tensor(T)),
}

new_linear :: proc(
	$T: typeid,
	in_feat, out_feat: uint,
	use_bias := true,
	init := true,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Linear(T) {
	w: ^tensor.Tensor(T)
	if init {
		w = tensor.randn(T, {in_feat, out_feat}, T(0), T(1), allocator, loc)
	} else {
		w = tensor.tensor_alloc(T, {in_feat, out_feat}, true, allocator, loc)
	}
	b: Maybe(^tensor.Tensor(T)) = nil
	if use_bias do b = tensor.zeros(T, []uint{out_feat}, allocator)
	return new_clone(Linear(T){w = w, b = b}, allocator)
}

free_linear :: proc(l: ^Linear($T), allocator := context.allocator) {
	tensor.free_tensor(l.w, allocator)
	b, ok := l.b.?
	if ok do tensor.free_tensor(b, allocator)
	free(l, allocator)
}

import "../trace"


forward_linear :: proc(
	l: ^Linear($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	forward_linear_trace := trace.TRACE_FUNCTION("forward_linear")
	defer trace.end_scoped_trace(forward_linear_trace)

	out := tensor.matmul(x, l.w, allocator, loc)


	// Add bias in-place
	if bias, has_bias := l.b.?; has_bias {
		out_features := out.shape[len(out.shape) - 1]
		total_elements := tensor.shape_to_size(out.shape)
		batch_elements := total_elements / out_features

		#no_bounds_check {
			when T == f32 {
				for i in 0 ..< batch_elements {
					base_idx := i * out_features
					j := uint(0)

					for ; j + 8 <= out_features; j += 8 {
						b0 := (^#simd[4]f32)(&bias.data[j])^
						b1 := (^#simd[4]f32)(&bias.data[j + 4])^

						o0 := (^#simd[4]f32)(&out.data[base_idx + j])^
						o1 := (^#simd[4]f32)(&out.data[base_idx + j + 4])^

						(^#simd[4]f32)(&out.data[base_idx + j])^ = o0 + b0
						(^#simd[4]f32)(&out.data[base_idx + j + 4])^ = o1 + b1
					}

					for ; j + 4 <= out_features; j += 4 {
						b := (^#simd[4]f32)(&bias.data[j])^
						o := (^#simd[4]f32)(&out.data[base_idx + j])^
						(^#simd[4]f32)(&out.data[base_idx + j])^ = o + b
					}

					for ; j < out_features; j += 1 {
						out.data[base_idx + j] += bias.data[j]
					}
				}

			} else when T == f64 {
				for i in 0 ..< batch_elements {
					base_idx := i * out_features
					j := uint(0)

					for ; j + 4 <= out_features; j += 4 {
						b0 := (^#simd[2]f64)(&bias.data[j])^
						b1 := (^#simd[2]f64)(&bias.data[j + 2])^

						o0 := (^#simd[2]f64)(&out.data[base_idx + j])^
						o1 := (^#simd[2]f64)(&out.data[base_idx + j + 2])^

						(^#simd[2]f64)(&out.data[base_idx + j])^ = o0 + b0
						(^#simd[2]f64)(&out.data[base_idx + j + 2])^ = o1 + b1
					}

					for ; j + 2 <= out_features; j += 2 {
						b := (^#simd[2]f64)(&bias.data[j])^
						o := (^#simd[2]f64)(&out.data[base_idx + j])^
						(^#simd[2]f64)(&out.data[base_idx + j])^ = o + b
					}

					for ; j < out_features; j += 1 {
						out.data[base_idx + j] += bias.data[j]
					}
				}

			} else {
				// Scalar fallback
				for i in 0 ..< batch_elements {
					base_idx := i * out_features
					for j in 0 ..< out_features {
						out.data[base_idx + j] += bias.data[j]
					}
				}
			}
		}
	}

	return out
}

import "core:testing"

@(test)
test_new_linear :: proc(t: ^testing.T) {
	l := new_linear(f32, 10, 10, allocator = context.temp_allocator)
	b, ok := l.b.?
	assert(ok)

	x := tensor.randn(f32, []uint{5, 10}, 0.0, 1.0, context.temp_allocator)
	out := forward_linear(l, x, context.temp_allocator)
	assert(out.shape[1] == 10)

	tensor.free_tensor(out, context.temp_allocator)
	tensor.free_tensor(x, context.temp_allocator)
	free_linear(l, context.temp_allocator)

	l = new_linear(f32, 10, 10, use_bias = false, allocator = context.temp_allocator)
	b, ok = l.b.?
	assert(!ok)

	free_linear(l, context.temp_allocator)
}
