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

	// Add bias if present
	if bias, has_bias := l.b.?; has_bias {
		// Get dimensions
		out_features := out.shape[len(out.shape) - 1]
		total_elements := tensor.shape_to_size(out.shape)
		batch_elements := total_elements / out_features

		// Add bias in-place
		for i in 0 ..< batch_elements {
			base_idx := i * out_features

			// Vectorized bias addition
			j := uint(0)
			for ; j + UNROLL_FACTOR <= out_features; j += UNROLL_FACTOR {
				#unroll for k in 0 ..< UNROLL_FACTOR {
					out.data[base_idx + j + uint(k)] += bias.data[j + uint(k)]
				}
			}

			// Handle remainder
			for ; j < out_features; j += 1 {
				out.data[base_idx + j] += bias.data[j]
			}
		}

		return out
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
