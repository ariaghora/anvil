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
	allocator := context.allocator,
) -> ^Linear(T) {
	w := tensor.randn(T, []uint{in_feat, out_feat}, T(0), T(1), allocator)
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

forward_linear :: proc(
	l: ^Linear($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	out := tensor.matmul(x, l.w, allocator, loc)
	return out
}

import "core:testing"

@(test)
test_new_linear :: proc(t: ^testing.T) {
	l := new_linear(f32, 10, 10)
	b, ok := l.b.?
	assert(ok)

	x := tensor.randn(f32, []uint{5, 10}, 0.0, 1.0)
	out := forward_linear(l, x)
	assert(out.shape[1] == 10)

	tensor.free_tensor(out)
	tensor.free_tensor(x)
	free_linear(l)

	l = new_linear(f32, 10, 10, use_bias = false)
	b, ok = l.b.?
	assert(!ok)

	free_linear(l)
}
