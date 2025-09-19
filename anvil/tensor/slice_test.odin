package tensor

import s "core:slice"
import "core:testing"

@(test)
slice_test :: proc(t: ^testing.T) {
	context.allocator = context.temp_allocator

	x := reshape(arange(f32, 9), {3, 3})
	xs := slice(x, {R(1, 3)})
	testing.expect(t, s.equal(xs.data, []f32{3, 4, 5, 6, 7, 8}))

	xs = slice(x, {{}, 1})
	testing.expect(t, s.equal(xs.data, []f32{1, 4, 7}))
	testing.expect(t, s.equal(xs.shape, []uint{3}))

	xs = slice(x, {-1}, keepdims = true)
	testing.expect(t, s.equal(xs.data, []f32{6, 7, 8}))
	testing.expect(t, s.equal(xs.shape, []uint{1, 3}))

	xs = slice(x, {-1, -2})
	testing.expect(t, s.equal(xs.data, []f32{7}))
	testing.expect(t, s.equal(xs.shape, []uint{}))

	// Every other column: [[1,3], [5,7], [9,11]]
	x = new_with_init([]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4})
	xs = slice(x, {{}, R(0, 0, 2)})
	testing.expect(t, s.equal(xs.data, []f32{1, 3, 5, 7, 9, 11}))
	testing.expect(t, s.equal(xs.shape, []uint{3, 2}))

	// Get last 2 rows: [[5,6,7,8], [9,10,11,12]]
	xs = slice(x, {R(-2, 0)})
	testing.expect(t, s.equal(xs.data, []f32{5, 6, 7, 8, 9, 10, 11, 12}))
	testing.expect(t, s.equal(xs.shape, []uint{2, 4}))
}
