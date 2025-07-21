package tensor

import "core:testing"

@(test)
test_strided_index :: proc(t: ^testing.T) {
	// Test 1D
	{
		shape := []uint{5}
		strides := []uint{1}
		testing.expect_value(t, compute_strided_index(shape, strides, 3), 3)
	}

	// Test 2D row-major
	{
		shape := []uint{3, 4} // 3x4 matrix
		strides := []uint{4, 1}
		// idx 7 = row 1, col 3 → offset = 1*4 + 3*1 = 7
		testing.expect_value(t, compute_strided_index(shape, strides, 7), 7)
	}

	// Test 3D
	{
		shape := []uint{2, 3, 4}
		strides := []uint{12, 4, 1}
		// idx 23 = [1,2,3] → offset = 1*12 + 2*4 + 3*1 = 23
		testing.expect_value(t, compute_strided_index(shape, strides, 23), 23)
	}

	// Test 4D
	{
		shape := []uint{2, 2, 3, 4}
		strides := []uint{24, 12, 4, 1}
		// idx 47 = [1,1,2,3] → offset = 1*24 + 1*12 + 2*4 + 3*1 = 47
		testing.expect_value(t, compute_strided_index(shape, strides, 47), 47)
	}

	// Test 5D (general case)
	{
		shape := []uint{2, 2, 2, 3, 4}
		strides := []uint{48, 24, 12, 4, 1}
		// idx 95 = [1,1,1,2,3] → offset = 1*48 + 1*24 + 1*12 + 2*4 + 3*1 = 95
		testing.expect_value(t, compute_strided_index(shape, strides, 95), 95)
	}

	// Test non-contiguous strides (e.g., transposed)
	{
		shape := []uint{3, 4}
		strides := []uint{1, 3} // column-major
		// idx 7: row=1, col=3 → offset = 1*1 + 3*3 = 10
		testing.expect_value(t, compute_strided_index(shape, strides, 7), 10)
	}
}

import "core:fmt"
@(test)
test_matmul :: proc(t: ^testing.T) {
	a := new_with_init([]f32{1, 2, 3, 4, 5, 6}, []uint{3, 2}, allocator = context.temp_allocator)
	b := new_with_init([]f32{1, 1}, []uint{2, 1}, allocator = context.temp_allocator)
	res := matmul(a, b, allocator = context.temp_allocator)

	expected := []f32{3, 7, 11}
	for i in 0 ..< len(expected) {
		assert(expected[i] == res.data[i])
	}
}
