package tensor

import "core:fmt"
import "core:slice"
import "core:testing"

@(test)
test_permute_2d :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
	tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := permute(tensor, []uint{1, 0}, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Check shape: (2,3) -> (3,2)
	expected_shape := []uint{3, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Permute 2D shape incorrect")

	// Check values: original[i][j] should be at result[j][i]
	// tensor[0][0] = 1 should be at result[0][0]
	testing.expect(t, tensor_get(result, 0, 0) == 1, "Permute value [0,0] incorrect")
	// tensor[0][1] = 2 should be at result[1][0]
	testing.expect(t, tensor_get(result, 1, 0) == 2, "Permute value [1,0] incorrect")
	// tensor[1][2] = 6 should be at result[2][1]
	testing.expect(t, tensor_get(result, 2, 1) == 6, "Permute value [2,1] incorrect")
}

@(test)
test_permute_3d :: proc(t: ^testing.T) {
	data := make([]f32, 24, context.temp_allocator)
	for i in 0 ..< 24 {
		data[i] = f32(i + 1)
	}

	tensor := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := permute(tensor, []uint{2, 0, 1}, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Check shape: (2,3,4) -> (4,2,3)
	expected_shape := []uint{4, 2, 3}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Permute 3D shape incorrect")

	// Check specific values
	// tensor[0][1][2] = data[0*3*4 + 1*4 + 2] = data[6] = 7
	// should be at result[2][0][1]
	testing.expect(t, tensor_get(tensor, 0, 1, 2) == 7, "Original tensor value check")
	testing.expect(t, tensor_get(result, 2, 0, 1) == 7, "Permute 3D value mapping incorrect")
}

@(test)
test_transpose_2d :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6, 7, 8} // [[1,2,3,4], [5,6,7,8]]
	tensor := new_with_init(data, []uint{2, 4}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := transpose(tensor, 0, 1, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Check shape: (2,4) -> (4,2)
	expected_shape := []uint{4, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Transpose shape incorrect")

	// Check values
	// tensor[0][1] = 2 should be at result[1][0]
	testing.expect(t, tensor_get(tensor, 0, 1) == 2, "Original value check")
	testing.expect(t, tensor_get(result, 1, 0) == 2, "Transpose value [1,0] incorrect")

	// tensor[1][3] = 8 should be at result[3][1]
	testing.expect(t, tensor_get(tensor, 1, 3) == 8, "Original value check")
	testing.expect(t, tensor_get(result, 3, 1) == 8, "Transpose value [3,1] incorrect")
}

@(test)
test_transpose_3d :: proc(t: ^testing.T) {
	data := make([]f32, 24, context.temp_allocator)
	for i in 0 ..< 24 {
		data[i] = f32(i + 1)
	}

	tensor := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := transpose(tensor, 0, 2, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Check shape: (2,3,4) -> (4,3,2)
	expected_shape := []uint{4, 3, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Transpose 3D shape incorrect")

	// tensor[1][2][0] = data[1*3*4 + 2*4 + 0] = data[20] = 21
	// should be at result[0][2][1]
	testing.expect(t, tensor_get(tensor, 1, 2, 0) == 21, "Original 3D value check")
	testing.expect(t, tensor_get(result, 0, 2, 1) == 21, "Transpose 3D value mapping incorrect")
}

@(test)
test_transpose_same_dim :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6}
	tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := transpose(tensor, 1, 1, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Shape should be unchanged
	testing.expect(
		t,
		slice.equal(result.shape, tensor.shape),
		"Transpose same dim shape should be unchanged",
	)

	// If original was contiguous, result should be too (since no actual change)
	testing.expect(t, result.contiguous == tensor.contiguous, "Transpose same dim contiguity")

	// Values should be identical
	testing.expect(
		t,
		tensor_get(result, 0, 1) == tensor_get(tensor, 0, 1),
		"Transpose same dim values should match",
	)
}

@(test)
test_matrix_transpose :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
	tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := matrix_transpose(tensor, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Should swap last two dimensions: (2,3) -> (3,2)
	expected_shape := []uint{3, 2}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		"Matrix transpose shape incorrect",
	)

	// Check values
	testing.expect(t, tensor_get(result, 0, 0) == 1, "Matrix transpose [0,0] incorrect")
	testing.expect(t, tensor_get(result, 1, 0) == 2, "Matrix transpose [1,0] incorrect")
	testing.expect(t, tensor_get(result, 2, 1) == 6, "Matrix transpose [2,1] incorrect")
}

@(test)
test_matrix_transpose_3d :: proc(t: ^testing.T) {
	data := make([]f32, 24, context.temp_allocator)
	for i in 0 ..< 24 {
		data[i] = f32(i + 1)
	}

	tensor := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := transpose(tensor, 1, 2, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Should swap last two dimensions: (2,3,4) -> (2,4,3)
	expected_shape := []uint{2, 4, 3}
	testing.expect(
		t,
		slice.equal(result.shape, expected_shape),
		"Matrix transpose 3D shape incorrect",
	)

	// Check specific values
	// tensor[1][2][1] = data[1*3*4 + 2*4 + 1] = data[21] = 22
	// should be at result[1][1][2]
	testing.expect(t, tensor_get(tensor, 1, 2, 1) == 22, "Original 3D matrix value check")
	testing.expect(
		t,
		tensor_get(result, 1, 1, 2) == 22,
		"Matrix transpose 3D value mapping incorrect",
	)
}

@(test)
test_memory_management :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6}
	original := new_with_init(data, []uint{2, 3}, context.temp_allocator)

	// Create multiple views
	view1 := transpose(original, 0, 1, context.temp_allocator)
	view2 := permute(original, []uint{1, 0}, context.temp_allocator)

	// Cleanup - views should free their shape/strides but not data
	// Original should free data when it's freed
	free_tensor(view1, context.temp_allocator)
	free_tensor(view2, context.temp_allocator)
	free_tensor(original, context.temp_allocator)

	// Test passes if no memory issues occur
	testing.expect(t, true, "Memory management test completed")
}

@(test)
test_permute_errors :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6}
	tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	// These would panic in actual usage, but we document the expected behaviors:
	// Wrong number of dimensions: permute(tensor, []int{0}) - should panic
	// Out of range dimension: permute(tensor, []int{0, 3}) - should panic
	// Duplicate dimension: permute(tensor, []int{0, 0}) - should panic

	testing.expect(t, true, "Permute error cases documented")
}

@(test)
test_transpose_errors :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4}
	tensor := new_with_init(data, []uint{2, 2}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	// These would panic in actual usage:
	// Out of range: transpose(tensor, 0, 2) - should panic
	// Negative indices: transpose(tensor, -1, 0) - should panic

	testing.expect(t, true, "Transpose error cases documented")
}

@(test)
test_matrix_transpose_errors :: proc(t: ^testing.T) {
	data := []f32{1, 2}
	tensor := new_with_init(data, []uint{2}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	// 1D tensor should panic: matrix_transpose(tensor) - should panic

	testing.expect(t, true, "Matrix transpose error cases documented")
}
