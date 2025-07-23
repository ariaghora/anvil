package tensor

import "core:fmt"
import "core:slice"
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

@(test)
test_get_strided_data :: proc(t: ^testing.T) {
	// Test get_strided_data with non-contiguous tensor
	original := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, allocator = context.temp_allocator)
	transposed := transpose(original, 0, 1, allocator = context.temp_allocator)

	// Extract data with strided access
	strided_data, allocated := get_strided_data(transposed, allocator = context.temp_allocator)
	testing.expect(t, allocated)

	// Expected: [1, 3, 2, 4] (transpose of [[1,2],[3,4]] is [[1,3],[2,4]])
	expected := []f32{1, 3, 2, 4}
	testing.expect_value(t, len(strided_data), len(expected))

	for i in 0 ..< len(expected) {
		testing.expect_value(t, strided_data[i], expected[i])
	}
}


@(test)
test_chunk :: proc(t: ^testing.T) {
	// Test chunking a 4D tensor along channel dimension
	data := make([]f32, 24, context.temp_allocator) // 1x6x2x2 tensor
	for i in 0 ..< 24 {
		data[i] = f32(i + 1)
	}
	tensor := new_with_init(data, []uint{1, 6, 2, 2}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	// Split into 2 groups along dimension 1 (channels)
	chunks := chunk(tensor, 2, 1, context.temp_allocator)
	defer {
		for chunk_tensor in chunks {
			free_tensor(chunk_tensor, context.temp_allocator)
		}
		delete(chunks, context.temp_allocator)
	}

	// Check we have 2 chunks
	testing.expect(t, len(chunks) == 2, "Should have 2 chunks")

	// Check first chunk shape: [1, 3, 2, 2]
	expected_shape := []uint{1, 3, 2, 2}
	testing.expect(t, slice.equal(chunks[0].shape, expected_shape), "First chunk shape incorrect")
	testing.expect(t, slice.equal(chunks[1].shape, expected_shape), "Second chunk shape incorrect")

	// Check first chunk data (first 12 elements: 1-12)
	for i in 0 ..< 12 {
		testing.expect(
			t,
			chunks[0].data[i] == f32(i + 1),
			fmt.tprintf(
				"First chunk data at %d incorrect: got %f, expected %f",
				i,
				chunks[0].data[i],
				f32(i + 1),
			),
		)
	}

	// Check second chunk data (next 12 elements: 13-24)
	for i in 0 ..< 12 {
		testing.expect(
			t,
			chunks[1].data[i] == f32(i + 13),
			fmt.tprintf(
				"Second chunk data at %d incorrect: got %f, expected %f",
				i,
				chunks[1].data[i],
				f32(i + 13),
			),
		)
	}
}

@(test)
test_cat :: proc(t: ^testing.T) {
	// Create two 2x2 tensors to concatenate
	data1 := []f32{1, 2, 3, 4}
	data2 := []f32{5, 6, 7, 8}

	tensor1 := new_with_init(data1, []uint{1, 2, 2}, context.temp_allocator)
	tensor2 := new_with_init(data2, []uint{1, 2, 2}, context.temp_allocator)
	defer free_tensor(tensor1, context.temp_allocator)
	defer free_tensor(tensor2, context.temp_allocator)

	tensors := []^Tensor(f32){tensor1, tensor2}

	// Concatenate along dimension 0 (batch dimension)
	result := cat(tensors, 0, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected shape: [2, 2, 2]
	expected_shape := []uint{2, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Cat result shape incorrect")

	// Expected data: [1, 2, 3, 4, 5, 6, 7, 8]
	expected_data := []f32{1, 2, 3, 4, 5, 6, 7, 8}
	for i in 0 ..< len(expected_data) {
		testing.expect(
			t,
			result.data[i] == expected_data[i],
			fmt.tprintf(
				"Cat result data at %d incorrect: got %f, expected %f",
				i,
				result.data[i],
				expected_data[i],
			),
		)
	}
}
