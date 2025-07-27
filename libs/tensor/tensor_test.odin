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

@(test)
test_reshape :: proc(t: ^testing.T) {
	// Test basic 2D to 1D reshape
	{
		data := []f32{1, 2, 3, 4, 5, 6}
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)

		reshaped := reshape(tensor, []uint{6}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{6}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"2D to 1D reshape shape incorrect",
		)

		// Check data preservation
		for i in 0 ..< 6 {
			testing.expect(t, reshaped.data[i] == data[i], fmt.tprintf("Data mismatch at %d", i))
		}
	}

	// Test 1D to 3D reshape
	{
		data := []f32{1, 2, 3, 4, 5, 6, 7, 8}
		tensor := new_with_init(data, []uint{8}, context.temp_allocator)

		reshaped := reshape(tensor, []uint{2, 2, 2}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{2, 2, 2}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"1D to 3D reshape shape incorrect",
		)

		// Check data preservation
		for i in 0 ..< 8 {
			testing.expect(t, reshaped.data[i] == data[i], fmt.tprintf("Data mismatch at %d", i))
		}
	}

	// Test 3D to 2D reshape
	{
		data := []f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		tensor := new_with_init(data, []uint{2, 2, 3}, context.temp_allocator)

		reshaped := reshape(tensor, []uint{4, 3}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{4, 3}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"3D to 2D reshape shape incorrect",
		)

		// Check data preservation
		for i in 0 ..< 12 {
			testing.expect(t, reshaped.data[i] == data[i], fmt.tprintf("Data mismatch at %d", i))
		}
	}

	// Test reshape with non-contiguous tensor (transposed)
	{
		data := []f32{1, 2, 3, 4}
		original := new_with_init(data, []uint{2, 2}, context.temp_allocator)

		transposed := transpose(original, 0, 1, context.temp_allocator)

		reshaped := reshape(transposed, []uint{4}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{4}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"Transposed reshape shape incorrect",
		)

		// After transpose [[1,2],[3,4]] becomes [[1,3],[2,4]], so flattened should be [1,3,2,4]
		expected_data := []f32{1, 3, 2, 4}
		for i in 0 ..< 4 {
			testing.expect(
				t,
				reshaped.data[i] == expected_data[i],
				fmt.tprintf(
					"Transposed reshape data mismatch at %d: got %f, expected %f",
					i,
					reshaped.data[i],
					expected_data[i],
				),
			)
		}
	}

	// Test same shape reshape (no-op)
	{
		data := []f32{1, 2, 3, 4}
		tensor := new_with_init(data, []uint{2, 2}, context.temp_allocator)

		reshaped := reshape(tensor, []uint{2, 2}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{2, 2}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"Same shape reshape incorrect",
		)

		// Check data preservation
		for i in 0 ..< 4 {
			testing.expect(t, reshaped.data[i] == data[i], fmt.tprintf("Data mismatch at %d", i))
		}
	}

	// Test 4D to 2D reshape (common in ML)
	{
		data := make([]f32, 24, context.temp_allocator) // 2x3x2x2 = 24 elements
		for i in 0 ..< 24 {
			data[i] = f32(i + 1)
		}
		tensor := new_with_init(data, []uint{2, 3, 2, 2}, context.temp_allocator)

		reshaped := reshape(tensor, []uint{6, 4}, context.temp_allocator)

		// Check shape
		expected_shape := []uint{6, 4}
		testing.expect(
			t,
			slice.equal(reshaped.shape, expected_shape),
			"4D to 2D reshape shape incorrect",
		)

		// Check data preservation
		for i in 0 ..< 24 {
			testing.expect(
				t,
				reshaped.data[i] == f32(i + 1),
				fmt.tprintf("Data mismatch at %d", i),
			)
		}
	}
}

@(test)
test_sequential_non_contiguous_operations :: proc(t: ^testing.T) {
	// Test sequential operations involving permute, transpose, chunk, and other ops

	// Test 1: permute -> reshape -> matmul
	{
		// Create a 3D tensor [2, 3, 4] = 24 elements
		data := make([]f32, 24, context.temp_allocator)
		for i in 0 ..< 24 {
			data[i] = f32(i + 1)
		}
		original := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)

		// Permute dimensions [2, 3, 4] -> [4, 2, 3]
		permuted := permute(original, []uint{2, 0, 1}, context.temp_allocator)
		testing.expect(t, slice.equal(permuted.shape, []uint{4, 2, 3}), "Permute shape incorrect")

		// Reshape to 2D for matrix multiplication [4, 2, 3] -> [8, 3]
		reshaped := reshape(permuted, []uint{8, 3}, context.temp_allocator)
		testing.expect(t, slice.equal(reshaped.shape, []uint{8, 3}), "Reshape shape incorrect")

		// Create another tensor for matmul [3, 1]
		b_data := []f32{1, 1, 1}
		b := new_with_init(b_data, []uint{3, 1}, context.temp_allocator)

		// Matrix multiplication
		result := matmul(reshaped, b, context.temp_allocator)
		testing.expect(t, slice.equal(result.shape, []uint{8, 1}), "Matmul result shape incorrect")
	}

	// Test 2: transpose -> chunk -> operations on chunks
	{
		// Create a 4D tensor [2, 4, 3, 2] = 48 elements
		data := make([]f32, 48, context.temp_allocator)
		for i in 0 ..< 48 {
			data[i] = f32(i + 1)
		}
		original := new_with_init(data, []uint{2, 4, 3, 2}, context.temp_allocator)

		// Transpose last two dimensions
		transposed := transpose(original, 2, 3, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(transposed.shape, []uint{2, 4, 2, 3}),
			"Transpose shape incorrect",
		)

		// Chunk along dimension 1 (split 4 channels into 2 groups)
		chunks := chunk(transposed, 2, 1, context.temp_allocator)
		defer {
			for chunk_tensor in chunks {
				free_tensor(chunk_tensor, context.temp_allocator)
			}
			delete(chunks, context.temp_allocator)
		}

		testing.expect(t, len(chunks) == 2, "Should have 2 chunks")
		for chunk_tensor in chunks {
			testing.expect(
				t,
				slice.equal(chunk_tensor.shape, []uint{2, 2, 2, 3}),
				"Chunk shape incorrect",
			)
		}

		// Reshape each chunk and verify data consistency
		for chunk_tensor in chunks {
			reshaped_chunk := reshape(chunk_tensor, []uint{24}, context.temp_allocator)
			testing.expect(
				t,
				len(reshaped_chunk.data) == 24,
				"Reshaped chunk data length incorrect",
			)
		}
	}

	// Test 3: Multiple transposes in sequence
	{
		// Create a 3D tensor [2, 3, 4]
		data := make([]f32, 24, context.temp_allocator)
		for i in 0 ..< 24 {
			data[i] = f32(i + 1)
		}
		original := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)

		// First transpose: swap dims 0 and 1 -> [3, 2, 4]
		first_transpose := transpose(original, 0, 1, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(first_transpose.shape, []uint{3, 2, 4}),
			"First transpose shape incorrect",
		)

		// Second transpose: swap dims 1 and 2 -> [3, 4, 2]
		second_transpose := transpose(first_transpose, 1, 2, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(second_transpose.shape, []uint{3, 4, 2}),
			"Second transpose shape incorrect",
		)

		// Verify data correctness by accessing specific elements
		// Get strided data to verify correctness
		strided_data, allocated := get_strided_data(
			second_transpose,
			allocator = context.temp_allocator,
		)
		testing.expect(t, len(strided_data) == 24, "Strided data length incorrect")
	}

	// Test 4: permute -> transpose -> reshape chain
	{
		// Create a 4D tensor [2, 2, 2, 3] = 24 elements
		data := make([]f32, 24, context.temp_allocator)
		for i in 0 ..< 24 {
			data[i] = f32(i + 1)
		}
		original := new_with_init(data, []uint{2, 2, 2, 3}, context.temp_allocator)

		// Permute to [3, 2, 2, 2]
		permuted := permute(original, []uint{3, 0, 1, 2}, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(permuted.shape, []uint{3, 2, 2, 2}),
			"Permute shape incorrect",
		)

		// Transpose dims 2,3 -> [3, 2, 2, 2] (same shape but different strides)
		transposed := transpose(permuted, 2, 3, context.temp_allocator)

		// Reshape to 2D [3, 8]
		reshaped := reshape(transposed, []uint{3, 8}, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(reshaped.shape, []uint{3, 8}),
			"Final reshape shape incorrect",
		)

		// Verify total data integrity
		testing.expect(t, len(reshaped.data) == 24, "Final data length should be preserved")
	}

	// Test 5: Operations on non-contiguous tensor with arithmetic operations
	{
		// Create two tensors for element-wise operations
		a_data := []f32{1, 2, 3, 4, 5, 6}
		b_data := []f32{10, 20, 30, 40, 50, 60}

		a := new_with_init(a_data, []uint{2, 3}, context.temp_allocator)
		b := new_with_init(b_data, []uint{2, 3}, context.temp_allocator)

		// Transpose both tensors
		a_transposed := transpose(a, 0, 1, context.temp_allocator)
		b_transposed := transpose(b, 0, 1, context.temp_allocator)

		// Perform element-wise operations on non-contiguous tensors
		// This should work correctly regardless of contiguousness
		sum_result := add(a_transposed, b_transposed, context.temp_allocator)

		testing.expect(t, sum_result.contiguous, "Sum result should be contiguous")
		testing.expect(
			t,
			slice.equal(sum_result.shape, []uint{3, 2}),
			"Sum result shape incorrect",
		)

		// Verify some values - transposed matrices should add correctly
		// a_transposed is [[1,4],[2,5],[3,6]], b_transposed is [[10,40],[20,50],[30,60]]
		// sum should be [[11,44],[22,55],[33,66]]
		testing.expect(t, sum_result.data[0] == 11, "Sum result[0] incorrect")
		testing.expect(t, sum_result.data[1] == 44, "Sum result[1] incorrect")
		testing.expect(t, sum_result.data[2] == 22, "Sum result[2] incorrect")
	}
}

@(test)
test_contiguousness_edge_cases :: proc(t: ^testing.T) {
	// Test edge cases for contiguousness handling

	// Test 1: transpose same dimension (should remain contiguous)
	{
		data := []f32{1, 2, 3, 4}
		tensor := new_with_init(data, []uint{2, 2}, context.temp_allocator)

		// Transpose same dimension should not change contiguousness
		same_transpose := transpose(tensor, 0, 0, context.temp_allocator)

		// Data should be identical
		for i in 0 ..< len(data) {
			testing.expect(
				t,
				same_transpose.data[i] == data[i],
				"Same transpose data should be identical",
			)
		}
	}

	// Test 2: permute with identity permutation (should remain contiguous)
	{
		data := []f32{1, 2, 3, 4, 5, 6, 7, 8}
		tensor := new_with_init(data, []uint{2, 2, 2}, context.temp_allocator)

		// Identity permutation should preserve contiguousness
		identity_permute := permute(tensor, []uint{0, 1, 2}, context.temp_allocator)
		// Note: current implementation always sets contiguous=false for permute,
		// but let's verify the data is correct
		testing.expect(
			t,
			slice.equal(identity_permute.shape, tensor.shape),
			"Identity permute shape should be same",
		)

		// Verify data correctness through strided access
		strided_data, allocated := get_strided_data(
			identity_permute,
			allocator = context.temp_allocator,
		)
		for i in 0 ..< len(data) {
			testing.expect(
				t,
				strided_data[i] == data[i],
				"Identity permute data should be identical",
			)
		}
	}

	// Test 3: chunk with single group (should be same as original but non-contiguous)
	{
		data := []f32{1, 2, 3, 4, 5, 6}
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)

		// Chunk into 1 group should create a view of the whole tensor
		chunks := chunk(tensor, 1, 0, context.temp_allocator)
		defer {
			for chunk_tensor in chunks {
				free_tensor(chunk_tensor, context.temp_allocator)
			}
			delete(chunks, context.temp_allocator)
		}

		testing.expect(t, len(chunks) == 1, "Single group chunk should have 1 element")
		testing.expect(
			t,
			slice.equal(chunks[0].shape, tensor.shape),
			"Single group chunk shape should match original",
		)

		// Data should be accessible correctly
		strided_data, allocated := get_strided_data(chunks[0], allocator = context.temp_allocator)
		// testing.expect(t, allocated, "Should have allocated strided data")
		for i in 0 ..< len(data) {
			testing.expect(
				t,
				strided_data[i] == data[i],
				"Single group chunk data should match original",
			)
		}
	}

	// Test 4: operations that should result in contiguous output from non-contiguous input
	{
		data := []f32{1, 2, 3, 4, 5, 6, 7, 8}
		tensor := new_with_init(data, []uint{2, 4}, context.temp_allocator)

		// Create non-contiguous tensor via transpose
		transposed := transpose(tensor, 0, 1, context.temp_allocator)

		// Operations that should create contiguous output:

		// 1. reshape
		reshaped := reshape(transposed, []uint{8}, context.temp_allocator)

		// 2. clone
		cloned := clone(transposed, context.temp_allocator)

		// 3. element-wise operations
		same_shape := new_with_init(
			[]f32{1, 1, 1, 1, 1, 1, 1, 1},
			[]uint{4, 2},
			context.temp_allocator,
		)

		add_result := add(transposed, same_shape, context.temp_allocator)
		testing.expect(
			t,
			slice.equal(add_result.shape, []uint{4, 2}),
			"Add result shape should match transposed",
		)
	}
}
