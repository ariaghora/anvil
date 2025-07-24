package tensor

import "../tensor"
import "core:slice"
import "core:testing"

@(test)
test_tensor_matmul_2d :: proc(t: ^testing.T) {
	a_data := []f32{1, 2, 3, 4} // [[1,2], [3,4]]
	b_data := []f32{5, 6, 7, 8, 9, 10} // [[5,6,7], [8,9,10]]

	a := new_with_init(a_data, []uint{2, 2}, context.temp_allocator)
	b := new_with_init(b_data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)

	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Expected: [[1*5+2*8, 1*6+2*9, 1*7+2*10], [3*5+4*8, 3*6+4*9, 3*7+4*10]]
	//         = [[21, 24, 27], [47, 54, 61]]
	expected := []f32{21, 24, 27, 47, 54, 61}
	expected_shape := []uint{2, 3}

	testing.expect(t, slice.equal(result.data, expected), "2D matmul values incorrect")
	testing.expect(t, slice.equal(result.shape, expected_shape), "2D matmul shape incorrect")
}

@(test)
test_tensor_matmul_3d_same_batch :: proc(t: ^testing.T) {
	// Batch 0: [[1,2], [3,4], [5,6]]  @ [[1,2,3,4], [5,6,7,8]]
	// Batch 1: [[7,8], [9,10], [11,12]] @ [[9,10,11,12], [13,14,15,16]]
	a_data := []f32 {
		1,
		2,
		3,
		4,
		5,
		6, // First batch
		7,
		8,
		9,
		10,
		11,
		12, // Second batch
	}
	b_data := []f32 {
		1,
		2,
		3,
		4,
		5,
		6,
		7,
		8, // First batch
		9,
		10,
		11,
		12,
		13,
		14,
		15,
		16, // Second batch
	}

	a := new_with_init(a_data, []uint{2, 3, 2}, context.temp_allocator)
	b := new_with_init(b_data, []uint{2, 2, 4}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)

	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	// Check shape
	expected_shape := []uint{2, 3, 4}
	testing.expect(t, slice.equal(result.shape, expected_shape), "3D same batch shape incorrect")

	// Check first few values (spot check)
	// Batch 0, row 0: [1,2] @ [[1,2,3,4],[5,6,7,8]] = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11,14,17,20]
	testing.expect(t, result.data[0] == 11, "3D batch matmul value 0 incorrect")
	testing.expect(t, result.data[1] == 14, "3D batch matmul value 1 incorrect")
	testing.expect(t, result.data[2] == 17, "3D batch matmul value 2 incorrect")
	testing.expect(t, result.data[3] == 20, "3D batch matmul value 3 incorrect")
}

@(test)
test_tensor_matmul_3d_broadcast :: proc(t: ^testing.T) {
	a_data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
	b_data := []f32 {
		1,
		2,
		3,
		4,
		5,
		6, // Batch 0: [[1,2], [3,4], [5,6]]
		7,
		8,
		9,
		10,
		11,
		12, // Batch 1: [[7,8], [9,10], [11,12]]
	}

	a := new_with_init(a_data, []uint{1, 2, 3}, context.temp_allocator)
	b := new_with_init(b_data, []uint{2, 3, 2}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)

	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	expected_shape := []uint{2, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "3D broadcast shape incorrect")

	// Verify broadcasting worked - both batches should use same A matrix
	// Batch 0: [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
	// Row 0: [1,2,3] @ [[1,2], [3,4], [5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	testing.expect(t, result.data[0] == 22, "3D broadcast batch 0 value 0 incorrect")
	testing.expect(t, result.data[1] == 28, "3D broadcast batch 0 value 1 incorrect")
}

@(test)
test_tensor_matmul_4d :: proc(t: ^testing.T) {
	a_data := []f32 {
		1,
		2,
		3,
		4, // Batch [0,0]: [[1,2], [3,4]]
		5,
		6,
		7,
		8, // Batch [1,0]: [[5,6], [7,8]]
	}
	b_data := []f32 {
		1,
		2,
		3,
		4, // Batch [0,0]: [[1,2], [3,4]]
		5,
		6,
		7,
		8, // Batch [0,1]: [[5,6], [7,8]]
		9,
		10,
		11,
		12, // Batch [0,2]: [[9,10], [11,12]]
	}

	a := new_with_init(a_data, []uint{2, 1, 2, 2}, context.temp_allocator)
	b := new_with_init(b_data, []uint{1, 3, 2, 2}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)

	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)

	expected_shape := []uint{2, 3, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "4D shape incorrect")

	// Spot check: batch [0,0] should be [[1,2],[3,4]] @ [[1,2],[3,4]]
	// = [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7,10], [15,22]]
	testing.expect(t, result.data[0] == 7, "4D batch [0,0] value 0 incorrect")
	testing.expect(t, result.data[1] == 10, "4D batch [0,0] value 1 incorrect")
	testing.expect(t, result.data[2] == 15, "4D batch [0,0] value 2 incorrect")
	testing.expect(t, result.data[3] == 22, "4D batch [0,0] value 3 incorrect")
}

@(test)
test_tensor_matmul_errors :: proc(t: ^testing.T) {
	// Test dimension validation
	{
		// 1D tensor should panic
		a_1d := new_with_init([]f32{1, 2}, []uint{2}, context.temp_allocator)
		b_2d := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(a_1d, context.temp_allocator)
		defer free_tensor(b_2d, context.temp_allocator)

		// Should panic - we can't easily test panics in Odin, so just document this
		// tensor_matmul(a_1d, b_2d, context.temp_allocator) // Should panic
	}

	// Test incompatible matrix dimensions
	{
		a := new_with_init([]f32{1, 2, 3}, []uint{1, 3}, context.temp_allocator)
		b := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(a, context.temp_allocator)
		defer free_tensor(b, context.temp_allocator)

		// Should panic due to dimension mismatch (3 != 2)
		// tensor_matmul(a, b, context.temp_allocator) // Should panic
	}
}

@(test)
test_tensor_matmul_higher_rank_first :: proc(t: ^testing.T) {
	// Test case where first tensor has higher rank than second tensor
	// a: shape [2, 3, 2, 3] - 4D tensor with batch dimensions [2, 3]
	// b: shape [3, 2] - 2D matrix
	// Expected result: [2, 3, 2, 2] - broadcasting b across batch dimensions
	
	a_data := []f32 {
		// Batch [0,0]: [[1,2,3], [4,5,6]]
		1, 2, 3,
		4, 5, 6,
		// Batch [0,1]: [[7,8,9], [10,11,12]]
		7, 8, 9,
		10, 11, 12,
		// Batch [0,2]: [[13,14,15], [16,17,18]]
		13, 14, 15,
		16, 17, 18,
		// Batch [1,0]: [[19,20,21], [22,23,24]]
		19, 20, 21,
		22, 23, 24,
		// Batch [1,1]: [[25,26,27], [28,29,30]]
		25, 26, 27,
		28, 29, 30,
		// Batch [1,2]: [[31,32,33], [34,35,36]]
		31, 32, 33,
		34, 35, 36,
	}
	
	b_data := []f32 {
		1, 2,  // [[1,2],
		3, 4,  //  [3,4],
		5, 6,  //  [5,6]]
	}
	
	a := new_with_init(a_data, []uint{2, 3, 2, 3}, context.temp_allocator)
	b := new_with_init(b_data, []uint{3, 2}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)
	
	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)
	
	// Expected shape: [2, 3, 2, 2]
	expected_shape := []uint{2, 3, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Higher rank first tensor shape incorrect")
	
	// Check first batch [0,0]: [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
	// Row 0: [1,2,3] @ [[1,2], [3,4], [5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	// Row 1: [4,5,6] @ [[1,2], [3,4], [5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
	testing.expect(t, result.data[0] == 22, "Higher rank batch [0,0] row 0 col 0 incorrect")
	testing.expect(t, result.data[1] == 28, "Higher rank batch [0,0] row 0 col 1 incorrect")
	testing.expect(t, result.data[2] == 49, "Higher rank batch [0,0] row 1 col 0 incorrect")
	testing.expect(t, result.data[3] == 64, "Higher rank batch [0,0] row 1 col 1 incorrect")
	
	// Check a different batch [1,2] to ensure broadcasting worked correctly
	// This should be at offset: 1*3*2*2 + 2*2*2 = 12 + 8 = 20
	// [[31,32,33], [34,35,36]] @ [[1,2], [3,4], [5,6]]
	// Row 0: [31,32,33] @ [[1,2], [3,4], [5,6]] = [31*1+32*3+33*5, 31*2+32*4+33*6] = [292, 358]
	// Let me double check: 31*1 + 32*3 + 33*5 = 31 + 96 + 165 = 292 ✓
	// 31*2 + 32*4 + 33*6 = 62 + 128 + 198 = 388 (not 358!)
	testing.expect(t, result.data[20] == 292, "Higher rank batch [1,2] row 0 col 0 incorrect")
	testing.expect(t, result.data[21] == 388, "Higher rank batch [1,2] row 0 col 1 incorrect")
}

@(test)
test_tensor_matmul_rank4_rank3 :: proc(t: ^testing.T) {
	// Test rank-4 @ rank-3: [2, 2, 3, 2] @ [2, 2, 4] -> [2, 2, 3, 4]
	// First tensor has batch dims [2, 2], second has batch dims [2]
	// Broadcasting: [2, 2] vs [2] -> [2, 2]
	
	a_data := []f32 {
		// Batch [0,0]: [[1,2], [3,4], [5,6]]
		1, 2,
		3, 4,
		5, 6,
		// Batch [0,1]: [[7,8], [9,10], [11,12]]
		7, 8,
		9, 10,
		11, 12,
		// Batch [1,0]: [[13,14], [15,16], [17,18]]
		13, 14,
		15, 16,
		17, 18,
		// Batch [1,1]: [[19,20], [21,22], [23,24]]
		19, 20,
		21, 22,
		23, 24,
	}
	
	b_data := []f32 {
		// Batch [0]: [[1,2,3,4], [5,6,7,8]]
		1, 2, 3, 4,
		5, 6, 7, 8,
		// Batch [1]: [[9,10,11,12], [13,14,15,16]]
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	
	a := new_with_init(a_data, []uint{2, 2, 3, 2}, context.temp_allocator)
	b := new_with_init(b_data, []uint{2, 2, 4}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)
	
	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)
	
	// Expected shape: [2, 2, 3, 4]
	expected_shape := []uint{2, 2, 3, 4}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Rank-4 @ rank-3 shape incorrect")
	
	// Check batch [0,0]: [[1,2], [3,4], [5,6]] @ [[1,2,3,4], [5,6,7,8]]
	// Row 0: [1,2] @ [[1,2,3,4], [5,6,7,8]] = [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
	testing.expect(t, result.data[0] == 11, "Rank-4@rank-3 batch [0,0] row 0 col 0 incorrect")
	testing.expect(t, result.data[1] == 14, "Rank-4@rank-3 batch [0,0] row 0 col 1 incorrect")
	testing.expect(t, result.data[2] == 17, "Rank-4@rank-3 batch [0,0] row 0 col 2 incorrect")
	testing.expect(t, result.data[3] == 20, "Rank-4@rank-3 batch [0,0] row 0 col 3 incorrect")
	
	// Check batch [1,1]: [[19,20], [21,22], [23,24]] @ [[9,10,11,12], [13,14,15,16]]
	// This should be at offset: 1*2*3*4 + 1*3*4 = 24 + 12 = 36
	// Row 0: [19,20] @ [[9,10,11,12], [13,14,15,16]] = [19*9+20*13, 19*10+20*14, 19*11+20*15, 19*12+20*16]
	// = [171+260, 190+280, 209+300, 228+320] = [431, 470, 509, 548]
	testing.expect(t, result.data[36] == 431, "Rank-4@rank-3 batch [1,1] row 0 col 0 incorrect")
}

@(test)
test_tensor_matmul_rank3_rank2 :: proc(t: ^testing.T) {
	// Test rank-3 @ rank-2: [3, 2, 3] @ [3, 2] -> [3, 2, 2]
	// First tensor has batch dims [3], second has no batch dims []
	// Broadcasting: [3] vs [] -> [3]
	
	a_data := []f32 {
		// Batch [0]: [[1,2,3], [4,5,6]]
		1, 2, 3,
		4, 5, 6,
		// Batch [1]: [[7,8,9], [10,11,12]]
		7, 8, 9,
		10, 11, 12,
		// Batch [2]: [[13,14,15], [16,17,18]]
		13, 14, 15,
		16, 17, 18,
	}
	
	b_data := []f32 {
		1, 2,  // [[1,2],
		3, 4,  //  [3,4],
		5, 6,  //  [5,6]]
	}
	
	a := new_with_init(a_data, []uint{3, 2, 3}, context.temp_allocator)
	b := new_with_init(b_data, []uint{3, 2}, context.temp_allocator)
	defer free_tensor(a, context.temp_allocator)
	defer free_tensor(b, context.temp_allocator)
	
	result := matmul(a, b, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)
	
	// Expected shape: [3, 2, 2]
	expected_shape := []uint{3, 2, 2}
	testing.expect(t, slice.equal(result.shape, expected_shape), "Rank-3 @ rank-2 shape incorrect")
	
	// Check batch [0]: [[1,2,3], [4,5,6]] @ [[1,2], [3,4], [5,6]]
	// Row 0: [1,2,3] @ [[1,2], [3,4], [5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	// Row 1: [4,5,6] @ [[1,2], [3,4], [5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
	testing.expect(t, result.data[0] == 22, "Rank-3@rank-2 batch [0] row 0 col 0 incorrect")
	testing.expect(t, result.data[1] == 28, "Rank-3@rank-2 batch [0] row 0 col 1 incorrect")
	testing.expect(t, result.data[2] == 49, "Rank-3@rank-2 batch [0] row 1 col 0 incorrect")
	testing.expect(t, result.data[3] == 64, "Rank-3@rank-2 batch [0] row 1 col 1 incorrect")
	
	// Check batch [2]: [[13,14,15], [16,17,18]] @ [[1,2], [3,4], [5,6]]
	// This should be at offset: 2*2*2 = 8
	// Row 0: [13,14,15] @ [[1,2], [3,4], [5,6]] = [13*1+14*3+15*5, 13*2+14*4+15*6]
	// = [13+42+75, 26+56+90] = [130, 172] (not 146!)
	testing.expect(t, result.data[8] == 130, "Rank-3@rank-2 batch [2] row 0 col 0 incorrect")
	testing.expect(t, result.data[9] == 172, "Rank-3@rank-2 batch [2] row 0 col 1 incorrect")
}

@(test)
test_tensor_matmul_noncontiguous :: proc(t: ^testing.T) {
	a := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, context.temp_allocator)
	b := new_with_init([]f32{1, 1, 2, 2}, []uint{2, 2}, context.temp_allocator)
	b = tensor.transpose(b, 0, 1, context.temp_allocator)
	c := tensor.matmul(a, b, context.temp_allocator)
	testing.expect(t, slice.equal(c.data, []f32{3, 6, 7, 14}))
}
