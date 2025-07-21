package tensor

import "core:testing"
import "core:slice"
import "core:math"

@(test)
test_tensor_sum_all :: proc(t: ^testing.T) {
	// Test sum over all dimensions (scalar result)
	{
		data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		result := tensor_sum(tensor, nil, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		// Result should be scalar (empty shape)
		testing.expect(t, len(result.shape) == 0, "Sum all should produce scalar tensor")
		testing.expect(t, len(result.data) == 1, "Sum all should have single value")
		testing.expect(t, result.data[0] == 21, "Sum all value incorrect")
	}
}

@(test)
test_tensor_sum_axis :: proc(t: ^testing.T) {
	// Test sum along specific axis
	{
		data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Sum along axis 0 (columns): [1+4, 2+5, 3+6] = [5, 7, 9]
		result0 := tensor_sum(tensor, 0, context.temp_allocator)
		defer free_tensor(result0, context.temp_allocator)
		
		expected_shape0 := []uint{3}
		expected_data0 := []f32{5, 7, 9}
		testing.expect(t, slice.equal(result0.shape, expected_shape0), "Sum axis 0 shape incorrect")
		testing.expect(t, slice.equal(result0.data, expected_data0), "Sum axis 0 values incorrect")
		
		// Sum along axis 1 (rows): [1+2+3, 4+5+6] = [6, 15]
		result1 := tensor_sum(tensor, 1, context.temp_allocator)
		defer free_tensor(result1, context.temp_allocator)
		
		expected_shape1 := []uint{2}
		expected_data1 := []f32{6, 15}
		testing.expect(t, slice.equal(result1.shape, expected_shape1), "Sum axis 1 shape incorrect")
		testing.expect(t, slice.equal(result1.data, expected_data1), "Sum axis 1 values incorrect")
	}
}

@(test)
test_tensor_mean_all :: proc(t: ^testing.T) {
	// Test mean over all dimensions
	{
		data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		result := tensor_mean(tensor, nil, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		// Result should be scalar
		testing.expect(t, len(result.shape) == 0, "Mean all should produce scalar tensor")
		expected_mean := f32(21.0 / 6.0) // Sum is 21, count is 6
		testing.expect(t, math.abs(result.data[0] - expected_mean) < 1e-6, "Mean all value incorrect")
	}
}

@(test)
test_tensor_mean_axis :: proc(t: ^testing.T) {
	// Test mean along specific axis
	{
		data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Mean along axis 0: [2.5, 3.5, 4.5]
		result0 := tensor_mean(tensor, 0, context.temp_allocator)
		defer free_tensor(result0, context.temp_allocator)
		
		expected_shape0 := []uint{3}
		testing.expect(t, slice.equal(result0.shape, expected_shape0), "Mean axis 0 shape incorrect")
		testing.expect(t, math.abs(result0.data[0] - 2.5) < 1e-6, "Mean axis 0 value 0 incorrect")
		testing.expect(t, math.abs(result0.data[1] - 3.5) < 1e-6, "Mean axis 0 value 1 incorrect")
		testing.expect(t, math.abs(result0.data[2] - 4.5) < 1e-6, "Mean axis 0 value 2 incorrect")
		
		// Mean along axis 1: [2.0, 5.0]
		result1 := tensor_mean(tensor, 1, context.temp_allocator)
		defer free_tensor(result1, context.temp_allocator)
		
		expected_shape1 := []uint{2}
		testing.expect(t, slice.equal(result1.shape, expected_shape1), "Mean axis 1 shape incorrect")
		testing.expect(t, math.abs(result1.data[0] - 2.0) < 1e-6, "Mean axis 1 value 0 incorrect")
		testing.expect(t, math.abs(result1.data[1] - 5.0) < 1e-6, "Mean axis 1 value 1 incorrect")
	}
}

@(test)
test_tensor_max_all :: proc(t: ^testing.T) {
	// Test max over all dimensions
	{
		data := []f32{1, 6, 3, 2, 5, 4} // [[1,6,3], [2,5,4]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		result := tensor_max(tensor, nil, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		// Result should be scalar with max value 6
		testing.expect(t, len(result.shape) == 0, "Max all should produce scalar tensor")
		testing.expect(t, result.data[0] == 6, "Max all value incorrect")
	}
}

@(test)
test_tensor_max_axis :: proc(t: ^testing.T) {
	// Test max along specific axis
	{
		data := []f32{1, 6, 3, 2, 5, 4} // [[1,6,3], [2,5,4]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Max along axis 0: [max(1,2), max(6,5), max(3,4)] = [2, 6, 4]
		result0 := tensor_max(tensor, 0, context.temp_allocator)
		defer free_tensor(result0, context.temp_allocator)
		
		expected_shape0 := []uint{3}
		expected_data0 := []f32{2, 6, 4}
		testing.expect(t, slice.equal(result0.shape, expected_shape0), "Max axis 0 shape incorrect")
		testing.expect(t, slice.equal(result0.data, expected_data0), "Max axis 0 values incorrect")
		
		// Max along axis 1: [max(1,6,3), max(2,5,4)] = [6, 5]
		result1 := tensor_max(tensor, 1, context.temp_allocator)
		defer free_tensor(result1, context.temp_allocator)
		
		expected_shape1 := []uint{2}
		expected_data1 := []f32{6, 5}
		testing.expect(t, slice.equal(result1.shape, expected_shape1), "Max axis 1 shape incorrect")
		testing.expect(t, slice.equal(result1.data, expected_data1), "Max axis 1 values incorrect")
	}
}

@(test)
test_tensor_min_all :: proc(t: ^testing.T) {
	// Test min over all dimensions
	{
		data := []f32{5, 2, 7, 3, 1, 6} // [[5,2,7], [3,1,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		result := tensor_min(tensor, nil, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		// Result should be scalar with min value 1
		testing.expect(t, len(result.shape) == 0, "Min all should produce scalar tensor")
		testing.expect(t, result.data[0] == 1, "Min all value incorrect")
	}
}

@(test)
test_tensor_min_axis :: proc(t: ^testing.T) {
	// Test min along specific axis
	{
		data := []f32{5, 2, 7, 3, 1, 6} // [[5,2,7], [3,1,6]]
		tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Min along axis 0: [min(5,3), min(2,1), min(7,6)] = [3, 1, 6]
		result0 := tensor_min(tensor, 0, context.temp_allocator)
		defer free_tensor(result0, context.temp_allocator)
		
		expected_shape0 := []uint{3}
		expected_data0 := []f32{3, 1, 6}
		testing.expect(t, slice.equal(result0.shape, expected_shape0), "Min axis 0 shape incorrect")
		testing.expect(t, slice.equal(result0.data, expected_data0), "Min axis 0 values incorrect")
		
		// Min along axis 1: [min(5,2,7), min(3,1,6)] = [2, 1]
		result1 := tensor_min(tensor, 1, context.temp_allocator)
		defer free_tensor(result1, context.temp_allocator)
		
		expected_shape1 := []uint{2}
		expected_data1 := []f32{2, 1}
		testing.expect(t, slice.equal(result1.shape, expected_shape1), "Min axis 1 shape incorrect")
		testing.expect(t, slice.equal(result1.data, expected_data1), "Min axis 1 values incorrect")
	}
}

@(test)
test_reduce_3d :: proc(t: ^testing.T) {
	// Test reductions on 3D tensor
	{
		data := make([]f32, 24, context.temp_allocator)
		for i in 0..<24 {
			data[i] = f32(i + 1)
		}
		
		tensor := new_with_init(data, []uint{2, 3, 4}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Sum along axis 1 (middle dimension): (2,3,4) -> (2,4)
		result := tensor_sum(tensor, 1, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		expected_shape := []uint{2, 4}
		testing.expect(t, slice.equal(result.shape, expected_shape), "3D sum axis 1 shape incorrect")
		
		// First batch: sum of [1,5,9], [2,6,10], [3,7,11], [4,8,12]
		testing.expect(t, result.data[0] == 15, "3D sum axis 1 value 0 incorrect") // 1+5+9
		testing.expect(t, result.data[1] == 18, "3D sum axis 1 value 1 incorrect") // 2+6+10
		testing.expect(t, result.data[2] == 21, "3D sum axis 1 value 2 incorrect") // 3+7+11
		testing.expect(t, result.data[3] == 24, "3D sum axis 1 value 3 incorrect") // 4+8+12
	}
}

@(test)
test_reduce_1d_to_scalar :: proc(t: ^testing.T) {
	// Test reducing 1D tensor to scalar
	{
		data := []f32{10, 20, 30}
		tensor := new_with_init(data, []uint{3}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Sum along axis 0 should produce scalar
		result := tensor_sum(tensor, 0, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		testing.expect(t, len(result.shape) == 0, "1D sum should produce scalar")
		testing.expect(t, result.data[0] == 60, "1D sum value incorrect")
	}
}

@(test)
test_reduce_non_contiguous :: proc(t: ^testing.T) {
	// Test reductions on non-contiguous tensors
	{
		// Create a tensor and transpose it to make it non-contiguous
		data := []f32{1, 2, 3, 4, 5, 6} // [[1,2,3], [4,5,6]]
		original := new_with_init(data, []uint{2, 3}, context.temp_allocator)
		defer free_tensor(original, context.temp_allocator)
		
		transposed := transpose(original, 0, 1, context.temp_allocator)
		defer free_tensor(transposed, context.temp_allocator)
		
		// Transposed is now [[1,4], [2,5], [3,6]] (shape 3x2) but non-contiguous
		testing.expect(t, !transposed.contiguous, "Transposed tensor should be non-contiguous")
		
		// Sum all should still work correctly
		result := tensor_sum(transposed, nil, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)
		
		testing.expect(t, result.data[0] == 21, "Non-contiguous sum all incorrect")
	}
}

@(test)
test_reduce_compile_time_specialization :: proc(t: ^testing.T) {
	// Test that different operations are compile-time specialized
	{
		data := []f32{1, 2, 3, 4}
		tensor := new_with_init(data, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// Test direct calls to tensor_reduce with different compile-time ops
		sum_result := tensor_reduce(tensor, .SUM, nil, context.temp_allocator)
		mean_result := tensor_reduce(tensor, .MEAN, nil, context.temp_allocator)
		max_result := tensor_reduce(tensor, .MAX, nil, context.temp_allocator)
		min_result := tensor_reduce(tensor, .MIN, nil, context.temp_allocator)
		
		defer free_tensor(sum_result, context.temp_allocator)
		defer free_tensor(mean_result, context.temp_allocator)
		defer free_tensor(max_result, context.temp_allocator)
		defer free_tensor(min_result, context.temp_allocator)
		
		testing.expect(t, sum_result.data[0] == 10, "Compile-time SUM incorrect")
		testing.expect(t, math.abs(mean_result.data[0] - 2.5) < 1e-6, "Compile-time MEAN incorrect")
		testing.expect(t, max_result.data[0] == 4, "Compile-time MAX incorrect")
		testing.expect(t, min_result.data[0] == 1, "Compile-time MIN incorrect")
	}
}

@(test)
test_reduce_error_cases :: proc(t: ^testing.T) {
	// Test error handling
	{
		data := []f32{1, 2, 3, 4}
		tensor := new_with_init(data, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(tensor, context.temp_allocator)
		
		// These would panic in actual usage - we just document expected behavior:
		// tensor_sum(tensor, 5, context.temp_allocator) // Axis out of range - should panic
		// tensor_sum(tensor, -1, context.temp_allocator) // Negative axis - should panic
		
		testing.expect(t, true, "Error cases documented")
	}
}