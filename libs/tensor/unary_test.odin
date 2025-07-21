package tensor

import "core:math"
import "core:slice"
import "core:testing"

@(test)
test_unary_operations :: proc(t: ^testing.T) {
	// Float data for all operations
	float_data := []f32{-2, -1, 0, 1, 2}
	float_tensor := new_with_init(float_data, []uint{5}, context.temp_allocator)
	defer free_tensor(float_tensor, context.temp_allocator)

	// Test negation
	neg_result := tensor_neg(float_tensor, context.temp_allocator)
	defer free_tensor(neg_result, context.temp_allocator)
	expected_neg := []f32{2, 1, 0, -1, -2}
	testing.expect(t, slice.equal(neg_result.data, expected_neg), "Negation incorrect")

	// Test ReLU
	relu_result := tensor_relu(float_tensor, context.temp_allocator)
	defer free_tensor(relu_result, context.temp_allocator)
	expected_relu := []f32{0, 0, 0, 1, 2}
	testing.expect(t, slice.equal(relu_result.data, expected_relu), "ReLU incorrect")

	// Test GELU
	gelu_result := tensor_gelu(float_tensor, context.temp_allocator)
	defer free_tensor(gelu_result, context.temp_allocator)
	testing.expect(t, math.abs(gelu_result.data[2]) < 1e-6, "GELU(0) should be ~0")
	testing.expect(t, gelu_result.data[4] > gelu_result.data[3], "GELU should be increasing")

	// Test integer types work for NEG/RELU
	int_data := []i32{-1, 0, 1}
	int_tensor := new_with_init(int_data, []uint{3}, context.temp_allocator)
	defer free_tensor(int_tensor, context.temp_allocator)

	int_neg := tensor_neg(int_tensor, context.temp_allocator)
	int_relu := tensor_relu(int_tensor, context.temp_allocator)
	defer free_tensor(int_neg, context.temp_allocator)
	defer free_tensor(int_relu, context.temp_allocator)

	testing.expect(t, int_neg.data[0] == 1, "Integer negation failed")
	testing.expect(t, int_relu.data[2] == 1, "Integer ReLU failed")
}

@(test)
test_unary_non_contiguous :: proc(t: ^testing.T) {
	data := []f32{-2, -1, 0, 1, 2, 3}
	original := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(original, context.temp_allocator)

	transposed := transpose(original, 0, 1, context.temp_allocator)
	defer free_tensor(transposed, context.temp_allocator)
	testing.expect(t, !transposed.contiguous, "Should be non-contiguous")

	result := tensor_neg(transposed, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)
	testing.expect(t, result.contiguous, "Result should be contiguous")
	testing.expect(t, result.data[0] == 2, "Non-contiguous negation failed")
}

@(test)
test_gelu_type_safety :: proc(t: ^testing.T) {
	data := []f32{0, 1}
	tensor := new_with_init(data, []uint{2}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	result := tensor_gelu(tensor, context.temp_allocator)
	defer free_tensor(result, context.temp_allocator)
	testing.expect(t, math.abs(result.data[0]) < 1e-6, "GELU type constraint works")
}

@(test)
test_shape_preservation :: proc(t: ^testing.T) {
	data := []f32{1, 2, 3, 4, 5, 6}
	tensor := new_with_init(data, []uint{2, 3}, context.temp_allocator)
	defer free_tensor(tensor, context.temp_allocator)

	neg_result := tensor_neg(tensor, context.temp_allocator)
	relu_result := tensor_relu(tensor, context.temp_allocator)
	gelu_result := tensor_gelu(tensor, context.temp_allocator)

	defer free_tensor(neg_result, context.temp_allocator)
	defer free_tensor(relu_result, context.temp_allocator)
	defer free_tensor(gelu_result, context.temp_allocator)

	testing.expect(t, slice.equal(neg_result.shape, tensor.shape), "NEG shape mismatch")
	testing.expect(t, slice.equal(relu_result.shape, tensor.shape), "RELU shape mismatch")
	testing.expect(t, slice.equal(gelu_result.shape, tensor.shape), "GELU shape mismatch")
}
