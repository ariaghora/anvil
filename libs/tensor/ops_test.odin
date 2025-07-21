package tensor

import "core:testing"
import "core:slice"

@(test)
test_shape_broadcastable :: proc(t: ^testing.T) {
	// Compatible shapes
	{
		shape_a := []uint{3, 1, 5}
		shape_b := []uint{4, 1}
		result, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, ok, "Should be broadcastable")
		testing.expect(t, len(result) == 3, "Result should have 3 dimensions")
		testing.expect(t, result[0] == 3, "First dim should be 3")
		testing.expect(t, result[1] == 4, "Second dim should be 4")
		testing.expect(t, result[2] == 5, "Third dim should be 5")
	}

	// Same shapes
	{
		shape_a := []uint{2, 3, 4}
		shape_b := []uint{2, 3, 4}
		result, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, ok, "Same shapes should be broadcastable")
		testing.expect(t, len(result) == 3, "Result should have 3 dimensions")
		testing.expect(t, slice.equal(result, shape_a), "Result should equal input shape")
	}

	// Scalar broadcasting
	{
		shape_a := []uint{3, 4, 5}
		shape_b := []uint{}
		result, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, ok, "Scalar should broadcast to any shape")
		testing.expect(t, slice.equal(result, shape_a), "Result should equal non-scalar shape")
	}

	// Incompatible shapes
	{
		shape_a := []uint{3, 4}
		shape_b := []uint{2, 5}
		_, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, !ok, "Should not be broadcastable")
	}

	// One dimension is 1
	{
		shape_a := []uint{1}
		shape_b := []uint{5}
		result, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, ok, "1 should broadcast to 5")
		testing.expect(t, result[0] == 5, "Result should be [5]")
	}

	// Empty shapes
	{
		shape_a := []uint{}
		shape_b := []uint{}
		result, ok := shape_broadcastable(shape_a, shape_b, context.temp_allocator)
		testing.expect(t, ok, "Empty shapes should be compatible")
		testing.expect(t, len(result) == 0, "Result should be empty")
	}
}

@(test)
test_broadcast_strides :: proc(t: ^testing.T) {
	// Broadcasting dimension of size 1
	{
		original_shape := []uint{3, 1}
		target_shape := []uint{3, 5}
		original_strides := []uint{4, 4}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []uint{4, 0}
		testing.expect(
			t,
			slice.equal(result, expected),
			"Broadcasted dimension should have stride 0",
		)
	}

	// Adding leading dimensions
	{
		original_shape := []uint{4}
		target_shape := []uint{2, 3, 4}
		original_strides := []uint{1}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []uint{0, 0, 1}
		testing.expect(t, slice.equal(result, expected), "Leading dimensions should have stride 0")
	}

	// No broadcasting needed
	{
		original_shape := []uint{2, 3, 4}
		target_shape := []uint{2, 3, 4}
		original_strides := []uint{12, 4, 1}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		testing.expect(
			t,
			slice.equal(result, original_strides),
			"No broadcasting should preserve strides",
		)
	}

	// Scalar broadcasting
	{
		original_shape := []uint{}
		target_shape := []uint{2, 3}
		original_strides := []uint{}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []uint{0, 0}
		testing.expect(
			t,
			slice.equal(result, expected),
			"Scalar should broadcast with all strides 0",
		)
	}

	// Complex case: mix of broadcasting and non-broadcasting
	{
		original_shape := []uint{1, 3, 1, 4}
		target_shape := []uint{2, 1, 3, 5, 4}
		original_strides := []uint{12, 4, 4, 1}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []uint{0, 0, 4, 0, 1}
		testing.expect(
			t,
			slice.equal(result, expected),
			"Mixed broadcasting should work correctly",
		)
	}
}

@(test)
test_tensor_add :: proc(t: ^testing.T) {
	// Same shapes - no broadcasting needed
	{
		a := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{5, 6, 7, 8}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(b, context.temp_allocator)

		result := tensor_add(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{6, 8, 10, 12}
		testing.expect(t, slice.equal(result.data, expected), "Same shape addition failed")
		testing.expect(t, slice.equal(result.shape, []uint{2, 2}), "Result shape incorrect")
	}

	// Broadcasting: scalar + tensor
	{
		a := new_with_init([]f32{5}, []uint{}, context.temp_allocator)  // scalar
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{1, 2, 3}, []uint{3}, context.temp_allocator)
		defer free_tensor(b, context.temp_allocator)

		result := tensor_add(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{6, 7, 8}
		testing.expect(t, slice.equal(result.data, expected), "Scalar broadcasting addition failed")
		testing.expect(t, slice.equal(result.shape, []uint{3}), "Result shape incorrect")
	}

	// Broadcasting: different dimensions
	{
		a := new_with_init([]f32{1, 2}, []uint{2, 1}, context.temp_allocator)  // 2x1
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{10, 20, 30}, []uint{3}, context.temp_allocator)  // 3
		defer free_tensor(b, context.temp_allocator)

		result := tensor_add(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{11, 21, 31, 12, 22, 32}  // broadcast to 2x3
		testing.expect(t, slice.equal(result.data, expected), "Dimension broadcasting addition failed")
		testing.expect(t, slice.equal(result.shape, []uint{2, 3}), "Result shape incorrect")
	}
}

@(test)
test_tensor_multiply :: proc(t: ^testing.T) {
	// Same shapes - no broadcasting needed
	{
		a := new_with_init([]f32{2, 3, 4, 5}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{3, 4, 5, 6}, []uint{2, 2}, context.temp_allocator)
		defer free_tensor(b, context.temp_allocator)

		result := tensor_multiply(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{6, 12, 20, 30}
		testing.expect(t, slice.equal(result.data, expected), "Same shape multiplication failed")
		testing.expect(t, slice.equal(result.shape, []uint{2, 2}), "Result shape incorrect")
	}

	// Broadcasting: scalar * tensor
	{
		a := new_with_init([]f32{3}, []uint{}, context.temp_allocator)  // scalar
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{1, 2, 3}, []uint{3}, context.temp_allocator)
		defer free_tensor(b, context.temp_allocator)

		result := tensor_multiply(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{3, 6, 9}
		testing.expect(t, slice.equal(result.data, expected), "Scalar broadcasting multiplication failed")
		testing.expect(t, slice.equal(result.shape, []uint{3}), "Result shape incorrect")
	}

	// Broadcasting: matrix * vector
	{
		a := new_with_init([]f32{1, 2, 3, 4}, []uint{2, 2}, context.temp_allocator)  // 2x2
		defer free_tensor(a, context.temp_allocator)
		
		b := new_with_init([]f32{10, 20}, []uint{2}, context.temp_allocator)  // 2 (broadcasts to 1x2)
		defer free_tensor(b, context.temp_allocator)

		result := tensor_multiply(a, b, context.temp_allocator)
		defer free_tensor(result, context.temp_allocator)

		expected := []f32{10, 40, 30, 80}  // broadcast to 2x2
		testing.expect(t, slice.equal(result.data, expected), "Matrix-vector broadcasting multiplication failed")
		testing.expect(t, slice.equal(result.shape, []uint{2, 2}), "Result shape incorrect")
	}
}