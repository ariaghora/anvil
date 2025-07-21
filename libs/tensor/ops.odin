package tensor

import "core:log"
import "core:math"
import "core:mem"
import "core:slice"

shape_broadcastable :: proc(
	shape_a, shape_b: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	[]uint,
	bool,
) {
	// Determine the maximum rank (number of dimensions)
	max_rank := max(len(shape_a), len(shape_b))
	if max_rank == 0 {
		return {}, true
	}

	// Allocate result shape
	result_shape := make([]uint, max_rank, allocator, loc)

	// Process dimensions from right to left (reverse order)
	for i in 0 ..< max_rank {
		// Get dimensions, using 1 if shape doesn't have this dimension
		dim_a: uint = 1
		dim_b: uint = 1

		if i < len(shape_a) {
			dim_a = shape_a[len(shape_a) - 1 - i]
		}
		if i < len(shape_b) {
			dim_b = shape_b[len(shape_b) - 1 - i]
		}

		// Check broadcasting compatibility
		if dim_a == dim_b {
			result_shape[max_rank - 1 - i] = dim_a
		} else if dim_a == 1 {
			result_shape[max_rank - 1 - i] = dim_b
		} else if dim_b == 1 {
			result_shape[max_rank - 1 - i] = dim_a
		} else {
			// Incompatible dimensions
			delete(result_shape, allocator, loc)
			return {}, false
		}
	}

	return result_shape, true
}

broadcast_strides :: proc(
	original_shape: []uint,
	target_shape: []uint,
	original_strides: []int,
	allocator := context.allocator,
	loc := #caller_location,
) -> []int {
	assert(len(original_shape) == len(original_strides))

	target_rank := len(target_shape)
	original_rank := len(original_shape)

	result_strides := make([]int, target_rank, allocator, loc)

	// Process from right to left
	for i in 0 ..< target_rank {
		target_dim := target_shape[target_rank - 1 - i]

		if i < original_rank {
			original_dim := original_shape[original_rank - 1 - i]
			original_stride := original_strides[original_rank - 1 - i]

			if original_dim == target_dim && target_dim != 1 {
				// Same dimension and not size 1, keep original stride
				result_strides[target_rank - 1 - i] = original_stride
			} else if original_dim == 1 || target_dim == 1 {
				// Broadcasting: stride becomes 0 (no movement in memory)
				result_strides[target_rank - 1 - i] = 0
			} else {
				// This should never happen if shape_broadcastable returned true
				panic("Invalid broadcast: dimension mismatch")
			}
		} else {
			// New leading dimension (original shape was shorter)
			// Stride is 0 since we repeat the entire original tensor
			result_strides[target_rank - 1 - i] = 0
		}
	}

	return result_strides
}

import "core:testing"

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
		original_strides := []int{4, 4}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []int{4, 0}
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
		original_strides := []int{1}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []int{0, 0, 1}
		testing.expect(t, slice.equal(result, expected), "Leading dimensions should have stride 0")
	}

	// No broadcasting needed
	{
		original_shape := []uint{2, 3, 4}
		target_shape := []uint{2, 3, 4}
		original_strides := []int{12, 4, 1}
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
		original_strides := []int{}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []int{0, 0}
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
		original_strides := []int{12, 4, 4, 1}
		result := broadcast_strides(
			original_shape,
			target_shape,
			original_strides,
			context.temp_allocator,
		)
		expected := []int{0, 0, 4, 0, 1}
		testing.expect(
			t,
			slice.equal(result, expected),
			"Mixed broadcasting should work correctly",
		)
	}
}
