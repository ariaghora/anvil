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
	original_strides: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> []uint {
	assert(len(original_shape) == len(original_strides))

	target_rank := len(target_shape)
	original_rank := len(original_shape)

	result_strides := make([]uint, target_rank, allocator, loc)

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

// Generic elementwise binary operation with broadcasting
elementwise_binary_op :: proc(
	a, b: ^Tensor($T),
	op: proc(a, b: T) -> T,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Check if shapes are broadcastable
	result_shape, broadcastable := shape_broadcastable(a.shape, b.shape, allocator, loc)
	if !broadcastable {
		panic("Tensors cannot be broadcasted together")
	}
	defer delete(result_shape, allocator, loc)

	// Create result tensor
	result := tensor_alloc(T, result_shape, allocator, loc)
	// Strides are already computed in tensor_alloc

	// Compute broadcasted strides for both tensors
	a_strides := broadcast_strides(a.shape, result_shape, a.strides, context.temp_allocator)
	defer delete(a_strides, context.temp_allocator)
	
	b_strides := broadcast_strides(b.shape, result_shape, b.strides, context.temp_allocator)
	defer delete(b_strides, context.temp_allocator)

	// Perform elementwise operation
	total_elements := shape_to_size(result_shape)
	for i in 0 ..< total_elements {
		// Convert linear index to multidimensional indices
		indices := make([]uint, len(result_shape), context.temp_allocator)
		defer delete(indices, context.temp_allocator)
		
		temp_i := i
		for dim := len(result_shape) - 1; dim >= 0; dim -= 1 {
			indices[dim] = temp_i % result_shape[dim]
			temp_i /= result_shape[dim]
		}

		// Compute linear indices for both tensors using broadcasted strides
		a_idx := compute_linear_index(indices, a_strides)
		b_idx := compute_linear_index(indices, b_strides)

		// Apply operation
		result.data[i] = op(a.data[a_idx], b.data[b_idx])
	}

	return result
}

// Addition operation
tensor_add :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	add_op :: proc(x, y: T) -> T { return x + y }
	return elementwise_binary_op(a, b, add_op, allocator, loc)
}

// Multiplication operation
tensor_multiply :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	mul_op :: proc(x, y: T) -> T { return x * y }
	return elementwise_binary_op(a, b, mul_op, allocator, loc)
}

