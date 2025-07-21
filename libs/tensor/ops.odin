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

// Binary operation types for compile-time dispatch
BinaryOp :: enum {
	ADD,
	MULTIPLY,
	SUBTRACT,
	DIVIDE,
}

// Generic elementwise binary operation with broadcasting using compile-time enum dispatch
elementwise_binary_op :: proc(
	a, b: ^Tensor($T),
	$op: BinaryOp,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Fast path 1: Same shape and both contiguous (most common case)
	if slice.equal(a.shape, b.shape) && a.contiguous && b.contiguous {
		result := tensor_alloc(T, a.shape, allocator, loc)
		total_elements := len(a.data)
		
		for i in 0 ..< total_elements {
			switch op {
			case .ADD:
				result.data[i] = a.data[i] + b.data[i]
			case .MULTIPLY:
				result.data[i] = a.data[i] * b.data[i]
			case .SUBTRACT:
				result.data[i] = a.data[i] - b.data[i]
			case .DIVIDE:
				result.data[i] = a.data[i] / b.data[i]
			}
		}
		return result
	}

	// Fast path 2: Scalar broadcasting (b is scalar)
	if len(b.shape) == 0 {
		result := tensor_alloc(T, a.shape, allocator, loc)
		scalar_val := b.data[0]
		total_elements := len(a.data)

		if a.contiguous {
			// Contiguous case
			for i in 0 ..< total_elements {
				switch op {
				case .ADD:
					result.data[i] = a.data[i] + scalar_val
				case .MULTIPLY:
					result.data[i] = a.data[i] * scalar_val
				case .SUBTRACT:
					result.data[i] = a.data[i] - scalar_val
				case .DIVIDE:
					result.data[i] = a.data[i] / scalar_val
				}
			}
		} else {
			// Non-contiguous case
			shape_size := shape_to_size(a.shape)
			for i in 0 ..< shape_size {
				a_idx := compute_strided_index(a.shape, a.strides, i)
				switch op {
				case .ADD:
					result.data[i] = a.data[a_idx] + scalar_val
				case .MULTIPLY:
					result.data[i] = a.data[a_idx] * scalar_val
				case .SUBTRACT:
					result.data[i] = a.data[a_idx] - scalar_val
				case .DIVIDE:
					result.data[i] = a.data[a_idx] / scalar_val
				}
			}
		}
		return result
	}

	// Fast path 3: Same shape and same strides (but not necessarily contiguous)
	if slice.equal(a.shape, b.shape) && slice.equal(a.strides, b.strides) {
		result := tensor_alloc(T, a.shape, allocator, loc)
		total_elements := shape_to_size(a.shape)
		
		for i in 0 ..< total_elements {
			idx := compute_strided_index(a.shape, a.strides, i)
			switch op {
			case .ADD:
				result.data[i] = a.data[idx] + b.data[idx]
			case .MULTIPLY:
				result.data[i] = a.data[idx] * b.data[idx]
			case .SUBTRACT:
				result.data[i] = a.data[idx] - b.data[idx]
			case .DIVIDE:
				result.data[i] = a.data[idx] / b.data[idx]
			}
		}
		return result
	}

	// General case: Full broadcasting
	result_shape, broadcastable := shape_broadcastable(a.shape, b.shape, allocator, loc)
	if !broadcastable {
		panic("Tensors cannot be broadcasted together")
	}
	defer delete(result_shape, allocator, loc)

	result := tensor_alloc(T, result_shape, allocator, loc)
	a_strides := broadcast_strides(a.shape, result_shape, a.strides, context.temp_allocator)
	defer delete(a_strides, context.temp_allocator)
	
	b_strides := broadcast_strides(b.shape, result_shape, b.strides, context.temp_allocator)
	defer delete(b_strides, context.temp_allocator)

	total_elements := shape_to_size(result_shape)
	for i in 0 ..< total_elements {
		a_idx := compute_strided_index(result_shape, a_strides, i)
		b_idx := compute_strided_index(result_shape, b_strides, i)

		switch op {
		case .ADD:
			result.data[i] = a.data[a_idx] + b.data[b_idx]
		case .MULTIPLY:
			result.data[i] = a.data[a_idx] * b.data[b_idx]
		case .SUBTRACT:
			result.data[i] = a.data[a_idx] - b.data[b_idx]
		case .DIVIDE:
			result.data[i] = a.data[a_idx] / b.data[b_idx]
		}
	}

	return result
}

// Addition operation
tensor_add :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .ADD, allocator, loc)
}

// Multiplication operation
tensor_multiply :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .MULTIPLY, allocator, loc)
}

// Subtraction operation
tensor_subtract :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .SUBTRACT, allocator, loc)
}

// Division operation
tensor_divide :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .DIVIDE, allocator, loc)
}

