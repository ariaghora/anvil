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

// Reduction operation types for compile-time dispatch
ReduceOp :: enum {
	SUM,
	MEAN, 
	MAX,
	MIN,
}

// Generic tensor reduction with compile-time specialization
// If axis is nil, reduce all dimensions (return scalar)
// If axis is specified, reduce along that axis only
tensor_reduce :: proc(
	tensor: ^Tensor($T),
	$op: ReduceOp,
	axis: Maybe(int) = nil,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Handle reduction along all dimensions (result is scalar)
	if axis == nil {
		result := tensor_alloc(T, []uint{}, allocator, loc) // Scalar tensor
		total_elements := data_len(tensor)
		
		if total_elements == 0 {
			panic("Cannot reduce empty tensor")
		}
		
		// Initialize with appropriate identity value
		initial_value: T
		switch op {
		case .SUM, .MEAN:
			initial_value = T(0)
		case .MAX, .MIN:
			// For max/min, initialize with first element
			initial_value = tensor.data[0] if len(tensor.data) > 0 else T(0)
		}
		
		result_value := initial_value
		
		// Fast path for contiguous tensors
		if tensor.contiguous {
			switch op {
			case .SUM, .MEAN:
				for i in 0 ..< len(tensor.data) {
					result_value += tensor.data[i]
				}
				if op == .MEAN {
					result_value /= T(total_elements)
				}
			case .MAX:
				result_value = tensor.data[0]
				for i in 1 ..< len(tensor.data) {
					if tensor.data[i] > result_value {
						result_value = tensor.data[i]
					}
				}
			case .MIN:
				result_value = tensor.data[0]
				for i in 1 ..< len(tensor.data) {
					if tensor.data[i] < result_value {
						result_value = tensor.data[i]
					}
				}
			}
		} else {
			// Strided access for non-contiguous tensors
			switch op {
			case .SUM, .MEAN:
				for i in 0 ..< total_elements {
					strided_idx := compute_strided_index(tensor.shape, tensor.strides, i)
					result_value += tensor.data[strided_idx]
				}
				if op == .MEAN {
					result_value /= T(total_elements)
				}
			case .MAX:
				strided_idx := compute_strided_index(tensor.shape, tensor.strides, 0)
				result_value = tensor.data[strided_idx]
				for i in 1 ..< total_elements {
					strided_idx = compute_strided_index(tensor.shape, tensor.strides, i)
					if tensor.data[strided_idx] > result_value {
						result_value = tensor.data[strided_idx]
					}
				}
			case .MIN:
				strided_idx := compute_strided_index(tensor.shape, tensor.strides, 0)
				result_value = tensor.data[strided_idx]
				for i in 1 ..< total_elements {
					strided_idx = compute_strided_index(tensor.shape, tensor.strides, i)
					if tensor.data[strided_idx] < result_value {
						result_value = tensor.data[strided_idx]
					}
				}
			}
		}
		
		result.data[0] = result_value
		return result
	}
	
	// Handle reduction along specific axis
	axis_val := axis.(int)
	if axis_val < 0 || axis_val >= len(tensor.shape) {
		panic("Axis out of range")
	}
	
	// Compute result shape (remove the reduced dimension)
	result_shape := make([]uint, len(tensor.shape) - 1, allocator)
	result_idx := 0
	for i in 0 ..< len(tensor.shape) {
		if i != axis_val {
			result_shape[result_idx] = tensor.shape[i]
			result_idx += 1
		}
	}
	
	// Handle edge case: 1D tensor reduction results in scalar
	if len(result_shape) == 0 {
		result_shape = []uint{}
	}
	
	result := tensor_alloc(T, result_shape, allocator, loc)
	result_size := data_len(result)
	axis_size := tensor.shape[axis_val]
	
	// Initialize result values
	switch op {
	case .SUM, .MEAN:
		for i in 0 ..< len(result.data) {
			result.data[i] = T(0)
		}
	case .MAX, .MIN:
		// Will be set in the first iteration of the reduction loop
	}
	
	// Perform reduction
	for result_linear_idx in 0 ..< result_size {
		// Convert result linear index to multi-dimensional coordinates
		result_coords := make([]uint, len(result.shape), context.temp_allocator)
		defer delete(result_coords, context.temp_allocator)
		
		temp_idx := result_linear_idx
		for dim := len(result.shape) - 1; dim >= 0; dim -= 1 {
			if len(result.shape) > 0 {
				result_coords[dim] = temp_idx % result.shape[dim]
				temp_idx /= result.shape[dim]
			}
		}
		
		// Map result coordinates to input tensor coordinates
		input_coords := make([]uint, len(tensor.shape), context.temp_allocator)
		defer delete(input_coords, context.temp_allocator)
		
		input_coord_idx := 0
		for dim in 0 ..< len(tensor.shape) {
			if dim != axis_val {
				input_coords[dim] = result_coords[input_coord_idx]
				input_coord_idx += 1
			}
		}
		
		// Reduce along the specified axis
		switch op {
		case .SUM, .MEAN:
			sum_value: T = 0
			for axis_idx in 0 ..< axis_size {
				input_coords[axis_val] = axis_idx
				linear_idx := compute_linear_index(input_coords, tensor.strides)
				sum_value += tensor.data[linear_idx]
			}
			result.data[result_linear_idx] = sum_value
			if op == .MEAN {
				result.data[result_linear_idx] /= T(axis_size)
			}
			
		case .MAX:
			input_coords[axis_val] = 0
			linear_idx := compute_linear_index(input_coords, tensor.strides)
			max_value := tensor.data[linear_idx]
			for axis_idx in 1 ..< axis_size {
				input_coords[axis_val] = axis_idx
				linear_idx = compute_linear_index(input_coords, tensor.strides)
				if tensor.data[linear_idx] > max_value {
					max_value = tensor.data[linear_idx]
				}
			}
			result.data[result_linear_idx] = max_value
			
		case .MIN:
			input_coords[axis_val] = 0
			linear_idx := compute_linear_index(input_coords, tensor.strides)
			min_value := tensor.data[linear_idx]
			for axis_idx in 1 ..< axis_size {
				input_coords[axis_val] = axis_idx
				linear_idx = compute_linear_index(input_coords, tensor.strides)
				if tensor.data[linear_idx] < min_value {
					min_value = tensor.data[linear_idx]
				}
			}
			result.data[result_linear_idx] = min_value
		}
	}
	
	return result
}

// Convenience wrapper functions

// Sum reduction - reduce along all axes or specific axis
tensor_sum :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .SUM, axis, allocator, loc)
}

// Mean reduction - reduce along all axes or specific axis
tensor_mean :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MEAN, axis, allocator, loc)
}

// Max reduction - reduce along all axes or specific axis
tensor_max :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MAX, axis, allocator, loc)
}

// Min reduction - reduce along all axes or specific axis
tensor_min :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MIN, axis, allocator, loc)
}

