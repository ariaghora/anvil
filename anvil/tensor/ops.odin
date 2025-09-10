package tensor

import "../trace"
import "core:log"
import "core:math"
import "core:mem"
import "core:simd"
import "core:slice"

// Binary operations
Binary_Op :: enum {
	ADD,
	MULTIPLY,
	SUBTRACT,
	DIVIDE,
}

// Reduction operations
Reduce_Op :: enum {
	SUM,
	MEAN,
	MAX,
	MIN,
}

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

// Generic elementwise binary operation with broadcasting using compile-time enum dispatch
elementwise_binary_op :: proc(
	a, b: ^Tensor($T),
	$op: Binary_Op,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Fast path 1: Same shape and both contiguous (most common case)
	if slice.equal(a.shape, b.shape) && a.contiguous && b.contiguous {
		result := tensor_alloc(T, a.shape, true, allocator, loc)
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
		result := tensor_alloc(T, a.shape, true, allocator, loc)
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
		result := tensor_alloc(T, a.shape, true, allocator, loc)
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

	result := tensor_alloc(T, result_shape, true, allocator, loc)
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
add :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .ADD, allocator, loc)
}

// Multiplication operation
mul :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .MULTIPLY, allocator, loc)
}

// Subtraction operation
sub :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .SUBTRACT, allocator, loc)
}

// Division operation
div :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_binary_op(a, b, .DIVIDE, allocator, loc)
}

// Helper function for all-axis reduction
@(private)
tensor_reduce_all_axes :: proc(
	tensor: ^Tensor($T),
	$op: Reduce_Op,
	keepdims: bool,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	result_shape: []uint
	if keepdims {
		result_shape = make([]uint, len(tensor.shape), allocator)
		for i in 0 ..< len(result_shape) {
			result_shape[i] = 1
		}
	} else {
		result_shape = []uint{} // Scalar tensor
	}
	result := tensor_alloc(T, result_shape, true, allocator, loc)
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

// Helper function to compute result shape for single-axis reduction
@(private)
compute_reduction_shape :: proc(
	input_shape: []uint,
	axis: int,
	keepdims: bool,
	allocator := context.allocator,
) -> []uint {
	if keepdims {
		result_shape := make([]uint, len(input_shape), allocator)
		for i in 0 ..< len(input_shape) {
			if i == axis {
				result_shape[i] = 1
			} else {
				result_shape[i] = input_shape[i]
			}
		}
		return result_shape
	} else {
		result_shape := make([]uint, len(input_shape) - 1, allocator)
		result_idx := 0
		for i in 0 ..< len(input_shape) {
			if i != axis {
				result_shape[result_idx] = input_shape[i]
				result_idx += 1
			}
		}
		if len(result_shape) == 0 {
			return []uint{}
		}
		return result_shape
	}
}

// Helper function to map result coordinates to input coordinates
@(private)
map_coordinates :: proc(result_coords: []uint, input_coords: []uint, axis: int, keepdims: bool) {
	if keepdims {
		for dim in 0 ..< len(input_coords) {
			if dim != axis {
				input_coords[dim] = result_coords[dim]
			}
		}
	} else {
		input_coord_idx := 0
		for dim in 0 ..< len(input_coords) {
			if dim != axis {
				input_coords[dim] = result_coords[input_coord_idx]
				input_coord_idx += 1
			}
		}
	}
}

// Helper function for single-axis reduction
@(private)
tensor_reduce_single_axis :: proc(
	tensor: ^Tensor($T),
	$op: Reduce_Op,
	axis: int,
	keepdims: bool,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	if axis < 0 || axis >= len(tensor.shape) {
		panic("Axis out of range")
	}

	result_shape := compute_reduction_shape(tensor.shape, axis, keepdims, allocator)
	defer if !keepdims && len(result_shape) > 0 do delete(result_shape, allocator)

	result := tensor_alloc(T, result_shape, true, allocator, loc)
	result_size := data_len(result)
	axis_size := tensor.shape[axis]

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

		map_coordinates(result_coords, input_coords, axis, keepdims)

		// Reduce along the specified axis
		switch op {
		case .SUM, .MEAN:
			sum_value: T = 0
			for axis_idx in 0 ..< axis_size {
				input_coords[axis] = axis_idx
				linear_idx := compute_linear_index(input_coords, tensor.strides)
				sum_value += tensor.data[linear_idx]
			}
			result.data[result_linear_idx] = sum_value
			if op == .MEAN {
				result.data[result_linear_idx] /= T(axis_size)
			}

		case .MAX:
			input_coords[axis] = 0
			linear_idx := compute_linear_index(input_coords, tensor.strides)
			max_value := tensor.data[linear_idx]
			for axis_idx in 1 ..< axis_size {
				input_coords[axis] = axis_idx
				linear_idx = compute_linear_index(input_coords, tensor.strides)
				if tensor.data[linear_idx] > max_value {
					max_value = tensor.data[linear_idx]
				}
			}
			result.data[result_linear_idx] = max_value

		case .MIN:
			input_coords[axis] = 0
			linear_idx := compute_linear_index(input_coords, tensor.strides)
			min_value := tensor.data[linear_idx]
			for axis_idx in 1 ..< axis_size {
				input_coords[axis] = axis_idx
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

// Generic tensor reduction with compile-time specialization
// If axis is nil, reduce all dimensions (return scalar)
// If axis is specified, reduce along that axis only
// keepdims: if true, keep reduced dimensions as size 1
tensor_reduce :: proc(
	tensor: ^Tensor($T),
	$op: Reduce_Op,
	axis: Maybe(int) = nil,
	keepdims: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	if axis == nil {
		return tensor_reduce_all_axes(tensor, op, keepdims, allocator, loc)
	} else {
		return tensor_reduce_single_axis(tensor, op, axis.(int), keepdims, allocator, loc)
	}
}

// Convenience wrapper functions

// Sum reduction - reduce along all axes or specific axis
tensor_sum :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	keepdims: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .SUM, axis, keepdims, allocator, loc)
}

// Mean reduction - reduce along all axes or specific axis
tensor_mean :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	keepdims: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MEAN, axis, keepdims, allocator, loc)
}

// Max reduction - reduce along all axes or specific axis
tensor_max :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	keepdims: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MAX, axis, keepdims, allocator, loc)
}

// Min reduction - reduce along all axes or specific axis
tensor_min :: proc(
	tensor: ^Tensor($T),
	axis: Maybe(int) = nil,
	keepdims: bool = false,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return tensor_reduce(tensor, .MIN, axis, keepdims, allocator, loc)
}

// Unary operation types for compile-time dispatch
UnaryOp :: enum {
	NEG, // Negation: -x
	RELU, // ReLU: max(x, 0)
	GELU, // GELU activation function
	SILU, // SILU activation function
	SQRT, // Square root: sqrt(x)
	SIGMOID,
	SIN,
	COS,
}

// Individual operation implementations - forced inline for zero overhead
@(private)
unary_neg :: #force_inline proc($T: typeid, x: T) -> T {
	return -x
}

@(private)
unary_relu :: #force_inline proc($T: typeid, x: T) -> T {
	return math.max(x, T(0))
}

@(private)
unary_gelu :: #force_inline proc($T: typeid, x: T) -> T where T == f32 || T == f64 || T == f16 {
	sqrt_2_over_pi := math.sqrt(T(2.0) / math.PI)
	inner := sqrt_2_over_pi * (x + T(0.044715) * x * x * x)
	return T(0.5) * x * (T(1.0) + math.tanh(inner))
}

@(private)
unary_silu :: #force_inline proc($T: typeid, x: T) -> T where T == f32 || T == f64 || T == f16 {
	return x / (T(1.0) + math.exp(-x))
}

@(private)
unary_sqrt :: #force_inline proc($T: typeid, x: T) -> T where T == f32 || T == f64 || T == f16 {
	return math.sqrt(x)
}

@(private)
unary_sigmoid :: #force_inline proc($T: typeid, x: T) -> T {
	return 1 / (1 + math.exp(-x))
}

@(private)
unary_sin :: #force_inline proc($T: typeid, x: T) -> T where T == f32 || T == f64 || T == f16 {
	return math.sin(x)
}

@(private)
unary_cos :: #force_inline proc($T: typeid, x: T) -> T where T == f32 || T == f64 || T == f16 {
	return math.cos(x)
}

// Generic elementwise unary operation with compile-time specialization
elementwise_unary_op :: proc(
	tensor: ^Tensor($T),
	$op: UnaryOp,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	result := tensor_alloc(T, tensor.shape, true, allocator, loc)

	for i in 0 ..< len(tensor.data) {
		switch op {
		case .NEG:
			result.data[i] = unary_neg(T, tensor.data[i])
		case .RELU:
			result.data[i] = unary_relu(T, tensor.data[i])
		case .GELU:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_gelu(T, tensor.data[i])
			} else {
				panic("GELU only supports f16, f32, f64")
			}
		case .SILU:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_silu(T, tensor.data[i])
			} else {
				panic("GELU only supports f16, f32, f64")
			}
		case .SQRT:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_sqrt(T, tensor.data[i])
			} else {
				panic("SQRT only supports f16, f32, f64")
			}
		case .SIGMOID:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_sigmoid(T, tensor.data[i])
			} else {
				panic("SIGMOID only supports f16, f32, f64")
			}
		case .SIN:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_sin(T, tensor.data[i])
			} else {
				panic("SIN only supports floating point types")
			}
		case .COS:
			when T == f32 || T == f64 || T == f16 {
				result.data[i] = unary_cos(T, tensor.data[i])
			} else {
				panic("COS only supports floating point types")
			}
		}
	}

	return result
}

// Convenience wrapper functions

// Negation operation: -x
neg :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_unary_op(tensor, .NEG, allocator, loc)
}

// ReLU activation: max(x, 0)
relu :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	return elementwise_unary_op(tensor, .RELU, allocator, loc)
}

sin :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	return elementwise_unary_op(tensor, .SIN, allocator)
}

cos :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	return elementwise_unary_op(tensor, .COS, allocator)
}

gelu :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	gelu_trace := trace.TRACE_FUNCTION("gelu")
	defer trace.end_scoped_trace(gelu_trace)

	return elementwise_unary_op(tensor, .GELU, allocator, loc)
}

silu :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	silu_trace := trace.TRACE_FUNCTION("silu")
	defer trace.end_scoped_trace(silu_trace)

	return elementwise_unary_op(tensor, .SILU, allocator, loc)
}

sqrt :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	return elementwise_unary_op(tensor, .SQRT, allocator, loc)
}

sigmoid :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) where T == f32 ||
	T == f64 ||
	T == f16 {
	return elementwise_unary_op(tensor, .SIGMOID, allocator, loc)
}

/*
 Fast variant of some activations
*/

tanh_fast_simd_4xf32 :: proc(x: #simd[4]f32) -> #simd[4]f32 {
	// NOTE(Aria): idk why this works okay-ish. Especially compared to huggingface's
	// SAM with tiny ViT and YOLOv8 ;p
	max_val := #simd[4]f32{3.0, 3.0, 3.0, 3.0}
	min_val := #simd[4]f32{-3.0, -3.0, -3.0, -3.0}
	x_clamped := simd.min(simd.max(x, min_val), max_val)

	x2 := simd.mul(x_clamped, x_clamped)
	c27 := #simd[4]f32{27.0, 27.0, 27.0, 27.0}
	c9 := #simd[4]f32{9.0, 9.0, 9.0, 9.0}

	numerator := simd.mul(x_clamped, simd.add(c27, x2))
	denominator := simd.fma(x2, c9, c27)

	return simd.div(numerator, denominator)
}
silu_fast :: proc(
	x: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	result := tensor_alloc(T, x.shape, true, allocator, loc)
	total_elements := len(x.data)

	when T == f32 {
		half := f32(0.5)
		one := f32(1.0)
		two := f32(2.0)

		i := 0

		// SIMD path for chunks of 4
		for ; i + 4 <= total_elements; i += 4 {
			v := (^#simd[4]f32)(&x.data[i])^

			// tanh(x/2)
			arg := simd.div(v, #simd[4]f32{two, two, two, two})
			tanh_result := tanh_fast_simd_4xf32(arg)

			// x * (tanh(x/2) + 1) / 2
			silu_result := simd.mul(
				v,
				simd.mul(
					simd.add(tanh_result, #simd[4]f32{one, one, one, one}),
					#simd[4]f32{half, half, half, half},
				),
			)

			(^#simd[4]f32)(&result.data[i])^ = silu_result
		}

		// Scalar fallback
		for ; i < total_elements; i += 1 {
			v := x.data[i]
			result.data[i] = v * (math.tanh(v / 2.0) + 1.0) / 2.0
		}
	} else {
		for i in 0 ..< total_elements {
			v := x.data[i]
			result.data[i] = v * (math.tanh(v / 2.0) + 1.0) / 2.0
		}
	}

	return result
}
