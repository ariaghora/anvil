package tensor


import "../matmul"
import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:slice"
import "core:strings"

// This structure implements a high level multidimensional tensor container using a
// linear tensor of data internally. The tensor is parametrized for any data type
// and supports strided access and broadcasting. The internal representation uses
// a C-contiguous storage layout (row-major order) with all the data stored in a
// single slice. A stride tensor is used to map N-dimensional coordinates to linear
// indices in the data tensor. The contiguous flag indicates if the tensor is stored
// in memory without gaps.
Tensor :: struct($T: typeid) where intrinsics.type_is_numeric(T) {
	data:       []T,
	shape:      []uint,
	strides:    []uint,
	contiguous: bool,
}

// Compute total size of an tensor by multiplying dimensions in shape
shape_to_size :: #force_inline proc(shape: []uint) -> uint {
	size: uint = 1
	for s in shape {size *= s}
	return size
}

// Create a new n-dimensional tensor with the given shape. For each dimension i
// in the tensor shape[i] represents the size of that dimension. After allocation
// the tensor elements are left uninitialized.
@(private)
tensor_alloc :: proc(
	$T: typeid,
	shape: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Tensor(T),
) {
	size := shape_to_size(shape)
	res = new(Tensor(T), allocator)
	res.data = make([]T, size, allocator, loc)
	res.shape = make([]uint, len(shape), allocator)
	res.strides = make([]uint, len(shape), allocator)
	res.contiguous = true

	// initialize shape and strides
	copy(res.shape, shape)
	stride: uint = 1
	for i := len(shape) - 1; i >= 0; i -= 1 {
		res.strides[i] = stride
		stride *= shape[i]
	}
	return res
}

// Get tensor data respecting strides. If shape and strides match the original
// tensor and data is contiguous, return a simple clone. Otherwise rearrange
// the data following target shape and strides to handle non-contiguous cases.
// Under the hood this function converts non-contiguous tensor storage into
// contiguous one.
get_strided_data :: proc(
	arr: ^Tensor($T),
	shape: []uint = nil,
	strides: []uint = nil,
	allocator := context.allocator,
) -> []T {
	target_strides := strides
	target_shape := shape
	if (strides == nil) {
		target_strides = arr.strides
	}
	if (shape == nil) {
		target_shape = arr.shape
	}

	if arr.contiguous &&
	   (slice.equal(arr.shape, target_shape) || shape == nil) &&
	   (slice.equal(arr.strides, target_strides) || strides == nil) {
		return slice.clone(arr.data, allocator)
	}

	// return strided
	size := shape_to_size(target_shape)
	data := make([]T, size, allocator)
	i: uint
	for i in 0 ..< size {
		i_strided := compute_strided_index(target_shape, target_strides, i)
		data[i] = arr.data[i_strided]
	}
	return data
}

// Deep copy of tensor data. The copy will be an exact replica of the original
// tensor, with exactly the same data, shape, strides and contiguous flag. The
// resulting tensor will be completely independent from the source.
clone :: proc(arr: ^Tensor($T), allocator := context.allocator) -> (res: ^Tensor(T)) {
	res = new(Tensor(T), allocator)
	res.data = make([]T, len(arr.data), allocator)
	res.shape = make([]uint, len(arr.shape), allocator)
	res.strides = make([]uint, len(arr.strides), allocator)
	res.contiguous = arr.contiguous

	copy(res.data, arr.data)
	copy(res.shape, arr.shape)
	copy(res.strides, arr.strides)

	return res
}

// Returns the total number of elements in the tensor by multiplying all dimensions
// together. Takes into account that some dimensions may be empty (0) in which
// case the total size will also be zero. This function does not care about strides
// or how data is laid out in memory, it is just about the logical size.
data_len :: proc(arr: ^Tensor($T)) -> uint {
	return shape_to_size(arr.shape)
}

// Create a new tensor filled with ones. tensors are initialized for all data types by casting
// 1 to the target type, so for example this works with floating point data types, integers
// and even complex data types like bool or void types.
ones :: proc($T: typeid, shape: []uint, allocator := context.allocator) -> (res: ^Tensor(T)) {
	res = tensor_alloc(T, shape, allocator)
	for _, i in res.data {res.data[i] = T(1)}
	return res
}


// Create an tensor with normally-distributed random values
randn :: proc(
	$T: typeid,
	shape: []uint,
	mean, stddev: T,
	allocator := context.allocator,
) -> (
	res: ^Tensor(T),
) {
	res = tensor_alloc(T, shape, allocator)
	for _, i in res.data {
		// Box-Muller transform to generate normal distribution
		u1 := rand.float64()
		u2 := rand.float64()
		z := math.sqrt(-2 * math.ln(u1)) * math.cos(2 * math.PI * u2)
		res.data[i] = T(z)
	}
	return res
}

// Create a new tensor filled with zeros. tensors are initialized for all data types by casting
// 0 to the target type, so for example this works with floating point data types, integers
// and even complex data types like bool or void types.
zeros :: proc($T: typeid, shape: []uint, allocator := context.allocator) -> (res: ^Tensor(T)) {
	res = tensor_alloc(T, shape, allocator)
	for _, i in res.data {res.data[i] = T(0)}
	return res
}

// Create a new tensor with given data and shape. This function performs a copy
// of the input data, so the original tensor is not referenced in the new one.
new_with_init :: proc(
	init: []$T,
	shape: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Tensor(T),
) {
	res = tensor_alloc(T, shape, allocator, loc)
	if len(res.data) != len(init) {
		panic("Input data length must match tensor size computed from shape")
	}

	copy(res.data, init)
	return res
}

@(private = "package")
compute_linear_index :: proc(indices: []uint, strides: []uint) -> uint {
	index: uint = 0
	for i in 0 ..< len(indices) {
		index += indices[i] * strides[i]
	}
	return index
}

free_tensor :: proc {
	free_tensor_one,
	free_tensor_many,
}

@(private = "file")
free_tensor_one :: proc(arr: ^Tensor($T), allocator := context.allocator) {
	delete(arr.data, allocator)
	delete(arr.shape, allocator)
	delete(arr.strides, allocator)
	free(arr, allocator)
}

@(private = "file")
free_tensor_many :: proc(arr: ^Tensor($T), rest: ..^Tensor(T), allocator := context.allocator) {
	free_tensor_one(arr, allocator)
	for r in rest {
		free_tensor_one(r, allocator)
	}
}


tensor_get :: proc {
	_tensor_get_1d,
	_tensor_get_2d,
	_tensor_get_3d,
	_tensor_get_4d,
	_tensor_get_nd,
}

@(private = "file")
_tensor_get_1d :: #force_inline proc(arr: ^Tensor($T), i0: uint) -> T {
	s0 := arr.strides[0]
	return arr.data[i0 * s0]
}

@(private = "file")
_tensor_get_2d :: #force_inline proc(arr: ^Tensor($T), row, col: uint) -> T {
	s0, s1 := arr.strides[0], arr.strides[1]
	return arr.data[row * s0 + col * s1]
}

@(private = "file")
_tensor_get_3d :: #force_inline proc(arr: ^Tensor($T), i0, i1, i2: uint) -> T {
	s0, s1, s2 := arr.strides[0], arr.strides[1], arr.strides[2]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2]
}

@(private = "file")
_tensor_get_4d :: #force_inline proc(arr: ^Tensor($T), i0, i1, i2, i3: uint) -> T {
	s0, s1, s2, s3 := arr.strides[0], arr.strides[1], arr.strides[2], arr.strides[3]
	return arr.data[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3]
}

@(private = "file")
_tensor_get_nd :: #force_inline proc(arr: ^Tensor($T), coord: []uint) -> T {
	index: uint = 0
	for i in 0 ..< len(coord) {
		index += coord[i] * arr.strides[i]
	}
	return arr.data[index]
}


@(private = "package")
compute_strided_index :: #force_inline proc(shape, strides: []uint, idx: uint) -> uint {
	#no_bounds_check {
		switch len(shape) {
		case 1:
			return idx * strides[0]
		case 2:
			s0, s1 := strides[0], strides[1]
			d1 := shape[1]
			return (idx / d1) * s0 + (idx % d1) * s1
		case 3:
			s0, s1, s2 := strides[0], strides[1], strides[2]
			d1, d2 := shape[1], shape[2]
			i2 := idx % d2
			tmp := idx / d2
			i1 := tmp % d1
			i0 := tmp / d1
			return i0 * s0 + i1 * s1 + i2 * s2
		case 4:
			s0, s1, s2, s3 := strides[0], strides[1], strides[2], strides[3]
			d1, d2, d3 := shape[1], shape[2], shape[3]
			i3 := idx % d3
			tmp := idx / d3
			i2 := tmp % d2
			tmp /= d2
			i1 := tmp % d1
			i0 := tmp / d1
			return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3
		case:
			// For N-dim, precompute products to avoid repeated divisions
			offset: uint = 0
			remaining := idx
			dim_product := uint(1)
			for i := len(shape) - 1; i >= 0; i -= 1 {
				coord := (remaining / dim_product) % shape[i]
				offset += coord * strides[i]
				dim_product *= shape[i]
			}
			return offset
		}
	}
}

reshape :: proc(
	arr: ^Tensor($T),
	new_shape: []uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	// Check if total size matches
	old_size := shape_to_size(arr.shape)
	new_size := shape_to_size(new_shape)
	if old_size != new_size {
		panic(fmt.tprintf("Cannot reshape tensor of size %v to shape %v", old_size, new_shape))
	}

	res := tensor_alloc(T, new_shape, allocator)
	copy(res.data, arr.data) // Since we're just changing shape, data can be copied directly
	return res
}


matmul :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	assert(len(a.shape) == 2 && len(b.shape) == 2)
	out := tensor_alloc(T, []uint{a.shape[0], b.shape[1]}, allocator = allocator, loc = loc)
	matmul.matmul(
		a.data,
		b.data,
		a.shape[0],
		b.shape[1],
		a.shape[1],
		out.data,
		allocator = allocator,
	)
	return out
}
