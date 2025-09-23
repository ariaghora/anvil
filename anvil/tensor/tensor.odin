package tensor

import "base:runtime"

import "../matmul_backend"
import "../simd_backend"
import "../trace"
import "base:intrinsics"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"
import "core:simd"
import "core:slice"
import "core:strings"

// NOTE(Aria): this is tuned based on Arm M chips
SIMD_ALIGNMENT :: 16

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
	owns_data:  bool,
}

// Compute total size of an tensor by multiplying dimensions in shape
shape_to_size :: #force_inline proc(shape: []uint) -> uint {
	return math.prod(shape)
}

// Create a new n-dimensional tensor with the given shape. For each dimension i
// in the tensor shape[i] represents the size of that dimension. After allocation
// the tensor elements are left uninitialized.
tensor_alloc :: proc(
	$T: typeid,
	shape: []uint,
	owns_data := true,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Tensor(T),
) {
	size := shape_to_size(shape)
	res = new(Tensor(T), allocator)
	if owns_data do res.data = runtime.make_aligned([]T, size, SIMD_ALIGNMENT, allocator, loc)

	res.shape = make([]uint, len(shape), allocator)
	res.strides = make([]uint, len(shape), allocator)
	res.contiguous = true
	res.owns_data = owns_data

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
) -> (
	res: []T,
	allocated: bool,
) {
	target_strides := strides if strides != nil else arr.strides
	target_shape := shape if shape != nil else arr.shape

	// Fast path - already contiguous
	if arr.contiguous &&
	   (slice.equal(arr.shape, target_shape) || shape == nil) &&
	   (slice.equal(arr.strides, target_strides) || strides == nil) {
		return arr.data, false
	}

	size := shape_to_size(target_shape)
	data := runtime.make_aligned([]T, size, SIMD_ALIGNMENT, allocator)

	// Specialized copy loops based on dimensionality
	switch len(target_shape) {
	case 1:
		copy_strided_1d(data, arr.data, target_shape, target_strides)
	case 2:
		copy_strided_2d(data, arr.data, target_shape, target_strides)
	case 3:
		copy_strided_3d(data, arr.data, target_shape, target_strides)
	case 4:
		copy_strided_4d(data, arr.data, target_shape, target_strides)
	case:
		copy_strided_nd(data, arr.data, target_shape, target_strides)
	}

	return data, true
}

@(private = "file")
copy_strided_1d :: proc(dst, src: []$T, shape, strides: []uint) {
	s0 := strides[0]

	when T == f32 {
		i := uint(0)

		// SIMD path for unit stride (contiguous)
		if s0 == 1 {
			// Both src and dst are contiguous
			for ; i + 8 <= shape[0]; i += 8 {
				vals1 := (^#simd[4]f32)(&src[i])^
				vals2 := (^#simd[4]f32)(&src[i + 4])^
				(^#simd[4]f32)(&dst[i])^ = vals1
				(^#simd[4]f32)(&dst[i + 4])^ = vals2
			}

			for ; i + 4 <= shape[0]; i += 4 {
				vals := (^#simd[4]f32)(&src[i])^
				(^#simd[4]f32)(&dst[i])^ = vals
			}
		} else if s0 == 2 {
			// Special case for stride 2 (every other element)
			// src is strided, but dst is consecutive
			for ; i + 4 <= shape[0]; i += 4 {
				vals := #simd[4]f32 {
					src[i * 2],
					src[(i + 1) * 2],
					src[(i + 2) * 2],
					src[(i + 3) * 2],
				}
				(^#simd[4]f32)(&dst[i])^ = vals
			}
		}

		// Scalar remainder
		for ; i < shape[0]; i += 1 {
			dst[i] = src[i * s0]
		}
	} else {
		// Slow scalar reference
		for i in 0 ..< shape[0] {
			dst[i] = src[i * s0]
		}
	}
}

@(private = "file")
copy_strided_2d :: proc(dst, src: []$T, shape, strides: []uint) {
	s0, s1 := strides[0], strides[1]
	d0, d1 := shape[0], shape[1]
	dst_idx := uint(0)

	when T == f32 {
		// Special case for row-major contiguous (s1 == 1)
		if s1 == 1 {
			for i in 0 ..< d0 {
				src_row_start := i * s0

				j := uint(0)
				// Copy entire rows with SIMD
				for ; j + 8 <= d1; j += 8 {
					vals1 := (^#simd[4]f32)(&src[src_row_start + j])^
					vals2 := (^#simd[4]f32)(&src[src_row_start + j + 4])^
					(^#simd[4]f32)(&dst[dst_idx])^ = vals1
					(^#simd[4]f32)(&dst[dst_idx + 4])^ = vals2
					dst_idx += 8
				}

				for ; j + 4 <= d1; j += 4 {
					vals := (^#simd[4]f32)(&src[src_row_start + j])^
					(^#simd[4]f32)(&dst[dst_idx])^ = vals
					dst_idx += 4
				}

				for ; j < d1; j += 1 {
					dst[dst_idx] = src[src_row_start + j]
					dst_idx += 1
				}
			}
		} else {
			// General strided case
			for i in 0 ..< d0 {
				src_row := i * s0

				j := uint(0)
				// Process 4 elements at a time with gather-like pattern
				for ; j + 4 <= d1; j += 4 {
					vals := #simd[4]f32 {
						src[src_row + j * s1],
						src[src_row + (j + 1) * s1],
						src[src_row + (j + 2) * s1],
						src[src_row + (j + 3) * s1],
					}
					(^#simd[4]f32)(&dst[dst_idx])^ = vals
					dst_idx += 4
				}

				for ; j < d1; j += 1 {
					dst[dst_idx] = src[src_row + j * s1]
					dst_idx += 1
				}
			}
		}
	} else {
		// Original scalar code
		for i in 0 ..< d0 {
			src_row := i * s0
			for j in 0 ..< d1 {
				dst[dst_idx] = src[src_row + j * s1]
				dst_idx += 1
			}
		}
	}
}

@(private = "file")
copy_strided_3d :: proc(dst, src: []$T, shape, strides: []uint) {
	s0, s1, s2 := strides[0], strides[1], strides[2]
	d0, d1, d2 := shape[0], shape[1], shape[2]
	dst_idx := uint(0)

	when T == f32 {
		// Optimize for contiguous inner dimension
		if s2 == 1 {
			for i in 0 ..< d0 {
				src_plane := i * s0
				for j in 0 ..< d1 {
					src_row_start := src_plane + j * s1

					k := uint(0)
					for ; k + 8 <= d2; k += 8 {
						vals1 := (^#simd[4]f32)(&src[src_row_start + k])^
						vals2 := (^#simd[4]f32)(&src[src_row_start + k + 4])^
						(^#simd[4]f32)(&dst[dst_idx])^ = vals1
						(^#simd[4]f32)(&dst[dst_idx + 4])^ = vals2
						dst_idx += 8
					}

					for ; k + 4 <= d2; k += 4 {
						vals := (^#simd[4]f32)(&src[src_row_start + k])^
						(^#simd[4]f32)(&dst[dst_idx])^ = vals
						dst_idx += 4
					}

					for ; k < d2; k += 1 {
						dst[dst_idx] = src[src_row_start + k]
						dst_idx += 1
					}
				}
			}
		} else {
			// General case
			for i in 0 ..< d0 {
				src_plane := i * s0
				for j in 0 ..< d1 {
					src_row := src_plane + j * s1
					for k in 0 ..< d2 {
						dst[dst_idx] = src[src_row + k * s2]
						dst_idx += 1
					}
				}
			}
		}
	} else {
		for i in 0 ..< d0 {
			src_plane := i * s0
			for j in 0 ..< d1 {
				src_row := src_plane + j * s1
				for k in 0 ..< d2 {
					dst[dst_idx] = src[src_row + k * s2]
					dst_idx += 1
				}
			}
		}
	}
}

@(private = "file")
copy_strided_4d :: proc(dst, src: []$T, shape, strides: []uint) {
	s0, s1, s2, s3 := strides[0], strides[1], strides[2], strides[3]
	d0, d1, d2, d3 := shape[0], shape[1], shape[2], shape[3]
	dst_idx := uint(0)

	when T == f32 {
		// Case 1: Fully contiguous (can do one big memcpy)
		if s0 == d1 * d2 * d3 && s1 == d2 * d3 && s2 == d3 && s3 == 1 {
			copy(dst, src[:d0 * d1 * d2 * d3])
			return
		}

		// Case 2: Last 2 dimensions contiguous (common for NCHW)
		if s2 == d3 && s3 == 1 {
			row_size := d2 * d3
			for i in 0 ..< d0 {
				src_batch := i * s0
				for j in 0 ..< d1 {
					src_plane_start := src_batch + j * s1

					// Copy entire HW plane at once
					copy(
						dst[dst_idx:dst_idx + row_size],
						src[src_plane_start:src_plane_start + row_size],
					)
					dst_idx += row_size
				}
			}
			return
		}

		// Case 3: Only last dimension contiguous
		if s3 == 1 {
			for i in 0 ..< d0 {
				src_batch := i * s0
				for j in 0 ..< d1 {
					src_plane := src_batch + j * s1
					for k in 0 ..< d2 {
						src_row_start := src_plane + k * s2

						l := uint(0)

						// Process 16 elements at a time
						for ; l + 16 <= d3; l += 16 {
							// Prefetch next row if available
							if k + 1 < d2 {
								_ = src[src_plane + (k + 1) * s2] // Prefetch hint
							}

							vals0 := (^#simd[4]f32)(&src[src_row_start + l])^
							vals1 := (^#simd[4]f32)(&src[src_row_start + l + 4])^
							vals2 := (^#simd[4]f32)(&src[src_row_start + l + 8])^
							vals3 := (^#simd[4]f32)(&src[src_row_start + l + 12])^

							(^#simd[4]f32)(&dst[dst_idx])^ = vals0
							(^#simd[4]f32)(&dst[dst_idx + 4])^ = vals1
							(^#simd[4]f32)(&dst[dst_idx + 8])^ = vals2
							(^#simd[4]f32)(&dst[dst_idx + 12])^ = vals3
							dst_idx += 16
						}

						for ; l + 4 <= d3; l += 4 {
							vals := (^#simd[4]f32)(&src[src_row_start + l])^
							(^#simd[4]f32)(&dst[dst_idx])^ = vals
							dst_idx += 4
						}

						for ; l < d3; l += 1 {
							dst[dst_idx] = src[src_row_start + l]
							dst_idx += 1
						}
					}
				}
			}
		} else {
			// General case, no contiguity
			for i in 0 ..< d0 {
				src_batch := i * s0
				for j in 0 ..< d1 {
					src_plane := src_batch + j * s1
					for k in 0 ..< d2 {
						src_row := src_plane + k * s2

						// Unroll the innermost loop
						l := uint(0)
						for ; l + 4 <= d3; l += 4 {
							dst[dst_idx] = src[src_row + l * s3]
							dst[dst_idx + 1] = src[src_row + (l + 1) * s3]
							dst[dst_idx + 2] = src[src_row + (l + 2) * s3]
							dst[dst_idx + 3] = src[src_row + (l + 3) * s3]
							dst_idx += 4
						}

						for ; l < d3; l += 1 {
							dst[dst_idx] = src[src_row + l * s3]
							dst_idx += 1
						}
					}
				}
			}
		}
	} else {
		// Non-SIMD types
		for i in 0 ..< d0 {
			src_batch := i * s0
			for j in 0 ..< d1 {
				src_plane := src_batch + j * s1
				for k in 0 ..< d2 {
					src_row := src_plane + k * s2
					for l in 0 ..< d3 {
						dst[dst_idx] = src[src_row + l * s3]
						dst_idx += 1
					}
				}
			}
		}
	}
}

@(private = "file")
copy_strided_nd :: proc(dst, src: []$T, shape, strides: []uint) {
	// For higher dimensions, use existing approach but optimize
	// by pre-computing as much as possible
	indices := make([]uint, len(shape), context.temp_allocator)

	for dst_idx in 0 ..< len(dst) {
		// Update indices
		carry := dst_idx
		for dim := len(shape) - 1; dim >= 0; dim -= 1 {
			indices[dim] = uint(carry % int(shape[dim]))
			carry /= int(shape[dim])
		}

		// Compute source index
		src_idx: uint = 0
		for dim in 0 ..< len(shape) {
			src_idx += indices[dim] * strides[dim]
		}

		dst[dst_idx] = src[src_idx]
	}
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
	res.owns_data = true

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
	res = tensor_alloc(T, shape, true, allocator)
	for _, i in res.data {res.data[i] = T(1)}
	return res
}


// Create an tensor with normally-distributed random values
// TODO(Aria): friggin slow, improve or abolish at all
randn :: proc(
	$T: typeid,
	shape: []uint,
	mean, stddev: T,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Tensor(T),
) {
	res = tensor_alloc(T, shape, true, allocator, loc)
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
zeros :: proc(
	$T: typeid,
	shape: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Tensor(T),
) {
	res = tensor_alloc(T, shape, true, allocator, loc)
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
	res = tensor_alloc(T, shape, true, allocator, loc)
	if len(res.data) != len(init) {
		panic("Input data length must match tensor size computed from shape")
	}

	copy(res.data, init)
	return res
}

// Create a new tensor with a given shape, and fill it with a constant value
full :: proc($T: typeid, value: T, shape: []uint, allocator := context.allocator) -> ^Tensor(T) {
	t := tensor_alloc(T, shape, true, allocator)
	for i in 0 ..< len(t.data) {
		t.data[i] = value
	}
	return t
}

@(private = "package")
compute_linear_index :: proc(indices: []uint, strides: []uint, loc := #caller_location) -> uint {
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
free_tensor_one :: proc(
	arr: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) {
	// Only free data if this tensor owns its data
	if arr.owns_data {
		delete(arr.data, allocator, loc)
	}
	// Always free shape and strides (each tensor owns these)
	delete(arr.shape, allocator, loc)
	delete(arr.strides, allocator, loc)
	free(arr, allocator, loc)
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
	loc := #caller_location,
) -> ^Tensor(T) {
	trace_reshape := trace.TRACE_FUNCTION("reshape")
	defer trace.end_scoped_trace(trace_reshape)

	// Check if total size matches
	old_size := shape_to_size(arr.shape)
	new_size := shape_to_size(new_shape)
	if old_size != new_size {
		panic(
			fmt.tprintf(
				"Cannot reshape tensor of size %v to shape %v (%c)",
				old_size,
				new_shape,
				loc,
			),
		)
	}

	res: ^Tensor(T)
	res_should_own_data := !arr.owns_data
	res = tensor_alloc(T, new_shape, res_should_own_data, allocator)
	if arr.owns_data {
		res.data = arr.data
	} else {
		arr_data, _ := get_strided_data(arr, allocator = context.temp_allocator)
		copy(res.data, arr_data) // Since we're just changing shape, data can be copied directly
	}
	return res
}


// Tensor-aware matrix multiplication with batch support and broadcasting
// Supports 2D+ tensors: (...batch_dims, m, k) @ (...batch_dims, k, n) -> (...batch_dims, m, n)
matmul :: proc(
	a, b: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Validate minimum dimensions
	if len(a.shape) < 2 || len(b.shape) < 2 {
		panic("tensor_matmul requires at least 2D tensors")
	}

	// Extract matrix dimensions (last 2 dimensions)
	a_matrix_dims := a.shape[len(a.shape) - 2:]
	b_matrix_dims := b.shape[len(b.shape) - 2:]
	a_m, a_k := a_matrix_dims[0], a_matrix_dims[1]
	b_k, b_n := b_matrix_dims[0], b_matrix_dims[1]

	// Validate inner matrix dimensions match
	if a_k != b_k {
		panic("Matrix dimensions incompatible for multiplication")
	}

	// Extract batch dimensions (all but last 2)
	a_batch := a.shape[:len(a.shape) - 2] if len(a.shape) > 2 else []uint{}
	b_batch := b.shape[:len(b.shape) - 2] if len(b.shape) > 2 else []uint{}

	// Check if batch dimensions are broadcastable
	result_batch, broadcastable := shape_broadcastable(a_batch, b_batch, context.temp_allocator)
	if !broadcastable {
		panic("Batch dimensions cannot be broadcasted")
	}

	// Construct result shape: [...batch_dims, m, n]
	result_shape := make([]uint, len(result_batch) + 2, context.temp_allocator, loc)
	copy(result_shape[:len(result_batch)], result_batch)
	result_shape[len(result_batch)] = a_m
	result_shape[len(result_batch) + 1] = b_n

	// Create result tensor
	result := tensor_alloc(T, result_shape, true, allocator, loc)

	// Calculate total batch size and matrix sizes
	batch_size := shape_to_size(result_batch) if len(result_batch) > 0 else 1
	matrix_size_a := a_m * a_k
	matrix_size_b := b_k * b_n
	matrix_size_result := a_m * b_n

	// Compute broadcasted strides for batch dimensions
	a_full_batch := make([]uint, len(result_batch), context.temp_allocator)
	b_full_batch := make([]uint, len(result_batch), context.temp_allocator)

	// Pad batch dimensions with 1s if needed for stride calculation
	if len(a_batch) < len(result_batch) {
		for i in 0 ..< len(result_batch) - len(a_batch) {
			a_full_batch[i] = 1
		}
		copy(a_full_batch[len(result_batch) - len(a_batch):], a_batch)
	} else {
		copy(a_full_batch, a_batch)
	}

	if len(b_batch) < len(result_batch) {
		for i in 0 ..< len(result_batch) - len(b_batch) {
			b_full_batch[i] = 1
		}
		copy(b_full_batch[len(result_batch) - len(b_batch):], b_batch)
	} else {
		copy(b_full_batch, b_batch)
	}

	// Create temporary strides for batch dimensions only
	a_batch_strides := make([]uint, len(result_batch), context.temp_allocator)
	b_batch_strides := make([]uint, len(result_batch), context.temp_allocator)

	// Compute strides for batch dimensions manually
	if len(result_batch) > 0 {
		a_stride: uint = matrix_size_a
		for i := len(result_batch) - 1; i >= 0; i -= 1 {
			if a_full_batch[i] == 1 {
				a_batch_strides[i] = 0 // Broadcasting: no movement
			} else {
				a_batch_strides[i] = a_stride
				a_stride *= a_full_batch[i]
			}
		}

		b_stride: uint = matrix_size_b
		for i := len(result_batch) - 1; i >= 0; i -= 1 {
			if b_full_batch[i] == 1 {
				b_batch_strides[i] = 0 // Broadcasting: no movement
			} else {
				b_batch_strides[i] = b_stride
				b_stride *= b_full_batch[i]
			}
		}
	}

	a_data, a_allocated := get_strided_data(a, a.shape, a.strides, context.temp_allocator)
	b_data, b_allocated := get_strided_data(b, b.shape, b.strides, context.temp_allocator)

	// Process each batch
	for batch_idx in 0 ..< batch_size {
		// Calculate batch indices
		batch_indices := make([]uint, len(result_batch), context.temp_allocator)

		temp_idx := batch_idx
		for dim := len(result_batch) - 1; dim >= 0; dim -= 1 {
			batch_indices[dim] = temp_idx % result_batch[dim]
			temp_idx /= result_batch[dim]
		}

		// Calculate linear offsets for this batch
		a_offset: uint = 0
		b_offset: uint = 0
		for i in 0 ..< len(result_batch) {
			a_offset += batch_indices[i] * a_batch_strides[i]
			b_offset += batch_indices[i] * b_batch_strides[i]
		}

		// Get slices for this batch's matrices
		a_matrix := a_data[a_offset:a_offset + matrix_size_a]
		b_matrix := b_data[b_offset:b_offset + matrix_size_b]
		result_matrix := result.data[batch_idx *
		matrix_size_result:(batch_idx + 1) *
		matrix_size_result]

		// Perform 2D matrix multiplication using BLAS
		matmul_backend.matmul_2d(a_matrix, b_matrix, a_m, b_n, a_k, result_matrix, allocator)
	}

	return result
}

gemm :: proc(
	a, b: ^Tensor($T),
	c: Maybe(^Tensor(T)),
	alpha, beta: T,
	trans_a, trans_b: bool,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	ensure(len(a.shape) == 2 && len(b.shape) == 2, "both inputs for gemm must be 2D tensors")
	a := trans_a ? transpose(a, 0, 1, allocator) : a
	b := trans_b ? transpose(b, 0, 1, allocator) : b

	defer {
		if trans_a do free_tensor(a, allocator)
		if trans_b do free_tensor(b, allocator)
	}

	res := matmul(a, b, allocator)
	if alpha != T(1) {
		for _, i in res.data do res.data[i] *= alpha
	}

	// Add bias in-place
	if bias, has_bias := c.?; has_bias {
		out_features := res.shape[len(res.shape) - 1]
		total_elements := shape_to_size(res.shape)
		batch_elements := total_elements / out_features

		#no_bounds_check {
			when T == f32 {
				for i in 0 ..< batch_elements {
					base_idx := i * out_features
					when ODIN_OS == .Darwin {
						simd_backend.addf_batch(
							res.data[base_idx:base_idx + out_features],
							res.data[base_idx:base_idx + out_features],
							bias.data,
						)
					} else {
						j := uint(0)
						for ; j + 4 <= out_features; j += 4 {
							b := (^#simd[4]f32)(&bias.data[j])^
							o := (^#simd[4]f32)(&res.data[base_idx + j])^
							(^#simd[4]f32)(&res.data[base_idx + j])^ = o + b
						}

						for ; j < out_features; j += 1 {
							res.data[base_idx + j] += bias.data[j]
						}
					}

				}
			} else {
				// Scalar fallback
				for i in 0 ..< batch_elements {
					base_idx := i * out_features
					for j in 0 ..< out_features {
						res.data[base_idx + j] += bias.data[j]
					}
				}
			}
		}
	}
	return res
}

pad_with_zero :: proc(
	x: ^Tensor($T),
	dim, left, right: uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	ensure(
		dim < len(x.shape),
		fmt.tprintf("dim %d out of bounds for tensor with %d dimensions", dim, len(x.shape)),
	)

	if left == 0 && right == 0 {
		return clone(x, allocator)
	} else if left == 0 {
		shape := slice.clone(x.shape, context.temp_allocator)
		shape[dim] = right
		right_zeros := zeros(T, shape, context.temp_allocator)
		return cat([]^Tensor(T){x, right_zeros}, dim, allocator)
	} else if right == 0 {
		shape := slice.clone(x.shape, context.temp_allocator)
		shape[dim] = left
		left_zeros := zeros(T, shape, context.temp_allocator)
		return cat([]^Tensor(T){left_zeros, x}, dim, allocator)
	} else {
		shape := slice.clone(x.shape, context.temp_allocator)
		shape[dim] = left
		left_zeros := zeros(T, shape, context.temp_allocator)
		shape[dim] = right
		right_zeros := zeros(T, shape, context.temp_allocator)
		return cat([]^Tensor(T){left_zeros, x, right_zeros}, dim, allocator)
	}
}

// General dimension permutation - specify new order of ALL dimensions
// Example: permute(tensor, [2, 0, 1]) reorders dims so that dim 0->2, dim 1->0, dim 2->1
permute :: proc(
	tensor: ^Tensor($T),
	dims: []uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	// Validate input
	if len(dims) != len(tensor.shape) {
		panic("Number of dims must equal tensor dimensions")
	}

	// Create new shape and strides
	new_shape := make([]uint, len(tensor.shape), context.temp_allocator)
	new_strides := make([]uint, len(tensor.strides), context.temp_allocator)

	for result_dim in 0 ..< len(dims) {
		source_dim := dims[result_dim]
		new_shape[result_dim] = tensor.shape[source_dim]
		new_strides[result_dim] = tensor.strides[source_dim]
	}

	// Create contiguous result tensor
	result := tensor_alloc(T, new_shape, true, allocator, loc)

	// Get strided data with the permuted layout
	data, allocated := get_strided_data(tensor, new_shape, new_strides, allocator)
	defer if allocated do delete(data, allocator)

	copy(result.data, data)
	return result
}

// Swap two specific dimensions (PyTorch-style transpose)
// Example: transpose(tensor, 0, 1) swaps dimensions 0 and 1
transpose :: proc(
	tensor: ^Tensor($T),
	dim0, dim1: int,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	trace_transpose := trace.TRACE_FUNCTION(fmt.tprint("transpose", tensor.shape))
	defer trace.end_scoped_trace(trace_transpose)

	if dim0 == dim1 {
		out := tensor_alloc(T, tensor.shape, false, allocator, loc)
		out.data = tensor.data
		return out
	}

	// Validate dimensions
	if dim0 < 0 || dim0 >= len(tensor.shape) || dim1 < 0 || dim1 >= len(tensor.shape) {
		panic("Dimension indices out of range")
	}

	if len(tensor.shape) == 2 {
		return matrix_transpose(tensor, allocator, loc)
	}

	// Create new shape and strides
	new_shape := make([]uint, len(tensor.shape), context.temp_allocator)
	new_strides := make([]uint, len(tensor.strides), context.temp_allocator)

	copy(new_shape, tensor.shape)
	copy(new_strides, tensor.strides)

	// Swap dimensions
	new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
	new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

	// Create contiguous result tensor
	result := tensor_alloc(T, new_shape, true, allocator, loc)

	// Get strided data with the transposed layout
	data, allocated := get_strided_data(tensor, new_shape, new_strides, allocator)
	defer if allocated do delete(data, allocator)

	copy(result.data, data)
	return result
}

// Matrix transpose, specialization for 2D
BLOCK_SIZE :: 16 // TODO(Aria): tune this
@(private = "file")
matrix_transpose_blocked :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	rows, cols := tensor.shape[0], tensor.shape[1]
	new_shape := []uint{cols, rows}
	out := tensor_alloc(T, new_shape, true, allocator, loc)

	// Process in blocks for better cache locality
	for row_block := uint(0); row_block < rows; row_block += BLOCK_SIZE {
		for col_block := uint(0); col_block < cols; col_block += BLOCK_SIZE {
			// Transpose within block
			row_end := min(row_block + BLOCK_SIZE, rows)
			col_end := min(col_block + BLOCK_SIZE, cols)

			for row in row_block ..< row_end {
				for col in col_block ..< col_end {
					out.data[col * rows + row] = tensor.data[row * cols + col]
				}
			}
		}
	}

	return out
}
matrix_transpose :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	if len(tensor.shape) != 2 {
		panic("Matrix transpose requires  2D tensor")
	}

	// TODO(Aria): fix SIMD version
	// when ODIN_OS == .Darwin && T == f32 {
	// 	rows, cols := tensor.shape[0], tensor.shape[1]
	// 	out := tensor_alloc(T, []uint{cols, rows}, true, allocator, loc)
	// 	simd_backend.transposef(out.data, tensor.data, rows, cols)
	// 	return out
	// } else {
	return matrix_transpose_blocked(tensor, allocator, loc)
	// }
}

// Split tensor into chunks along specified dimension
chunk :: proc(
	tensor: ^Tensor($T),
	groups: uint,
	dim: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> []^Tensor(T) {
	if dim >= uint(len(tensor.shape)) {
		panic("Dimension index out of range")
	}

	dim_size := tensor.shape[dim]
	if dim_size % groups != 0 {
		panic("Dimension size must be divisible by number of groups")
	}

	chunk_size := dim_size / groups
	chunks := make([]^Tensor(T), groups, allocator)

	for i in 0 ..< groups {
		// Create shape and strides for this chunk
		chunk_shape := make([]uint, len(tensor.shape), context.temp_allocator)
		chunk_strides := make([]uint, len(tensor.strides), context.temp_allocator)

		copy(chunk_shape, tensor.shape)
		copy(chunk_strides, tensor.strides)
		chunk_shape[dim] = chunk_size

		// Create contiguous result tensor
		chunk_tensor := tensor_alloc(T, chunk_shape, true, allocator, loc)

		// Calculate offset for this chunk
		offset := i * chunk_size * tensor.strides[dim]

		// Get strided data for this chunk
		data, allocated := get_strided_data(
			&Tensor(T){data = tensor.data[offset:], shape = chunk_shape, strides = chunk_strides},
			chunk_shape,
			chunk_strides,
			allocator,
		)
		defer if allocated do delete(data, allocator)

		copy(chunk_tensor.data, data)
		chunks[i] = chunk_tensor
	}

	return chunks
}

tensor_cast :: proc(x: ^Tensor($T), $U: typeid, allocator := context.allocator) -> ^Tensor(U) {
	out := tensor_alloc(U, x.shape, true, allocator)
	for v, i in x.data do out.data[i] = U(v)
	return out
}

/*
	Concatenates tensors along an existing dimension.
	
	All tensors must have identical shapes except in the concatenation dimension.
	The result has the sum of sizes in the concatenation dimension.
	
	Parameters:
	  tensors: Array of tensors to concatenate
	  dim: Dimension along which to concatenate (0 to rank-1)
	
	Returns tensor with combined data along specified dimension.
	
	Example:
	  // Create two 2x3 tensors
	  a := new_with_init([]f32{1,2,3,4,5,6}, {2,3})
	  b := new_with_init([]f32{7,8,9,10,11,12}, {2,3})
	
	  // Concatenate along dim 0 (rows): shape [4, 3]
	  // Result: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
	  c0 := cat({a, b}, 0)
	
	  // Concatenate along dim 1 (cols): shape [2, 6]
	  // Result: [[1,2,3,7,8,9], [4,5,6,10,11,12]]
	  c1 := cat({a, b}, 1)
	
	  // Concatenating tensors with different sizes in concat dim
	  x := new_with_init([]f32{1,2,3,4}, {2,2})     // shape [2,2]
	  y := new_with_init([]f32{5,6,7,8,9,10}, {2,3}) // shape [2,3]
	  z := cat({x, y}, 1)  // shape [2,5]: [[1,2,5,6,7], [3,4,8,9,10]]
*/
cat :: proc(
	tensors: []^Tensor($T),
	dim: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	if len(tensors) == 0 {
		panic("Cannot concatenate empty tensor list")
	}

	first := tensors[0]
	if dim >= uint(len(first.shape)) {
		panic("Dimension index out of range")
	}

	// Verify all tensors have compatible shapes
	for tensor in tensors[1:] {
		if len(tensor.shape) != len(first.shape) {
			panic("All tensors must have same number of dimensions")
		}
		for i in 0 ..< len(first.shape) {
			if i != int(dim) && tensor.shape[i] != first.shape[i] {
				panic("All tensors must have same size in non-concatenated dimensions")
			}
		}
	}

	// Calculate output shape
	output_shape := make([]uint, len(first.shape), context.temp_allocator)
	copy(output_shape, first.shape)

	total_dim_size: uint = 0
	for tensor in tensors {
		total_dim_size += tensor.shape[dim]
	}
	output_shape[dim] = total_dim_size

	result := tensor_alloc(T, output_shape, true, allocator, loc)

	// Get all strided data upfront
	tensor_datas := make([][]T, len(tensors), context.temp_allocator)

	for tensor, i in tensors {
		tensor_datas[i], _ = get_strided_data(tensor, allocator = context.temp_allocator)
	}

	// Calculate sizes for proper copying
	outer_size := uint(1)
	for i in 0 ..< dim {
		outer_size *= first.shape[i]
	}

	inner_size := uint(1)
	for i in dim + 1 ..< uint(len(first.shape)) {
		inner_size *= first.shape[i]
	}

	// Copy with correct interleaving
	result_offset := uint(0)
	for outer in 0 ..< outer_size {
		for tensor, tensor_idx in tensors {
			chunk_size := tensor.shape[dim] * inner_size
			src_offset := outer * chunk_size

			copy(
				result.data[result_offset:result_offset + chunk_size],
				tensor_datas[tensor_idx][src_offset:src_offset + chunk_size],
			)
			result_offset += chunk_size
		}
	}

	return result
}

arange :: proc($T: typeid, n: uint, allocator := context.allocator) -> ^Tensor(T) {
	result := tensor_alloc(T, []uint{n}, true, allocator)
	for i in 0 ..< n {
		result.data[i] = T(i)
	}
	return result
}

/*
	Stacks tensors along a new dimension.

	Creates a new tensor with an additional dimension at the specified axis.
	All input tensors must have identical shapes.

	Parameters:
	  tensors: Array of tensors to stack (must all have same shape)
	  axis: Position where new dimension is inserted (0 to rank inclusive)

	Returns tensor with shape [...dims_before, len(tensors), ...dims_after].

	Example:
	  // Create three 2x3 tensors
	  a := new_with_init([]f32{1,2,3,4,5,6}, {2,3})
	  b := new_with_init([]f32{7,8,9,10,11,12}, {2,3})
	  c := new_with_init([]f32{13,14,15,16,17,18}, {2,3})

	  // Stack along axis 0: shape [3, 2, 3]
	  // Result: [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]], [[13,14,15],[16,17,18]]]
	  s0 := stack({a, b, c}, 0)

	  // Stack along axis 1: shape [2, 3, 3]
	  // Result: [[[1,2,3],[7,8,9],[13,14,15]], [[4,5,6],[10,11,12],[16,17,18]]]
	  s1 := stack({a, b, c}, 1)

	  // Stack along axis 2: shape [2, 3, 3]
	  // Result: [[[1,7,13],[2,8,14],[3,9,15]], [[4,10,16],[5,11,17],[6,12,18]]]
	  s2 := stack({a, b, c}, 2)
*/
stack :: proc(tensors: []^Tensor($T), axis: int, allocator := context.allocator) -> ^Tensor(T) {
	if len(tensors) == 0 {
		panic("Cannot stack empty tensor list")
	}

	// All tensors must have same shape
	first_shape := tensors[0].shape
	for t in tensors[1:] {
		if !slice.equal(t.shape, first_shape) {
			panic("All tensors must have same shape for stack")
		}
	}

	// Create new shape with extra dimension
	new_shape := make([]uint, len(first_shape) + 1, allocator)
	for i in 0 ..< axis {
		new_shape[i] = first_shape[i]
	}
	new_shape[axis] = uint(len(tensors))
	for i in axis ..< len(first_shape) {
		new_shape[i + 1] = first_shape[i]
	}

	result := tensor_alloc(T, new_shape, true, allocator)

	// Get all strided data upfront
	tensor_datas := make([][]T, len(tensors), context.temp_allocator)

	for tensor, i in tensors {
		tensor_datas[i], _ = get_strided_data(tensor, allocator = context.temp_allocator)
	}

	// Calculate sizes for proper striding
	outer_size := uint(1)
	for i in 0 ..< axis {
		outer_size *= first_shape[i]
	}

	inner_size := uint(1)
	for i in axis ..< len(first_shape) {
		inner_size *= first_shape[i]
	}

	// Copy data with correct interleaving
	result_offset := uint(0)
	for outer in 0 ..< outer_size {
		for tensor_idx in 0 ..< len(tensors) {
			src_offset := outer * inner_size
			copy(
				result.data[result_offset:result_offset + inner_size],
				tensor_datas[tensor_idx][src_offset:src_offset + inner_size],
			)
			result_offset += inner_size
		}
	}

	return result
}

broadcast_as :: proc(
	tensor: ^Tensor($T),
	target_shape: []uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	// Check if shapes are broadcastable
	_, broadcastable := shape_broadcastable(tensor.shape, target_shape, allocator)
	if !broadcastable {
		panic("Cannot broadcast to target shape")
	}

	// Create result with target shape
	result := tensor_alloc(T, target_shape, true, allocator)

	// Compute broadcast strides
	broadcast_strides := broadcast_strides(
		tensor.shape,
		target_shape,
		tensor.strides,
		context.temp_allocator,
	)

	// Fill result using broadcast indexing
	total_elements := shape_to_size(target_shape)
	for i in 0 ..< total_elements {
		src_idx := compute_strided_index(target_shape, broadcast_strides, i)
		result.data[i] = tensor.data[src_idx]
	}

	return result
}

Range :: struct {
	start: int,
	end:   int,
	step:  int,
}

Slice :: union {
	int,
	Range,
}

// Procedure group to help constructing ranges to be used for tensor slicing
R :: proc {
	R_upper,
	R_lower_upper,
	R_lower_upper_step,
}
@(private = "file")
R_upper :: proc(upper: int) -> Range {
	return Range{0, upper, 1}
}
@(private = "file")
R_lower_upper :: proc(lower, upper: int) -> Range {
	return Range{lower, upper, 1}
}
@(private = "file")
R_lower_upper_step :: proc(lower, upper, step: int) -> Range {
	return Range{lower, upper, step}
}

scalar :: proc(v: $T, allocator := context.allocator) -> ^Tensor(T) {
	return full(T, v, {}, allocator)
}

scale :: proc(x: ^Tensor($T), v: T) {
	for v, i in x.data {
		x.data[i] *= v
	}
}

/*
	Extracts a sub-tensor by slicing along each dimension.

	Supports two slice types per dimension:
	  - Range{start, end, step}: Preserves dimension, selects elements [start:end:step).
	    NOTE: use `{}` to slice the entire dimension.
	  - int: Selects single index and squeezes dimension (unless keepdims=true)

	Negative indices count from the end (-1 is last element).
	Range.end=0 means slice to the end of that dimension.
	Fewer slices than tensor rank will slice only leading dimensions.

	Parameters:
	  input: Source tensor to slice
	  slices: Array of Range or int values for each dimension
	  keepdims: If true, dimensions indexed with int retain size 1 instead of being squeezed
	  allocator: Memory allocator for output tensor

	Returns new tensor containing the sliced data.

	Example:
	  // Create 3x4 tensor: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
	  t := new_with_init([]f32{1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4})
	
	  // Get row 1 (second row): [5,6,7,8]
	  row := slice(t, {1})                    // shape [4] - first dim squeezed
	
	  // Get column 2 (third column): [3,7,11]
	  col := slice(t, {{}, 2})                // shape [3] - second dim squeezed
	
	  // Get 2x2 submatrix from rows 1:3, cols 1:3: [[6,7], [10,11]]
	  sub := slice(t, {R(1, 3), R(1, 3)})     // shape [2, 2]
	
	  // Get last 2 rows: [[5,6,7,8], [9,10,11,12]]
	  last := slice(t, {R(-2, 0)})            // shape [2, 4]
	
	  // Every other column: [[1,3], [5,7], [9,11]]
	  skip := slice(t, {{}, R(0, 0, 2)})      // shape [3, 2]
*/
slice :: proc(
	input: ^Tensor($T),
	slices: []Slice,
	keepdims := false,
	allocator := context.allocator,
) -> ^Tensor(T) {
	trace_slice := trace.TRACE_FUNCTION("slice")
	defer trace.end_scoped_trace(trace_slice)

	rank := len(input.shape)
	assert(len(slices) <= rank, "ranges exceeding tensor rank")

	slices_dyn := slice.to_dynamic(slices, context.temp_allocator)
	// We add trailing slices with ranges covering the entire dimension
	for len(slices_dyn) < rank {
		append(&slices_dyn, Range{})
	}

	output_shape: [16]uint
	squeezed_mask: [16]bool
	starts: [16]int
	steps: [16]int
	input_strides := input.strides

	assert(rank <= 16, "rank too high")

	// Normalize ranges and calculate output shape
	out_rank := rank
	for i in 0 ..< rank {
		slice := slices_dyn[i]
		range: Range
		switch s in slice {
		case Range:
			range = s
		case int:
			squeezed_mask[i] = true
			out_rank -= 1
			range = Range{s, s + 1, 1}
		}

		dim_size := int(input.shape[i])
		start := range.start if range.start >= 0 else range.start + dim_size
		end := range.end if range.end != 0 else dim_size
		if end < 0 do end += dim_size
		step := range.step if range.step != 0 else 1

		start = clamp(start, 0, dim_size)
		end = clamp(end, 0, dim_size)

		output_shape[i] = uint(max(0, (end - start + step - 1) / step))
		starts[i] = start
		steps[i] = step
	}
	output_shape_squeezed := make([dynamic]uint, context.temp_allocator)
	for must_squeeze, i in squeezed_mask {
		if !must_squeeze do append(&output_shape_squeezed, output_shape[i])
	}
	final_output_shape := keepdims ? output_shape[:rank] : output_shape_squeezed[:out_rank]

	output := tensor_alloc(T, final_output_shape, true, allocator)

	// Find innermost contiguous dimensions
	contiguous_from := rank
	for i := rank - 1; i >= 0; i -= 1 {
		if steps[i] != 1 {
			contiguous_from = i + 1
			break
		}
	}

	when T == f32 {
		if contiguous_from < rank {
			// SIMD path for contiguous inner dimensions
			contiguous_elements := 1
			for i in contiguous_from ..< rank {
				contiguous_elements *= int(output_shape[i])
			}

			outer_elements := len(output.data) / contiguous_elements
			indices: [16]int

			for outer_idx in 0 ..< outer_elements {
				// Calculate base addresses
				temp := outer_idx
				for i := contiguous_from - 1; i >= 0; i -= 1 {
					indices[i] = temp % int(output_shape[i])
					temp /= int(output_shape[i])
				}

				input_base := 0
				for i in 0 ..< contiguous_from {
					input_pos := starts[i] + indices[i] * steps[i]
					input_base += input_pos * int(input_strides[i])
				}
				for i in contiguous_from ..< rank {
					input_base += starts[i] * int(input_strides[i])
				}

				output_base := outer_idx * contiguous_elements

				// SIMD copy of contiguous chunk
				for i := 0; i + 4 <= contiguous_elements; i += 4 {
					v := (^#simd[4]f32)(uintptr(&input.data[input_base]) + uintptr(i * 4))^
					(^#simd[4]f32)(uintptr(&output.data[output_base]) + uintptr(i * 4))^ = v
				}
				// remainder
				for i := 0; i < contiguous_elements; i += 1 {
					(^f32)(uintptr(&output.data[output_base]) + uintptr(i * 4))^ = (^f32)(
						uintptr(&input.data[input_base]) + uintptr(i * 4),
					)^
				}
			}
			return output
		}
	}

	// Fallback slicing
	total_elements := len(output.data)
	indices: [16]int

	for out_idx in 0 ..< total_elements {
		// Convert flat output index to n-dimensional indices
		temp := out_idx
		for i := rank - 1; i >= 0; i -= 1 {
			indices[i] = temp % int(output_shape[i])
			temp /= int(output_shape[i])
		}

		// Map to input index
		input_idx := 0
		for i in 0 ..< rank {
			input_pos := starts[i] + indices[i] * steps[i]
			input_idx += input_pos * int(input_strides[i])
		}

		output.data[out_idx] = input.data[input_idx]
	}

	return output
}

// Repeat tensor along specified dimensions (tiling)
// e.g., tensor shape (2, 3) with repeats (2, 1) -> (4, 3)
repeat :: proc(
	tensor: ^Tensor($T),
	repeats: []uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	if len(repeats) != len(tensor.shape) {
		panic("Repeats must have same length as tensor dimensions")
	}

	// Calculate new shape
	new_shape := make([]uint, len(tensor.shape), context.temp_allocator)
	for i in 0 ..< len(tensor.shape) {
		new_shape[i] = tensor.shape[i] * repeats[i]
	}

	result := tensor_alloc(T, new_shape, true, allocator)

	// Get source data
	src_data, src_allocated := get_strided_data(tensor, allocator = allocator)
	defer if src_allocated do delete(src_data, allocator)

	// Calculate strides for iterating through result
	result_strides := make([]uint, len(new_shape), context.temp_allocator)
	stride := uint(1)
	for i := len(new_shape) - 1; i >= 0; i -= 1 {
		result_strides[i] = stride
		stride *= new_shape[i]
	}

	// Fill result by tiling
	total_elements := shape_to_size(new_shape)
	for i in 0 ..< total_elements {
		// Calculate coordinates in result tensor
		coords := make([]uint, len(new_shape), context.temp_allocator)
		temp := i
		for dim := len(new_shape) - 1; dim >= 0; dim -= 1 {
			coords[dim] = temp % new_shape[dim]
			temp /= new_shape[dim]
		}

		// Map to source coordinates
		src_coords := make([]uint, len(tensor.shape), context.temp_allocator)
		for dim in 0 ..< len(tensor.shape) {
			src_coords[dim] = coords[dim] % tensor.shape[dim]
		}

		// Calculate source index
		src_idx := uint(0)
		for dim in 0 ..< len(tensor.shape) {
			src_idx += src_coords[dim] * tensor.strides[dim]
		}

		result.data[i] = src_data[src_idx]
	}

	return result
}

repeat_interleave :: proc(
	tensor: ^Tensor($T),
	repeats: uint,
	dim: uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	if !tensor.contiguous {
		panic("repeat_interleave requires contiguous tensor")
	}
	if dim >= uint(len(tensor.shape)) {
		panic("Dimension out of bounds")
	}

	// Calculate new shape
	new_shape := make([]uint, len(tensor.shape), allocator)
	copy(new_shape, tensor.shape)
	new_shape[dim] *= repeats

	result := tensor_alloc(T, new_shape, true, allocator)

	// Calculate sizes for iteration
	size_before := uint(1) // Product of dimensions before 'dim'
	size_at := tensor.shape[dim] // Size of dimension to repeat
	size_after := uint(1) // Product of dimensions after 'dim'

	for i in 0 ..< dim {
		size_before *= tensor.shape[i]
	}
	for i in dim + 1 ..< uint(len(tensor.shape)) {
		size_after *= tensor.shape[i]
	}

	// Process all data
	dst_idx := uint(0)

	// Iterate over all positions before the repeat dimension
	for before in 0 ..< size_before {
		// For each element at the repeat dimension
		for at in 0 ..< size_at {
			src_offset := (before * size_at + at) * size_after

			// Repeat 'repeats' times
			for r in 0 ..< repeats {
				// Copy the chunk after the dimension
				if size_after == 1 {
					// Scalar case - direct assignment
					result.data[dst_idx] = tensor.data[src_offset]
					dst_idx += 1
				} else if size_after <= 16 {
					// Small chunk - unrolled copy for better performance
					#no_bounds_check for i in 0 ..< size_after {
						result.data[dst_idx + i] = tensor.data[src_offset + i]
					}
					dst_idx += size_after
				} else {
					// Large chunk - use memcpy
					copy(
						result.data[dst_idx:dst_idx + size_after],
						tensor.data[src_offset:src_offset + size_after],
					)
					dst_idx += size_after
				}
			}
		}
	}

	return result
}

// Special case for dim=0 (batch dimension) - most common case
repeat_interleave_batch :: proc(
	tensor: ^Tensor($T),
	repeats: uint,
	allocator := context.allocator,
) -> ^Tensor(T) {
	if !tensor.contiguous {
		panic("repeat_interleave requires contiguous tensor")
	}

	batch_size := tensor.shape[0]
	elements_per_batch := len(tensor.data) / int(batch_size)

	new_shape := make([]uint, len(tensor.shape), allocator)
	copy(new_shape, tensor.shape)
	new_shape[0] *= repeats

	result := tensor.tensor_alloc(T, new_shape, true, allocator)

	// Optimized for cache-friendly access pattern
	dst_offset := uint(0)
	for b in 0 ..< batch_size {
		src_start := b * uint(elements_per_batch)
		src_end := src_start + uint(elements_per_batch)

		for r in 0 ..< repeats {
			copy(
				result.data[dst_offset:dst_offset + uint(elements_per_batch)],
				tensor.data[src_start:src_end],
			)
			dst_offset += uint(elements_per_batch)
		}
	}

	return result
}

squeeze :: proc(
	tensor: ^Tensor($T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	shape_out := make([dynamic]uint, context.temp_allocator)
	for s in tensor.shape {
		if s > 1 do append(&shape_out, s)
	}
	return reshape(tensor, shape_out[:], allocator, loc)
}

unsqueeze :: proc(tensor: ^Tensor($T), dim: uint, allocator := context.allocator) -> ^Tensor(T) {
	if dim > uint(len(tensor.shape)) {
		panic("Dimension out of bounds for unsqueeze")
	}

	new_shape := make([]uint, len(tensor.shape) + 1, context.temp_allocator)
	for i in 0 ..< dim do new_shape[i] = tensor.shape[i]

	new_shape[dim] = 1
	for i in dim ..< uint(len(tensor.shape)) do new_shape[i + 1] = tensor.shape[i]

	result := tensor_alloc(T, new_shape, false, allocator)
	result.data = tensor.data

	return result
}

flatten :: proc(
	t: ^Tensor($T),
	from: uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> ^Tensor(T) {
	assert(from < len(t.shape) - 1, "flatten dimension out of bound")
	new_ndim := from + 1
	new_shape := make([dynamic]uint, context.temp_allocator)
	rest_size := 1
	for i in 0 ..< len(t.shape) {
		if uint(i) < from {
			append(&new_shape, uint(t.shape[i]))
		} else {
			rest_size *= int(t.shape[i])
		}
	}
	append(&new_shape, uint(rest_size))
	result := tensor_alloc(T, new_shape[:], false, allocator)
	result.data = t.data
	return result
}

// Flatten all dimensions to 1D
flatten_all :: proc(tensor: ^Tensor($T), allocator := context.allocator) -> ^Tensor(T) {
	total_size := shape_to_size(tensor.shape)
	result := tensor_alloc(T, []uint{total_size}, true, allocator)

	data, alloc := get_strided_data(tensor, allocator = allocator)
	defer if alloc do delete(data, allocator)

	copy(result.data, data)
	return result
}

// Applies softmax along the last dimension of a tensor
// For a tensor of shape [..., N], applies softmax to each [...] slice over N elements
softmax_last_dim_inplace :: proc(t: ^Tensor($T)) {
	softmax_trace := trace.TRACE_FUNCTION("softmax_inplace")
	defer trace.end_scoped_trace(softmax_trace)

	// Calculate the number of softmax operations needed
	// This is the product of all dimensions except the last
	num_rows := uint(1)
	for i in 0 ..< len(t.shape) - 1 {
		num_rows *= t.shape[i]
	}

	last_dim := t.shape[len(t.shape) - 1]
	dim_stride := t.strides[uint(len(t.shape) - 1)]

	#no_bounds_check for row_idx in 0 ..< num_rows {
		row_offset := row_idx * last_dim
		row_data := t.data[row_offset:][:last_dim]

		when T == f32 {
			col := uint(0)
			max_vec := #simd[4]f32 {
				math.inf_f32(-1),
				math.inf_f32(-1),
				math.inf_f32(-1),
				math.inf_f32(-1),
			}

			// Pass 1: Find max
			for ; col + 4 <= last_dim; col += 4 {
				vals := #simd[4]f32 {
					row_data[col + 0 * dim_stride],
					row_data[col + 1 * dim_stride],
					row_data[col + 2 * dim_stride],
					row_data[col + 3 * dim_stride],
				}
				max_vec = simd_backend.max_f32(max_vec, vals)
			}

			// Reduce the vector to scalar
			max_val := simd.reduce_max(max_vec)

			// Handle remainder
			for ; col < last_dim; col += 1 {
				max_val = max(max_val, row_data[col])
			}

			// Pass 2: Compute exp(x - max) and accumulate sum
			sum := f32(0)
			col = 0
			sum_vec := #simd[4]f32{0, 0, 0, 0}
			vals := #simd[4]f32{0, 0, 0, 0}

			for ; col + 4 <= last_dim; col += 4 {
				vals := #simd[4]f32 {
					row_data[col + 0 * dim_stride],
					row_data[col + 1 * dim_stride],
					row_data[col + 2 * dim_stride],
					row_data[col + 3 * dim_stride],
				}
				vals = simd.sub(vals, #simd[4]f32{max_val, max_val, max_val, max_val})

				exp_vals: #simd[4]f32
				simd_backend.expf_4(&exp_vals, &vals)

				#unroll for i in 0 ..< 4 {
					row_data[col + uint(i)] = simd.extract(exp_vals, i)
				}
				sum_vec = simd.add(
					sum_vec,
					#simd[4]f32 {
						row_data[col + 0 * dim_stride],
						row_data[col + 1 * dim_stride],
						row_data[col + 2 * dim_stride],
						row_data[col + 3 * dim_stride],
					},
				)
			}

			// Reduce by SIMD
			sum = simd.reduce_add_bisect(sum_vec)
			// Reduce remainders
			for ; col < last_dim; col += 1 {
				val := math.exp(row_data[col] - max_val)
				row_data[col] = val
				sum += val
			}

			inv_sum := f32(1) / sum
			inv_sum_vec := #simd[4]f32{inv_sum, inv_sum, inv_sum, inv_sum}

			col = 0
			for ; col + 4 <= last_dim; col += 4 {
				vals := #simd[4]f32 {
					row_data[col + 0 * dim_stride],
					row_data[col + 1 * dim_stride],
					row_data[col + 2 * dim_stride],
					row_data[col + 3 * dim_stride],
				}
				vals = simd.mul(vals, inv_sum_vec)
				#unroll for i in 0 ..< 4 {
					row_data[col + uint(i)] = simd.extract(vals, i)
				}
			}

			for ; col < last_dim; col += 1 {
				row_data[col] *= inv_sum
			}

		} else {
			max_val := row_data[0]
			for col in 1 ..< last_dim {
				if row_data[col] > max_val {
					max_val = row_data[col]
				}
			}

			sum := T(0)
			for col in 0 ..< last_dim {
				val := math.exp(row_data[col] - max_val)
				row_data[col] = val
				sum += val
			}

			inv_sum := T(1) / sum
			for col in 0 ..< last_dim {
				row_data[col] *= inv_sum
			}
		}
	}
}

softmax_last_dim :: proc(t: ^Tensor($T), allocator := context.allocator) -> ^Tensor(T) {
	result := clone(t, allocator)
	softmax_last_dim_inplace(result)
	return result
}

// Applies softmax along the specified dimension of a tensor
// Requires contiguous tensor for predictable performance
softmax_inplace :: proc(t: ^Tensor($T), dim: uint) {
	softmax_trace := trace.TRACE_FUNCTION("softmax_inplace")
	defer trace.end_scoped_trace(softmax_trace)

	ensure(t.contiguous, "softmax requires contiguous tensor")
	ensure(dim < uint(len(t.shape)), "dimension index out of range for softmax")

	dim_size := t.shape[dim]
	ensure(dim_size > 0, "cannot apply softmax to dimension of size 0")

	// Last dimension is special - we can process contiguous chunks
	if dim == uint(len(t.shape) - 1) {
		softmax_last_dim_inplace(t)
		return
	}

	// For middle dimensions, we need to jump around in memory
	// Example: shape [2, 3, 4, 5], dim=2
	// We process 2*3*5=30 slices, each of size 4

	dim_stride := t.strides[dim]

	// Total number of 1D slices to softmax
	num_slices := uint(1)
	for i in 0 ..< len(t.shape) {
		if i != int(dim) {
			num_slices *= t.shape[i]
		}
	}

	// Size of the "inner block" - dimensions after our target dim
	// This helps us calculate where each slice starts
	inner_size := uint(1)
	for i := int(dim) + 1; i < len(t.shape); i += 1 {
		inner_size *= t.shape[i]
	}

	#no_bounds_check for slice_idx in 0 ..< num_slices {
		// Figure out where this slice starts in the flat array
		// block_idx: which "outer" repetition are we in?
		// inner_idx: offset within that block
		block_idx := slice_idx / inner_size
		inner_idx := slice_idx % inner_size
		base_idx := block_idx * dim_stride * dim_size + inner_idx

		when T == f32 {
			slice_data := t.data[base_idx:]

			// Pass 1: Find max
			// We gather strided values into SIMD registers for comparison
			max_val := math.inf_f32(-1)
			i := uint(0)
			max_vec := #simd[4]f32 {
				math.inf_f32(-1),
				math.inf_f32(-1),
				math.inf_f32(-1),
				math.inf_f32(-1),
			}

			// Process 4 elements at a time by gathering from strided locations
			for ; i + 4 <= dim_size; i += 4 {
				idx := i * dim_stride
				vals := #simd[4]f32 {
					slice_data[idx + 0 * dim_stride],
					slice_data[idx + 1 * dim_stride],
					slice_data[idx + 2 * dim_stride],
					slice_data[idx + 3 * dim_stride],
				}
				max_vec = simd_backend.max_f32(max_vec, vals)
			}

			// Reduce the vector to scalar
			max_val = simd.reduce_max(max_vec)

			// Handle remainder
			for ; i < dim_size; i += 1 {
				val := slice_data[i * dim_stride]
				max_val = max(max_val, val)
			}

			// Pass 2: Compute exp(x - max) and accumulate sum
			sum := f32(0)
			i = 0
			sum_vec := #simd[4]f32{0, 0, 0, 0}

			// Check if data is consecutive
			if dim_stride == 1 {
				// Fast path: consecutive data, real SIMD benefit
				for ; i + 4 <= dim_size; i += 4 {
					// Load 4 consecutive values
					vals := (^#simd[4]f32)(&slice_data[i])^

					// Subtract max
					vals = simd.sub(vals, #simd[4]f32{max_val, max_val, max_val, max_val})

					exp_vals: #simd[4]f32
					simd_backend.expf_4(&exp_vals, (^#simd[4]f32)(&vals))

					// SIMD store
					(^#simd[4]f32)(&slice_data[i])^ = exp_vals
					sum_vec = simd.add(sum_vec, exp_vals)
				}

				sum = simd.reduce_add_bisect(sum_vec)

				// Handle remainder
				for ; i < dim_size; i += 1 {
					val := math.exp(slice_data[i] - max_val)
					slice_data[i] = val
					sum += val
				}
			} else {
				// Strided data: just go scalar, it's cleaner and likely faster
				for i in 0 ..< dim_size {
					idx := i * dim_stride
					val := math.exp(slice_data[idx] - max_val)
					slice_data[idx] = val
					sum += val
				}
			}

			// Pass 3: Normalize by sum
			inv_sum := f32(1) / sum
			for i in 0 ..< dim_size {
				slice_data[i * dim_stride] *= inv_sum
			}
		} else {
			// No SIMD for other types
			softmax_slice_scalar(t.data[base_idx:], dim_size, dim_stride, T)
		}
	}
}

@(private)
softmax_slice_scalar :: proc(slice_data: []$T, dim_size: uint, dim_stride: uint, $type: typeid) {
	// Standard 3-pass softmax: max, exp, normalize
	max_val := slice_data[0]
	for i in 1 ..< dim_size {
		idx := i * dim_stride
		if slice_data[idx] > max_val {
			max_val = slice_data[idx]
		}
	}

	sum := T(0)
	for i in 0 ..< dim_size {
		idx := i * dim_stride
		val := math.exp(slice_data[idx] - max_val)
		slice_data[idx] = val
		sum += val
	}

	inv_sum := T(1) / sum
	for i in 0 ..< dim_size {
		slice_data[i * dim_stride] *= inv_sum
	}
}

// Allocating version
softmax :: proc(t: ^Tensor($T), dim: uint, allocator := context.allocator) -> ^Tensor(T) {
	result := clone(t, allocator)
	softmax_inplace(result, dim)
	return result
}
