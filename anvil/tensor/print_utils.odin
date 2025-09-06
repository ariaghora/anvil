package tensor

import "core:fmt"
import "core:strings"

// Pretty print tensor with numpy-like formatting
print :: proc(
	arr: ^Tensor($T),
	prefix: string = "tensor",
	backward_fn_name: string = "",
	max_print_elements_per_dim: uint = 6,
) {
	if len(arr.shape) == 0 {
		fmt.printf("%s()", prefix)
		return
	}

	builder := strings.builder_make()
	defer strings.builder_destroy(&builder)

	backward_fn_str :=
		len(backward_fn_name) > 0 ? fmt.tprintf(", backward_fn=%s", backward_fn_name) : ""
	strings.write_string(
		&builder,
		fmt.tprintf("%s(type=%v, shape=%v%s)\n", prefix, typeid_of(T), arr.shape, backward_fn_str),
	)

	indices := make([]uint, len(arr.shape))
	defer delete(indices)
	print_recursive(
		arr,
		arr.shape,
		arr.strides,
		0,
		indices,
		1,
		&builder,
		max_print_elements_per_dim,
	)
	strings.write_string(&builder, "\n")

	fmt.println(strings.to_string(builder))
}


print_value :: proc(builder: ^strings.Builder, value: $T, loc := #caller_location) {
	when T == f32 || T == f64 {
		strings.write_string(builder, fmt.tprintf("%.6f", value))
	} else {
		strings.write_string(builder, fmt.tprintf("% 6v", value))
	}}

@(private = "file")
print_recursive :: proc(
	arr: ^Tensor($T),
	shape: []uint,
	strides: []uint,
	depth: int,
	indices: []uint,
	indent: int,
	builder: ^strings.Builder,
	max_print_elements_per_dim: uint,
	loc := #caller_location,
) {
	if depth == len(shape) - 1 {
		// Innermost dimension
		strings.write_byte(builder, '[')

		dim_size := shape[depth]
		if dim_size > max_print_elements_per_dim {
			// First 3 elements
			for i in 0 ..< 3 {
				if i > 0 {strings.write_byte(builder, ' ')}
				indices[depth] = uint(i)
				index := compute_linear_index(indices, strides, loc)
				print_value(builder, arr.data[index], loc)
			}

			strings.write_string(builder, " ... ")

			// Last 3 elements
			for i in dim_size - 3 ..< dim_size {
				indices[depth] = uint(i)
				index := compute_linear_index(indices, strides)
				print_value(builder, arr.data[index])
				if i < dim_size - 1 {strings.write_byte(builder, ' ')}
			}
		} else {
			// Print all elements
			for i in 0 ..< dim_size {
				if i > 0 {strings.write_byte(builder, ' ')}
				indices[depth] = uint(i)
				index := compute_linear_index(indices, strides)
				print_value(builder, arr.data[index])
			}
		}
		strings.write_byte(builder, ']')
	} else {
		// Outer dimensions
		strings.write_byte(builder, '[')

		dim_size := shape[depth]
		if dim_size > max_print_elements_per_dim {
			// First 3
			for i in 0 ..< 3 {
				if i > 0 {
					strings.write_string(builder, ",\n")
					for j in 0 ..< depth + 1 {
						strings.write_string(builder, " ")
					}
				}
				indices[depth] = uint(i)
				print_recursive(
					arr,
					shape,
					strides,
					depth + 1,
					indices,
					indent + 1,
					builder,
					max_print_elements_per_dim,
				)
			}

			// Ellipsis line
			strings.write_string(builder, ",\n")
			for j in 0 ..< depth + 1 {
				strings.write_string(builder, " ")
			}
			strings.write_string(builder, "...")

			// Last 3
			for i in dim_size - 3 ..< dim_size {
				strings.write_string(builder, ",\n")
				for j in 0 ..< depth + 1 {
					strings.write_string(builder, " ")
				}
				indices[depth] = uint(i)
				print_recursive(
					arr,
					shape,
					strides,
					depth + 1,
					indices,
					indent + 1,
					builder,
					max_print_elements_per_dim,
				)
			}
		} else {
			// Print all
			for i in 0 ..< dim_size {
				if i > 0 {
					strings.write_string(builder, ",\n")
					for j in 0 ..< depth + 1 {
						strings.write_string(builder, " ")
					}
				}
				indices[depth] = uint(i)
				print_recursive(
					arr,
					shape,
					strides,
					depth + 1,
					indices,
					indent + 1,
					builder,
					max_print_elements_per_dim,
				)
			}
		}
		strings.write_string(builder, "]")
	}
}
