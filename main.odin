package main

import "core:fmt"
import "core:net"
import "libs/matmul"

import "libs/tensor"
import tf "libs/transformer"


main :: proc() {
	sam := tf.new_tiny(f32)
	tf.free_tiny(sam)

	t1 := tensor.ones(f32, []uint{8, 8}, context.temp_allocator)
	tensor.print(t1, max_print_elements_per_dim = 6)
	tensor.free_tensor(t1, allocator = context.temp_allocator)

	t2 := tensor.new_with_init([]i32{3, -2, 10}, []uint{3, 1}, allocator = context.temp_allocator)
	tensor.print(t2, max_print_elements_per_dim = 4)
	tensor.free_tensor(t2, allocator = context.temp_allocator)


	// // f32
	// a32 := []f32{1, 2, 3, 4}
	// b32 := []f32{5, 6, 7, 8}
	// c32 := matmul.matmul(a32, b32, 2, 2, 2)
	// fmt.println(c32)

	// // f64
	// a64 := []f64{1, 2, 3, 4}
	// b64 := []f64{5, 6, 7, 8}
	// c64 := matmul.matmul(a64, b64, 2, 2, 2)
	// fmt.println(c64)
}
