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

}
