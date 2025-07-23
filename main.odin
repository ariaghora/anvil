package main

import "core:fmt"
import "core:net"
import "libs/matmul"

import "libs/safetensors"
import "libs/tensor"
import tf "libs/transformer"


main :: proc() {
	sam_safetensors, err := safetensors.read_from_file(
		f32,
		"./models/mobile_sam-tiny-vitt.safetensors",
		context.temp_allocator,
	)
	assert(err == nil)
	for k, _ in sam_safetensors.tensors {
		fmt.println(k)
	}

	sam := tf.new_tiny(f32)
	defer tf.free_tiny(sam)

	t1 := tensor.ones(f32, []uint{8, 8}, context.temp_allocator)
	tensor.print(t1, max_print_elements_per_dim = 6)

	t2 := tensor.new_with_init([]i32{3, -2, 10}, []uint{3, 1}, allocator = context.temp_allocator)
	tensor.print(t2, max_print_elements_per_dim = 4)
}
