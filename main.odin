package main

import "core:fmt"
import "core:math/rand"
import vmem "core:mem/virtual"
import "libs/tensor"
import "libs/trace"
import tf "libs/transformer"

main :: proc() {
	arena: vmem.Arena
	arena_err := vmem.arena_init_growing(&arena)
	ensure(arena_err == nil)
	arena_alloc := vmem.arena_allocator(&arena)
	defer vmem.arena_destroy(&arena)

	trace.init_trace()
	defer trace.finish_trace()

	main_trace := trace.TRACE_FUNCTION("main")
	defer trace.end_scoped_trace(main_trace)

	imgsz := uint(1024)

	model_init_trace := trace.TRACE_SECTION("model_initialization")
	model := tf.new_tiny_vit_5m(f32, imgsz, arena_alloc)
	trace.end_scoped_trace(model_init_trace)

	input_prep_trace := trace.TRACE_SECTION("input_preparation")
	input_size := 1 * 3 * imgsz * imgsz
	input_data := make([]f32, input_size, arena_alloc)

	for i in 0 ..< len(input_data) {
		input_data[i] = (rand.float32() - 0.5) * 0.1 // Small values [-0.05, 0.05]
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, imgsz, imgsz}, arena_alloc)
	trace.end_scoped_trace(input_prep_trace)

	trace.trace_instant("starting_forward_pass")
	forward_trace := trace.TRACE_SECTION("tiny_vit_forward_pass")
	output := tf.forward_tiny_vit_5m(model, input, arena_alloc)
	trace.end_scoped_trace(forward_trace)
	trace.trace_instant("forward_pass_completed")

}
