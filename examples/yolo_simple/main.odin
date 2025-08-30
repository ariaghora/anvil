package main

import "../../anvil/models/yolo"
import st "../../anvil/safetensors/"
import "core:fmt"
import "core:mem"
import rl "vendor:raylib"


main :: proc() {
	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		context.allocator = mem.tracking_allocator(&track)

		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}

	safetensors, err_st := st.read_from_file(f32, "weights/yolov8n.safetensors")
	ensure(err_st == nil)
	defer st.free_safe_tensors(safetensors)

	// NANO
	multiples := yolo.Multiples {
		depth = 0.33,
		width = 0.25,
		ratio = 2.0,
	}
	num_classes := uint(80)
	model := yolo.load_yolo(safetensors, multiples, num_classes, context.allocator)
	defer yolo.free_yolo(model)

	result := yolo.forward_yolo(model, nil)

}
