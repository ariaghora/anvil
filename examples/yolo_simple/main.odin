package main

import "../../anvil/models/yolo"
import st "../../anvil/safetensors/"
import "../../anvil/tensor"
import "../../anvil/trace"
import "core:fmt"
import "core:mem"
import "core:slice"
import "core:time"

import "../../anvil/plot"


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

	trace.init_trace()
	defer trace.finish_trace()

	safetensors_ref, err_st_ref := st.read_from_file(f32, "weights/yolo_tensors.safetensors")
	ensure(err_st_ref == nil)
	defer st.free_safe_tensors(safetensors_ref)
	input_t := safetensors_ref.tensors["original_input"]


	///
	// plot.visualize_tensor(
	// 	tensor.squeeze(
	// 		tensor.upsample_nearest_2d(input_t, 20, 20, context.temp_allocator),
	// 		context.temp_allocator,
	// 	),
	// 	"resize",
	// )


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

	t := time.now()
	// result := yolo.forward_yolo(model, input_t, context.allocator)

	// DarkNet Forward
	x2, x3, x5 := yolo.forward_dark_net(model.net, input_t, context.temp_allocator)

	// Neck Forward
	head_1, head_2, head_3 := yolo.forward_neck(model.fpn, x2, x3, x5, context.temp_allocator)

	// head Forward
	det1, det2, det3 := yolo.forward_head(
		model.head,
		head_1,
		head_2,
		head_3,
		context.temp_allocator,
	)

	fmt.println(time.since(t))

	// Plot references
	t_own := det3
	t_ref := safetensors_ref.tensors["det_3"]

	vmin, vmax, _ := slice.min_max(t_own.data)
	// for v, i in t_own.data do t_own.data[i] = (t_own.data[i] - vmin) / (vmax - vmin)
	vmin, vmax, _ = slice.min_max(t_ref.data)
	// for v, i in t_ref.data do t_ref.data[i] = (t_ref.data[i] - vmin) / (vmax - vmin)
	fmt.println(t_own.shape, t_ref.shape)

	t_ref_stack := tensor.squeeze(
		tensor.cat([]^tensor.Tensor(f32){t_own, t_ref}, 3, context.temp_allocator),
		context.temp_allocator,
	)

	c_sliced := 3
	plot.visualize_tensor(
		tensor.slice(t_ref_stack, {{c_sliced, c_sliced + 1, 1}, {}, {}}, context.temp_allocator),
		"comparison",
	)

}
