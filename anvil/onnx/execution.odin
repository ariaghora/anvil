package onnx

import "../tensor"
import "core:fmt"
import "core:slice"

run :: proc(
	model: ^ONNX($T),
	inputs: map[string]^tensor.Tensor(T),
	allocator := context.allocator,
) -> ONNX_Error {
	// set inputs to the models
	for k, v in inputs do model.graph.initializers[k] = v

	input_names := slice.map_keys(inputs, context.temp_allocator) or_return
	orders := determine_execution_order(model.graph, input_names, allocator)
	for op_idx in orders {
		op := model.graph.nodes[op_idx]

		// Sanity check for each inputs before node execution
		for iname in input_names {
			ensure_tensor_non_nil(model.graph.initializers[iname], iname, op.op_type, true)
		}

		switch op.op_type {
		case "Conv":
			run_conv(op.inputs[:], op.outputs[:], model.graph, model.opset_version) or_return
		case:
			return Unsupported_Op{op.op_type}
		}
	}
	return nil
}

run_conv :: proc(
	inputs, outputs: []string,
	graph: ^Graph($T),
	opset: i64,
	allocator := context.allocator,
) -> (
	err: ONNX_Error,
) {
	x := graph.initializers[inputs[0]]
	ensure(
		len(x.shape) == 4,
		fmt.tprintf(
			"Conv only supports 2D convolutions. Expected input dimension is 4, got $d",
			len(x.shape),
		),
	)
	ensure_batched_image_shape(x, inputs[0], "Conv", is_input = true)
	w := graph.initializers[inputs[1]]
	b := graph.initializers[inputs[2]] or_else nil

	return
}

@(private)
ensure_batched_image_shape :: proc(x: ^tensor.Tensor($T), name, op_type: string, is_input: bool) {
	ensure(
		len(x.shape) == 4,
		fmt.tprintf("Input for %s must be a 4D tensor, got %dD", op_type, len(x.shape)),
	)
}

@(private)
ensure_tensor_non_nil :: proc(x: ^tensor.Tensor($T), name, op_type: string, is_input: bool) {
	ensure(
		x != nil,
		fmt.tprintfln(
			"The %s %s, `%s`, is nil",
			op_type,
			is_input ? "input for" : "output of",
			name,
		),
	)
}

@(private = "file")
determine_execution_order :: proc(
	graph: ^Graph($T),
	input_names: []string,
	allocator := context.allocator,
) -> []int {
	ready_tensors := make(map[string]bool, allocator)

	// Mark initializers as ready
	for name, _ in graph.initializers {
		ready_tensors[name] = true
	}

	// Mark user inputs as ready
	for input_name in input_names {
		ready_tensors[input_name] = true
	}

	// Build a map of which nodes produce which tensors
	tensor_producers := make(map[string]int, allocator) // tensor name -> node index
	for node, idx in graph.nodes {
		for output in node.outputs {
			tensor_producers[output] = idx
		}
	}

	// Find ACTUAL required inputs (not produced by any node, not initializers)
	required_inputs := make(map[string]bool, allocator)
	for node in graph.nodes {
		for input in node.inputs {
			if input == "" do continue
			_, is_init := graph.initializers[input]
			_, is_produced := tensor_producers[input]
			_, is_provided := ready_tensors[input]

			if !is_init && !is_produced && !is_provided {
				required_inputs[input] = true
			}
		}
	}

	if len(required_inputs) > 0 {
		fmt.printf("ERROR: Missing required inputs:\n")
		for input, _ in required_inputs {
			fmt.printf("  - %s\n", input)
		}
		panic("Graph execution impossible without these inputs")
	}

	// Topological sort
	execution_order := make([dynamic]int, allocator)
	nodes_remaining := make(map[int]bool, allocator)

	// Initialize remaining nodes
	for i in 0 ..< len(graph.nodes) {
		nodes_remaining[i] = true
	}

	// Keep looking for nodes that can execute
	for len(nodes_remaining) > 0 {
		made_progress := false

		for idx in nodes_remaining {
			node := graph.nodes[idx]

			// Check if all inputs are ready
			can_execute := true
			for input in node.inputs {
				if input == "" do continue // Optional input
				if !ready_tensors[input] {
					can_execute = false
					break
				}
			}

			if can_execute {
				append(&execution_order, idx)

				// Mark outputs as ready
				for output in node.outputs {
					ready_tensors[output] = true
				}

				delete_key(&nodes_remaining, idx)
				made_progress = true
				break // Restart to maintain order
			}
		}

		if !made_progress {
			// Find what's blocking us
			fmt.printf(
				"ERROR: Cannot schedule %d nodes. Dependency issues:\n",
				len(nodes_remaining),
			)
			for idx in nodes_remaining {
				node := graph.nodes[idx]
				fmt.printf("  Node %d (%s) waiting for: ", idx, node.op_type)
				missing_count := 0
				for input in node.inputs {
					if input != "" && !ready_tensors[input] {
						fmt.printf("%s ", input)
						missing_count += 1
					}
				}
				if missing_count == 0 {
					fmt.printf("Cycle detected")
				}
				fmt.println()
			}
			panic("Graph has cycles or unresolvable dependencies")
		}
	}

	return execution_order[:]
}
