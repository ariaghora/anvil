package onnx

import "../tensor"
import "base:runtime"
import "core:fmt"
import "core:slice"

run :: proc(model: ^ONNX($T), inputs: map[string]^tensor.Tensor(T)) -> ONNX_Error {
	allocator := model.allocator

	// set inputs to the models
	for k, v in inputs do model.graph.tensors[k] = v

	input_names := slice.map_keys(inputs, context.temp_allocator) or_return
	orders := determine_execution_order(model.graph, input_names, context.temp_allocator)
	for op_idx, i in orders {
		op := model.graph.nodes[op_idx]

		// Sanity check for each inputs before node execution
		for iname in op.inputs {
			ensure_tensor_non_nil(model.graph.tensors[iname], iname, op.op_type, true) or_return
		}

		switch op.op_type {
		case "Conv":
			run_conv(
				op.inputs[:],
				op.outputs[:],
				op.attributes,
				model.graph,
				model.opset_version,
				allocator,
			) or_return
		case "Relu":
			run_relu(
				op.inputs[:],
				op.outputs[:],
				op.attributes,
				model.graph,
				model.opset_version,
				allocator,
			) or_return
		case "MaxPool":
			run_max_pool(
				op.inputs[:],
				op.outputs[:],
				op.attributes,
				model.graph,
				model.opset_version,
				allocator,
			) or_return
		case "Add":
			run_add(op, model, allocator) or_return
		case "Flatten":
			run_flatten(op, model, allocator) or_return
		case "Gemm":
			run_gemm(op, model, allocator) or_return
		case "GlobalAveragePool":
			run_global_average_pool(op, model, allocator) or_return
		case:
			return Unsupported_Op{op.op_type}
		}
	}
	return nil
}


@(private, require_results)
ensure_batched_image_shape :: proc(
	x: ^tensor.Tensor($T),
	name, op_type: string,
	is_input: bool,
	loc := #caller_location,
) -> ONNX_Error {
	if len(x.shape) != 4 {
		return Value_Error {
			fmt.tprintf("Input for %s must be a 4D tensor, got %dD", op_type, len(x.shape)),
		}
	}
	return nil
}

@(private, require_results)
ensure_tensor_non_nil :: proc(
	x: ^tensor.Tensor($T),
	name, op_type: string,
	is_input: bool,
) -> ONNX_Error {
	if x == nil {
		return Value_Error {
			fmt.tprintfln(
				"The %s %s, `%s`, is nil",
				is_input ? "input for" : "output of",
				op_type,
				name,
			),
		}
	}
	return nil
}

@(private = "file")
determine_execution_order :: proc(
	graph: ^Graph($T),
	input_names: []string,
	allocator := context.allocator,
) -> []int {
	ready_tensors := make(map[string]bool, allocator)

	// Mark initializers as ready
	for name, _ in graph.tensors {
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
			_, is_init := graph.tensors[input]
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

run_conv :: proc(
	inputs, outputs: []string,
	attributes: map[string]Attribute($T),
	graph: ^Graph(T),
	opset: i64,
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := graph.tensors[inputs[0]]
	ensure(
		len(x.shape) == 4,
		fmt.tprintf(
			"Conv only supports 2D convolutions. Expected input dimension is 4, got $d",
			len(x.shape),
		),
	)

	w := graph.tensors[inputs[1]]
	b := graph.tensors[inputs[2]] or_else nil
	auto_pad: string
	dilations, kernel_shape, pads, strides: []i64
	groups: uint

	output: ^tensor.Tensor(T)

	if opset < 1 {
		return Unsupported_Opset{"Conv", opset}
	} else if opset <= 22 {
		auto_pad = attributes["auto_pad"].(string) or_else "NOTSET"
		if auto_pad != "NOTSET" do return Unsupported_Attribute{"Conv", "auto_pad", auto_pad, opset}

		dilations = attributes["dilations"].([]i64) or_else {1}
		if !slice.all_of(dilations, dilations[0]) do return Malformed_Attribute{"Conv only support symmetrical dilations"}

		pads = attributes["pads"].([]i64) or_else {0}
		if !slice.all_of(pads, pads[0]) do return Malformed_Attribute{"Conv only support symmetrical paddings"}

		strides = attributes["strides"].([]i64) or_else {1}
		if !slice.all_of(strides, strides[0]) do return Malformed_Attribute{"Conv only support symmetrical strides"}

		// Kernel shape is omitted since we don't use it directly and can just infer from
		// the kernel tensor itself.
		// kernel_shape = ...

		// ensure(slice.all_of(dilations, dilations[0]), "Conv only support symmetrical dilations")
		stride := uint(strides[0])
		dilation := uint(dilations[0])
		padding := uint(pads[0])
		groups = uint(attributes["group"].(i64) or_else 1)

		// Current implementation
		output = tensor.conv2d_xwb(
			x,
			w,
			b,
			stride = stride,
			dilation = dilation,
			padding = padding,
			groups = groups,
			allocator = allocator,
		)
	} else {
		return Unsupported_Opset{"Conv", opset}
	}

	graph.tensors[outputs[0]] = output

	return
}

run_relu :: proc(
	inputs, outputs: []string,
	attributes: map[string]Attribute($T),
	graph: ^Graph(T),
	opset: i64,
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := graph.tensors[inputs[0]]
	graph.tensors[outputs[0]] = tensor.relu(x, allocator)
	return
}

run_max_pool :: proc(
	inputs, outputs: []string,
	attributes: map[string]Attribute($T),
	graph: ^Graph(T),
	opset: i64,
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := graph.tensors[inputs[0]]
	ensure_batched_image_shape(x, inputs[0], "MaxPool", true) or_return

	auto_pad: string
	output: ^tensor.Tensor(T)
	dilations, kernel_shape, pads, strides: []i64
	if opset < 1 {
		return Unsupported_Opset{"Conv", opset}
	} else if opset <= 22 {
		auto_pad = attributes["auto_pad"].(string) or_else "NOTSET"
		if auto_pad != "NOTSET" do return Unsupported_Attribute{"Conv", "auto_pad", auto_pad, opset}

		ceil_mode := attributes["ceil_mode"].(i64) or_else 0
		if ceil_mode != 0 do return Value_Error{"nonzero ceil_mode for MaxPool is not supported"}

		dilations = attributes["dilations"].([]i64) or_else {1}
		if !slice.all_of(dilations, dilations[0]) do return Malformed_Attribute{"Conv only support symmetrical dilations"}

		pads = attributes["pads"].([]i64) or_else {0}
		if !slice.all_of(pads, pads[0]) do return Malformed_Attribute{"Conv only support symmetrical paddings"}

		strides = attributes["strides"].([]i64) or_else {1}
		if !slice.all_of(strides, strides[0]) do return Malformed_Attribute{"Conv only support symmetrical strides"}

		kernel_size := attributes["kernel_shape"].([]i64)
		kernel_size_uint := [2]uint{uint(kernel_size[0]), uint(kernel_size[1])}

		// ensure(slice.all_of(dilations, dilations[0]), "Conv only support symmetrical dilations")
		stride := uint(strides[0])
		dilation := uint(dilations[0])
		if dilation != 1 do return Malformed_Attribute{"MaxPool Opset 11 can only support dilation of 1"}
		padding := uint(pads[0])

		// Current implementation
		output = tensor.max_pool_2d(x, kernel_size_uint, stride, padding, allocator = allocator)
	} else {
		return Unsupported_Opset{"Conv", opset}
	}
	graph.tensors[outputs[0]] = output
	return
}

run_flatten :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	axis := op.attributes["axis"].(i64) or_else 1
	if axis < 0 do return Value_Error{"Flatten axis must be positive"}
	model.graph.tensors[op.outputs[0]] = tensor.flatten(x, uint(axis), allocator)
	return
}
run_gemm :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	w := model.graph.tensors[op.inputs[1]]
	b := model.graph.tensors[op.inputs[2]] or_else nil
	if len(x.shape) != 2 || len(w.shape) != 2 do return Value_Error{"gemm requires both tensors have 2D"}
	// model.graph.tensors[op.outputs[0]] = tensor.global_avg_pool_2d(x, allocator)
	return
}

run_global_average_pool :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	ensure_batched_image_shape(x, op.inputs[0], "GlobalAveragePool", true) or_return
	model.graph.tensors[op.outputs[0]] = tensor.global_avg_pool_2d(x, allocator)
	return
}

run_add :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	inputs, outputs := op.inputs, op.outputs
	x, y := model.graph.tensors[inputs[0]], model.graph.tensors[inputs[1]]
	model.graph.tensors[outputs[0]] = tensor.add(x, y, allocator)
	return
}
