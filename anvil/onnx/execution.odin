package onnx

import "../tensor"
import "base:runtime"
import "core:fmt"
import "core:math"
import "core:slice"
import "core:time"

run :: proc(model: ^ONNX($T), inputs: map[string]^tensor.Tensor(T)) -> ONNX_Error {
	allocator := model.allocator
	// set inputs to the models. We shall clone and manage ownership by our own.
	// TODO(Aria): no need to clone.
	for k, v in inputs do model.graph.tensors[k] = tensor.clone(v, allocator)

	input_names := slice.map_keys(inputs, context.temp_allocator) or_return
	orders := determine_execution_order(model.graph, input_names, context.temp_allocator)
	
	// odinfmt:disable
	for op_idx, i in orders {
		// Sanity check for each inputs before node execution
		op := model.graph.nodes[op_idx]
		for iname in op.inputs do ensure_tensor_non_nil(model.graph.tensors[iname], iname, op.op_type, true) or_return

		switch op.op_type {
/*===================================================================================================*/	
//      SUPPORTED OPERATIONS
/*===================================================================================================*/	
		case "Add"               : run_add(op, model, allocator) or_return
		case "Concat"            : run_concat(op, model, allocator) or_return
		case "Conv"              : run_conv(op, model, allocator) or_return
		case "Div"               : run_div(op, model, allocator) or_return
		case "Exp"               : run_exp(op, model, allocator) or_return
		case "Flatten"           : run_flatten(op, model, allocator) or_return
		case "Gemm"              : run_gemm(op, model, allocator) or_return
		case "GlobalAveragePool" : run_global_average_pool(op, model, allocator) or_return
		case "MaxPool"           : run_max_pool(op, model, allocator) or_return
		case "Mul"               : run_mul(op, model, allocator) or_return
		case "Relu"              : run_relu(op, model, allocator) or_return
		case "Reshape"           : run_reshape(op, model, allocator) or_return
		case "Slice"             : run_slice(op, model, allocator) or_return
		case "Softmax"           : run_softmax(op, model, allocator) or_return
		case "Sub"               : run_sub(op, model, allocator) or_return
		case "Transpose"         : run_transpose(op, model, allocator) or_return
/*====================================================================================================*/	
		case /*OTHER*/          : return Unsupported_Op{op.op_type}
		}
	}
	// odinfmt:enable
	return nil
}

fetch_tensor :: proc(model: ^ONNX($T), name: string) -> (^tensor.Tensor(T), ONNX_Error) {
	out, ok := model.graph.tensors[name]
	if !ok do return nil, Value_Error{fmt.tprintf("Tensor `%s` is either not found or not computed yet. Have you run the model?", name)}
	return out, nil
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
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	ensure(
		len(x.shape) == 4,
		fmt.tprintf(
			"Conv only supports 2D convolutions. Expected input dimension is 4, got $d",
			len(x.shape),
		),
	)

	w := model.graph.tensors[op.inputs[1]]
	b := model.graph.tensors[op.inputs[2]] or_else nil
	auto_pad: string
	dilations, kernel_shape, pads, strides: []i64
	groups: uint

	output: ^tensor.Tensor(T)

	if model.opset_version < 1 {
		return Unsupported_Opset{"Conv", model.opset_version}
	} else if model.opset_version <= 22 {
		auto_pad = op.attributes["auto_pad"].(string) or_else "NOTSET"
		if auto_pad != "NOTSET" do return Unsupported_Attribute{"Conv", "auto_pad", auto_pad, model.opset_version}

		dilations = op.attributes["dilations"].([]i64) or_else {1}
		if !slice.all_of(dilations, dilations[0]) do return Malformed_Attribute{"Conv only support symmetrical dilations"}

		pads = op.attributes["pads"].([]i64) or_else {0}
		if !slice.all_of(pads, pads[0]) do return Malformed_Attribute{"Conv only support symmetrical paddings"}

		strides = op.attributes["strides"].([]i64) or_else {1}
		if !slice.all_of(strides, strides[0]) do return Malformed_Attribute{"Conv only support symmetrical strides"}

		// NOTE(Aria): Kernel shape is omitted since we don't use it directly and can just infer from
		// the kernel tensor itself.
		// kernel_shape = ...

		// ensure(slice.all_of(dilations, dilations[0]), "Conv only support symmetrical dilations")
		stride := uint(strides[0])
		dilation := uint(dilations[0])
		padding := uint(pads[0])
		groups = uint(op.attributes["group"].(i64) or_else 1)

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
		return Unsupported_Opset{"Conv", model.opset_version}
	}

	model.graph.tensors[op.outputs[0]] = output

	return
}

run_div :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	inputs, outputs := op.inputs, op.outputs
	x, y := model.graph.tensors[inputs[0]], model.graph.tensors[inputs[1]]
	model.graph.tensors[outputs[0]] = tensor.div(x, y, allocator)
	return
}

run_exp :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	model.graph.tensors[op.outputs[0]] = tensor.exp(x, allocator)
	return
}

run_relu :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	model.graph.tensors[op.outputs[0]] = tensor.relu(x, allocator)
	return
}

run_reshape :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	if model.opset_version <= 5 do return Unsupported_Opset{"Reshape", model.opset_version}
	allowzero := op.attributes["beta"].(i64) or_else 0
	if allowzero != 0 do return Unsupported_Attribute{"Reshape", "allowzero", fmt.tprint(allowzero), model.opset_version}

	x := model.graph.tensors[op.inputs[0]]
	out_shape_f := model.graph.tensors[op.inputs[1]].data
	out_shape := make([]uint, len(out_shape_f), context.temp_allocator)

	out_size := abs(int(math.prod(out_shape_f)))
	inferred_dimsize := len(x.data) / out_size
	for s, i in out_shape_f {
		if s < 0 { 	// -1 detected!
			out_shape[i] = uint(inferred_dimsize)
		} else {
			out_shape[i] = uint(s)
		}
	}
	reshaped := tensor.reshape(x, out_shape, allocator)
	model.graph.tensors[op.outputs[0]] = reshaped
	return
}

run_max_pool :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	ensure_batched_image_shape(x, op.inputs[0], "MaxPool", true) or_return

	auto_pad: string
	output: ^tensor.Tensor(T)
	dilations, kernel_shape, pads, strides: []i64
	opset := model.opset_version
	attributes := op.attributes

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
	model.graph.tensors[op.outputs[0]] = output
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
	if len(x.shape) != 2 || len(w.shape) != 2 do return Value_Error{"gemm requires both tensors have 2D"}
	bias: Maybe(^tensor.Tensor(T)) = nil

	if len(op.inputs) > 2 do bias = model.graph.tensors[op.inputs[2]]
	alpha := op.attributes["alpha"].(f32) or_else 1.0
	beta := op.attributes["beta"].(f32) or_else 1.0
	trans_a := (op.attributes["transA"].(i64) or_else 0) == 1
	trans_b := (op.attributes["transB"].(i64) or_else 0) == 1

	output := tensor.gemm(x, w, bias, alpha, beta, trans_a, trans_b, allocator)
	model.graph.tensors[op.outputs[0]] = output

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

run_mul :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	inputs, outputs := op.inputs, op.outputs
	x, y := model.graph.tensors[inputs[0]], model.graph.tensors[inputs[1]]
	model.graph.tensors[outputs[0]] = tensor.mul(x, y, allocator)
	return
}

@(private = "file")
_range :: proc($T: typeid, n: uint, allocator := context.allocator) -> []T {
	res := make([]T, n, allocator)
	for i in 0 ..< n do res[i] = T(i)
	return res
}

run_slice :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]

	opset := model.opset_version
	attributes := op.attributes


	slices := make([]tensor.Slice, len(x.shape))
	for _, i in slices do slices[i] = tensor.Range{}

	if opset < 10 {
		// 1..9: starts and ends are in attributes
		starts, ok_starts := attributes["starts"].([]i64)
		if !ok_starts do return Missing_Required_Attribute{"starts"}
		ends, ok_ends := attributes["ends"].([]i64)
		if !ok_ends do return Missing_Required_Attribute{"ends"}
		axes, ok_axes := attributes["axes"].([]i64)

		for start, i in starts {
			slices[axes[i]] = tensor.Range {
				start = int(starts[i]),
				end   = int(ends[i]),
				step  = 1, // earlier ONNX version doesn't specify step
			}
		}
	} else {
		// 10..current: starts, ends, and steps are input tensors
		panic("TODO (Aria): implemented yet slice for opset >= 10")
	}

	out := tensor.slice(x, slices, allocator = allocator)
	model.graph.tensors[op.outputs[0]] = out
	return
}

run_softmax :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	axis_i := op.attributes["axis"].(i64)

	axis: uint
	if axis_i < 0 {
		axis = uint(int(len(x.data)) - int(axis_i))
	}

	out := tensor.softmax(x, axis, allocator)
	model.graph.tensors[op.outputs[0]] = out
	return
}

run_sub :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	inputs, outputs := op.inputs, op.outputs
	x, y := model.graph.tensors[inputs[0]], model.graph.tensors[inputs[1]]
	model.graph.tensors[outputs[0]] = tensor.sub(x, y, allocator)
	return
}

run_transpose :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	x := model.graph.tensors[op.inputs[0]]
	perm, ok := op.attributes["perm"].([]i64)
	if !ok do return Missing_Required_Attribute{"perm"}

	perm_uint := make([]uint, len(perm), context.temp_allocator)
	for v, i in perm do perm_uint[i] = uint(v)

	out := tensor.permute(x, perm_uint, allocator)
	model.graph.tensors[op.outputs[0]] = out
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

run_concat :: proc(
	op: ^Node($T),
	model: ^ONNX(T),
	allocator: runtime.Allocator,
) -> (
	err: ONNX_Error,
) {
	inputs, outputs := op.inputs, op.outputs
	inputs_tensor := make([dynamic]^tensor.Tensor(T), len(inputs), context.temp_allocator)
	for input_name, i in inputs do inputs_tensor[i] = model.graph.tensors[input_name]

	axis, ok := op.attributes["axis"].(i64)
	if !ok do return Missing_Required_Attribute{"axis"}

	out := tensor.cat(inputs_tensor[:], uint(axis), allocator)
	model.graph.tensors[outputs[0]] = out
	return
}
