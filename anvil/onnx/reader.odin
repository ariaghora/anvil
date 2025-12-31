// ONNX Model Parser
//
// This file implements a zero-dependency protobuf parser specifically for ONNX models.
// We parse protobuf manually rather than using a code generator because:
// 1. ONNX models only use a small subset of protobuf features
// 2. We want zero-copy string handling (strings point directly into raw_bytes)
// 3. We need control over memory allocation for the tensor runtime
//
// Protobuf wire format basics (see https://protobuf.dev/programming-guides/encoding/):
//   Each field is encoded as: [tag][payload]
//   Tag = (field_number << 3) | wire_type
//   Wire types:
//     0 = varint (int32, int64, uint32, uint64, bool, enum)
//     1 = 64-bit fixed (fixed64, sfixed64, double)
//     2 = length-delimited (string, bytes, embedded messages, packed repeated)
//     5 = 32-bit fixed (fixed32, sfixed32, float)
//     3,4 = deprecated group start/end, we reject these
//
// ONNX protobuf schema reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//
// =============================================================================
// GOTCHAS AND WARNINGS
// =============================================================================
//
// Memory ownership:
//   - All strings (node names, tensor names, op types) point directly into raw_bytes.
//     Do NOT free raw_bytes while the model is still in use.
//   - raw_embedded=true means caller owns raw_bytes; false means we own it.
//
// Platform assumptions:
//   - Float parsing assumes little-endian byte order (x86, ARM).
//     Will silently produce wrong values on big-endian systems.
//   - Pointer casts like (^f32)(&bytes[i])^ assume the address is 4-byte aligned.
//     ONNX files from standard exporters satisfy this, but hand-crafted files might not.
//
// Protobuf quirks:
//   - Repeated int fields can be packed (one blob) or unpacked (separate varints).
//     We handle both, but they use different field numbers (e.g. 7 vs 8 for ints).
//   - Protobuf allows fields in any order. We don't assume ordering.
//   - Unknown fields are skipped, not rejected (forward compatibility).
//
// ONNX quirks:
//   - Tensor data can be in raw_data, float_data, int32_data, or int64_data.
//     Modern exporters use raw_data; legacy ones use the typed arrays.
//   - Graph attributes (for If/Loop ops) are not supported yet.
//   - Non-standard opsets (vendor extensions) return an error.
//
package onnx

import "../tensor"
import "base:runtime"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

IO_Error :: struct {
	msg: string,
}

ONNX_Format_Error :: struct {
	msg: string,
}

Unsupported_Op :: struct {
	msg: string,
}

Unsupported_Opset :: struct {
	op_name: string,
	opset:   i64,
}

Unsupported_Attribute :: struct {
	op_name, attribute_name, attribute_value_as_string: string,
	opset:                                              i64,
}

Malformed_Attribute :: struct {
	msg: string,
}

Truncated_Data :: struct {}

Value_Error :: struct {
	msg: string,
}

Missing_Required_Attribute :: struct {
	name: string,
}

ONNX_Error :: union {
	runtime.Allocator_Error,
	IO_Error,
	ONNX_Format_Error,
	Truncated_Data,
	Unsupported_Op,
	Unsupported_Opset,
	Unsupported_Attribute,
	Malformed_Attribute,
	Value_Error,
	Missing_Required_Attribute,
}

// The root ONNX model structure.
// T is the compute type for tensors (typically f32 or f16).
//
// Memory ownership: raw_bytes holds the entire file contents. All strings in
// the model (node names, input/output names, etc.) are slices pointing into
// raw_bytes, so raw_bytes must outlive the model. When raw_embedded=true,
// the caller owns raw_bytes. When false, we own it and free_onnx will delete it.
ONNX :: struct($T: typeid) {
	opset_version:    i64,
	producer_name:    string, // e.g. "pytorch", points into raw_bytes
	producer_version: string,
	raw_bytes:        []u8,
	raw_embedded:     bool, // if true, caller owns raw_bytes
	graph:            ^Graph(T),
	allocator:        runtime.Allocator,
}

// ONNX operators can have attributes like kernel_size=[3,3] or epsilon=1e-5.
// This union covers the attribute types we actually encounter in practice.
// Note: Graph attributes (for control flow ops like If/Loop) are not yet supported.
Attribute :: union($T: typeid) {
	i64,
	f32,
	string,
	[]i64,
	[]f32,
	^tensor.Tensor(T),
}

// From onnx.proto AttributeProto.AttributeType
Attribute_Type :: enum {
	Undefined = 0,
	Float     = 1,
	Int       = 2,
	String    = 3,
	Tensor    = 4,
	Graph     = 5,
	Floats    = 6,
	Ints      = 7,
	Strings   = 8,
}

// From onnx.proto TensorProto.DataType
// These values are fixed by the ONNX spec and must match exactly.
ONNX_DataType :: enum i32 {
	Undefined  = 0,
	Float      = 1,
	Uint8      = 2,
	Int8       = 3,
	Uint16     = 4,
	Int16      = 5,
	Int32      = 6,
	Int64      = 7,
	String     = 8,
	Bool       = 9,
	Float16    = 10,
	Double     = 11,
	Uint32     = 12,
	Uint64     = 13,
	Complex64  = 14,
	Complex128 = 15,
	BFloat16   = 16,
}

// A single operation in the computation graph.
// inputs/outputs are tensor names that connect nodes together.
Node :: struct($T: typeid) {
	op_type:    string, // e.g. "Conv", "Relu", "MatMul"
	name:       string, // optional, for debugging
	inputs:     [dynamic]string,
	outputs:    [dynamic]string,
	attributes: map[string]Attribute(T),
}

// The computation graph. Nodes are stored in topological order (guaranteed by ONNX spec).
// tensors map holds both:
//   - initializers: weight tensors baked into the model file
//   - intermediates: filled in during inference by the executor
Graph :: struct($T: typeid) {
	raw_bytes: []u8,
	nodes:     [dynamic]^Node(T),
	tensors:   map[string]^tensor.Tensor(T),
}

// Load an ONNX model from a file. Reads the entire file into memory,
// then parses it. The returned model owns the file bytes.
read_from_file :: proc(
	$T: typeid,
	fn: string,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^ONNX(T),
	er: ONNX_Error,
) {
	raw_bytes, ok := os.read_entire_file(fn, allocator)
	if !ok do return nil, IO_Error{msg = fmt.tprintf("Failed to read ONNX from %s", fn)}
	return read_from_bytes(T, raw_bytes, false, allocator, loc)
}

// Parse an ONNX model from a byte slice. This is the core parser.
//
// raw_embedded: if true, caller owns raw_bytes and must keep it alive.
//               if false, we own it and free_onnx will delete it.
//
// The top-level ONNX protobuf is onnx.ModelProto. Field numbers:
//   2 = producer_name (string, e.g. "pytorch")
//   3 = producer_version (string)
//   7 = graph (GraphProto, the actual computation graph)
//   8 = opset_import (repeated OperatorSetIdProto)
//   14 = metadata_props (repeated StringStringEntryProto, key-value pairs)
read_from_bytes :: proc(
	$T: typeid,
	raw_bytes: []u8,
	raw_embedded := true,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^ONNX(T),
	er: ONNX_Error,
) {
	offset := 0
	producer_name, producer_version: string
	opset_version: i64
	graph: ^Graph(T)

	for offset < len(raw_bytes) {
		tag, new_offset := read_varint(raw_bytes, offset) or_return
		offset = new_offset

		wire_type := u8(tag & 0x7)
		field_num := u32(tag >> 3)

		// wire types 3 and 4 are deprecated (start/end group)
		wire_type_is_deprecated := slice.contains([]u8{3, 4}, wire_type)
		ensure(!wire_type_is_deprecated, fmt.tprintf("wire type %d is unsupported", wire_type))

		switch wire_type {
		case 0:
			_, new_offset := read_varint(raw_bytes, offset) or_return
			offset = new_offset

		case 2:
			length, new_offset := read_varint(raw_bytes, offset) or_return
			offset = new_offset

			if offset + int(length) > len(raw_bytes) do return nil, Truncated_Data{}
			payload := raw_bytes[offset:offset + int(length)]
			offset += int(length)

			switch field_num {
			case 2:
				producer_name = string(payload)
			case 3:
				producer_version = string(payload)
			case 7:
				graph = parse_graph(T, payload, allocator) or_return
			case 8:
				opset_version = parse_opset(payload) or_return
			case 14: // metadata_props, skip
			case:
				fmt.panicf("Field num %d handling is not implemented yet", field_num)
			}
		case:
			fmt.panicf("wire type %d is unsupported", wire_type)
		}
	}

	return new_clone(
			ONNX(T) {
				raw_bytes = raw_bytes,
				producer_name = producer_name,
				producer_version = producer_version,
				graph = graph,
				opset_version = opset_version,
				raw_embedded = raw_embedded,
				allocator = allocator,
			},
			allocator,
		),
		nil
}

// Protobuf varint decoder. Varints encode integers in 7-bit chunks, with the
// MSB of each byte indicating whether more bytes follow. This is a compact
// encoding for small values (1 byte for 0-127) but can expand to 10 bytes
// for full 64-bit values. We cap at 10 iterations since that's the max for u64.
@(private = "file")
read_varint :: proc(data: []u8, pos: int) -> (value: u64, new_pos: int, err: ONNX_Error) {
	value = 0
	shift := u64(0)
	new_pos = pos

	for i := 0; i < 10 && new_pos < len(data); i += 1 {
		b := data[new_pos]
		new_pos += 1
		value |= u64(b & 0x7F) << shift
		if (b & 0x80) == 0 {
			return
		}
		shift += 7
	}

	// TODO(Aria): better handling, maybe?
	fmt.panicf("Varint is too long (%d), invalid ONNX format\n", value)
}

// Parse onnx.GraphProto. Field numbers from onnx.proto:
//   1 = node (repeated NodeProto)
//   5 = initializer (repeated TensorProto, the model weights)
//   8 = value_info (repeated ValueInfoProto, intermediate tensor shapes)
//   11 = input (repeated ValueInfoProto)
//   12 = output (repeated ValueInfoProto)
// We only need nodes and initializers for inference. The value_info/input/output
// fields describe shapes but we infer those dynamically.
@(private = "file")
parse_graph :: proc(
	$T: typeid,
	graph_bytes: []u8,
	allocator := context.allocator,
) -> (
	graph: ^Graph(T),
	err: ONNX_Error,
) {
	nodes := make([dynamic]^Node(T), allocator)
	initializers := make(map[string]^tensor.Tensor(T), allocator)
	offset := 0
	for offset < len(graph_bytes) {
		tag_value, new_offset := read_varint(graph_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 2:
			length, new_offset := read_varint(graph_bytes, offset) or_return
			offset = new_offset
			payload := graph_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				node := parse_node(T, payload, allocator) or_return
				append(&nodes, node)
			case 5:
				t, name := parse_tensor(T, payload, allocator) or_return
				initializers[name] = t
			case 8, 11, 12: // value_info, input, output: skip, we infer shapes
			}
			offset += int(length)

		// Skip unknown wire types. ONNX graphs are mostly wire type 2, but we
		// handle the others for forward compatibility with new spec versions.
		case 0:
			_, new_offset = read_varint(graph_bytes, offset) or_return
			offset = new_offset
		case 1:
			offset += 8
		case 5:
			offset += 4
		case:
			fmt.panicf("Found unhandled wire type during graph decoding: %d", wire_type)
		}
	}
	graph = new_clone(
		Graph(T){raw_bytes = graph_bytes, nodes = nodes, tensors = initializers},
		allocator,
	)

	return graph, nil
}


// Parse onnx.NodeProto. Field numbers from onnx.proto:
//   1 = input (repeated string, tensor names this op reads from)
//   2 = output (repeated string, tensor names this op writes to)
//   3 = name (optional string, for debugging)
//   4 = op_type (string, e.g. "Conv", "MatMul", "Relu")
//   5 = attribute (repeated AttributeProto)
//   6 = doc_string (skip)
//   7 = domain (skip, for custom ops)
@(private = "file")
parse_node :: proc(
	$T: typeid,
	node_bytes: []u8,
	allocator := context.allocator,
) -> (
	node: ^Node(T),
	err: ONNX_Error,
) {
	inputs := make([dynamic]string, allocator)
	outputs := make([dynamic]string, allocator)
	attributes := make(map[string]Attribute(T), allocator)

	op_type: string
	name: string

	offset := 0
	for offset < len(node_bytes) {
		tag_value, new_offset := read_varint(node_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			_, new_offset = read_varint(node_bytes, offset) or_return
			offset = new_offset
		case 1:
			offset += 8
		case 2:
			length, new_offset := read_varint(node_bytes, offset) or_return
			offset = new_offset
			payload := node_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				append(&inputs, string(payload))
			case 2:
				append(&outputs, string(payload))
			case 3:
				name = string(payload)
			case 4:
				op_type = string(payload)
			case 5:
				attr_name, attr_value := parse_attribute(T, payload, allocator) or_return
				attributes[attr_name] = attr_value
			case 6, 7: // doc_string, domain
			}

			offset += int(length)
		case 5:
			offset += 4
		}
	}

	node = new_clone(
		Node(T) {
			op_type = op_type,
			name = name,
			inputs = inputs,
			outputs = outputs,
			attributes = attributes,
		},
		allocator,
	)

	return node, nil
}

// Parse onnx.AttributeProto. Field numbers from onnx.proto:
//   1 = name (string)
//   2 = f (float, wire type 5 = fixed32)
//   3 = i (int64, wire type 0 = varint)
//   4 = s (bytes/string)
//   5 = t (TensorProto, for constant tensors like shape parameters)
//   7 = ints (repeated int64, packed in length-delimited field)
//   8 = ints (repeated int64, unpacked as individual varints, older format)
//   20 = type (AttributeType enum, tells us which field to expect)
//
// Note: ints can appear as either field 7 (packed) or field 8 (unpacked).
// Packed means all values are concatenated in one length-delimited blob.
// Unpacked means each value is a separate varint field. We handle both.
@(private = "file")
parse_attribute :: proc(
	$T: typeid,
	attr_bytes: []u8,
	allocator := context.allocator,
) -> (
	name: string,
	value: Attribute(T),
	err: ONNX_Error,
) {
	attr_type: Attribute_Type
	ints_list: [dynamic]i64 // collects unpacked ints (field 8)

	offset := 0
	for offset < len(attr_bytes) {
		tag_value, new_offset := read_varint(attr_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			val, new_offset := read_varint(attr_bytes, offset) or_return
			offset = new_offset

			switch field_num {
			case 20:
				attr_type = Attribute_Type(val)
			case 3:
				value = i64(val)
			case 8:
				// unpacked repeated int64
				if ints_list == nil {
					ints_list = make([dynamic]i64, allocator)
				}
				append(&ints_list, i64(val))
			case:
				fmt.panicf(
					"attribute of wire_type %d field_num %d not supported yet",
					wire_type,
					field_num,
				)
			}

		case 2:
			length, new_offset := read_varint(attr_bytes, offset) or_return
			offset = new_offset
			payload := attr_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				name = string(payload)
			case 4:
				value = string(payload)
			case 5:
				t, _ := parse_tensor(T, payload, allocator) or_return
				value = t
			case 7:
				// packed repeated int64
				ints := make([dynamic]i64, allocator)
				p_offset := 0
				for p_offset < len(payload) {
					val, new_p_offset := read_varint(payload, p_offset) or_return
					append(&ints, i64(val))
					p_offset = new_p_offset
				}
				value = ints[:]
			case:
				fmt.panicf(
					"attribute of wire_type %d field_num %d not supported yet",
					wire_type,
					field_num,
				)
			}

			offset += int(length)

		case 5:
			// Fixed 32-bit. Field 2 is float.
			// WARNING: This assumes little-endian and 4-byte alignment.
			// Works on x86/ARM but may break on exotic platforms.
			if field_num == 2 {
				value = (^f32)(&attr_bytes[offset])^
			}
			offset += 4
		}
	}

	// If we got unpacked ints (field 8), use those as the value
	if len(ints_list) > 0 {
		value = ints_list[:]
	}

	return name, value, nil
}

// Parse onnx.OperatorSetIdProto. Field numbers:
//   1 = domain (string, empty = standard ONNX ops)
//   2 = version (int64)
// ONNX models can import multiple opsets (standard + vendor extensions).
// We only care about the standard opset (empty domain) for version checking.
parse_opset :: proc(opset_bytes: []u8) -> (version: i64, err: ONNX_Error) {
	domain: string

	offset := 0
	for offset < len(opset_bytes) {
		tag_value, new_offset := read_varint(opset_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			val, new_offset := read_varint(opset_bytes, offset) or_return
			offset = new_offset
			if field_num == 2 {
				version = i64(val)
			}
		case 2:
			length, new_offset := read_varint(opset_bytes, offset) or_return
			offset = new_offset
			if field_num == 1 {
				domain = string(opset_bytes[offset:offset + int(length)])
			}
			offset += int(length)
		}
	}

	// empty domain = standard ONNX opset, non-empty = vendor extension (skip)
	if domain == "" {
		return version, nil
	}
	return 0, ONNX_Format_Error{"Non-empty domain"}
}

// Parse onnx.TensorProto and convert to our internal tensor format.
// The ONNX tensor data can be stored in multiple ways:
//   - raw_data (field 9): binary blob, most compact, used by modern exporters
//   - float_data (field 4): repeated float, older format
//   - int32_data (field 5): repeated int32
//   - int64_data (field 7): repeated int64
// We allocate a tensor with the caller's type T and convert from whatever
// format the file uses.
@(private = "file")
parse_tensor :: proc(
	$T: typeid,
	tensor_bytes: []u8,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	t: ^tensor.Tensor(T),
	name: string,
	err: ONNX_Error,
) {
	tensor_fields := parse_tensor_fields(tensor_bytes) or_return

	shape := make([]uint, len(tensor_fields.dims), context.temp_allocator)
	for dim, i in tensor_fields.dims do shape[i] = uint(dim)

	t = tensor.tensor_alloc(T, shape, owns_data = true, allocator = allocator, loc = loc)

	total_elements := 1
	for d in tensor_fields.dims do total_elements *= int(d)

	// Empty tensors (e.g. shape [0]) are valid, just return early
	if total_elements == 0 {
		return t, tensor_fields.name, nil
	}

	convert_tensor_data(T, t, tensor_fields, total_elements)

	return t, tensor_fields.name, nil
}

// Intermediate struct to collect TensorProto fields before conversion.
// Uses temp_allocator for dims since they're only needed during parsing.
Tensor_Fields :: struct {
	name:       string,
	dims:       [dynamic]i64,
	data_type:  i32, // ONNX_DataType enum value
	raw_data:   []u8, // binary blob, slice into original bytes
	float_data: [dynamic]f32, // legacy format
	int32_data: [dynamic]i32,
	int64_data: [dynamic]i64,
}

// Parse onnx.TensorProto fields. Field numbers:
//   1 = dims (repeated int64, can be packed or unpacked)
//   2 = data_type (int32, ONNX_DataType enum)
//   4 = float_data (repeated float, legacy)
//   5 = int32_data (repeated int32, legacy)
//   7 = int64_data (repeated int64, legacy)
//   8 = name (string)
//   9 = raw_data (bytes, preferred format)
@(private = "file")
parse_tensor_fields :: proc(tensor_bytes: []u8) -> (fields: Tensor_Fields, err: ONNX_Error) {
	offset := 0
	for offset < len(tensor_bytes) {
		tag_value, new_offset := read_varint(tensor_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			val, new_offset := read_varint(tensor_bytes, offset) or_return
			offset = new_offset

			switch field_num {
			case 1:
				// dims as unpacked varints (one field per dimension)
				if fields.dims == nil do fields.dims = make([dynamic]i64, context.temp_allocator)
				append(&fields.dims, i64(val))
			case 2:
				fields.data_type = i32(val)
			}

		case 2:
			length, new_offset := read_varint(tensor_bytes, offset) or_return
			offset = new_offset
			payload := tensor_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				// dims as packed varints (all dimensions in one blob)
				fields.dims = parse_packed_varints(payload) or_return
			case 4:
				fields.float_data = parse_packed_floats(payload)
			case 5:
				fields.int32_data = parse_packed_int32s(payload) or_return
			case 7:
				fields.int64_data = parse_packed_int64s(payload) or_return
			case 8:
				fields.name = string(payload)
			case 9:
				// raw_data is just a slice into the original bytes, no copy
				fields.raw_data = payload
			}

			offset += int(length)

		case 5:
			offset += 4
		}
	}

	return fields, nil
}

// Decode a packed repeated field of varints into a dynamic array.
// Packed encoding means all values are concatenated in one length-delimited blob
// with no per-element tags, which is more compact than unpacked.
@(private = "file")
parse_packed_varints :: proc(payload: []u8) -> (result: [dynamic]i64, err: ONNX_Error) {
	result = make([dynamic]i64, context.temp_allocator)
	offset := 0
	for offset < len(payload) {
		val, new_offset := read_varint(payload, offset) or_return
		append(&result, i64(val))
		offset = new_offset
	}
	return result, nil
}

// Decode packed floats. Unlike varints, floats are fixed 4 bytes each.
// WARNING: assumes little-endian byte order.
@(private = "file")
parse_packed_floats :: proc(payload: []u8) -> [dynamic]f32 {
	result := make([dynamic]f32, len(payload) / 4, context.temp_allocator)
	for i := 0; i < len(payload); i += 4 {
		result[i / 4] = (^f32)(&payload[i])^
	}
	return result
}

// Decode packed int32s. Despite being "int32", protobuf encodes these as varints.
@(private = "file")
parse_packed_int32s :: proc(payload: []u8) -> (result: [dynamic]i32, err: ONNX_Error) {
	result = make([dynamic]i32, context.temp_allocator)
	offset := 0
	for offset < len(payload) {
		val, new_offset := read_varint(payload, offset) or_return
		append(&result, i32(val))
		offset = new_offset
	}
	return result, nil
}

@(private = "file")
parse_packed_int64s :: proc(payload: []u8) -> (result: [dynamic]i64, err: ONNX_Error) {
	return parse_packed_varints(payload)
}

// Convert parsed tensor fields into our tensor format. Handles the different
// ways ONNX can store tensor data (raw_data vs typed arrays).
@(private = "file")
convert_tensor_data :: proc(
	$T: typeid,
	t: ^tensor.Tensor(T),
	fields: Tensor_Fields,
	total_elements: int,
) {
	// Prefer raw_data if available (most common in modern models)
	if len(fields.raw_data) > 0 {
		convert_raw_data(T, t.data, fields.raw_data, fields.data_type, fields.name)
	} else if len(fields.float_data) > 0 {
		for val, i in fields.float_data {
			t.data[i] = T(val)
		}
	} else if len(fields.int32_data) > 0 {
		for val, i in fields.int32_data {
			t.data[i] = T(val)
		}
	} else if len(fields.int64_data) > 0 {
		for val, i in fields.int64_data {
			t.data[i] = T(val)
		}
	} else {
		panic(
			fmt.tprintf(
				"Tensor '%s' (dtype=%d, shape=%v) has no data!",
				fields.name if fields.name != "" else "<unnamed>",
				fields.data_type,
				fields.dims,
			),
		)
	}
}

// Convert raw_data bytes to our tensor type T. The data_type field tells us
// how to interpret the bytes. We reinterpret the byte slice as the source type,
// then convert element-by-element to T.
@(private = "file")
convert_raw_data :: proc($T: typeid, dest: []T, raw_data: []u8, data_type: i32, name: string) {
	#partial switch ONNX_DataType(data_type) {
	case .Float:
		src_data := mem.slice_data_cast([]f32, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Uint8:
		src_data := mem.slice_data_cast([]u8, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Int8:
		src_data := mem.slice_data_cast([]i8, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Int32:
		src_data := mem.slice_data_cast([]i32, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Int64:
		src_data := mem.slice_data_cast([]i64, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Float16:
		src_data := mem.slice_data_cast([]f16, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case .Double:
		src_data := mem.slice_data_cast([]f64, raw_data)
		for src_val, i in src_data do dest[i] = T(src_val)
	case:
		panic(fmt.tprintf("Unsupported ONNX data type %d for tensor %s", data_type, name))
	}
}

// Free all memory associated with an ONNX model.
// The allocator param is for raw_bytes (when read from file).
// All other allocations use model.allocator (stored at parse time).
//
// Note: strings (node names, tensor names, etc.) point into raw_bytes
// and don't need separate freeing.
free_onnx :: proc(model: ^ONNX($T), allocator := context.allocator) {
	raw_bytes_allocator := allocator
	main_allocator := model.allocator

	for node in model.graph.nodes {
		// Only slice attributes need freeing; scalars and strings are inline or borrowed
		for _, av in node.attributes {
			#partial switch v in av {
			case []i64:
				delete(v, main_allocator)
			case []f32:
				delete(v, main_allocator)
			// TODO: tensor attributes also need freeing
			}
		}
		delete(node.inputs)
		delete(node.outputs)
		delete(node.attributes)
		free(node, main_allocator)
	}

	for _, t in model.graph.tensors {
		tensor.free_tensor(t, allocator = main_allocator)
	}

	// Only free raw_bytes if we own it (read from file, not embedded)
	if !model.raw_embedded do delete(model.raw_bytes, raw_bytes_allocator)

	delete(model.graph.nodes)
	delete(model.graph.tensors)
	free(model.graph, main_allocator)

	free(model, main_allocator)
}
