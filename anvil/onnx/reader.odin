// References:
// - https://protobuf.dev/programming-guides/encoding/
package onnx

import "../tensor"
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

Truncated_Data :: struct {}

ONNX_Error :: union {
	IO_Error,
	ONNX_Format_Error,
	Truncated_Data,
}

ONNX :: struct($T: typeid) {
	opset_version:    i64,
	producer_name:    string,
	producer_version: string,
	raw_bytes:        []u8,
	graph:            ^Graph(T),
}

Attribute :: union($T: typeid) {
	i64,
	f32,
	string,
	[dynamic]i64,
	[dynamic]f32,
	^tensor.Tensor(T),
}

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

// Data type enum from ONNX
ONNX_DataType :: enum i32 {
	UNDEFINED  = 0,
	FLOAT      = 1, // float32
	UINT8      = 2,
	INT8       = 3,
	UINT16     = 4,
	INT16      = 5,
	INT32      = 6,
	INT64      = 7,
	STRING     = 8,
	BOOL       = 9,
	FLOAT16    = 10,
	DOUBLE     = 11, // float64
	UINT32     = 12,
	UINT64     = 13,
	COMPLEX64  = 14,
	COMPLEX128 = 15,
	BFLOAT16   = 16,
}

Node :: struct($T: typeid) {
	op_type:    string,
	name:       string,
	inputs:     [dynamic]string,
	outputs:    [dynamic]string,
	attributes: map[string]Attribute(T),
}

Graph :: struct($T: typeid) {
	raw_bytes:    []u8, // This is ONLY A REFERENCE to the original bytes
	nodes:        [dynamic]^Node(T),
	initializers: [dynamic]^tensor.Tensor(T),
}

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
	return read_from_bytes(T, raw_bytes, allocator, loc)
}

read_from_bytes :: proc(
	$T: typeid,
	raw_bytes: []u8,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^ONNX(T),
	er: ONNX_Error,
) {
	offset := 0
	producer_name, producer_version: string
	opset_version: i64
	graph: ^Graph(T) // Main (root) graph

	for offset < len(raw_bytes) {
		// step 1: get the tag, i.e., the first byte of currently parsed sequence
		// then move offset one bit
		tag, new_offset := read_varint(raw_bytes, offset) or_return
		offset = new_offset

		// step 2: get field number.
		// take the last three bits to get the wire type
		// and then right-shift by three to get the field number.
		wire_type := u8(tag & 0x7) // 0x7 == 0b00000111 --> filter-in last 3 bits
		field_num := u32(tag >> 3)

		// Let's just panic upon deprecated wire types (3 and 4)
		wire_type_is_deprecated := slice.contains([]u8{3, 4}, wire_type)
		ensure(!wire_type_is_deprecated, fmt.tprintf("wire type %d is unsupported", wire_type))

		// step 3: read the actual field data based on wire type
		switch wire_type {
		// Varint: int32, int64, uint32, uint64, sint32, sint64, bool, enum
		case 0:
			// TODO(Aria): dispatch based on field_num to store in appropriate struct field
			value, new_offset := read_varint(raw_bytes, offset) or_return
			offset = new_offset

		// Length-delimited: string, bytes, embedded messages, packed repeated fields
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
				// opset_import
				opset_version = parse_opset(payload) or_return
			case 14: // metadata_props, key-value pairs like {"author": "someone"}, skip it for now
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
			},
			allocator,
		),
		nil
}

@(private = "file")
read_varint :: proc(data: []u8, pos: int) -> (value: u64, new_pos: int, err: ONNX_Error) {
	value = 0
	shift := u64(0)
	new_pos = pos

	// Varints can be up to 10 bytes for 64-bit values
	for i := 0; i < 10 && new_pos < len(data); i += 1 {
		b := data[new_pos]
		new_pos += 1

		// Take lower 7 bits and shift them into position
		value |= u64(b & 0x7F) << shift

		// If MSB is 0, we're done
		if (b & 0x80) == 0 {
			return
		}

		shift += 7
	}

	// If we get here, varint was too long and we're basically screwed up
	// TODO(Aria): better handling, maybe?
	fmt.panicf("Varint is too long (%d), invalid ONNX format\n", value)
}

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
	initializers := make([dynamic]^tensor.Tensor(T), allocator)
	offset := 0
	for offset < len(graph_bytes) {
		tag_value, new_offset := read_varint(graph_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		// Wire type 2 is the most common in ONNX graphs, since most fields
		// are length-delimited, such as nodes, initializers, inputs, outputs, etc.
		case 2:
			// Length-delimited
			length, new_offset := read_varint(graph_bytes, offset) or_return
			offset = new_offset
			payload := graph_bytes[offset:offset + int(length)]

			switch field_num {
			// node
			case 1:
				node := parse_node(T, payload, allocator) or_return
				append(&nodes, node)
			// initializer
			case 5:
				t := parse_tensor(T, payload, allocator) or_return
			// fmt.println("Got Initializer (weights)!")
			// value_info
			case 8:
			// fmt.println("Got ValueInfo (intermediate shapes)!")
			// input
			case 11:
			// fmt.println("Got Input!")
			// output
			case 12:
			// fmt.println("Got Output!")
			}
			offset += int(length) // skip by payload length

		// For the other cases, just in case ONNX does something funny in the future,
		// let's just offsets according to several likely possible cases.
		case 0:
			// Varint, skip according to found offset
			_, new_offset = read_varint(graph_bytes, offset) or_return
			offset = new_offset
		case 1:
			offset += 8 // 64-bit, skip 8 bytes
		case 5:
			offset += 4 // 32-bit, skip 4 bytes
		case:
			// give up ¯\_(ツ)_/¯
			fmt.panicf("Found unhandlend wire type during graph decoding: %d", wire_type)
		}
	}
	graph = new_clone(
		Graph(T){raw_bytes = graph_bytes, nodes = nodes, initializers = initializers},
		allocator,
	)

	return graph, nil
}


@(private = "file")
parse_node :: proc(
	$T: typeid,
	node_bytes: []u8,
	allocator := context.allocator,
) -> (
	node: ^Node(T),
	err: ONNX_Error,
) {
	// Temporary storage for repeated fields
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
			// Varint
			_, new_offset = read_varint(node_bytes, offset) or_return
			offset = new_offset

		case 1:
			// 64-bit
			offset += 8

		case 2:
			// Length-delimited
			length, new_offset := read_varint(node_bytes, offset) or_return
			offset = new_offset
			payload := node_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				// input (repeated string)
				// String is directly a slice of node_bytes, no allocation
				input_name := string(payload)
				append(&inputs, input_name)

			case 2:
				// output (repeated string)
				output_name := string(payload)
				append(&outputs, output_name)

			case 3:
				// name
				name = string(payload)

			case 4:
				// op_type
				op_type = string(payload)

			case 5:
				// attribute
				// Attributes are complex, parse them separately
				attr_name, attr_value := parse_attribute(T, payload, allocator) or_return
				attributes[attr_name] = attr_value

			case 6: // doc_string - skip
			case 7: // domain - skip
			}

			offset += int(length)

		case 5:
			// 32-bit
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

	// For repeated fields that might come unpacked
	ints_list: [dynamic]i64
	floats_list: [dynamic]f32

	offset := 0
	for offset < len(attr_bytes) {
		tag_value, new_offset := read_varint(attr_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			// Varint
			val, new_offset := read_varint(attr_bytes, offset) or_return
			offset = new_offset

			switch field_num {
			case 20:
				// type (AttributeProto.AttributeType enum)
				attr_type = Attribute_Type(val)
			case 3:
				// i (int64) - sometimes stored as varint
				value = i64(val)
			case 8:
				// ints (repeated int64, unpacked)
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
			// Length-delimited
			length, new_offset := read_varint(attr_bytes, offset) or_return
			offset = new_offset

			payload := attr_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				// name
				name = string(payload)

			case 4:
				// s (string)
				value = string(payload)

			case 7:
				// ints (repeated int64)
				ints := make([dynamic]i64, allocator)
				p_offset := 0
				for p_offset < len(payload) {
					val, new_p_offset := read_varint(payload, p_offset) or_return
					append(&ints, i64(val))
					p_offset = new_p_offset
				}
				value = ints
			case:
				fmt.panicf(
					"attribute of wire_type %d field_num %d not supported yet",
					wire_type,
					field_num,
				)
			}

			offset += int(length)

		case 5:
			// 32-bit
			if field_num == 2 { 	// f (float)
				value = (^f32)(&attr_bytes[offset])^
			}
			offset += 4
		}
	}

	// If we collected unpacked repeated values, use them
	if len(ints_list) > 0 {
		value = ints_list
	}

	return name, value, nil
}

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
			// Varint
			if field_num == 2 { 	// version
				val, new_offset := read_varint(opset_bytes, offset) or_return
				version = i64(val)
				offset = new_offset
			}

		case 2:
			// Length-delimited
			length, new_offset := read_varint(opset_bytes, offset) or_return
			offset = new_offset

			if field_num == 1 { 	// domain
				domain = string(opset_bytes[offset:offset + int(length)])
			}

			offset += int(length)
		}
	}

	// Empty domain means default ONNX
	if domain == "" {
		return version, nil
	}

	// Non-empty domain is custom opset, ignore for now
	return 0, ONNX_Format_Error{"Non-empty domain"}
}

@(private = "file")
parse_tensor :: proc(
	$T: typeid,
	tensor_bytes: []u8,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	t: ^tensor.Tensor(T),
	err: ONNX_Error,
) {
	name: string
	dims: [dynamic]i64 // Collect dims, might come unpacked
	data_type: i32
	raw_data: []u8

	offset := 0
	for offset < len(tensor_bytes) {
		tag_value, new_offset := read_varint(tensor_bytes, offset) or_return
		offset = new_offset

		wire_type := tag_value & 0x7
		field_num := tag_value >> 3

		switch wire_type {
		case 0:
			// Varint
			val, new_offset := read_varint(tensor_bytes, offset) or_return
			offset = new_offset

			switch field_num {
			case 1:
				// dims - UNPACKED repeated field!
				if dims == nil do dims = make([dynamic]i64, context.temp_allocator)
				append(&dims, i64(val))

			case 2:
				// data_type
				data_type = i32(val)
			}

		case 2:
			// Length-delimited
			length, new_offset := read_varint(tensor_bytes, offset) or_return
			offset = new_offset
			payload := tensor_bytes[offset:offset + int(length)]

			switch field_num {
			case 1:
				// dims - PACKED repeated field
				if dims == nil do dims = make([dynamic]i64, context.temp_allocator)

				p_offset := 0
				for p_offset < len(payload) {
					dim, new_p_offset := read_varint(payload, p_offset) or_return
					append(&dims, i64(dim))
					p_offset = new_p_offset
				}

			case 8:
				// name
				name = string(payload)

			case 9:
				// raw_data
				raw_data = payload
			}

			offset += int(length)

		case 5:
			// 32-bit
			offset += 4
		}
	}

	// Convert dims to shape (using uint as you specified)
	shape := make([]uint, len(dims), context.temp_allocator)
	for dim, i in dims do shape[i] = uint(dim)

	// Allocate tensor
	t = tensor.tensor_alloc(T, shape, owns_data = true, allocator = allocator, loc = loc)

	// Convert based on ONNX data type
	if len(raw_data) > 0 {
		#partial switch ONNX_DataType(data_type) {
		case .FLOAT:
			// 1
			src_data := mem.slice_data_cast([]f32, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .UINT8:
			// 2
			src_data := mem.slice_data_cast([]u8, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .INT8:
			// 3
			src_data := mem.slice_data_cast([]i8, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .INT32:
			// 6
			src_data := mem.slice_data_cast([]i32, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .INT64:
			// 7
			src_data := mem.slice_data_cast([]i64, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .FLOAT16:
			// 10
			src_data := mem.slice_data_cast([]f16, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case .DOUBLE:
			// 11
			src_data := mem.slice_data_cast([]f64, raw_data)
			for src_val, i in src_data {
				t.data[i] = T(src_val)
			}

		case:
			panic(fmt.tprintf("Unsupported ONNX data type %d for tensor %s", data_type, name))
		}
	} else {
		panic(fmt.tprintf("Tensor %s has no raw_data", name))
	}

	fmt.println(name)

	return t, nil
}
