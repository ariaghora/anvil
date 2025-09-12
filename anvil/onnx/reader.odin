// References:
// - https://protobuf.dev/programming-guides/encoding/
package onnx

import "core:fmt"
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

ONNX :: struct($T: typeid) {}

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
	for offset < len(raw_bytes) {
		// step 1: get the tag, i.e., the first byte of currently parsed sequence
		tag := raw_bytes[offset]
		offset += 1 // Move one bit

		// step 2: get field number.
		// take the last three bits to get the wire type
		// and then right-shift by three to get the field number.
		wire_type := tag & 0b00000111
		field_num := tag >> 3

		// Let's just panic upon deprecated wire types (3 and 4)
		wire_type_is_deprecated := slice.contains([]u8{3, 4}, wire_type)
		ensure(!wire_type_is_deprecated, fmt.tprintf("wire type %d is unsupported", wire_type))

		// step 3: read the actual field data based on wire type
		switch wire_type {
		// Varint: int32, int64, uint32, uint64, sint32, sint64, bool, enum
		case 0:
			value, new_offset := read_varint(raw_bytes, offset) or_return
			offset = new_offset
			// TODO(Aria): dispatch based on field_num to store in appropriate struct field
			{}
		// Length-delimited: string, bytes, embedded messages, packed repeated fields
		case 2:
			length, new_offset := read_varint(raw_bytes, offset) or_return
			offset = new_offset

			if offset + int(length) > len(raw_bytes) do return nil, Truncated_Data{}
			payload := raw_bytes[offset:offset + int(length)]
			offset += int(length)

			switch field_num {
			case 2:
				fmt.println("Producer Name:", string(payload))
			case 3:
				fmt.println("Producer Version:", string(payload))
			// case 7:
			// TODO(Aria): parse graph recursively
			case:
				fmt.panicf("Field num %d handling is not implemented yet", field_num)
			}
		case:
			fmt.panicf("wire type %d is unsupported", wire_type)
		}


	}
	return nil, nil
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

	// If we get here, varint was too long or data ended
	panic("Varint is too long, invalid ONNX format")
}
