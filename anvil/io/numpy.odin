package file_io

import "../tensor"
import "base:runtime"
import "base:intrinsics"
import "core:os"
import "core:io"
import "core:bufio"
import "core:mem"
import "core:strings"
import "core:slice"
import "core:strconv"
import "core:encoding/endian"

// from https://github.com/numpy/numpy/blob/main/numpy/lib/_format_impl.py
MAGIC_NPY :: []u8{0x93, 'N', 'U', 'M', 'P', 'Y'}
NPY_BUFFER_READER_SIZE :: 1024
MAGIC_NPY_LEN := len(MAGIC_NPY)

@(private = "file")
get_alignment :: proc(np_type_char: string) -> uint {
	alignment : uint
	switch np_type_char {
		// bool, ('?', dtype('bool'))
		// byte, ('b', dtype('int8'))
		// int8, ('b', dtype('int8'))
		case "i1" : alignment = 1
		// short, ('h', dtype('int16'))
		// int16, ('h', dtype('int16'))
		case "i2" : alignment = 2
		// intc, ('i', dtype('int32'))
		// int, ('l', dtype('int32'))
		// int32, ('l', dtype('int32'))
		case "i4" : alignment = 4
		// longlong, ('q', dtype('int64'))
		// int64, ('q', dtype('int64'))
		case "i8" : alignment = 8
		// uint8, ('B', dtype('uint8'))
		// ubyte, ('B', dtype('uint8'))
		case "u1" : alignment = 1
		// ushort, ('H', dtype('uint16'))
		case "u2" : alignment = 2
		// uintc, ('I', dtype('uint32'))
		case "u4" : alignment = 4
		// ulonglong, ('Q', dtype('uint64'))
		case "u8" : alignment = 8
		// half, ('e', dtype('float16'))
		// float16, ('e', dtype('float16'))
		case "f2" : alignment = 2
		// single, ('f', dtype('float32'))
		// float32, ('f', dtype('float32'))
		case "f4" : alignment = 4
		// double, ('d', dtype('float64'))
		// longdouble, ('g', dtype('float64'))
		// float64, ('d', dtype('float64'))
		case "f8" : alignment = 8
		// csingle, ('F', dtype('complex64'))
		// complex64, ('F', dtype('complex64'))
		case "c8" : alignment = 4
		// cdouble, ('D', dtype('complex127'))
		// clongdouble, ('G', dtype('complex128'))
		// complex128, ('D', dtype('complex128'))
		case "c16": alignment = 8
	}
	return alignment
}

ArrayTypes :: union {
	b8,
	u8,
	i8,
	i16,
	u16,
	i32,
	u32,
	i64,
	u64,
	f16,
	f32,
	f64,
	f16be,
	f16le,
	complex32,
	complex64,
}

NumpyHeader :: struct #packed {
	magic         : string,
	version       : [2]u8, // [major, minor]
	header_length : u16le,
	descr         : string,
	fortran_order : bool,
	shape         : []uint,
	endianess     : endian.Byte_Order,
}

@(private = "file")
parse_npy_header :: proc(
	h: ^NumpyHeader,
	header: string,
	allocator := context.allocator
) -> (err: ParseError) {

	// Clean up header string
	clean_header := strings.trim_space(header)
	is_alloc : bool
	// Replace single quotes
	clean_header, is_alloc = strings.replace(clean_header, "'", "\"", -1)
	clean_header, is_alloc = strings.replace(clean_header, "(", "[", -1)
	clean_header, is_alloc = strings.replace(clean_header, ")", "]", -1)

	// Enhanced descriptor parsing
	if descr_start := strings.index(clean_header, "\"descr\":"); descr_start != -1 {
		descr_start += 8 // exactly the length of ` "descr": `
		descr_end := strings.index_byte(clean_header[descr_start:], ',')
		if descr_end == -1 do return .NPY_Malformed_Header
		descr_str := strings.trim(clean_header[descr_start:descr_start+descr_end], " \"")
		// Handle native/byte-order-agnostic types
		switch {
		case strings.has_prefix(descr_str, "|"):
			h.endianess = endian.PLATFORM_BYTE_ORDER
			descr, clone_err := strings.clone(descr_str[:])
			h.descr = descr
		case strings.has_prefix(descr_str, "<") :
			// Existing endian-sensitive types
			h.endianess = endian.Byte_Order.Little
			descr, clone_err := strings.clone(descr_str[:])
			h.descr = descr
		case strings.has_prefix(descr_str, ">") :
			// Existing endian-sensitive types
			h.endianess = endian.Byte_Order.Big
			descr, clone_err := strings.clone(descr_str[:])
			h.descr = descr
		case: // Handle non-byte-ordered types
			h.endianess = endian.PLATFORM_BYTE_ORDER
			descr, clone_err := strings.clone(descr_str[:])
			h.descr = descr
		}
	}

	// Parse fortran_order
	if fo_start := strings.index(clean_header, "\"fortran_order\":"); fo_start != -1 {
		fo_start += 16  // Skip `"fortran_order": `
		fo_str := clean_header[fo_start:]
		h.fortran_order = strings.has_prefix(fo_str, "True")
	}

	// Parse shape tuple
	if shape_start := strings.index(clean_header, "\"shape\":"); shape_start != -1 {

		shape_start += 8  // Skip `"shape": `
		shape_end := strings.index_byte(clean_header[shape_start:], ']')

		if shape_end == -1 do return .NPY_Shape_Parse_Failed

		shape_str := clean_header[shape_start:shape_start+shape_end]
		shape_str = strings.trim_space(shape_str)
		shape_str = strings.trim(shape_str, "[]")

		// Split and parse integers
		parts := strings.split(shape_str, ",", allocator)
		defer delete(parts)
		h.shape = make([]uint, len(parts), allocator)

		count := uint(0)
		for part in parts {
			trimmed := strings.trim_space(part)
			if trimmed == "" { continue }
			value, ok := strconv.parse_int(trimmed)
			if !ok do return .NPY_Shape_Parse_Failed
			h.shape[count] = cast(uint)value
			count += 1
        }
		h.shape = h.shape[:count]

    }

    return nil
}

import "core:fmt"
import "core:testing"

