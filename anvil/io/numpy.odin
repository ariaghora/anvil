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

Array_Type :: union {
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

NPY_Array_Header :: struct #packed {
	magic         : string,
	version       : [2]u8, // [major, minor]
	header_length : u16le,
	descr         : string, // 'endian'+type-char, e.g. "<f8"
	fortran_order : bool,
	shape         : []uint,
	endianess     : endian.Byte_Order,
	alignment     : uint
}

delete_np_header :: proc(h: ^NPY_Array_Header) {
	delete(h.magic)
	delete(h.shape)
	delete(h.descr)
}

// read_numpy_array_from_file
// Read NumPy's npy file as `anvil.tensor.Tensor`. The produced `Tensor` would have
// same shape with the input array with type of `T`. It only support numeric types
// and boolean, other NumPy's dtype is not supported, yet.
read_numpy_array_from_file :: proc(
	$T: typeid,
	file_name: string,
	bufreader_size: int = NPY_BUFFER_READER_SIZE,
	allocator:= context.allocator,
	loc := #caller_location,
) -> (
	^tensor.Tensor(T),
	IO_Error,
) where intrinsics.type_is_numeric(T) || T == b8 {

	// define bufio_reader, and io.Stream
	bufio_reader : bufio.Reader
	reader       : io.Stream
	ok           : bool
	// create an handler
	npy_header   := NPY_Array_Header{}
	defer delete_np_header(&npy_header)

	{ // scoping the stream and readers
		handle, open_error := os.open(file_name, os.O_RDONLY)
		if open_error != os.ERROR_NONE do return nil, NPY_Open_Error{file_name, open_error}

		// create a stream
		stream := os.stream_from_handle(handle)

		// create a reader
		reader, ok = io.to_reader(stream)
		if !ok do return nil, NPY_Reader_Creation_Error{file_name, stream}

		bufio.reader_init(&bufio_reader, reader, bufreader_size, allocator)
		bufio_reader.max_consecutive_empty_reads = 1
	}

	magic : [6]u8
	// read magic magic
	read, rerr := io.read(reader, magic[:], &MAGIC_NPY_LEN)
	if rerr != nil || read != 6 do return nil, NPY_Invalid_Header_Error{"Invalid magic number"}
	if !slice.equal(magic[:], MAGIC_NPY) do return nil, NPY_Invalid_Header_Error{"Invalid magic number"}

	clone_err : mem.Allocator_Error
	npy_header.magic, clone_err = strings.clone_from_bytes(magic[:])
	if clone_err != nil do return nil, nil

	// read version
	version : [2]u8
	read, rerr = io.read(reader, version[:])
	if rerr != nil || read != 2 do return nil, NPY_Invalid_Version_Error{"Invalid version", version}
	npy_header.version = version

	header_lenght : [2]u8
	// read header length
	read, rerr = io.read(reader, header_lenght[:])
	if rerr != nil || read != 2 do return nil, NPY_Invalid_Header_Length{header_lenght}
	npy_header.header_length = transmute(u16le)header_lenght

	// TODO(Rey): not sure about keeping this len_header thingy
	len_header := cast(int)transmute(u16le)header_lenght
	header_desc := make([]u8, len_header)
	read, rerr = io.read(reader, header_desc[:])
	if rerr != nil || read != len_header do return nil, NPY_Invalid_Header_Length{header_lenght}

	// parsed_header : Descriptor
	parr_err := parse_npy_header(&npy_header, string( header_desc ))
	if parr_err != nil do return nil, parr_err
	if npy_header.fortran_order do return nil, NPY_Not_Implemented{"Array with fortran order is not supported yet"}

	out := tensor.tensor_alloc(T, npy_header.shape[:], true, allocator, loc)

	type_char := npy_header.descr[1:]
	npy_header.alignment = get_alignment(type_char)

	n_elem : uint
	if len(npy_header.shape) > 1 {
		n_elem = tensor.shape_to_size(cast([]uint)npy_header.shape)
	} else {
		n_elem = npy_header.shape[0]
	}
	n_elem *= npy_header.alignment

	ok = recreate_npy_array(
		T,
		&npy_header,
		&bufio_reader,
		out,
		n_elem,
		allocator = allocator
	)
	if !ok do return nil, NPY_Read_Array_Error{"Cannot parse data array, possible curropted data type is not supported yet"}
	return out, nil
}

@(private = "file")
recreate_npy_array :: proc(
	$T: typeid,
	np_header: ^NPY_Array_Header,
	reader: ^bufio.Reader,
	tensor : ^tensor.Tensor(T),
	n_elem : uint,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	bool
) where intrinsics.type_is_numeric(T) || T == b8 {

	count     := uint(0)
	i         := uint(0)
	alignment := np_header.alignment
	endianess := np_header.endianess

	read_bytes_err : io.Error
	raw_bytes_pos  : int

	// TODO(Rey) : should we defer delet this and copy the content to
	// the tensor.data instead? NEED advice.
	raw_bytes_container := make([]u8, n_elem, allocator=allocator, loc=loc)

	raw_bytes_pos, read_bytes_err = bufio.reader_read(reader, raw_bytes_container[:])

    switch np_header.descr[1:] {
	case "b1" :
		#no_bounds_check for ; i < n_elem; i += alignment {
			tensor.data[count] = cast(T)raw_bytes_container[i]
			count += 1
		}
		return true

	case "u1" :
		#no_bounds_check for ; i < n_elem; i += alignment {
			tensor.data[count] = cast(T)raw_bytes_container[i]
			count += 1
		}
		return true

	case "i1" :
		#no_bounds_check for ; i < n_elem; i += alignment {
			tensor.data[count] = cast(T)raw_bytes_container[i]
			count += 1
		}
		return true

	case "i2" :
		casted_data : i16
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_i16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "u2" :
		casted_data : u16
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "u4" :
		casted_data : u32
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "i4" :
		casted_data : i32
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_i32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "u8" :
		casted_data : u16
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "i8" :
		casted_data : i64
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_i64(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "f2" :
		casted_data : f16
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_f16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "c8" :
		casted_data : f32
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_f32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "c16" :
		casted_data : f64
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem-uint(alignment/2); i += alignment {
			casted_data, cast_ok := endian.get_f64(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "f4" :
		casted_data : f32
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_f32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok

	case "f8" :
		casted_data : f64
		cast_ok : bool = true
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok := endian.get_f64(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = cast(T)casted_data
			count += 1
		}
		return cast_ok
    }
    return false
}

// parse_npy_header
@(private = "file")
parse_npy_header :: proc(
	h: ^NPY_Array_Header,
	header: string,
	allocator := context.allocator
) -> (err: NPY_Parse_Error) {

	// Clean up header string
	clean_header := strings.trim_space(header)
	is_alloc : bool
	// Replace single quotes
	clean_header, is_alloc = strings.replace(clean_header, "'", "\"", -1)
	clean_header, is_alloc = strings.replace(clean_header, "(", "[", -1)
	clean_header, is_alloc = strings.replace(clean_header, ")", "]", -1)

	// Enhanced descriptor parsing
	if descr_start := strings.index(clean_header, "\"descr\":"); descr_start != -1 {
		descr_start += 8 // offset exactly the length of ` "descr": `
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
		shape_start += 8  // Skip ` "shape": `
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

@(test)
read_numpy_array_from_file_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/longdouble_5x5.npy
	// ```
	// import numpy as np
	// clongdouble      = np.arange(1, 6, 1).astype(np.clongdouble)
	// clongdouble_5x5  = np.array(list(clongdouble + x for x in range(5)))
	// np.save("assets/test_np_arrays/longdouble_5x5.npy", clongdouble_5x5)
	// ```

	np_tensor, err := read_numpy_array_from_file(f32, "assets/test_np_arrays/longdouble_5x5.npy")
	testing.expect(t, err == nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
	testing.expect(t, slice.equal(np_tensor.shape, []uint{5, 5}))
	testing.expect(
		t,
		slice.equal(
			np_tensor.data,
			[]f32{
				1, 2, 3, 4, 5,
				2, 3, 4, 5, 6,
				3, 4, 5, 6, 7,
				4, 5, 6, 7, 8,
				5, 6, 7, 8, 9
			}
		)
	)
}
