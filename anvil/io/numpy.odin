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
get_alignment :: proc(np_type_char: string) -> (alignment: uint,  ok : IO_Error ) {
	ok = nil
	switch np_type_char {
	// double, ('d', dtype('float64'))
	// longdouble, ('g', dtype('float64'))
	// float64, ('d', dtype('float64'))
	case "f8" : alignment = 8
	// longlong, ('q', dtype('int64'))
	// int64, ('q', dtype('int64'))
	case "i8" : alignment = 8
	// single, ('f', dtype('float32'))
	// float32, ('f', dtype('float32'))
	case "f4" : alignment = 4
	// half, ('e', dtype('float16'))
	// float16, ('e', dtype('float16'))
	case "f2" : alignment = 2
	// intc, ('i', dtype('int32'))
	// int, ('l', dtype('int32'))
	// int32, ('l', dtype('int32'))
	case "i4" : alignment = 4
	// short, ('h', dtype('int16'))
	// int16, ('h', dtype('int16'))
	case "i2" : alignment = 2
	// int8, ('b', dtype('int8'))
	case "i1" : alignment = 1
	// // cdouble, ('D', dtype('complex127'))
	// // clongdouble, ('G', dtype('complex128'))
	// // complex128, ('D', dtype('complex128'))
	// case "c16": alignment = 8
	// // csingle, ('F', dtype('complex64'))
	// // complex64, ('F', dtype('complex64'))
	// case "c8" : alignment = 4
	// ulonglong, ('Q', dtype('uint64'))
	case "u8" : alignment = 8
	// uintc, ('I', dtype('uint32'))
	case "u4" : alignment = 4
	// ushort, ('H', dtype('uint16'))
	case "u2" : alignment = 2
	// uint8, ('B', dtype('uint8'))
	// ubyte, ('B', dtype('uint8'))
	case "u1" : alignment = 1
	case:
		alignment = 255 // not supported
		ok = NPY_Not_Implemented{"Array with non-numeric type is not supported!"}

	}
	return
}

NPY_Array_Header :: struct #packed {
	magic         : string,
	version       : [2]u8,  // [major, minor]
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

// read_numpy_array_from_npy_file
// Read NumPy's `npy` file as `anvil.tensor.Tensor`. The produced `Tensor` would have
// same shape with the input array with type of `T`. It only support numeric types
// except for complex numbers, other NumPy's dtype is not supported, yet.
// Note: For unsigned data types, they will be casted to `T`.
read_numpy_array_from_npy_file :: proc(
	$T: typeid,
	file_name: string,
	bufreader_size: int = NPY_BUFFER_READER_SIZE,
	allocator:= context.allocator,
	loc := #caller_location,
) -> (
	out : ^tensor.Tensor(T),
	parse_numpy_npy_error : IO_Error,
) where intrinsics.type_is_numeric(T) {

	// define bufio, os, and io objects
	bufio_reader     : bufio.Reader
	handle           : os.Handle
	os_open_error    : os.Error
	reader           : io.Stream
	ok               : bool             // general usage
	
	// NOTE: npy in header is NOT an abreviation for NumPy
	// it is referring exactly to `.npy` file format
	npy_header       := new(NPY_Array_Header) // numpy npy file header

	{ // scoping the stream and readers
		handle, os_open_error = os.open(file_name, os.O_RDONLY)
		if os_open_error != os.ERROR_NONE do return nil, NPY_Open_Error{file_name, os_open_error}
	
		// create a stream
		stream := os.stream_from_handle(handle)
	
		// create a reader
		reader, ok = io.to_reader(stream)
		if !ok do return nil, NPY_Reader_Creation_Error{file_name, stream}
		bufio.reader_init(&bufio_reader, reader, bufreader_size, allocator, loc)
		// guard the reader to stop early when there is no progress when reading file
		bufio_reader.max_consecutive_empty_reads = 1
	}
	
	// read and validate header section
	parse_numpy_npy_error = parse_and_validate_npy_header(&reader, npy_header, allocator, loc)
	if parse_numpy_npy_error != nil do return nil, parse_numpy_npy_error
	
	// tensor allocation
	out = tensor.tensor_alloc(T, npy_header.shape[:], true, allocator, loc)
	
	// figure out length of bytes in the array
	n_elem : uint
	if len(npy_header.shape) > 1 {
		n_elem = tensor.shape_to_size(cast([]uint)npy_header.shape)
	} else {
		n_elem = npy_header.shape[0]
	}
	n_elem *= npy_header.alignment
	
	// start numpy array parsing
	ok = parse_npy_array_values(
		T,
		npy_header,
		&bufio_reader,
		out,
		n_elem,
		allocator = allocator,
		loc = loc
	)
	if !ok {
		return nil, NPY_Read_Array_Error{
			"Cannot parse array values, possible curropted data, or data saved with type that is not supported yet"
		}
	}
	delete_np_header(npy_header)
	return out, nil
}

@(private = "file")
parse_npy_array_values :: proc(
	$T        : typeid,
	np_header : ^NPY_Array_Header,
	reader    : ^bufio.Reader,
	tensor    : ^tensor.Tensor(T),
	n_elem    : uint,
	allocator := context.allocator,
	loc       := #caller_location,
) -> ( bool ) where intrinsics.type_is_numeric(T) {

	count     := uint(0)
	i         := uint(0)
	alignment := np_header.alignment
	endianess := np_header.endianess

	read_bytes_err : io.Error
	raw_bytes_pos  : int
	cast_ok : bool = true

	raw_bytes_container := make([]u8, n_elem, allocator=allocator, loc=loc)
	raw_bytes_pos, read_bytes_err = bufio.reader_read(reader, raw_bytes_container[:])

	switch np_header.descr[1:] {

	case "f8" :
		casted_data : f64
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_f64(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "i8" :
		casted_data : i64
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_i64(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "f4" :
		casted_data : f32
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_f32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "i4" :
		casted_data : i32
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_i32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "f2" :
		casted_data : f16
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_f16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "i2" :
		casted_data : i16
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_i16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "u2" :
		casted_data : u16
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "u4" :
		casted_data : u32
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u32(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "u8" :
		casted_data : u16
		#no_bounds_check for ; i < n_elem; i += alignment {
			casted_data, cast_ok = endian.get_u16(raw_bytes_container[i:i+alignment], endianess)
			if !cast_ok do break
			tensor.data[count] = T(casted_data)
			count += 1
		}
		return cast_ok

	case "u1" :
		#no_bounds_check for ; i < n_elem; i += alignment {
			tensor.data[count] = T(raw_bytes_container[i])
			count += 1
		}
		return true

	case "i1" :
		#no_bounds_check for ; i < n_elem; i += alignment {
			tensor.data[count] = T(raw_bytes_container[i])
			count += 1
		}
		return true
	case : return false
	}
}

// parse_npy_header
@(private = "file")
parse_and_validate_npy_header :: proc(
	reader     : ^io.Stream,
	npy_header : ^NPY_Array_Header,
	allocator  := context.allocator,
	loc        := #caller_location,
) -> (err: IO_Error) {

	// Layout. (source: https://numpy.org/neps/nep-0001-npy-format.html)
	//
	// ---
	//
	// First 6 bytes are a magic string: exactly “x93NUMPY”, `MAGIC_NPY`
	//
	// ---
	//
	// Next  1 byte  is an unsigned byte: major version of NumPy
	// Next  1 byte  is an unsigned byte: minor version of NumPy
	// We directly read major and minor version in one go in variable `version : [2]u8`
	//
	// ---
	//
	// Next  2 bytes is a little-endian unsigned short int: in variable `header_length : [2]u8`
	// These 2 bytes tell us how many bytes that we should read in order to get all
	// the header. The header descriptions are ASCII strings which contains
	// Python's literal experssion of dictionary (map/hashmap). Header is terminated
	// by new line character `\n`.
	//
	//
	// EXAMPLE:
	// header : "{'descr': '|b1', 'fortran_order': False, 'shape': (5, 5), }                                                          \n"
	// with length of: 118
	//
	// Note that `descr` later will be refered as `type char` in this procedure.
	//
	//     more about above example can be seen in
	//     https://github.com/kelreeeeey/python-numpy-npy-in-odin#generate-test-data
	//
	// ---
	//
	// This procedure only cares for the first (6 + (1 + 1) + 2 + `header_length`)
	// bytes. The rest of the bytes will handled by `parse_npy_array_values`
	// procedure.


	// read magic header (6 first bytes)
	// these bytes indicating whether the file is produced by NumPy or not.
	// read the first six bytes of the header, it should be equal to `MAGIC_NPY`
	// otherwise the file is not valid numpy's npy file.
	magic : [6]u8
	read, rerr := io.read(reader^, magic[:], &MAGIC_NPY_LEN)
	if (rerr != nil || read != 6 || !slice.equal(magic[:], MAGIC_NPY)) {
		delete_np_header(npy_header)
		return NPY_Invalid_Header_Error{"Invalid NumPy's npy file."}
	}
	clone_err : mem.Allocator_Error
	npy_header.magic, clone_err = strings.clone_from_bytes(magic[:], allocator, loc)
	if clone_err != nil {
		delete_np_header(npy_header)
		return clone_err
	}

	// read version
	// Version info is exactly 2 bytes right after magic header (MAGIC_NPY)
	// it should be either one of these
	//   - {1, 0}
	//   - {2, 0}
	//   - {3, 0}
	// otherwise the file is not
	// a valid numpy's npy file.
	version : [2]u8
	read, rerr = io.read(reader^, version[:])
	if rerr != nil || read != 2 {
		delete_np_header(npy_header)
		return NPY_Invalid_Header_Error{"Invalid NumPy's npy file"}
	}
	if (version[1] >= version[0]) || (version[0] > 4 || version[1] != 0) {
		delete_np_header(npy_header)
		return NPY_Invalid_Header_Error{"Invalid NumPy's npy file. Version is not supported"}
	}
	npy_header.version = version

	// Infer header's length
	header_length : [2]u8
	header := header_length[:]
	read, rerr = io.read(reader^, header)
	if rerr != nil || read != 2 {
		delete(header)
		delete_np_header(npy_header)
		return NPY_Invalid_Header_Length{header_length}
	}
	npy_header.header_length = transmute(u16le)header_length
	len_header := cast(int)npy_header.header_length
	header_desc := make([]u8, len_header)

	// Actually reading the header, convert the header to string, take the
	// necessary information, and put them in `npy_header`
	read, rerr = io.read(reader^, header_desc[:])
	if rerr != nil || read != len_header {
		delete_np_header(npy_header)
		return NPY_Invalid_Header_Length{header_length}
	}

	// From the point, all the following processes are string manipulations since the header
	// are literal ASCII character of Python's dictionary (map/hashmap) expression.

	// Clean up header string
	clean_header := strings.trim_space(string(header_desc))
	is_alloc : bool
	// Replace single quotes
	clean_header, is_alloc = strings.replace(clean_header, "'", "\"", -1, allocator, loc)
	clean_header, is_alloc = strings.replace(clean_header, "(", "[", -1 , allocator, loc)
	clean_header, is_alloc = strings.replace(clean_header, ")", "]", -1 , allocator, loc)
	if !is_alloc {
		delete_np_header(npy_header)
		return nil
	}

	// Parse the byte order and type char
	if descr_start := strings.index(clean_header, "\"descr\":"); descr_start != -1 {
		descr_start += 8 // offset exactly the length of ` "descr": `
		descr_end := strings.index_byte(clean_header[descr_start:], ',')
		if descr_end == -1 {
			delete_np_header(npy_header)
			return NPY_Malformed_Header{}
		}

		descr_str := strings.trim(clean_header[descr_start:descr_start+descr_end], " \"")
		descr := " "
		clone_err : runtime.Allocator_Error
		// Handle native/byte-order-agnostic types
		switch {
		case strings.has_prefix(descr_str, "|"):
			npy_header.endianess = endian.PLATFORM_BYTE_ORDER
			descr, clone_err := strings.clone(descr_str[:], allocator, loc)
			if clone_err != nil {
				delete(descr)
				delete_np_header(npy_header)
				return clone_err
			}
			npy_header.descr = descr
		case strings.has_prefix(descr_str, "<") :
			npy_header.endianess = endian.Byte_Order.Little
			descr, clone_err := strings.clone(descr_str[:], allocator, loc)
			if clone_err != nil {
				delete(descr)
				delete_np_header(npy_header)
				return clone_err
			}
			npy_header.descr = descr
		case strings.has_prefix(descr_str, ">") :
			npy_header.endianess = endian.Byte_Order.Big
			descr, clone_err := strings.clone(descr_str[:], allocator, loc)
			if clone_err != nil {
				delete(descr)
				delete_np_header(npy_header)
				return clone_err
			}
			npy_header.descr = descr
		case:
			npy_header.endianess = endian.PLATFORM_BYTE_ORDER
			descr, clone_err := strings.clone(descr_str[:], allocator, loc)
			if clone_err != nil {
				delete(descr)
				delete_np_header(npy_header)
				return clone_err
			}
			npy_header.descr = descr
		}
		// take the type char only, e.g. take f8 from <f8, if the type char is
		// is not numeric, return NPY_Not_Implemented
		ok : IO_Error
		npy_header.alignment, ok = get_alignment(npy_header.descr[1:])
		if ok != nil {
			delete_np_header(npy_header)
			return ok
		}
	}

	// Parse fortran_order
	npy_header.fortran_order = true // first assumption
	if fo_start := strings.index(clean_header, "\"fortran_order\":"); fo_start != -1 {
		fo_start += 16  // Skip `"fortran_order": `
		fo_str := clean_header[fo_start:]
		is_fortran := strings.has_prefix(fo_str, "True")
		if is_fortran {
			delete_np_header(npy_header)
			return NPY_Not_Implemented{
				"Array with fortran order is not supported yet"
			}
		}
	}
	

	// Parse shape tuple
	if shape_start := strings.index(clean_header, "\"shape\":"); shape_start != -1 {
		shape_start += 8  // Skip ` "shape": `
		shape_end := strings.index_byte(clean_header[shape_start:], ']')
		if shape_end == -1 {
			delete_np_header(npy_header)
			return NPY_Shape_Parse_Failed{}
		}
		shape_str := clean_header[shape_start:shape_start+shape_end]
		shape_str = strings.trim_space(shape_str)
		shape_str = strings.trim(shape_str, "[]")

		// Split and parse integers
		parts := strings.split(shape_str, ",", allocator)
		defer delete(parts)
		npy_header.shape = make([]uint, len(parts), allocator)

		count := uint(0)
		for part in parts {
			trimmed := strings.trim_space(part)
			if trimmed == "" { continue }
			value, ok := strconv.parse_int(trimmed)
			if !ok {
				delete_np_header(npy_header)
				return NPY_Shape_Parse_Failed{}
			}
			npy_header.shape[count] = cast(uint)value
			count += 1
        }
		npy_header.shape = npy_header.shape[:count]
    }
    return nil
}

import "core:fmt"
import "core:testing"

@(test)
read_numpy_array_from_npy_file_longdouble_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/longdouble_5x5.npy
	// ```python
	// import numpy as np
	// clongdouble      = np.arange(1, 6, 1).astype(np.clongdouble)
	// clongdouble_5x5  = np.array([clongdouble+x for x in range(5)])
	// np.save("assets/test_np_arrays/longdouble_5x5.npy", clongdouble_5x5)
	// ```
	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		f32,
		"assets/test_np_arrays/longdouble_5x5.npy",
		allocator=context.allocator,
	)
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


@(test)
read_numpy_array_from_npy_file_complex_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/complex128_5x5.npy
	// ```python
	// import numpy as np;
	// complex128 = np.arange(1, 6, 1).astype(np.complex128)
	// complex128_5x5 = np.array([complex128 +x for x in range(5)])
	// np.save("assets/test_np_arrays/complex128_5x5.npy", complex128)
	// ```
	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		f64,
		"assets/test_np_arrays/complex128_5x5.npy",
		allocator=context.allocator,
	)
	testing.expect(t, err != nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
}

@(test)
read_numpy_array_from_npy_file_boolean_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/boolean_5x5.npy
	// ```python
	// import numpy as np;
	// b_5     = np.array([1, 0, 1, 0, 1]).astype(np.bool_)
	// b_5x5   = np.array([b_5 for _ in range(5)])
	// np.save("assets/test_np_arrays/boolean_5x5.npy", b_5x5)
	// ```
	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		f32,
		"assets/test_np_arrays/boolean_5x5.npy",
		allocator=context.allocator,
	)
	testing.expect(t, err != nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
}

@(test)
read_numpy_array_from_npy_file_ubyte_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/ubyte_5x5.npy
	// ```python
	// import numpy as np;
    // ubyte_5     = np.array([1, 0, 1, 0, 1]).astype(np.ubyte)
    // ubyte_5x5   = np.array([ubyte_5 for _ in range(5)])
	// np.save("assets/test_np_arrays/ubyte_5x5.npy", ubyte_5x5)
	// ```

	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		f32,
		"assets/test_np_arrays/ubyte_5x5.npy",
		allocator=context.allocator,
	)
	testing.expect(t, err == nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
}

@(test)
read_numpy_array_from_npy_file_BCHW_array_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/array_bchw.npy
	// ```python
	// import numpy as np
	// a = np.arange(0, 16, 1).astype("int32")
	// x,y = np.meshgrid(a,a)
	// b = np.concatenate(
	//     [
	//         x[np.newaxis, np.newaxis, ...],
	//         y[np.newaxis, np.newaxis, ...]
	//     ],
	//     axis=1
	// )
	// np.save("assets/test_np_arrays/array_bchw.npy", b)
	// ```
	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		i32,
		"assets/test_np_arrays/array_bchw.npy",
		allocator=context.allocator,
	)
	testing.expect(t, err == nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
	testing.expect(t, slice.equal(np_tensor.shape, []uint{1, 2, 16, 16}))
	testing.expect(
		t,
		slice.equal(
			np_tensor.data,
			[]i32{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
				0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
				4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
				5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
				6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
				7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
				8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
				9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
				10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
				11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
				12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
				13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
				14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
				15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
			}
		)
	)
}

@(test)
read_numpy_array_from_npy_file_nested_shaped_array_test :: proc(t: ^testing.T) {
	// creation of assets/test_np_arrays/nested_shape.npy
	// ```python
	// import numpy as np
	// a = np.arange(0, 16, 1).astype("int32")
	// a = a[
	//     np.newaxis, np.newaxis, np.newaxis,
	//     np.newaxis, np.newaxis, ...
	// ]
	// b = np.concatenate([ a, a ], axis=3)
	// np.save("assets/test_np_arrays/nested_shape.npy", b)
	// ```
	context.allocator = context.temp_allocator
	np_tensor, err := read_numpy_array_from_npy_file(
		i64,
		"assets/test_np_arrays/nested_shape.npy",
		allocator=context.allocator,
	)
	testing.expect(t, err == nil, fmt.tprint(err))
	defer tensor.free_tensor(np_tensor)
	testing.expect(t, slice.equal(np_tensor.shape, []uint{1, 1, 1, 2, 1, 16}))
	testing.expect(
		t,
		slice.equal(
			np_tensor.data,
			[]i64{
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
				0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
			}
		)
	)
}
