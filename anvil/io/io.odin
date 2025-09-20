// package io is a collection of utilities to aid with basic file reading and its conversion to
// anvil tensor.
package file_io

import "core:encoding/csv"
import "core:image"
import "core:os"
import "core:io"
import "core:bufio"
import "core:mem"

CSV_Read_Error :: csv.Error
CSV_Format_Conversion_Error :: struct {}
CSV_Empty_Row :: struct {}
CSV_Inconsistent_Column_Count :: struct {}

Image_Load_Error :: image.Error
Cannot_Read_File :: struct {}
Invalid_Image_Format :: struct {}

NPY_Open_Error :: struct {
	file_name: string,
	error: os.Errno,
}

NPY_Reader_Creation_Error :: struct {
	file_name: string,
	stream: io.Stream,
}

NPY_Reader_Read_Byte_Error :: struct {
	file_name: string,
	reader: bufio.Reader,
}

NPY_Invalid_Header_Error :: struct {
	message: string,
}

NPY_Invalid_Version_Error :: struct {
	message: string,
	version: [2]u8,
}

NPY_Invalid_Header_Length_Error :: struct {
	message: string,
	length: [2]u8,
}

NPY_Not_Implemented :: struct {
	message: string
}

NPY_Parse_Error :: enum {
	NPY_Invalid_Descriptor,
	NPY_Malformed_Header,
	NPY_Shape_Parse_Failed,
}

NPY_Read_Array_Error :: struct {
	message: string
}

IO_Error :: union {
	CSV_Read_Error,
	CSV_Format_Conversion_Error,
	CSV_Empty_Row,
	CSV_Inconsistent_Column_Count,
	Image_Load_Error,

	NPY_Open_Error,

	// NOTE(Rey): allocation error for bufio in numpy parser
	// TODO(Rey): maybe alias this one?
	mem.Allocator_Error,

	NPY_Reader_Creation_Error,
	NPY_Reader_Read_Byte_Error,
	NPY_Invalid_Header_Error,
	NPY_Invalid_Version_Error,
	NPY_Invalid_Header_Length_Error,
	NPY_Parse_Error,
	NPY_Read_Array_Error,
	NPY_Not_Implemented,

	Cannot_Read_File,
	Invalid_Image_Format,

}
