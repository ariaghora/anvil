// package io is a collection of utilities to aid with basic file reading and its conversion to
// anvil tensor.
package file_io

import "core:bufio"
import "core:encoding/csv"
import "core:image"
import "core:io"
import "core:mem"
import "core:os"

CSV_Read_Error :: csv.Error
CSV_Format_Conversion_Error :: struct {}
CSV_Empty_Row :: struct {}
CSV_Inconsistent_Column_Count :: struct {}

Image_Load_Error :: image.Error
Image_Cannot_Read_File :: struct {}
Image_Invalid_Format :: struct {}
Image_Unsupported_Output_Extension :: struct {}
Image_Write_Failed :: struct {}

NPY_Open_Error :: struct {
	file_name: string,
	error:     os.Errno,
}
NPY_Reader_Creation_Error :: struct {
	file_name: string,
	stream:    io.Stream,
}
NPY_Reader_Read_Byte_Error :: struct {
	file_name: string,
	reader:    bufio.Reader,
}
NPY_Invalid_Header_Error :: struct {
	message: string,
}
NPY_Invalid_Version_Error :: struct {
	message: string,
	version: [2]u8,
}
NPY_Invalid_Header_Length :: struct {
	length: [2]u8,
}
// NOTE(Rey) : maybe `NPY_Not_Implemented` should be an enum instead of struct
NPY_Not_Implemented :: struct {
	message: string,
}
NPY_Invalid_Descriptor :: struct {}
NPY_Malformed_Header :: struct {}
NPY_Shape_Parse_Failed :: struct {}
NPY_Read_Array_Error :: struct {
	message: string,
}

IO_Error :: union {
	mem.Allocator_Error,
	CSV_Read_Error,
	CSV_Format_Conversion_Error,
	CSV_Empty_Row,
	CSV_Inconsistent_Column_Count,
	Image_Load_Error,
	NPY_Open_Error,
	NPY_Reader_Creation_Error,
	NPY_Reader_Read_Byte_Error,
	NPY_Invalid_Header_Error,
	NPY_Invalid_Version_Error,
	NPY_Invalid_Header_Length,
	NPY_Invalid_Descriptor,
	NPY_Malformed_Header,
	NPY_Shape_Parse_Failed,
	NPY_Read_Array_Error,
	NPY_Not_Implemented,
	Image_Cannot_Read_File,
	Image_Invalid_Format,
	Image_Unsupported_Output_Extension,
	Image_Write_Failed,
}
