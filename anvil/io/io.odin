// package io is a collection of utilities to aid with basic file reading and its conversion to
// anvil tensor.
package file_io

import "core:encoding/csv"
import "core:image"

CSV_Read_Error :: csv.Error
CSV_Format_Conversion_Error :: struct {}
CSV_Empty_Row :: struct {}
CSV_Inconsistent_Column_Count :: struct {}

Image_Load_Error :: image.Error

IO_Error :: union {
	CSV_Read_Error,
	CSV_Format_Conversion_Error,
	CSV_Empty_Row,
	CSV_Inconsistent_Column_Count,
	Image_Load_Error,
}
