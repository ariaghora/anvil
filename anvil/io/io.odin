package file_io

import "core:encoding/csv"
import "core:image"

CSV_Read_Error :: csv.Error
CSV_Format_Conversion_Error :: struct {}
CSV_Empty_Row :: struct {}
CSV_Inconsistent_Column_Count :: struct {}

Image_Load_Error :: image.Error
Image_File_Not_Found :: struct {
	path: string,
}
Image_Format_Not_Supported :: struct {}

IO_Error :: union {
	CSV_Read_Error,
	CSV_Format_Conversion_Error,
	CSV_Empty_Row,
	CSV_Inconsistent_Column_Count,
	Image_Load_Error,
	Image_File_Not_Found,
	Image_Format_Not_Supported,
}
