package file_io

import "core:encoding/csv"

CSV_Read_Error :: csv.Error
CSV_Format_Conversion_Error :: struct {}
CSV_Empty_Row :: struct {}
CSV_Inconsistent_Column_Count :: struct {}

IO_Error :: union {
	CSV_Read_Error,
	CSV_Format_Conversion_Error,
	CSV_Empty_Row,
	CSV_Inconsistent_Column_Count,
}
