package file_io

import "../tensor"
import "core:encoding/csv"
import "core:fmt"
import "core:slice"
import "core:strconv"

// Parses a CSV string into a 2D tensor of type T.
// Assumes all rows have the same number of columns as the first row.
// This uses temp_allocator for CSV parsing, specified allocator for output tensor.
read_csv :: proc(
	$T: typeid,
	csv_string: string,
	n_skip_first_rows: uint = 0,
	n_skip_first_cols: uint = 0,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	^tensor.Tensor(T),
	IO_Error,
) {
	records, _ := csv.read_all_from_string(
		csv_string,
		context.temp_allocator,
		context.temp_allocator,
	)

	n_row := uint(len(records))
	if n_row == 0 do return nil, CSV_Empty_Row{}
	n_col := uint(len(records[0])) // Assume correct number of cols is based on the 1st row
	out := tensor.tensor_alloc(
		T,
		{n_row - n_skip_first_rows, n_col - n_skip_first_cols},
		true,
		allocator = allocator,
		loc = loc,
	)

	offset := 0
	for row, i in records {
		if i < int(n_skip_first_rows) do continue

		if len(row) != int(n_col) {
			// Inconsistent!
			return nil, CSV_Inconsistent_Column_Count{}
		}
		for col, j in row {
			if j < int(n_skip_first_cols) do continue

			val: T
			ok: bool

			when T == f32 {
				val, ok = strconv.parse_f32(col)
				if !ok do return nil, CSV_Format_Conversion_Error{}
			} else {
				fmt.panicf("unsupported target data type %v", typeid_of(T))
			}

			out.data[offset] = val
			offset += 1
		}
	}

	return out, nil
}

import "core:testing"

@(test)
read_csv_test :: proc(t: ^testing.T) {
	context.allocator = context.temp_allocator

	res, err := read_csv(f32, "1,2,3\n4,5,6")
	testing.expect(t, err == nil, fmt.tprint(err))
	testing.expect(t, slice.equal(res.shape, []uint{2, 3}))
	testing.expect(t, slice.equal(res.data, []f32{1, 2, 3, 4, 5, 6}))

	res, err = read_csv(f32, "1,2,3\n4,6")
	testing.expect(t, res == nil, fmt.tprint(err))
	testing.expect(t, err != nil, fmt.tprint(err))
}

@(test)
read_csv_skip_test :: proc(t: ^testing.T) {
	context.allocator = context.temp_allocator

	res, err := read_csv(f32, "1,2,3\n4,5,6", n_skip_first_rows = 1)
	testing.expect(t, err == nil, fmt.tprint(err))
	testing.expect(t, slice.equal(res.shape, []uint{1, 3}))
	testing.expect(t, slice.equal(res.data, []f32{4, 5, 6}))

	res, err = read_csv(f32, "1,2,3\n4,5,6", n_skip_first_cols = 1)
	testing.expect(t, err == nil, fmt.tprint(err))
	testing.expect(t, slice.equal(res.shape, []uint{2, 2}))
	testing.expect(t, slice.equal(res.data, []f32{2, 3, 5, 6}))
}
