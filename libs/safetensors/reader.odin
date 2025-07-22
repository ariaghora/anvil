package safetensors

import "../tensor"
import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

Assignment_Incompatible_Shape :: struct {}
Tensor_Not_Found :: struct {}
Non_Contiguous_Tensor :: struct {}
IO_Error :: struct {
	msg: string,
}

Safe_Tensors_Error :: union {
	json.Unmarshal_Error,
	Assignment_Incompatible_Shape,
	IO_Error,
	Tensor_Not_Found,
	Non_Contiguous_Tensor,
}

Tensor_Info :: struct {
	dtype:        string,
	shape:        []uint,
	data_offsets: [2]uint,
}

// Each tensor's data in tensors will point to some offset `raw_bytes` with a
// certain length. It means that the tensors in Safe_Tensors will not own any data.
Safe_Tensors :: struct($T: typeid) {
	raw_bytes: []u8,
	tensors:   map[string]^tensor.Tensor(T),
}

read_from_file :: proc(
	$T: typeid,
	fn: string,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Safe_Tensors(T),
	er: Safe_Tensors_Error,
) {
	raw_bytes, ok := os.read_entire_file(fn, allocator)
	if !ok do return nil, IO_Error{msg = fmt.tprintf("Failed to read safe tensor from %s", fn)}

	header_size := uint(mem.slice_data_cast([]u64, raw_bytes[:8])[0])
	header_content := transmute(string)raw_bytes[8:8 + header_size]
	tensors_proto: map[string]Tensor_Info
	json.unmarshal_string(
		header_content,
		&tensors_proto,
		allocator = context.temp_allocator,
	) or_return

	// Exclude __metadata__ entry because who cares (for now)
	delete_key(&tensors_proto, "__metadata__")

	tensor_data_start := header_size + 8
	res = new_clone(Safe_Tensors(T){raw_bytes = raw_bytes}, allocator, loc)

	for k, v in tensors_proto {
		data := mem.slice_data_cast(
			[]T,
			raw_bytes[tensor_data_start + v.data_offsets[0]:tensor_data_start + v.data_offsets[1]],
		)
		t := tensor.tensor_alloc(T, v.shape, owns_data = false, allocator = allocator, loc = loc)
		t.data = data
		res.tensors[k] = t
	}

	return res, nil
}

free_safe_tensors :: proc(
	st: ^Safe_Tensors($T),
	allocator := context.allocator,
	loc := #caller_location,
) {
	for k, v in st.tensors {
		tensor.free_tensor(v, allocator)
	}
	delete(st.raw_bytes, allocator)
	delete_map(st.tensors)
	free(st, allocator)
}

tensor_assign_from_safe_tensors :: proc(
	t: ^tensor.Tensor($T),
	tensor_name: string,
	safe_tensors: ^Safe_Tensors(T),
) -> Safe_Tensors_Error {
	t_from, ok := safe_tensors.tensors[tensor_name]
	if !ok do return Tensor_Not_Found{}
	if !slice.equal(t.shape, t_from.shape) do return Assignment_Incompatible_Shape{}
	if !t.contiguous do return Non_Contiguous_Tensor{}

	if t.owns_data {
		copy(t.data, t_from.data)
	} else {
		t.data = t_from.data
	}
	return nil
}
