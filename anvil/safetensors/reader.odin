package safetensors

import "../tensor"
import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

Assignment_Incompatible_Shape :: struct {
	source_shape, target_shape: []uint,
}
Tensor_Not_Found :: struct {
	key: string,
}
Tensors_Names_Length_Mismatch :: struct {}
IO_Error :: struct {
	msg: string,
}

Safe_Tensors_Error :: union {
	json.Marshal_Error,
	json.Unmarshal_Error,
	Assignment_Incompatible_Shape,
	IO_Error,
	Tensor_Not_Found,
	Tensors_Names_Length_Mismatch,
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
	return read_from_bytes(T, raw_bytes, allocator, loc)
}

read_from_bytes :: proc(
	$T: typeid,
	raw_bytes: []u8,
	allocator := context.allocator,
	loc := #caller_location,
) -> (
	res: ^Safe_Tensors(T),
	er: Safe_Tensors_Error,
) {
	header_size := uint(mem.slice_data_cast([]u64, raw_bytes[:8])[0])
	header_content := transmute(string)raw_bytes[8:8 + header_size]
	tensors_proto: map[string]Tensor_Info
	json.unmarshal_string(
		header_content,
		&tensors_proto,
		allocator = context.temp_allocator,
	) or_return

	delete_key(&tensors_proto, "__metadata__")

	tensor_data_start := header_size + 8
	res = new_clone(Safe_Tensors(T){raw_bytes = raw_bytes}, allocator, loc)

	for k, v in tensors_proto {
		raw_data := raw_bytes[tensor_data_start +
		v.data_offsets[0]:tensor_data_start +
		v.data_offsets[1]]

		t := tensor.tensor_alloc(T, v.shape, owns_data = true, allocator = allocator, loc = loc)

		// Convert based on source dtype
		switch v.dtype {
		case "U8", "u8", "uint8":
			src_data := mem.slice_data_cast([]u8, raw_data)
			for i in 0 ..< len(src_data) {
				t.data[i] = T(src_data[i])
			}
		case "F16", "f16", "float16":
			src_data := mem.slice_data_cast([]f16, raw_data)
			for i in 0 ..< len(src_data) {
				t.data[i] = T(src_data[i])
			}
		case "F32", "f32", "float32":
			src_data := mem.slice_data_cast([]f32, raw_data)
			for i in 0 ..< len(src_data) {
				t.data[i] = T(src_data[i])
			}
		case "F64", "f64", "float64":
			src_data := mem.slice_data_cast([]f64, raw_data)
			for i in 0 ..< len(src_data) {
				t.data[i] = T(src_data[i])
			}
		case "I32", "i32", "int32":
			src_data := mem.slice_data_cast([]i32, raw_data)
			for i in 0 ..< len(src_data) {
				t.data[i] = T(src_data[i])
			}
		case:
			panic(fmt.tprintf("Unsupported dtype: %s", v.dtype))
		}

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

tensor_assign_from_safe_tensors :: proc {
	tensor_assign_from_safe_tensors_one,
	tensor_assign_from_safe_tensors_many,
}

tensor_assign_from_safe_tensors_one :: proc(
	t: ^tensor.Tensor($T),
	tensor_name: string,
	safe_tensors: ^Safe_Tensors(T),
	should_transpose := false,
	loc := #caller_location,
) -> Safe_Tensors_Error {
	t_from, ok := safe_tensors.tensors[tensor_name]
	if !ok do return Tensor_Not_Found{tensor_name}

	if should_transpose do t_from = tensor.transpose(t_from, 0, 1)

	if !slice.equal(t.shape, t_from.shape) {
		fmt.println(loc)
		return Assignment_Incompatible_Shape{source_shape = t_from.shape, target_shape = t.shape}
	}
	if !t.contiguous do return Tensors_Names_Length_Mismatch{}


	if t.owns_data {
		copy(t.data, t_from.data)
	} else {
		if should_transpose {
			copy(t.data, t_from.data)
			tensor.free_tensor(t_from)
		} else {
			t.data = t_from.data
		}
	}
	return nil
}

tensor_assign_from_safe_tensors_many :: proc(
	tensors: []^tensor.Tensor($T),
	tensor_names: []string,
	safe_tensors: ^Safe_Tensors(T),
	should_transpose := false,
) -> Safe_Tensors_Error {
	if len(tensors) != len(tensor_names) do return Tensors_Names_Length_Mismatch{}
	for i in 0 ..< len(tensors) {
		tensor_assign_from_safe_tensors(tensors[i], tensor_names[i], safe_tensors) or_return
	}
	return nil
}
