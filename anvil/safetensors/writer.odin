package safetensors

import "../tensor"
import "core:encoding/json"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"

write_tensors_to_file :: proc(st: ^Safe_Tensors($T), file_name: string) -> Safe_Tensors_Error {
	// We collect and sort names for convenience and determinism, just in case.
	// Though it should be no issues for safetensors itself.
	// TODO(aria): verify, maybe
	names := make([dynamic]string, context.temp_allocator)
	for name in st.tensors {
		append(&names, name)
	}
	slice.sort(names[:])

	header_map := make(map[string]Tensor_Info, allocator = context.temp_allocator)
	defer delete(header_map)

	current_offset: uint = 0
	for name in names {
		t := st.tensors[name]
		if !t.contiguous {
			return IO_Error{msg = fmt.tprintf("Tensor '%s' must be contiguous", name)}
		}

		data_size := uint(len(t.data) * size_of(T))
		header_map[name] = Tensor_Info {
			dtype        = get_dtype_string(T),
			shape        = t.shape,
			data_offsets = {current_offset, current_offset + data_size},
		}
		current_offset += data_size
	}

	// Marshal header to JSON
	header_bytes, err := json.marshal(header_map, allocator = context.temp_allocator)
	if err != nil {
		return json.Marshal_Error{}
	}

	file, open_err := os.open(file_name, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if open_err != 0 {
		return IO_Error{msg = fmt.tprintf("Failed to open file: %s", file_name)}
	}
	defer os.close(file)

	// Write header size. Assume 8 bytes, little-endian. Huggingface's rust
	// implementation should do the same. Most systems do too, except the broken ones.
	header_size := u64(len(header_bytes))
	header_size_bytes := transmute([8]u8)header_size
	os.write(file, header_size_bytes[:])
	os.write(file, header_bytes)

	for name in names {
		t := st.tensors[name]
		tensor_bytes := mem.slice_data_cast([]u8, t.data)
		os.write(file, tensor_bytes)
	}

	return nil
}

get_dtype_string :: proc($T: typeid) -> string {
	when T == f32 {
		return "F32"
	} else when T == f64 {
		return "F64"
	} else when T == i32 {
		return "I32"
	} else when T == i64 {
		return "I64"
	} else {
		#panic("Unsupported dtype for safetensors")
	}
}
