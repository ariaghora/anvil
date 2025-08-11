package transformer

import st "../safetensors"
import "../tensor"
import "core:fmt"
import "core:strings"

Var_Builder :: struct($T: typeid) {
	name:        string,
	safetensors: ^st.Safe_Tensors(T),
	parent:      ^Var_Builder(T),
}

vb_make :: proc($T: typeid, name: string, parent: ^Var_Builder(T)) -> Var_Builder(T) {
	return Var_Builder(T){name = name, parent = parent, safetensors = parent.safetensors}
}

@(private = "file")
vb_resolve_preceding_path :: proc(vb: ^Var_Builder($T), allocator := context.allocator) -> string {
	prec := strings.concatenate(
		{
			vb.parent == nil ? "" : fmt.tprintf("%s.", vb_resolve_preceding_path(vb.parent, allocator)),
			vb.name,
		},
	)

	return prec
}

vb_assignt_to_tensor :: proc(
	vb: ^Var_Builder($T),
	leaf_name: string,
	target: ^tensor.Tensor(T),
	should_transpose := false,
) {
	path := strings.concatenate(
		{vb_resolve_preceding_path(vb, context.temp_allocator), ".", leaf_name},
	)
	err := st.tensor_assign_from_safe_tensors(target, path, vb.safetensors, should_transpose)
	if err != nil {
		fmt.panicf("Error assigning %s to target tensor: %v", path, err)
	}
}
