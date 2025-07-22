package safetensors

import "../tensor"
import "core:testing"

@(test)
test_tensor_max_all :: proc(t: ^testing.T) {
	safe_tensors, ok := read_from_file(f32, "models/mnist.safetensors")
	defer free_safe_tensors(safe_tensors)

	testing.expect(t, safe_tensors != nil)

	// 3 layers with 2 tensors each = 6 tensors
	testing.expect(t, len(safe_tensors.tensors) == 6)

	i_w := tensor.tensor_alloc(f32, []uint{256, 784})
	defer tensor.free_tensor(i_w)

	ok = tensor_assign_from_safe_tensors(i_w, "input_layer.weight", safe_tensors)
	testing.expect(t, ok == nil)

	i_b := tensor.tensor_alloc(f32, []uint{666})
	defer tensor.free_tensor(i_b)

	ok = tensor_assign_from_safe_tensors(i_b, "input_layer.bias", safe_tensors)
	#partial switch v in ok {
	case Assignment_Incompatible_Shape:
		break
	case:
		panic("Should be incompatible")
	}
}
