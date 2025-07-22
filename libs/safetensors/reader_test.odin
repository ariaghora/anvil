package safetensors

import "../tensor"
import "core:log"
import "core:slice"
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

@(test)
test_tensor_ff :: proc(t: ^testing.T) {
	safe_tensors, ok := read_from_file(f32, "models/mnist.safetensors")
	defer free_safe_tensors(safe_tensors)

	i_w := tensor.tensor_alloc(f32, []uint{256, 784})
	defer tensor.free_tensor(i_w)
	ok = tensor_assign_from_safe_tensors(i_w, "input_layer.weight", safe_tensors)
	testing.expect(t, ok == nil)

	i_wt := tensor.transpose(i_w, 0, 1)
	defer tensor.free_tensor(i_wt)

	i_b := tensor.tensor_alloc(f32, []uint{256})
	defer tensor.free_tensor(i_b)
	ok = tensor_assign_from_safe_tensors(i_b, "input_layer.bias", safe_tensors)
	testing.expect(t, ok == nil)

	m_w := tensor.tensor_alloc(f32, []uint{256, 256})
	defer tensor.free_tensor(m_w)
	ok = tensor_assign_from_safe_tensors(m_w, "mid_layer.weight", safe_tensors)
	testing.expect(t, ok == nil)

	m_wt := tensor.transpose(m_w, 0, 1)
	defer tensor.free_tensor(m_wt)

	m_b := tensor.tensor_alloc(f32, []uint{256})
	defer tensor.free_tensor(m_b)
	ok = tensor_assign_from_safe_tensors(m_b, "mid_layer.bias", safe_tensors)
	testing.expect(t, ok == nil)

	o_w := tensor.tensor_alloc(f32, []uint{10, 256})
	defer tensor.free_tensor(o_w)
	ok = tensor_assign_from_safe_tensors(o_w, "output_layer.weight", safe_tensors)
	testing.expect(t, ok == nil)

	o_wt := tensor.transpose(o_w, 0, 1)
	defer tensor.free_tensor(o_wt)

	o_b := tensor.tensor_alloc(f32, []uint{10})
	defer tensor.free_tensor(o_b)
	ok = tensor_assign_from_safe_tensors(o_b, "output_layer.bias", safe_tensors)
	testing.expect(t, ok == nil)

	// Digit `1`
	input := make([]f32, 28 * 28, context.temp_allocator)
	for row in 0 ..< 28 {
		for col in 0 ..< 28 {
			if row > 3 && row < 25 && col > 12 && col < 16 {
				input[row * 28 + col] = 1.0
			}
		}
	}
	input_tensor := tensor.new_with_init(input, []uint{1, 28 * 28})
	defer tensor.free_tensor(input_tensor)

	h1 := tensor.tensor_relu(
		tensor.tensor_add(
			tensor.matmul(input_tensor, i_wt, context.temp_allocator),
			i_b,
			context.temp_allocator,
		),
		context.temp_allocator,
	)
	h2 := tensor.tensor_relu(
		tensor.tensor_add(
			tensor.matmul(h1, m_wt, context.temp_allocator),
			m_b,
			context.temp_allocator,
		),
		context.temp_allocator,
	)
	out := tensor.tensor_relu(
		tensor.tensor_add(
			tensor.matmul(h2, o_wt, context.temp_allocator),
			o_b,
			context.temp_allocator,
		),
		context.temp_allocator,
	)
	class, ok_maxidx := slice.max_index(out.data)
	testing.expect(t, ok_maxidx)
	testing.expect(t, class == 1)
}
