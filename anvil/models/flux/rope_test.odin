package flux

import "../../tensor"
import "core:math"
import "core:testing"

@(test)
test_rope_freqs_shape :: proc(t: ^testing.T) {
	freqs := compute_rope_freqs(f32, 512, 128, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	testing.expect(t, freqs.shape[0] == 512, "Expected seq dim = 512")
	testing.expect(t, freqs.shape[1] == 128, "Expected dim = 128")
}

@(test)
test_rope_freqs_values :: proc(t: ^testing.T) {
	freqs := compute_rope_freqs(f32, 4, 8, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	// Position 0 should have cos(0)=1, sin(0)=0 for all frequencies
	for i in 0 ..< 4 {
		cos_val := freqs.data[2 * i]
		sin_val := freqs.data[2 * i + 1]
		testing.expect(t, math.abs(cos_val - 1.0) < 1e-5, "cos(0) should be 1")
		testing.expect(t, math.abs(sin_val) < 1e-5, "sin(0) should be 0")
	}
}

@(test)
test_rope_freqs_2d_shape :: proc(t: ^testing.T) {
	freqs := compute_rope_freqs_2d(f32, 16, 16, 128, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	testing.expect(t, freqs.shape[0] == 256, "Expected seq = 16*16 = 256")
	testing.expect(t, freqs.shape[1] == 128, "Expected dim = 128")
}

@(test)
test_rope_apply_changes_values :: proc(t: ^testing.T) {
	// Create Q tensor [batch=1, seq=4, heads*dim=2*8=16]
	q := tensor.randn(f32, []uint{1, 4, 16}, 0, 1, context.temp_allocator)
	defer tensor.free_tensor(q, context.temp_allocator)

	freqs := compute_rope_freqs(f32, 4, 8, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	// Clone original
	q_before := tensor.clone(q, context.temp_allocator)
	defer tensor.free_tensor(q_before, context.temp_allocator)

	apply_rope_inplace(q, freqs, 2, 8)

	// Values should change (except possibly position 0 where rotation is identity)
	different := false
	// Check position 1+ where rotation should differ
	for i in 16 ..< len(q.data) {
		if q.data[i] != q_before.data[i] {
			different = true
			break
		}
	}
	testing.expect(t, different, "RoPE should change tensor values")
}

@(test)
test_rope_apply_position_zero_identity :: proc(t: ^testing.T) {
	// At position 0, angle=0, so rotation is identity: cos(0)=1, sin(0)=0
	// [1, 0; 0, 1] @ [x0, x1] = [x0, x1]
	q := tensor.new_with_init(
		[]f32{1, 2, 3, 4, 5, 6, 7, 8},  // 1 head, 8 dim
		[]uint{1, 1, 8},
		context.temp_allocator,
	)
	defer tensor.free_tensor(q, context.temp_allocator)

	freqs := compute_rope_freqs(f32, 1, 8, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	apply_rope_inplace(q, freqs, 1, 8)

	// Position 0 should be unchanged
	testing.expect(t, math.abs(q.data[0] - 1.0) < 1e-5)
	testing.expect(t, math.abs(q.data[1] - 2.0) < 1e-5)
}

@(test)
test_rope_apply_non_inplace :: proc(t: ^testing.T) {
	q := tensor.randn(f32, []uint{1, 4, 16}, 0, 1, context.temp_allocator)
	defer tensor.free_tensor(q, context.temp_allocator)

	freqs := compute_rope_freqs(f32, 4, 8, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	original_first := q.data[0]

	result := apply_rope(q, freqs, 2, 8, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	// Original should be unchanged
	testing.expect(t, q.data[0] == original_first, "Original tensor should not change")
	// Result should be different pointer
	testing.expect(t, result != q, "Should return new tensor")
}

@(test)
test_rope_freqs_joint_shape :: proc(t: ^testing.T) {
	// 4x4 image + 8 text tokens
	freqs := compute_rope_freqs_joint(f32, 4, 4, 8, 64, 10000.0, context.temp_allocator)
	defer tensor.free_tensor(freqs, context.temp_allocator)

	testing.expect(t, freqs.shape[0] == 24, "Expected 16 img + 8 txt = 24")
	testing.expect(t, freqs.shape[1] == 64, "Expected dim = 64")
}
