package flux

import "../../tensor"
import "core:math"
import "core:testing"

@(test)
test_silu_zero :: proc(t: ^testing.T) {
	// silu(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
	result := silu(f32(0))
	testing.expect(t, math.abs(result) < 1e-6, "silu(0) should be 0")
}

@(test)
test_silu_large_positive :: proc(t: ^testing.T) {
	// For large x, silu(x) ≈ x (sigmoid approaches 1)
	result := silu(f32(10))
	testing.expect(t, math.abs(result - 10.0) < 0.1, "silu(10) should be ~10")
}

@(test)
test_silu_large_negative :: proc(t: ^testing.T) {
	// For large negative x, silu(x) ≈ 0 (sigmoid approaches 0)
	result := silu(f32(-10))
	testing.expect(t, math.abs(result) < 0.01, "silu(-10) should be ~0")
}

@(test)
test_silu_one :: proc(t: ^testing.T) {
	// silu(1) = 1 / (1 + exp(-1)) ≈ 0.731
	result := silu(f32(1))
	expected := f32(1.0) / (1.0 + math.exp(f32(-1.0)))
	testing.expect(t, math.abs(result - expected) < 1e-5, "silu(1) mismatch")
}

@(test)
test_rms_norm_shape :: proc(t: ^testing.T) {
	x := tensor.randn(f32, []uint{2, 4}, 0, 1, context.temp_allocator)
	defer tensor.free_tensor(x, context.temp_allocator)

	weight := tensor.ones(f32, []uint{4}, context.temp_allocator)
	defer tensor.free_tensor(weight, context.temp_allocator)

	result := rms_norm(x, weight, 1e-6, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	testing.expect(t, result.shape[0] == 2)
	testing.expect(t, result.shape[1] == 4)
}

@(test)
test_rms_norm_unit_rms :: proc(t: ^testing.T) {
	// After RMSNorm with weight=1, each row should have RMS ≈ 1
	x := tensor.new_with_init([]f32{1, 2, 3, 4}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(x, context.temp_allocator)

	weight := tensor.ones(f32, []uint{4}, context.temp_allocator)
	defer tensor.free_tensor(weight, context.temp_allocator)

	result := rms_norm(x, weight, 1e-6, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	// Compute RMS of result
	rms: f32 = 0
	for v in result.data {
		rms += v * v
	}
	rms = math.sqrt(rms / 4)

	testing.expect(t, math.abs(rms - 1.0) < 0.1, "RMS should be ~1 after normalization")
}

@(test)
test_rms_norm_with_weight :: proc(t: ^testing.T) {
	x := tensor.new_with_init([]f32{1, 1, 1, 1}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(x, context.temp_allocator)

	// Weight of 2 should scale output by 2
	weight := tensor.new_with_init([]f32{2, 2, 2, 2}, []uint{4}, context.temp_allocator)
	defer tensor.free_tensor(weight, context.temp_allocator)

	result := rms_norm(x, weight, 1e-6, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	// All values equal 1, RMS = 1, normalized = 1, scaled by 2 = 2
	for v in result.data {
		testing.expect(t, math.abs(v - 2.0) < 1e-5, "Expected scaled output")
	}
}

@(test)
test_rms_norm_batch :: proc(t: ^testing.T) {
	// Test with batch dimension
	x := tensor.new_with_init(
		[]f32{1, 2, 3, 4, 5, 6, 7, 8},
		[]uint{2, 4},
		context.temp_allocator,
	)
	defer tensor.free_tensor(x, context.temp_allocator)

	weight := tensor.ones(f32, []uint{4}, context.temp_allocator)
	defer tensor.free_tensor(weight, context.temp_allocator)

	result := rms_norm(x, weight, 1e-6, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	// Each row should be independently normalized
	// Row 0: [1,2,3,4], Row 1: [5,6,7,8]
	// Check both rows have RMS ≈ 1
	for row in 0 ..< 2 {
		rms: f32 = 0
		for col in 0 ..< 4 {
			v := result.data[row * 4 + col]
			rms += v * v
		}
		rms = math.sqrt(rms / 4)
		testing.expect(t, math.abs(rms - 1.0) < 0.1, "Each row RMS should be ~1")
	}
}
