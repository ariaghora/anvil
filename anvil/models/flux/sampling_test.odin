package flux

import "../../tensor"
import "core:math"
import "core:testing"

@(test)
test_flux_schedule_length :: proc(t: ^testing.T) {
	schedule := flux_schedule(4, 256, context.temp_allocator)
	defer delete(schedule, context.temp_allocator)

	// num_steps + 1 entries (including start and end)
	testing.expect(t, len(schedule) == 5, "Expected 5 entries for 4 steps")
}

@(test)
test_flux_schedule_range :: proc(t: ^testing.T) {
	schedule := flux_schedule(4, 256, context.temp_allocator)
	defer delete(schedule, context.temp_allocator)

	// Should start at 0 and end at ~1 (shifted)
	testing.expect(t, schedule[0] >= 0, "Schedule should start >= 0")
	testing.expect(t, schedule[len(schedule) - 1] <= 1.5, "Schedule should end <= 1.5")
}

@(test)
test_flux_schedule_monotonic :: proc(t: ^testing.T) {
	schedule := flux_schedule(10, 256, context.temp_allocator)
	defer delete(schedule, context.temp_allocator)

	// Should be monotonically increasing
	for i in 1 ..< len(schedule) {
		testing.expect(t, schedule[i] >= schedule[i - 1], "Schedule should be monotonic")
	}
}

@(test)
test_flux_schedule_shift_effect :: proc(t: ^testing.T) {
	// Larger image should have larger shift
	schedule_small := flux_schedule(4, 64, context.temp_allocator)
	defer delete(schedule_small, context.temp_allocator)

	schedule_large := flux_schedule(4, 1024, context.temp_allocator)
	defer delete(schedule_large, context.temp_allocator)

	// With larger shift, middle values should be higher
	testing.expect(t, schedule_large[2] >= schedule_small[2], "Larger image should have larger shifted values")
}

@(test)
test_init_noise_shape :: proc(t: ^testing.T) {
	noise := init_noise(1, 32, 16, 16, 42, context.temp_allocator)
	defer tensor.free_tensor(noise, context.temp_allocator)

	testing.expect(t, noise.shape[0] == 1, "Batch dim")
	testing.expect(t, noise.shape[1] == 32, "Channel dim")
	testing.expect(t, noise.shape[2] == 16, "Height dim")
	testing.expect(t, noise.shape[3] == 16, "Width dim")
}

@(test)
test_init_noise_deterministic :: proc(t: ^testing.T) {
	noise1 := init_noise(1, 4, 4, 4, 42, context.temp_allocator)
	defer tensor.free_tensor(noise1, context.temp_allocator)

	noise2 := init_noise(1, 4, 4, 4, 42, context.temp_allocator)
	defer tensor.free_tensor(noise2, context.temp_allocator)

	// Same seed should give same noise
	testing.expect(t, noise1.data[0] == noise2.data[0], "Same seed should give same first value")
	testing.expect(t, noise1.data[10] == noise2.data[10], "Same seed should give same 10th value")
}

@(test)
test_init_noise_different_seeds :: proc(t: ^testing.T) {
	noise1 := init_noise(1, 4, 4, 4, 42, context.temp_allocator)
	defer tensor.free_tensor(noise1, context.temp_allocator)

	noise2 := init_noise(1, 4, 4, 4, 123, context.temp_allocator)
	defer tensor.free_tensor(noise2, context.temp_allocator)

	// Different seeds should give different noise
	different := false
	for i in 0 ..< len(noise1.data) {
		if noise1.data[i] != noise2.data[i] {
			different = true
			break
		}
	}
	testing.expect(t, different, "Different seeds should give different noise")
}

@(test)
test_init_noise_distribution :: proc(t: ^testing.T) {
	// Large sample to test distribution
	noise := init_noise(1, 64, 32, 32, 42, context.temp_allocator)
	defer tensor.free_tensor(noise, context.temp_allocator)

	// Compute mean and std
	mean: f32 = 0
	for v in noise.data {
		mean += v
	}
	mean /= f32(len(noise.data))

	variance: f32 = 0
	for v in noise.data {
		variance += (v - mean) * (v - mean)
	}
	variance /= f32(len(noise.data))
	std := math.sqrt(variance)

	// Should be approximately standard normal (mean ~0, std ~1)
	testing.expect(t, math.abs(mean) < 0.1, "Mean should be ~0")
	testing.expect(t, math.abs(std - 1.0) < 0.1, "Std should be ~1")
}

@(test)
test_apply_cfg :: proc(t: ^testing.T) {
	v_cond := tensor.new_with_init([]f32{1, 2, 3, 4}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(v_cond, context.temp_allocator)

	v_uncond := tensor.new_with_init([]f32{0, 0, 0, 0}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(v_uncond, context.temp_allocator)

	// guidance_scale = 1 means no change from v_cond
	result := apply_cfg(v_cond, v_uncond, 1.0, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	testing.expect(t, result.data[0] == 1.0, "CFG scale 1 should equal v_cond")
}

@(test)
test_apply_cfg_scale_2 :: proc(t: ^testing.T) {
	v_cond := tensor.new_with_init([]f32{2, 2, 2, 2}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(v_cond, context.temp_allocator)

	v_uncond := tensor.new_with_init([]f32{1, 1, 1, 1}, []uint{1, 4}, context.temp_allocator)
	defer tensor.free_tensor(v_uncond, context.temp_allocator)

	// v_guided = v_uncond + 2 * (v_cond - v_uncond) = 1 + 2*(2-1) = 3
	result := apply_cfg(v_cond, v_uncond, 2.0, context.temp_allocator)
	defer tensor.free_tensor(result, context.temp_allocator)

	testing.expect(t, math.abs(result.data[0] - 3.0) < 1e-5, "CFG scale 2 calculation")
}
