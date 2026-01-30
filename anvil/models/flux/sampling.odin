// FLUX Sampling Implementation
//
// Euler sampler for rectified flow ODEs.
// FLUX uses a simple Euler discretization with a shifted noise schedule.

package flux

import "../../tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

// Compute FLUX noise schedule
// Based on flux2.c: shifted sigmoid schedule
flux_schedule :: proc(
	num_steps: int,
	image_seq_len: int,
	allocator := context.allocator,
) -> []f32 {
	schedule := make([]f32, num_steps + 1, allocator)

	// FLUX uses a shifted schedule based on image resolution
	// shift = 1.0 + (image_seq_len / 256) * 0.5
	shift := f32(1.0) + f32(image_seq_len) / f32(256) * f32(0.5)

	for i in 0 ..= num_steps {
		t := f32(i) / f32(num_steps)
		// Shifted sigmoid: sigma(t) = 1 / (1 + exp(-shift * (t - 0.5)))
		// But for rectified flow, we use linear interpolation with shift
		schedule[i] = shift_time(t, shift)
	}

	return schedule
}

// Apply shift to timestep
@(private = "file")
shift_time :: proc(t: f32, shift: f32) -> f32 {
	// shift(t) = (shift * t) / (1 + (shift - 1) * t)
	return (shift * t) / (f32(1.0) + (shift - f32(1.0)) * t)
}

// Initialize random noise tensor
init_noise :: proc(
	batch, channels, height, width: uint,
	seed: i64,
	allocator := context.allocator,
) -> ^tensor.Tensor(f32) {
	noise := tensor.tensor_alloc(f32, []uint{batch, channels, height, width}, true, allocator)

	// Seed RNG if provided
	if seed >= 0 {
		rand.reset(u64(seed))
	}

	// Generate Gaussian noise using Box-Muller transform
	i := 0
	for i < len(noise.data) {
		u1 := f32(rand.float64())
		u2 := f32(rand.float64())

		// Ensure u1 > 0 to avoid log(0)
		for u1 <= 0 {
			u1 = f32(rand.float64())
		}

		// Box-Muller transform
		mag := math.sqrt(f32(-2.0) * math.ln(u1))
		z0 := mag * math.cos(f32(2.0) * math.PI * u2)
		z1 := mag * math.sin(f32(2.0) * math.PI * u2)

		noise.data[i] = z0
		i += 1
		if i < len(noise.data) {
			noise.data[i] = z1
			i += 1
		}
	}

	return noise
}

// Euler sampler for rectified flow
// z_t = (1-t) * z_0 + t * x  (where x is data, z_0 is noise)
// v = x - z_0 (velocity field)
// ODE: dz/dt = v
euler_sample :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T), // Initial noise
	txt_emb: ^tensor.Tensor(T), // Text embeddings
	schedule: []T, // Timestep schedule
	num_steps: int,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Clone initial noise as working tensor
	z_t := tensor.clone(z, allocator)

	for step in 0 ..< num_steps {
		// Current and next timesteps
		t := schedule[step]
		t_next := schedule[step + 1]
		dt := t_next - t

		// Predict velocity at current timestep
		v := transformer_forward(tf, z_t, txt_emb, t, allocator)
		defer tensor.free_tensor(v, allocator)

		// Euler step: z_{t+dt} = z_t + dt * v
		for i in 0 ..< len(z_t.data) {
			z_t.data[i] += dt * v.data[i]
		}
	}

	return z_t
}

// Euler sampler with reference image (for img2img)
// Mixes reference latent with noise based on t_offset
euler_sample_with_ref :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T), // Initial noise
	txt_emb: ^tensor.Tensor(T), // Text embeddings
	ref_latent: ^tensor.Tensor(T), // Reference image latent
	schedule: []T,
	num_steps: int,
	t_offset: int, // Skip first t_offset steps (strength control)
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Mix noise with reference based on t_offset
	// At t_offset, we start from a mix of ref and noise
	t_start := schedule[t_offset]

	z_t := tensor.tensor_alloc(T, z.shape, true, allocator)

	// z_t = (1-t_start) * noise + t_start * ref
	for i in 0 ..< len(z_t.data) {
		z_t.data[i] = (T(1.0) - t_start) * z.data[i] + t_start * ref_latent.data[i]
	}

	// Continue sampling from t_offset
	for step in t_offset ..< num_steps {
		t := schedule[step]
		t_next := schedule[step + 1]
		dt := t_next - t

		v := transformer_forward(tf, z_t, txt_emb, t, allocator)
		defer tensor.free_tensor(v, allocator)

		for i in 0 ..< len(z_t.data) {
			z_t.data[i] += dt * v.data[i]
		}
	}

	return z_t
}

// Multi-reference sampling (for style transfer with multiple refs)
euler_sample_multi_ref :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T),
	txt_emb: ^tensor.Tensor(T),
	ref_latents: []^tensor.Tensor(T),
	ref_weights: []T, // Weights for each reference
	schedule: []T,
	num_steps: int,
	t_offset: int,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Weighted sum of reference latents
	ref_combined := tensor.tensor_alloc(T, z.shape, true, allocator)
	defer tensor.free_tensor(ref_combined, allocator)

	total_weight: T = 0
	for w in ref_weights {
		total_weight += w
	}

	for i, ref in ref_latents {
		w := ref_weights[i] / total_weight
		for j in 0 ..< len(ref_combined.data) {
			ref_combined.data[j] += w * ref.data[j]
		}
	}

	// Use combined reference for sampling
	return euler_sample_with_ref(tf, z, txt_emb, ref_combined, schedule, num_steps, t_offset, allocator)
}

// DDIM-like sampler (alternative to Euler)
// More stable for fewer steps
ddim_sample :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T),
	txt_emb: ^tensor.Tensor(T),
	schedule: []T,
	num_steps: int,
	eta: T, // Stochasticity (0 = deterministic)
	seed: i64,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	z_t := tensor.clone(z, allocator)

	// Generate noise for stochastic sampling
	noise: ^tensor.Tensor(T)
	if eta > 0 {
		shape := z.shape
		noise = init_noise(shape[0], shape[1], shape[2], shape[3], seed, allocator)
		defer tensor.free_tensor(noise, allocator)
	}

	for step in 0 ..< num_steps {
		t := schedule[step]
		t_next := schedule[step + 1]

		// Predict velocity
		v := transformer_forward(tf, z_t, txt_emb, t, allocator)
		defer tensor.free_tensor(v, allocator)

		// DDIM update
		// For rectified flow, this simplifies to:
		// z_{t+1} = z_t + (t_next - t) * v + eta * sqrt(dt) * noise

		dt := t_next - t
		sigma := eta * math.sqrt(T(math.abs(f64(dt))))

		for i in 0 ..< len(z_t.data) {
			z_t.data[i] += dt * v.data[i]
			if eta > 0 && noise != nil {
				z_t.data[i] += sigma * noise.data[i]
			}
		}
	}

	return z_t
}

// Guidance scale application (classifier-free guidance)
// v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
apply_cfg :: proc(
	v_cond: ^tensor.Tensor($T),
	v_uncond: ^tensor.Tensor(T),
	guidance_scale: T,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	result := tensor.tensor_alloc(T, v_cond.shape, true, allocator)

	for i in 0 ..< len(result.data) {
		result.data[i] = v_uncond.data[i] + guidance_scale * (v_cond.data[i] - v_uncond.data[i])
	}

	return result
}

// Euler sampler with classifier-free guidance
euler_sample_cfg :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T),
	txt_emb: ^tensor.Tensor(T),
	null_emb: ^tensor.Tensor(T), // Empty/null text embedding
	schedule: []T,
	num_steps: int,
	guidance_scale: T,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	z_t := tensor.clone(z, allocator)

	for step in 0 ..< num_steps {
		t := schedule[step]
		t_next := schedule[step + 1]
		dt := t_next - t

		// Conditional prediction
		v_cond := transformer_forward(tf, z_t, txt_emb, t, allocator)
		defer tensor.free_tensor(v_cond, allocator)

		// Unconditional prediction
		v_uncond := transformer_forward(tf, z_t, null_emb, t, allocator)
		defer tensor.free_tensor(v_uncond, allocator)

		// Apply CFG
		v := apply_cfg(v_cond, v_uncond, guidance_scale, allocator)
		defer tensor.free_tensor(v, allocator)

		// Euler step
		for i in 0 ..< len(z_t.data) {
			z_t.data[i] += dt * v.data[i]
		}
	}

	return z_t
}
