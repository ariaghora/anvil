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

// Compute empirical mu for FLUX.2 official schedule
// Matches Python's get_schedule() from official flux2 code
@(private = "file")
compute_empirical_mu :: proc(image_seq_len: int, num_steps: int) -> f32 {
	a1 : f32 = 8.73809524e-05
	b1 : f32 = 1.89833333
	a2 : f32 = 0.00016927
	b2 : f32 = 0.45666666

	if image_seq_len > 4300 {
		return a2 * f32(image_seq_len) + b2
	}

	m_200 := a2 * f32(image_seq_len) + b2
	m_10 := a1 * f32(image_seq_len) + b1

	a := (m_200 - m_10) / 190.0
	b := m_200 - 200.0 * a
	return a * f32(num_steps) + b
}

// Generalized time SNR shift
@(private = "file")
generalized_time_snr_shift :: proc(t: f32, mu: f32, sigma: f32) -> f32 {
	if t <= 0.0 do return 0.0
	if t >= 1.0 do return 1.0
	return math.exp(mu) / (math.exp(mu) + math.pow(1.0 / t - 1.0, sigma))
}

// FLUX.2 official schedule with empirical mu calculation
// Matches antirez's flux_official_schedule
flux_schedule :: proc(
	num_steps: int,
	image_seq_len: int,
	allocator := context.allocator,
) -> []f32 {
	schedule := make([]f32, num_steps + 1, allocator)
	mu := compute_empirical_mu(image_seq_len, num_steps)

	for i in 0 ..= num_steps {
		t := 1.0 - f32(i) / f32(num_steps)  // Linear from 1 to 0
		schedule[i] = generalized_time_snr_shift(t, mu, 1.0)
	}

	return schedule
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

// Patchify: reshape [B, C, H, W] -> [B, H*W, C]
@(private = "file")
patchify :: proc(x: ^tensor.Tensor($T), allocator := context.allocator) -> ^tensor.Tensor(T) {
	batch := x.shape[0]
	channels := x.shape[1]
	h := x.shape[2]
	w := x.shape[3]
	seq_len := h * w

	result := tensor.tensor_alloc(T, []uint{batch, seq_len, channels}, true, allocator)

	for b in 0 ..< int(batch) {
		for y in 0 ..< int(h) {
			for x_pos in 0 ..< int(w) {
				seq_idx := y * int(w) + x_pos
				for c in 0 ..< int(channels) {
					src := b * int(channels * h * w) + c * int(h * w) + y * int(w) + x_pos
					dst := b * int(seq_len * channels) + seq_idx * int(channels) + c
					result.data[dst] = x.data[src]
				}
			}
		}
	}
	return result
}

// Unpatchify: reshape [B, H*W, C] -> [B, C, H, W]
@(private = "file")
unpatchify :: proc(x: ^tensor.Tensor($T), h, w: uint, allocator := context.allocator) -> ^tensor.Tensor(T) {
	batch := x.shape[0]
	channels := x.shape[2]
	seq_len := h * w

	result := tensor.tensor_alloc(T, []uint{batch, channels, h, w}, true, allocator)

	for b in 0 ..< int(batch) {
		for y in 0 ..< int(h) {
			for x_pos in 0 ..< int(w) {
				seq_idx := y * int(w) + x_pos
				for c in 0 ..< int(channels) {
					src := b * int(seq_len * channels) + seq_idx * int(channels) + c
					dst := b * int(channels * h * w) + c * int(h * w) + y * int(w) + x_pos
					result.data[dst] = x.data[src]
				}
			}
		}
	}
	return result
}

// Euler sampler for rectified flow
// z_t = (1-t) * z_0 + t * x  (where x is data, z_0 is noise)
// v = x - z_0 (velocity field)
// ODE: dz/dt = v
euler_sample :: proc(
	tf: ^Transformer($T),
	z: ^tensor.Tensor(T), // Initial noise [B, C, H, W]
	txt_emb: ^tensor.Tensor(T), // Text embeddings
	schedule: []T, // Timestep schedule
	num_steps: int,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	// Get spatial dimensions for unpatchify later
	h := z.shape[2]
	w := z.shape[3]

	// Patchify to sequence format [B, C, H, W] -> [B, H*W, C]
	z_seq := patchify(z, allocator)
	defer tensor.free_tensor(z_seq, allocator)

	// Clone as working tensor
	z_t := tensor.clone(z_seq, allocator)

	for step in 0 ..< num_steps {
		// Current and next timesteps
		t := schedule[step]
		t_next := schedule[step + 1]
		dt := t_next - t

		// Predict velocity at current timestep
		v := transformer_forward(tf, z_t, txt_emb, t, h, w, allocator)
		defer tensor.free_tensor(v, allocator)

		// Euler step: z_{t+dt} = z_t + dt * v
		for i in 0 ..< len(z_t.data) {
			z_t.data[i] += dt * v.data[i]
		}
	}

	// Unpatchify back to spatial format [B, H*W, C] -> [B, C, H, W]
	result := unpatchify(z_t, h, w, allocator)
	tensor.free_tensor(z_t, allocator)

	return result
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
	img_height, img_width: uint, // Image dimensions for RoPE
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

		v := transformer_forward(tf, z_t, txt_emb, t, img_height, img_width, allocator)
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
	img_height, img_width: uint, // Image dimensions for RoPE
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
		v := transformer_forward(tf, z_t, txt_emb, t, img_height, img_width, allocator)
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
	img_height, img_width: uint, // Image dimensions for RoPE
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	z_t := tensor.clone(z, allocator)

	for step in 0 ..< num_steps {
		t := schedule[step]
		t_next := schedule[step + 1]
		dt := t_next - t

		// Conditional prediction
		v_cond := transformer_forward(tf, z_t, txt_emb, t, img_height, img_width, allocator)
		defer tensor.free_tensor(v_cond, allocator)

		// Unconditional prediction
		v_uncond := transformer_forward(tf, z_t, null_emb, t, img_height, img_width, allocator)
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
