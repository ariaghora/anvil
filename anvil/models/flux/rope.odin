// 4-Axis Rotary Position Embedding (RoPE) for FLUX
//
// FLUX uses 4-axis RoPE with head_dim=128 split into 4 axes of 32 dims each:
// - Axis 0 (dims 0-31): T position (always 0 for both img and txt)
// - Axis 1 (dims 32-63): H position (y for images, 0 for text)
// - Axis 2 (dims 64-95): W position (x for images, 0 for text)
// - Axis 3 (dims 96-127): L position (0 for images, seq_idx for text)
//
// Each axis has 16 frequency pairs (half of 32 dims).
// Rotation pairs elements [d, d+16] within each 32-dim axis.

package flux

import "../../tensor"
import "core:math"
import "core:mem"

// Precompute 1D RoPE frequencies (for Qwen3 text encoder, not FLUX transformer)
compute_rope_freqs :: proc(
	$T: typeid,
	max_seq: uint,
	dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	half_dim := dim / 2

	// Allocate [max_seq, half_dim * 2] for cos/sin pairs
	freqs := tensor.tensor_alloc(T, []uint{max_seq, half_dim * 2}, true, allocator)

	// Compute inverse frequencies: 1 / (theta^(2i/dim))
	inv_freqs := make([]T, half_dim, context.temp_allocator)
	for i in 0 ..< int(half_dim) {
		inv_freqs[i] = T(1.0) / math.pow(T(theta), T(2 * i) / T(dim))
	}

	// Compute cos/sin for each position
	for pos in 0 ..< int(max_seq) {
		for d in 0 ..< int(half_dim) {
			angle := T(pos) * inv_freqs[d]
			freqs.data[pos * int(half_dim) * 2 + d * 2] = math.cos(angle)
			freqs.data[pos * int(half_dim) * 2 + d * 2 + 1] = math.sin(angle)
		}
	}

	return freqs
}

// Compute 4-axis RoPE frequencies for FLUX joint txt+img sequence
// Returns [total_seq, head_dim*2] tensor with separate cos and sin arrays
// Order: text tokens first, then image tokens (matching antirez)
// Per antirez: cos[d] == cos[d+1] (repeat_interleave for pairs)
compute_rope_freqs_flux :: proc(
	$T: typeid,
	txt_len: uint,
	img_height, img_width: uint,
	head_dim: uint,  // Should be 128
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	img_seq := img_height * img_width
	total_seq := txt_len + img_seq
	axis_dim : uint = 32  // head_dim / 4
	half_axis : uint = 16  // axis_dim / 2

	// Allocate [total_seq, head_dim*2] - cos then sin for each position
	freqs := tensor.tensor_alloc(T, []uint{total_seq, head_dim * 2}, true, allocator)

	// Compute base frequencies for each axis (16 frequencies per axis)
	base_freqs := make([]T, half_axis, context.temp_allocator)
	for i in 0 ..< int(half_axis) {
		base_freqs[i] = T(1.0) / math.pow(T(theta), T(2 * i) / T(axis_dim))
	}

	// Text tokens: only axis 3 (L) rotates, axes 0,1,2 are identity
	for t in 0 ..< int(txt_len) {
		cos_offset := t * int(head_dim * 2)
		sin_offset := cos_offset + int(head_dim)

		// Axis 0 (dims 0-31): T=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + d] = T(1.0)
			freqs.data[sin_offset + d] = T(0.0)
		}

		// Axis 1 (dims 32-63): H=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + int(axis_dim) + d] = T(1.0)
			freqs.data[sin_offset + int(axis_dim) + d] = T(0.0)
		}

		// Axis 2 (dims 64-95): W=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + int(axis_dim) * 2 + d] = T(1.0)
			freqs.data[sin_offset + int(axis_dim) * 2 + d] = T(0.0)
		}

		// Axis 3 (dims 96-127): L=t -> rotate
		for d in 0 ..< int(half_axis) {
			angle := T(t) * base_freqs[d]
			c := math.cos(angle)
			s := math.sin(angle)
			freqs.data[cos_offset + int(axis_dim) * 3 + d * 2] = c
			freqs.data[cos_offset + int(axis_dim) * 3 + d * 2 + 1] = c
			freqs.data[sin_offset + int(axis_dim) * 3 + d * 2] = s
			freqs.data[sin_offset + int(axis_dim) * 3 + d * 2 + 1] = s
		}
	}

	// Image tokens: axes 1 (H) and 2 (W) rotate, axes 0 and 3 are identity
	for y in 0 ..< int(img_height) {
		for x in 0 ..< int(img_width) {
			pos := int(txt_len) + y * int(img_width) + x
			cos_offset := pos * int(head_dim * 2)
			sin_offset := cos_offset + int(head_dim)

			// Axis 0 (dims 0-31): T=0 -> identity
			for d in 0 ..< int(axis_dim) {
				freqs.data[cos_offset + d] = T(1.0)
				freqs.data[sin_offset + d] = T(0.0)
			}

			// Axis 1 (dims 32-63): H=y -> rotate
			for d in 0 ..< int(half_axis) {
				angle := T(y) * base_freqs[d]
				c := math.cos(angle)
				s := math.sin(angle)
				freqs.data[cos_offset + int(axis_dim) + d * 2] = c
				freqs.data[cos_offset + int(axis_dim) + d * 2 + 1] = c
				freqs.data[sin_offset + int(axis_dim) + d * 2] = s
				freqs.data[sin_offset + int(axis_dim) + d * 2 + 1] = s
			}

			// Axis 2 (dims 64-95): W=x -> rotate
			for d in 0 ..< int(half_axis) {
				angle := T(x) * base_freqs[d]
				c := math.cos(angle)
				s := math.sin(angle)
				freqs.data[cos_offset + int(axis_dim) * 2 + d * 2] = c
				freqs.data[cos_offset + int(axis_dim) * 2 + d * 2 + 1] = c
				freqs.data[sin_offset + int(axis_dim) * 2 + d * 2] = s
				freqs.data[sin_offset + int(axis_dim) * 2 + d * 2 + 1] = s
			}

			// Axis 3 (dims 96-127): L=0 -> identity
			for d in 0 ..< int(axis_dim) {
				freqs.data[cos_offset + int(axis_dim) * 3 + d] = T(1.0)
				freqs.data[sin_offset + int(axis_dim) * 3 + d] = T(0.0)
			}
		}
	}

	return freqs
}

// Apply 1D RoPE to query/key tensors in-place (for Qwen3)
// x: [batch, seq, num_heads * head_dim]
// freqs: [seq, half_dim * 2] with [cos, sin] pairs
// Pairs x[d] with x[d + half_dim]
apply_rope_inplace :: proc(
	x: ^tensor.Tensor($T),
	freqs: ^tensor.Tensor(T),
	num_heads: uint,
	head_dim: uint,
) {
	batch := x.shape[0]
	seq_len := x.shape[1]
	half_dim := head_dim / 2

	for b in 0 ..< int(batch) {
		for s in 0 ..< int(seq_len) {
			// Get frequency row for this position
			freq_offset := min(s, int(freqs.shape[0]) - 1) * int(half_dim) * 2

			for h in 0 ..< int(num_heads) {
				head_offset := b * int(seq_len * num_heads * head_dim) + s * int(num_heads * head_dim) + h * int(head_dim)

				// Apply rotation: pair x[d] with x[d + half_dim]
				for d in 0 ..< int(half_dim) {
					cos_val := freqs.data[freq_offset + d * 2]
					sin_val := freqs.data[freq_offset + d * 2 + 1]

					x0 := x.data[head_offset + d]
					x1 := x.data[head_offset + d + int(half_dim)]

					// Rotate
					x.data[head_offset + d] = x0 * cos_val - x1 * sin_val
					x.data[head_offset + d + int(half_dim)] = x0 * sin_val + x1 * cos_val
				}
			}
		}
	}
}

// Apply 4-axis RoPE for FLUX transformer
// x: [batch, seq, num_heads * head_dim] where head_dim=128
// freqs: [seq, head_dim*2] with cos array then sin array
// Per antirez: cos[d] == cos[d+1] due to repeat_interleave
apply_rope_flux_inplace :: proc(
	x: ^tensor.Tensor($T),
	freqs: ^tensor.Tensor(T),
	num_heads: uint,
	head_dim: uint,
) {
	batch := x.shape[0]
	seq_len := x.shape[1]

	for b in 0 ..< int(batch) {
		for s in 0 ..< int(seq_len) {
			freq_s := min(s, int(freqs.shape[0]) - 1)
			cos_offset := freq_s * int(head_dim * 2)
			sin_offset := cos_offset + int(head_dim)

			for h in 0 ..< int(num_heads) {
				head_offset := b * int(seq_len * num_heads * head_dim) + s * int(num_heads * head_dim) + h * int(head_dim)

				// Process consecutive pairs across all 128 dims
				// Pairs: (0,1), (2,3), ..., (126,127)
				for d := 0; d < int(head_dim); d += 2 {
					cos_val := freqs.data[cos_offset + d]  // cos[d] == cos[d+1]
					sin_val := freqs.data[sin_offset + d]  // sin[d] == sin[d+1]

					x0 := x.data[head_offset + d]
					x1 := x.data[head_offset + d + 1]

					x.data[head_offset + d] = x0 * cos_val - x1 * sin_val
					x.data[head_offset + d + 1] = x1 * cos_val + x0 * sin_val
				}
			}
		}
	}
}

// Apply RoPE, returning a new tensor
apply_rope :: proc(
	x: ^tensor.Tensor($T),
	freqs: ^tensor.Tensor(T),
	num_heads: uint,
	head_dim: uint,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	result := tensor.clone(x, allocator)
	apply_rope_inplace(result, freqs, num_heads, head_dim)
	return result
}

// Compute RoPE frequencies for IMAGE tokens only (double blocks)
// Returns [img_seq, head_dim*2] tensor with separate cos and sin arrays
// Axes 1 (H) and 2 (W) rotate, axes 0 and 3 are identity
// Per antirez: cos[d] == cos[d+1] (repeat_interleave for pairs)
compute_rope_freqs_img :: proc(
	$T: typeid,
	img_height, img_width: uint,
	head_dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	img_seq := img_height * img_width
	axis_dim : uint = 32
	half_axis : uint = 16

	// Store cos and sin separately, each with head_dim elements (duplicated for pairs)
	freqs := tensor.tensor_alloc(T, []uint{img_seq, head_dim * 2}, true, allocator)

	base_freqs := make([]T, half_axis, context.temp_allocator)
	for i in 0 ..< int(half_axis) {
		base_freqs[i] = T(1.0) / math.pow(T(theta), T(2 * i) / T(axis_dim))
	}

	for y in 0 ..< int(img_height) {
		for x in 0 ..< int(img_width) {
			pos := y * int(img_width) + x
			cos_offset := pos * int(head_dim * 2)
			sin_offset := cos_offset + int(head_dim)

			// Axis 0 (dims 0-31): T=0 -> identity (cos=1, sin=0)
			for d in 0 ..< int(axis_dim) {
				freqs.data[cos_offset + d] = T(1.0)
				freqs.data[sin_offset + d] = T(0.0)
			}

			// Axis 1 (dims 32-63): H=y -> rotate
			for d in 0 ..< int(half_axis) {
				angle := T(y) * base_freqs[d]
				c := math.cos(angle)
				s := math.sin(angle)
				// Duplicate for consecutive pairs
				freqs.data[cos_offset + int(axis_dim) + d * 2] = c
				freqs.data[cos_offset + int(axis_dim) + d * 2 + 1] = c
				freqs.data[sin_offset + int(axis_dim) + d * 2] = s
				freqs.data[sin_offset + int(axis_dim) + d * 2 + 1] = s
			}

			// Axis 2 (dims 64-95): W=x -> rotate
			for d in 0 ..< int(half_axis) {
				angle := T(x) * base_freqs[d]
				c := math.cos(angle)
				s := math.sin(angle)
				freqs.data[cos_offset + int(axis_dim) * 2 + d * 2] = c
				freqs.data[cos_offset + int(axis_dim) * 2 + d * 2 + 1] = c
				freqs.data[sin_offset + int(axis_dim) * 2 + d * 2] = s
				freqs.data[sin_offset + int(axis_dim) * 2 + d * 2 + 1] = s
			}

			// Axis 3 (dims 96-127): L=0 -> identity
			for d in 0 ..< int(axis_dim) {
				freqs.data[cos_offset + int(axis_dim) * 3 + d] = T(1.0)
				freqs.data[sin_offset + int(axis_dim) * 3 + d] = T(0.0)
			}
		}
	}

	return freqs
}

// Compute RoPE frequencies for TEXT tokens only (double blocks)
// Returns [txt_len, head_dim*2] tensor with separate cos and sin arrays
// Only axis 3 (L) rotates, axes 0,1,2 are identity
// Per antirez: cos[d] == cos[d+1] (repeat_interleave for pairs)
compute_rope_freqs_txt :: proc(
	$T: typeid,
	txt_len: uint,
	head_dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	axis_dim : uint = 32
	half_axis : uint = 16

	freqs := tensor.tensor_alloc(T, []uint{txt_len, head_dim * 2}, true, allocator)

	base_freqs := make([]T, half_axis, context.temp_allocator)
	for i in 0 ..< int(half_axis) {
		base_freqs[i] = T(1.0) / math.pow(T(theta), T(2 * i) / T(axis_dim))
	}

	for t in 0 ..< int(txt_len) {
		cos_offset := t * int(head_dim * 2)
		sin_offset := cos_offset + int(head_dim)

		// Axis 0 (dims 0-31): T=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + d] = T(1.0)
			freqs.data[sin_offset + d] = T(0.0)
		}

		// Axis 1 (dims 32-63): H=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + int(axis_dim) + d] = T(1.0)
			freqs.data[sin_offset + int(axis_dim) + d] = T(0.0)
		}

		// Axis 2 (dims 64-95): W=0 -> identity
		for d in 0 ..< int(axis_dim) {
			freqs.data[cos_offset + int(axis_dim) * 2 + d] = T(1.0)
			freqs.data[sin_offset + int(axis_dim) * 2 + d] = T(0.0)
		}

		// Axis 3 (dims 96-127): L=t -> rotate
		for d in 0 ..< int(half_axis) {
			angle := T(t) * base_freqs[d]
			c := math.cos(angle)
			s := math.sin(angle)
			freqs.data[cos_offset + int(axis_dim) * 3 + d * 2] = c
			freqs.data[cos_offset + int(axis_dim) * 3 + d * 2 + 1] = c
			freqs.data[sin_offset + int(axis_dim) * 3 + d * 2] = s
			freqs.data[sin_offset + int(axis_dim) * 3 + d * 2 + 1] = s
		}
	}

	return freqs
}


