// 2D Rotary Position Embedding (RoPE) for FLUX
//
// FLUX uses 2D RoPE for image patches - positions are encoded
// based on (x, y) coordinates in the image grid.

package flux

import "../../tensor"
import "core:math"
import "core:mem"

// Precompute RoPE frequencies for a given max sequence length
// Returns tensor of shape [max_seq, dim] containing cos/sin pairs
compute_rope_freqs :: proc(
	$T: typeid,
	max_seq: uint,
	dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	half_dim := dim / 2

	// Allocate [max_seq, dim] - stores interleaved cos/sin
	freqs := tensor.tensor_alloc(T, []uint{max_seq, dim}, true, allocator)

	// Compute inverse frequencies: 1 / (theta^(2i/dim))
	inv_freqs := make([]T, half_dim, context.temp_allocator)
	for i in 0 ..< int(half_dim) {
		inv_freqs[i] = T(1.0) / math.pow(T(theta), T(2 * i) / T(dim))
	}

	// Compute cos/sin for each position
	for pos in 0 ..< int(max_seq) {
		for i in 0 ..< int(half_dim) {
			angle := T(pos) * inv_freqs[i]
			// Store as [cos, sin] pairs
			freqs.data[pos * int(dim) + 2 * i] = math.cos(angle)
			freqs.data[pos * int(dim) + 2 * i + 1] = math.sin(angle)
		}
	}

	return freqs
}

// Compute 2D RoPE frequencies for image patches
// Positions are based on (x, y) grid coordinates
compute_rope_freqs_2d :: proc(
	$T: typeid,
	height, width: uint,
	dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	seq_len := height * width
	half_dim := dim / 2
	quarter_dim := dim / 4

	// Allocate [seq_len, dim]
	freqs := tensor.tensor_alloc(T, []uint{seq_len, dim}, true, allocator)

	// Compute inverse frequencies
	inv_freqs := make([]T, quarter_dim, context.temp_allocator)
	for i in 0 ..< int(quarter_dim) {
		inv_freqs[i] = T(1.0) / math.pow(T(theta), T(4 * i) / T(dim))
	}

	// For each position in the 2D grid
	for y in 0 ..< int(height) {
		for x in 0 ..< int(width) {
			pos := y * int(width) + x

			// First half: x-axis frequencies
			for i in 0 ..< int(quarter_dim) {
				angle_x := T(x) * inv_freqs[i]
				freqs.data[pos * int(dim) + 2 * i] = math.cos(angle_x)
				freqs.data[pos * int(dim) + 2 * i + 1] = math.sin(angle_x)
			}

			// Second half: y-axis frequencies
			for i in 0 ..< int(quarter_dim) {
				angle_y := T(y) * inv_freqs[i]
				freqs.data[pos * int(dim) + int(half_dim) + 2 * i] = math.cos(angle_y)
				freqs.data[pos * int(dim) + int(half_dim) + 2 * i + 1] = math.sin(angle_y)
			}
		}
	}

	return freqs
}

// Apply RoPE to query/key tensors in-place
// x: [batch, seq, num_heads * head_dim]
// freqs: [seq, head_dim] with interleaved cos/sin
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
			freq_offset := min(s, int(freqs.shape[0]) - 1) * int(head_dim)

			for h in 0 ..< int(num_heads) {
				head_offset := b * int(seq_len * num_heads * head_dim) + s * int(num_heads * head_dim) + h * int(head_dim)

				// Apply rotation to pairs of elements
				for i in 0 ..< int(half_dim) {
					cos_val := freqs.data[freq_offset + 2 * i]
					sin_val := freqs.data[freq_offset + 2 * i + 1]

					x0 := x.data[head_offset + 2 * i]
					x1 := x.data[head_offset + 2 * i + 1]

					// Rotate: [cos, -sin; sin, cos] @ [x0, x1]
					x.data[head_offset + 2 * i] = x0 * cos_val - x1 * sin_val
					x.data[head_offset + 2 * i + 1] = x0 * sin_val + x1 * cos_val
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

// Compute RoPE frequencies for joint img+txt sequence
// Image positions use 2D coordinates, text uses sequential 1D positions
compute_rope_freqs_joint :: proc(
	$T: typeid,
	img_height, img_width: uint,
	txt_len: uint,
	dim: uint,
	theta: f32,
	allocator := context.allocator,
) -> ^tensor.Tensor(T) {
	img_seq := img_height * img_width
	total_seq := img_seq + txt_len
	half_dim := dim / 2
	quarter_dim := dim / 4

	freqs := tensor.tensor_alloc(T, []uint{total_seq, dim}, true, allocator)

	// Compute inverse frequencies
	inv_freqs := make([]T, quarter_dim, context.temp_allocator)
	for i in 0 ..< int(quarter_dim) {
		inv_freqs[i] = T(1.0) / math.pow(T(theta), T(4 * i) / T(dim))
	}

	// Image positions: 2D coordinates
	for y in 0 ..< int(img_height) {
		for x in 0 ..< int(img_width) {
			pos := y * int(img_width) + x

			for i in 0 ..< int(quarter_dim) {
				angle_x := T(x) * inv_freqs[i]
				freqs.data[pos * int(dim) + 2 * i] = math.cos(angle_x)
				freqs.data[pos * int(dim) + 2 * i + 1] = math.sin(angle_x)

				angle_y := T(y) * inv_freqs[i]
				freqs.data[pos * int(dim) + int(half_dim) + 2 * i] = math.cos(angle_y)
				freqs.data[pos * int(dim) + int(half_dim) + 2 * i + 1] = math.sin(angle_y)
			}
		}
	}

	// Text positions: 1D sequential (starting after image)
	for t in 0 ..< int(txt_len) {
		pos := int(img_seq) + t

		for i in 0 ..< int(half_dim) {
			// Use same frequency pattern but with sequential position
			angle := T(t) * inv_freqs[i % int(quarter_dim)]
			freqs.data[pos * int(dim) + 2 * i] = math.cos(angle)
			freqs.data[pos * int(dim) + 2 * i + 1] = math.sin(angle)
		}
	}

	return freqs
}
