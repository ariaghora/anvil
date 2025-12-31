package nn

import "../tensor"
import "../trace"
import "core:fmt"
import "core:math"
import "core:simd"
import "core:slice"

UNROLL_FACTOR :: 16

// BatchNorm2d - 2D Batch Normalization for convolutional layers
// Input shape: (N, C, H, W) - normalizes over (N, H, W) dimensions per channel
Batch_Norm_2d :: struct($T: typeid) {
	weight:       ^tensor.Tensor(T), // (C,) - learnable scale parameter γ
	bias:         ^tensor.Tensor(T), // (C,) - learnable shift parameter β
	running_mean: ^tensor.Tensor(T), // (C,) - pre-computed running mean
	running_var:  ^tensor.Tensor(T), // (C,) - pre-computed running variance
	num_features: uint, // number of channels C
	eps:          T, // small constant for numerical stability
}

new_batch_norm_2d :: proc(
	$T: typeid,
	num_features: uint,
	allocator := context.allocator,
	eps := 1e-5,
) -> ^Batch_Norm_2d(T) {
	weight := tensor.ones(T, []uint{num_features}, allocator)
	bias := tensor.zeros(T, []uint{num_features}, allocator)
	running_mean := tensor.zeros(T, []uint{num_features}, allocator)
	running_var := tensor.ones(T, []uint{num_features}, allocator)

	return new_clone(
		Batch_Norm_2d(T) {
			weight = weight,
			bias = bias,
			running_mean = running_mean,
			running_var = running_var,
			num_features = num_features,
			eps = T(eps),
		},
		allocator,
	)
}

free_batch_norm_2d :: proc(bn: ^Batch_Norm_2d($T), allocator := context.allocator) {
	tensor.free_tensor(bn.weight, allocator)
	tensor.free_tensor(bn.bias, allocator)
	tensor.free_tensor(bn.running_mean, allocator)
	tensor.free_tensor(bn.running_var, allocator)
	free(bn, allocator)
}


forward_batch_norm_2d :: proc(
	bn: ^Batch_Norm_2d($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	assert(len(x.shape) == 4, "expected 4-dim tensor with NCHW format")
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	spatial_size := h * w

	result := tensor.zeros(T, x.shape, allocator)

	// Precompute ALL scales and shifts at once
	scales := make([]T, c, context.temp_allocator)
	shifts := make([]T, c, context.temp_allocator)

	#no_bounds_check for ch in 0 ..< c {
		mean := bn.running_mean.data[ch]
		var := bn.running_var.data[ch]
		weight := bn.weight.data[ch]
		bias := bn.bias.data[ch]

		scale := weight / math.sqrt(var + bn.eps)
		scales[ch] = scale
		shifts[ch] = bias - mean * scale
	}

	// Process in batch-first order for better cache locality
	#no_bounds_check {
		when T == f32 {
			for batch in 0 ..< n {
				batch_offset := batch * c * spatial_size
				for ch in 0 ..< c {
					scale_vec := #simd[4]f32{scales[ch], scales[ch], scales[ch], scales[ch]}
					shift_vec := #simd[4]f32{shifts[ch], shifts[ch], shifts[ch], shifts[ch]}

					base_idx := batch_offset + ch * spatial_size

					i := uint(0)
					// Process 8 elements at a time using two SIMD registers
					for ; i + 8 <= spatial_size; i += 8 {
						// SIMD loads instead of manual construction
						vals1 := (^#simd[4]f32)(&x.data[base_idx + i])^
						vals2 := (^#simd[4]f32)(&x.data[base_idx + i + 4])^

						// Apply batch norm: result = x * scale + shift
						results1 := simd.fma(vals1, scale_vec, shift_vec)
						results2 := simd.fma(vals2, scale_vec, shift_vec)
						(^#simd[4]f32)(&result.data[base_idx + i])^ = results1
						(^#simd[4]f32)(&result.data[base_idx + i + 4])^ = results2
					}

					// Process 4 elements at a time
					for ; i + 4 <= spatial_size; i += 4 {
						vals := (^#simd[4]f32)(&x.data[base_idx + i])^
						results := simd.fma(vals, scale_vec, shift_vec)
						(^#simd[4]f32)(&result.data[base_idx + i])^ = results
					}

					// Handle remainder
					for ; i < spatial_size; i += 1 {
						idx := base_idx + i
						result.data[idx] = x.data[idx] * scales[ch] + shifts[ch]
					}
				}
			}
		} else {
			// Original scalar code for other types
			for batch in 0 ..< n {
				batch_offset := batch * c * spatial_size
				for ch in 0 ..< c {
					scale := scales[ch]
					shift := shifts[ch]

					base_idx := batch_offset + ch * spatial_size

					i := uint(0)
					for ; i + UNROLL_FACTOR <= spatial_size; i += UNROLL_FACTOR {
						#unroll for j in 0 ..< UNROLL_FACTOR {
							idx := base_idx + i + uint(j)
							result.data[idx] = x.data[idx] * scale + shift
						}
					}

					for ; i < spatial_size; i += 1 {
						idx := base_idx + i
						result.data[idx] = x.data[idx] * scale + shift
					}
				}
			}
		}
	}

	return result
}

// BatchNorm1d - 1D Batch Normalization for linear layers
// Input shape: (N, C) - normalizes over N dimension per feature
Batch_Norm_1d :: struct($T: typeid) {
	weight:       ^tensor.Tensor(T), // (C,) - learnable scale parameter γ
	bias:         ^tensor.Tensor(T), // (C,) - learnable shift parameter β
	running_mean: ^tensor.Tensor(T), // (C,) - pre-computed running mean
	running_var:  ^tensor.Tensor(T), // (C,) - pre-computed running variance
	num_features: uint, // number of features C
	eps:          T, // small constant for numerical stability
}

new_batch_norm_1d :: proc(
	$T: typeid,
	num_features: uint,
	allocator := context.allocator,
	eps := 1e-5,
) -> ^Batch_Norm_1d(T) {
	weight := tensor.ones(T, []uint{num_features}, allocator)
	bias := tensor.zeros(T, []uint{num_features}, allocator)
	running_mean := tensor.zeros(T, []uint{num_features}, allocator)
	running_var := tensor.ones(T, []uint{num_features}, allocator)

	return new_clone(
		Batch_Norm_1d(T) {
			weight = weight,
			bias = bias,
			running_mean = running_mean,
			running_var = running_var,
			num_features = num_features,
			eps = T(eps),
		},
		allocator,
	)
}

free_batch_norm_1d :: proc(bn: ^Batch_Norm_1d($T), allocator := context.allocator) {
	tensor.free_tensor(bn.weight, allocator)
	tensor.free_tensor(bn.bias, allocator)
	tensor.free_tensor(bn.running_mean, allocator)
	tensor.free_tensor(bn.running_var, allocator)
	free(bn, allocator)
}

forward_batch_norm_1d :: proc(
	bn: ^Batch_Norm_1d($T),
	x: ^tensor.Tensor(T), // Input: (N, C)
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Validate input shape
	if len(x.shape) != 2 {
		panic("BatchNorm1d input must be 2D tensor (N, C)")
	}

	if x.shape[1] != bn.num_features {
		panic("Input features mismatch with num_features")
	}

	n, c := x.shape[0], x.shape[1]
	result := tensor.zeros(T, x.shape, allocator, loc)

	// Precompute scales and shifts for all channels
	scales := make([]T, c, context.temp_allocator)
	shifts := make([]T, c, context.temp_allocator)

	#no_bounds_check for ch in 0 ..< c {
		mean := bn.running_mean.data[ch]
		var := bn.running_var.data[ch]
		weight := bn.weight.data[ch]
		bias := bn.bias.data[ch]

		scale := weight / math.sqrt(var + bn.eps)
		scales[ch] = scale
		shifts[ch] = bias - mean * scale
	}

	#no_bounds_check {
		when T == f32 {
			// Process multiple samples and channels with SIMD
			for batch in 0 ..< n {
				base_idx := batch * c

				ch := uint(0)
				// Process 8 channels at a time
				for ; ch + 8 <= c; ch += 8 {
					// Load scales and shifts
					scale_vec1 := (^#simd[4]f32)(&scales[ch])^
					scale_vec2 := (^#simd[4]f32)(&scales[ch + 4])^
					shift_vec1 := (^#simd[4]f32)(&shifts[ch])^
					shift_vec2 := (^#simd[4]f32)(&shifts[ch + 4])^

					// Load input values
					vals1 := (^#simd[4]f32)(&x.data[base_idx + ch])^
					vals2 := (^#simd[4]f32)(&x.data[base_idx + ch + 4])^

					// Apply batch norm: result = x * scale + shift
					results1 := simd.fma(vals1, scale_vec1, shift_vec1)
					results2 := simd.fma(vals2, scale_vec2, shift_vec2)

					// Store results
					(^#simd[4]f32)(&result.data[base_idx + ch])^ = results1
					(^#simd[4]f32)(&result.data[base_idx + ch + 4])^ = results2
				}

				// Process 4 channels at a time
				for ; ch + 4 <= c; ch += 4 {
					scale_vec := (^#simd[4]f32)(&scales[ch])^
					shift_vec := (^#simd[4]f32)(&shifts[ch])^
					vals := (^#simd[4]f32)(&x.data[base_idx + ch])^
					results := simd.fma(vals, scale_vec, shift_vec)
					(^#simd[4]f32)(&result.data[base_idx + ch])^ = results
				}

				// Handle remainder
				for ; ch < c; ch += 1 {
					idx := base_idx + ch
					result.data[idx] = x.data[idx] * scales[ch] + shifts[ch]
				}
			}
		} else when T == f64 {
			// Process multiple samples and channels with SIMD
			for batch in 0 ..< n {
				base_idx := batch * c

				ch := uint(0)
				// Process 4 channels at a time
				for ; ch + 4 <= c; ch += 4 {
					scale_vec1 := (^#simd[2]f64)(&scales[ch])^
					scale_vec2 := (^#simd[2]f64)(&scales[ch + 2])^
					shift_vec1 := (^#simd[2]f64)(&shifts[ch])^
					shift_vec2 := (^#simd[2]f64)(&shifts[ch + 2])^

					vals1 := (^#simd[2]f64)(&x.data[base_idx + ch])^
					vals2 := (^#simd[2]f64)(&x.data[base_idx + ch + 2])^

					results1 := simd.fma(vals1, scale_vec1, shift_vec1)
					results2 := simd.fma(vals2, scale_vec2, shift_vec2)

					(^#simd[2]f64)(&result.data[base_idx + ch])^ = results1
					(^#simd[2]f64)(&result.data[base_idx + ch + 2])^ = results2
				}

				// Process 2 channels at a time
				for ; ch + 2 <= c; ch += 2 {
					scale_vec := (^#simd[2]f64)(&scales[ch])^
					shift_vec := (^#simd[2]f64)(&shifts[ch])^
					vals := (^#simd[2]f64)(&x.data[base_idx + ch])^
					results := simd.fma(vals, scale_vec, shift_vec)
					(^#simd[2]f64)(&result.data[base_idx + ch])^ = results
				}

				// Handle remainder
				for ; ch < c; ch += 1 {
					idx := base_idx + ch
					result.data[idx] = x.data[idx] * scales[ch] + shifts[ch]
				}
			}
		} else {
			// Scalar code for other types
			for batch in 0 ..< n {
				base_idx := batch * c

				ch := uint(0)
				// Unroll by 4 for better performance
				for ; ch + 4 <= c; ch += 4 {
					#unroll for j in 0 ..< 4 {
						idx := base_idx + ch + uint(j)
						result.data[idx] =
							x.data[idx] * scales[ch + uint(j)] + shifts[ch + uint(j)]
					}
				}

				// Handle remainder
				for ; ch < c; ch += 1 {
					idx := base_idx + ch
					result.data[idx] = x.data[idx] * scales[ch] + shifts[ch]
				}
			}
		}
	}

	return result
}
// LayerNorm - Layer Normalization
// Normalizes over the last dimension(s) of the input
Layer_Norm :: struct($T: typeid) {
	weight:           ^tensor.Tensor(T), // learnable scale parameter γ
	bias:             ^tensor.Tensor(T), // learnable shift parameter β
	normalized_shape: []uint, // shape of the normalization dimensions
	num_channels:     u64,
	eps:              T, // small constant for numerical stability
}

new_layer_norm :: proc(
	$T: typeid,
	normalized_shape: []uint,
	eps: T,
	allocator := context.allocator,
) -> ^Layer_Norm(T) {
	// Create weight and bias tensors with the normalized shape
	weight := tensor.ones(T, normalized_shape, allocator)
	bias := tensor.zeros(T, normalized_shape, allocator)

	// Copy normalized_shape
	norm_shape_copy := make([]uint, len(normalized_shape), allocator)
	copy(norm_shape_copy, normalized_shape)

	return new_clone(
		Layer_Norm(T) {
			weight = weight,
			bias = bias,
			normalized_shape = norm_shape_copy,
			eps = T(eps),
		},
		allocator,
	)
}

free_layer_norm :: proc(ln: ^Layer_Norm($T), allocator := context.allocator) {
	tensor.free_tensor(ln.weight, allocator)
	tensor.free_tensor(ln.bias, allocator)
	delete(ln.normalized_shape, allocator)
	free(ln, allocator)
}

// Helper for 1D LayerNorm (most common case)
forward_layer_norm_1d :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	forward_layer_norm_trace := trace.global_scoped("forward_layer_norm_1d")
	defer trace.global_end_scoped(forward_layer_norm_trace)

	// Get dimensions
	normalized_dim := ln.normalized_shape[0]
	total_elements := tensor.shape_to_size(x.shape)
	num_groups := total_elements / normalized_dim

	result := tensor.zeros(T, x.shape, allocator, loc)

	#no_bounds_check for group in 0 ..< num_groups {
		base_idx := group * normalized_dim

		// Compute mean
		mean := T(0)
		for i in 0 ..< normalized_dim {
			mean += x.data[base_idx + i]
		}
		mean /= T(normalized_dim)

		// Compute variance
		variance := T(0)
		for i in 0 ..< normalized_dim {
			diff := x.data[base_idx + i] - mean
			variance += diff * diff
		}
		variance /= T(normalized_dim)

		// Compute scale factor: 1 / sqrt(variance + eps)
		inv_std := T(1) / math.sqrt(variance + ln.eps)

		// Apply normalization, scale and bias in one pass
		i := uint(0)
		for ; i + 8 <= normalized_dim; i += 8 {
			#unroll for j in 0 ..< 8 {
				idx := base_idx + i + uint(j)
				normalized := (x.data[idx] - mean) * inv_std
				result.data[idx] =
					normalized * ln.weight.data[i + uint(j)] + ln.bias.data[i + uint(j)]
			}
		}

		// Handle remainder
		for ; i < normalized_dim; i += 1 {
			idx := base_idx + i
			normalized := (x.data[idx] - mean) * inv_std
			result.data[idx] = normalized * ln.weight.data[i] + ln.bias.data[i]
		}
	}

	return result
}

forward_layer_norm_2d :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T), // Expected shape: [B, C, H, W]
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) where T == f32 || T == f64 {
	// return result
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	spatial_size := h * w

	reshaped := tensor.reshape(x, []uint{n * c, spatial_size}, context.temp_allocator)
	spatial_ln := new_layer_norm(T, []uint{spatial_size}, ln.eps, context.temp_allocator)
	normalized := forward_layer_norm_1d(spatial_ln, reshaped, context.temp_allocator)
	normalized = tensor.reshape(normalized, []uint{n, c, h, w}, context.temp_allocator)

	result := tensor.zeros(T, x.shape, allocator, loc)
	when T == f32 {
		#no_bounds_check for batch in 0 ..< n {
			for ch in 0 ..< c {
				weight_val := ln.weight.data[ch]
				bias_val := ln.bias.data[ch]
				base_idx := (batch * c + ch) * spatial_size

				i := uint(0)
				// Process 8 at a time with SIMD
				for ; i + 8 <= spatial_size; i += 8 {
					weight_vec := #simd[8]f32 {
						weight_val,
						weight_val,
						weight_val,
						weight_val,
						weight_val,
						weight_val,
						weight_val,
						weight_val,
					}
					bias_vec := #simd[8]f32 {
						bias_val,
						bias_val,
						bias_val,
						bias_val,
						bias_val,
						bias_val,
						bias_val,
						bias_val,
					}

					vals := (^#simd[8]f32)(&normalized.data[base_idx + i])^
					results := simd.fma(vals, weight_vec, bias_vec)
					(^#simd[8]f32)(&result.data[base_idx + i])^ = results
				}

				// Process 4 at a time
				for ; i + 4 <= spatial_size; i += 4 {
					weight_vec := #simd[4]f32{weight_val, weight_val, weight_val, weight_val}
					bias_vec := #simd[4]f32{bias_val, bias_val, bias_val, bias_val}

					vals := (^#simd[4]f32)(&normalized.data[base_idx + i])^
					results := simd.fma(vals, weight_vec, bias_vec)
					(^#simd[4]f32)(&result.data[base_idx + i])^ = results
				}

				// Scalar remainder
				for ; i < spatial_size; i += 1 {
					result.data[base_idx + i] =
						normalized.data[base_idx + i] * weight_val + bias_val
				}
			}
		}
	}

	return result
}

// Common convenience constructor for ViT-style LayerNorm (1D)
new_layer_norm_1d :: proc(
	$T: typeid,
	embed_dim: uint,
	eps: T,
	allocator := context.allocator,
) -> ^Layer_Norm(T) {
	return new_layer_norm(T, []uint{embed_dim}, eps, allocator)
}

new_layer_norm_2d :: proc(
	$T: typeid,
	num_channels: uint,
	eps: f32 = 1e-5,
	allocator := context.allocator,
) -> ^Layer_Norm(T) {
	return new_layer_norm_1d(T, num_channels, eps, allocator)
}

// Based on
// https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
// facebook & huggingface named this LayerNorm and LayerNorm2d while in fact this is nonstandard
// way to do layer normalization. So I created a separated struct doing different thing.
Channel_Layer_Norm :: struct($T: typeid) {
	weight:       ^tensor.Tensor(T), // (C,)
	bias:         ^tensor.Tensor(T), // (C,)
	num_channels: uint,
	eps:          T,
}

new_channel_layer_norm :: proc(
	$T: typeid,
	num_channels: uint,
	eps: T, // ConvNeXt default
	allocator := context.allocator,
) -> ^Channel_Layer_Norm(T) {
	weight := tensor.ones(T, []uint{num_channels}, allocator)
	bias := tensor.zeros(T, []uint{num_channels}, allocator)

	return new_clone(
		Channel_Layer_Norm(T) {
			weight = weight,
			bias = bias,
			num_channels = num_channels,
			eps = eps,
		},
		allocator,
	)
}

forward_channel_layer_norm :: proc(
	ln: ^Channel_Layer_Norm($T),
	x: ^tensor.Tensor(T), // (N, C, H, W)
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Normalizes across channels at each (n, h, w) position
	// This is what ConvNeXt/SAM uses
	assert(len(x.shape) == 4)
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]

	result := tensor.zeros(T, x.shape, allocator, loc)

	when T == f32 {
		#no_bounds_check for batch in 0 ..< n {
			for spatial in 0 ..< (h * w) {
				h_idx := spatial / w
				w_idx := spatial % w

				// Mean across channels
				mean := f32(0)
				for ch in 0 ..< c {
					idx := ((batch * c + ch) * h + h_idx) * w + w_idx
					mean += x.data[idx]
				}
				mean /= f32(c)

				// Variance across channels
				variance := f32(0)
				for ch in 0 ..< c {
					idx := ((batch * c + ch) * h + h_idx) * w + w_idx
					diff := x.data[idx] - mean
					variance += diff * diff
				}
				variance /= f32(c)

				inv_std := f32(1) / math.sqrt(variance + ln.eps)

				// Apply normalization and channel weights
				for ch in 0 ..< c {
					idx := ((batch * c + ch) * h + h_idx) * w + w_idx
					normalized := (x.data[idx] - mean) * inv_std
					result.data[idx] = normalized * ln.weight.data[ch] + ln.bias.data[ch]
				}
			}
		}
	}

	return result
}

free_channel_layer_norm :: proc(ln: ^Channel_Layer_Norm($T), allocator := context.allocator) {
	tensor.free_tensor(ln.weight, allocator)
	tensor.free_tensor(ln.bias, allocator)
	free(ln, allocator)
}
