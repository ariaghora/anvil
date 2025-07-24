package nn

import "../tensor"
import "../trace"
import "core:math"

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


// NOTE(Aria): This is optimized implementation, so we don't use tensor abstraction and broadcasting
// too much. Not a clean code. Deal with it.
forward_batch_norm_2d :: proc(
	bn: ^Batch_Norm_2d($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
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
	#no_bounds_check for batch in 0 ..< n {
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

			// Handle remainder
			for ; i < spatial_size; i += 1 {
				idx := base_idx + i
				result.data[idx] = x.data[idx] * scale + shift
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

	// For broadcasting: (C,) -> (1, C)
	mean_reshaped := tensor.reshape(
		bn.running_mean,
		[]uint{1, bn.num_features},
		context.temp_allocator,
	)
	var_reshaped := tensor.reshape(
		bn.running_var,
		[]uint{1, bn.num_features},
		context.temp_allocator,
	)
	weight_reshaped := tensor.reshape(
		bn.weight,
		[]uint{1, bn.num_features},
		context.temp_allocator,
	)
	bias_reshaped := tensor.reshape(bn.bias, []uint{1, bn.num_features}, context.temp_allocator)

	// Compute: (x - mean) / sqrt(var + eps) * weight + bias
	x_centered := tensor.sub(x, mean_reshaped, context.temp_allocator)

	eps_tensor := tensor.new_with_init([]T{bn.eps}, []uint{1, 1}, context.temp_allocator)
	var_eps := tensor.add(var_reshaped, eps_tensor, context.temp_allocator)
	std := tensor.sqrt(var_eps, context.temp_allocator)

	x_normalized := tensor.div(x_centered, std, context.temp_allocator)
	x_scaled := tensor.mul(x_normalized, weight_reshaped, context.temp_allocator)
	result := tensor.add(x_scaled, bias_reshaped, allocator, loc)

	return result
}

// LayerNorm - Layer Normalization
// Normalizes over the last dimension(s) of the input
Layer_Norm :: struct($T: typeid) {
	weight:           ^tensor.Tensor(T), // learnable scale parameter γ
	bias:             ^tensor.Tensor(T), // learnable shift parameter β
	normalized_shape: []uint, // shape of the normalization dimensions
	eps:              T, // small constant for numerical stability
}

new_layer_norm :: proc(
	$T: typeid,
	normalized_shape: []uint,
	allocator := context.allocator,
	eps := 1e-5,
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

forward_layer_norm :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	forward_layer_norm_trace := trace.TRACE_FUNCTION("forward_layer_norm")
	defer trace.end_scoped_trace(forward_layer_norm_trace)

	// Validate that the last dimensions match normalized_shape
	input_rank := len(x.shape)
	norm_rank := len(ln.normalized_shape)

	if input_rank < norm_rank {
		panic("Input tensor has fewer dimensions than normalized_shape")
	}

	// Check that the last norm_rank dimensions match
	for i in 0 ..< norm_rank {
		input_dim := x.shape[input_rank - norm_rank + i]
		norm_dim := ln.normalized_shape[i]
		if input_dim != norm_dim {
			panic("Input shape mismatch with normalized_shape")
		}
	}

	// Compute statistics over the normalized dimensions
	if norm_rank == 1 {
		return forward_layer_norm_1d(ln, x, allocator, loc)
	} else {
		return forward_layer_norm_nd(ln, x, allocator, loc)
	}
}

// Helper for 1D LayerNorm (most common case)
forward_layer_norm_1d :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	forward_layer_norm_trace := trace.TRACE_FUNCTION("forward_layer_norm_1d")
	defer trace.end_scoped_trace(forward_layer_norm_trace)

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


// Helper for multi-dimensional LayerNorm
forward_layer_norm_nd :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	forward_layer_norm_trace := trace.TRACE_FUNCTION("forward_layer_norm_nd")
	defer trace.end_scoped_trace(forward_layer_norm_trace)
	input_rank := len(x.shape)
	norm_rank := len(ln.normalized_shape)

	// Create axes to reduce over (the last norm_rank dimensions)
	reduce_axes := make([]int, norm_rank, context.temp_allocator)
	for i in 0 ..< norm_rank {
		reduce_axes[i] = input_rank - norm_rank + i
	}

	// Compute mean over the normalized dimensions
	// We need to reduce sequentially over each axis
	mean := x
	for axis in reduce_axes {
		mean = tensor.tensor_mean(mean, axis, true, context.temp_allocator)
	}

	// Compute variance: E[(x - mean)^2]
	x_centered := tensor.sub(x, mean, context.temp_allocator)
	x_squared := tensor.mul(x_centered, x_centered, context.temp_allocator)

	variance := x_squared
	for axis in reduce_axes {
		variance = tensor.tensor_mean(variance, axis, true, context.temp_allocator)
	}

	// Add eps and take sqrt
	eps_tensor := tensor.new_with_init([]T{ln.eps}, []uint{1}, context.temp_allocator)
	var_eps := tensor.add(variance, eps_tensor, context.temp_allocator)
	std := tensor.sqrt(var_eps, context.temp_allocator)

	// Normalize: (x - mean) / std
	x_norm := tensor.div(x_centered, std, context.temp_allocator)

	// Scale and shift: x_norm * weight + bias
	// Broadcasting handled automatically by tensor operations
	x_scaled := tensor.mul(x_norm, ln.weight, context.temp_allocator)
	result := tensor.add(x_scaled, ln.bias, allocator, loc)

	return result
}

// Common convenience constructor for ViT-style LayerNorm (1D)
new_layer_norm_1d :: proc(
	$T: typeid,
	embed_dim: uint,
	allocator := context.allocator,
	eps := 1e-5,
) -> ^Layer_Norm(T) {
	return new_layer_norm(T, []uint{embed_dim}, allocator, eps)
}

// Common convenience constructor for 2D LayerNorm (typically for spatial normalization)
// Normalizes over spatial dimensions (H, W) for input shape (N, C, H, W)
new_layer_norm_2d :: proc(
	$T: typeid,
	normalized_shape: []uint, // Typically [H, W] for spatial normalization
	allocator := context.allocator,
	eps := 1e-5,
) -> ^Layer_Norm(T) {
	return new_layer_norm(T, normalized_shape, allocator, eps)
}
