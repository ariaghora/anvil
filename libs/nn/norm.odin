package nn

import "../tensor"
import "core:math"


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
	x: ^tensor.Tensor(T), // Input: (N, C, H, W)
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
	// Validate input shape
	if len(x.shape) != 4 {
		panic("BatchNorm2d input must be 4D tensor (N, C, H, W)")
	}

	if x.shape[1] != bn.num_features {
		panic("Input channels mismatch with num_features")
	}

	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]

	// Reshape tensors for broadcasting: (C,) -> (1, C, 1, 1)
	mean_reshaped := tensor.reshape(bn.running_mean, []uint{1, c, 1, 1}, context.temp_allocator)
	var_reshaped := tensor.reshape(bn.running_var, []uint{1, c, 1, 1}, context.temp_allocator)
	weight_reshaped := tensor.reshape(bn.weight, []uint{1, c, 1, 1}, context.temp_allocator)
	bias_reshaped := tensor.reshape(bn.bias, []uint{1, c, 1, 1}, context.temp_allocator)

	// Compute: (x - mean) / sqrt(var + eps) * weight + bias
	// Step 1: x - mean
	x_centered := tensor.sub(x, mean_reshaped, context.temp_allocator)

	// Step 2: sqrt(var + eps)
	eps_tensor := tensor.new_with_init([]T{bn.eps}, []uint{1, 1, 1, 1}, context.temp_allocator)
	var_eps := tensor.add(var_reshaped, eps_tensor, context.temp_allocator)
	std := tensor.sqrt(var_eps, context.temp_allocator)

	// Step 3: (x - mean) / sqrt(var + eps)
	x_normalized := tensor.div(x_centered, std, context.temp_allocator)

	// Step 4: * weight + bias
	x_scaled := tensor.mul(x_normalized, weight_reshaped, context.temp_allocator)
	result := tensor.add(x_scaled, bias_reshaped, allocator, loc)

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
	// For input shape (..., normalized_shape[0])
	// Compute mean and variance over the last dimension
	last_dim := len(x.shape) - 1

	// Compute mean: sum over last dimension and divide by size
	mean := tensor.tensor_mean(x, last_dim, true, context.temp_allocator)

	// Compute variance: E[(x - mean)^2]
	x_centered := tensor.sub(x, mean, context.temp_allocator)
	x_squared := tensor.mul(x_centered, x_centered, context.temp_allocator)
	variance := tensor.tensor_mean(x_squared, last_dim, true, context.temp_allocator)

	// Add eps and take sqrt
	eps_tensor := tensor.new_with_init([]T{ln.eps}, []uint{1}, context.temp_allocator)
	var_eps := tensor.add(variance, eps_tensor, context.temp_allocator)
	std := tensor.sqrt(var_eps, context.temp_allocator)

	// Normalize: (x - mean) / std
	x_norm := tensor.div(x_centered, std, context.temp_allocator)

	// Scale and shift: x_norm * weight + bias
	x_scaled := tensor.mul(x_norm, ln.weight, context.temp_allocator)
	result := tensor.add(x_scaled, ln.bias, allocator, loc)

	return result
}


// Helper for multi-dimensional LayerNorm
forward_layer_norm_nd :: proc(
	ln: ^Layer_Norm($T),
	x: ^tensor.Tensor(T),
	allocator := context.allocator,
	loc := #caller_location,
) -> ^tensor.Tensor(T) {
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
