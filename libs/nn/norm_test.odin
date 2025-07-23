package nn

import "../tensor"
import "core:testing"

@(test)
test_batch_norm_2d :: proc(t: ^testing.T) {
	// Test BatchNorm2d creation and forward pass
	bn := new_batch_norm_2d(f32, 3, allocator=context.temp_allocator)
	
	// Check parameter shapes
	testing.expect(t, len(bn.weight.shape) == 1, "Weight should be 1D")
	testing.expect(t, bn.weight.shape[0] == 3, "Weight should have 3 elements")
	testing.expect(t, bn.num_features == 3, "Should have 3 features")
	
	// Create input tensor: (2, 3, 4, 4)
	input_data := make([]f32, 96, context.temp_allocator) // 2*3*4*4 = 96
	for i in 0..<96 {
		input_data[i] = f32(i + 1)
	}
	input := tensor.new_with_init(input_data, []uint{2, 3, 4, 4}, context.temp_allocator)
	
	// Forward pass
	output := forward_batch_norm_2d(bn, input, context.temp_allocator)
	
	// Check output shape matches input
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	for i in 0..<len(input.shape) {
		testing.expect(t, output.shape[i] == input.shape[i], "Output shape should match input")
	}
}

@(test)
test_batch_norm_1d :: proc(t: ^testing.T) {
	// Test BatchNorm1d creation and forward pass
	bn := new_batch_norm_1d(f32, 5, allocator=context.temp_allocator)
	
	// Check parameter shapes
	testing.expect(t, len(bn.weight.shape) == 1, "Weight should be 1D")
	testing.expect(t, bn.weight.shape[0] == 5, "Weight should have 5 elements")
	
	// Create input tensor: (3, 5)
	input_data := make([]f32, 15, context.temp_allocator) // 3*5 = 15
	for i in 0..<15 {
		input_data[i] = f32(i + 1) * 0.1
	}
	input := tensor.new_with_init(input_data, []uint{3, 5}, context.temp_allocator)
	
	// Forward pass
	output := forward_batch_norm_1d(bn, input, context.temp_allocator)
	
	// Check output shape matches input
	testing.expect(t, len(output.shape) == 2, "Output should be 2D")
	for i in 0..<len(input.shape) {
		testing.expect(t, output.shape[i] == input.shape[i], "Output shape should match input")
	}
}

@(test)
test_layer_norm :: proc(t: ^testing.T) {
	// Test LayerNorm creation and forward pass (ViT case)
	ln := new_layer_norm_1d(f32, 4, allocator=context.temp_allocator)
	
	// Check parameter shapes
	testing.expect(t, len(ln.weight.shape) == 1, "Weight should be 1D") 
	testing.expect(t, ln.weight.shape[0] == 4, "Weight should have 4 elements")
	testing.expect(t, len(ln.normalized_shape) == 1, "Should normalize over 1 dimension")
	testing.expect(t, ln.normalized_shape[0] == 4, "Should normalize over last 4 elements")
	
	// Create input tensor: (2, 3, 4) - batch_size=2, seq_len=3, embed_dim=4
	input_data := make([]f32, 24, context.temp_allocator) // 2*3*4 = 24
	for i in 0..<24 {
		input_data[i] = f32(i + 1) * 0.1
	}
	input := tensor.new_with_init(input_data, []uint{2, 3, 4}, context.temp_allocator)
	
	// Forward pass
	output := forward_layer_norm(ln, input, context.temp_allocator)
	
	// Check output shape matches input
	testing.expect(t, len(output.shape) == 3, "Output should be 3D")
	for i in 0..<len(input.shape) {
		testing.expect(t, output.shape[i] == input.shape[i], "Output shape should match input")
	}
}

@(test)
test_layer_norm_2d :: proc(t: ^testing.T) {
	// Test LayerNorm with 2D input (N, C)
	ln := new_layer_norm_1d(f32, 5, allocator=context.temp_allocator)
	
	// Create input tensor: (3, 5)
	input_data := []f32{
		1.0, 2.0, 3.0, 4.0, 5.0,
		6.0, 7.0, 8.0, 9.0, 10.0,
		11.0, 12.0, 13.0, 14.0, 15.0,
	}
	input := tensor.new_with_init(input_data, []uint{3, 5}, context.temp_allocator)
	
	// Forward pass
	output := forward_layer_norm(ln, input, context.temp_allocator)
	
	// Check output shape
	testing.expect(t, len(output.shape) == 2, "Output should be 2D")
	testing.expect(t, output.shape[0] == 3, "Batch dimension should be preserved")
	testing.expect(t, output.shape[1] == 5, "Feature dimension should be preserved")
}

@(test)
test_layer_norm_multi_dim :: proc(t: ^testing.T) {
	// Test LayerNorm with multi-dimensional normalized_shape
	// Input: (2, 3, 4, 4), normalize over last 2 dims (4, 4)
	ln := new_layer_norm_2d(f32, []uint{4, 4}, allocator=context.temp_allocator)
	
	// Create input tensor: (2, 3, 4, 4)
	input_data := make([]f32, 96, context.temp_allocator) // 2*3*4*4 = 96
	for i in 0..<96 {
		input_data[i] = f32(i + 1) * 0.1
	}
	input := tensor.new_with_init(input_data, []uint{2, 3, 4, 4}, context.temp_allocator)
	
	// Forward pass
	output := forward_layer_norm(ln, input, context.temp_allocator)
	
	// Check output shape matches input
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	for i in 0..<len(input.shape) {
		testing.expect(t, output.shape[i] == input.shape[i], "Output shape should match input")
	}
}