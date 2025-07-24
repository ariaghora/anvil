package transformer

import "../nn"
import "../tensor"
import "core:math"
import "core:math/rand"
import "core:testing"

@(test)
test_conv_2d_bn :: proc(t: ^testing.T) {
	// Test Conv2dBN basic functionality
	layer := new_conv_2d_bn(f32, 3, 64, 3, 1, 1, 1, context.temp_allocator)

	// Input: (1, 3, 32, 32)
	input_data := make([]f32, 1 * 3 * 32 * 32, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32() * 2 - 1 // Random values [-1, 1]
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, 32, 32}, context.temp_allocator)

	// Forward pass
	output := forward_conv_2d_bn(layer, input, context.temp_allocator)

	// Check output shape: should be (1, 64, 32, 32) with padding=1
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")
	testing.expect(t, output.shape[1] == 64, "Output channels should be 64")
	testing.expect(t, output.shape[2] == 32, "Height should be preserved with padding=1")
	testing.expect(t, output.shape[3] == 32, "Width should be preserved with padding=1")

}

@(test)
test_patch_embed :: proc(t: ^testing.T) {
	// Test PatchEmbed
	pe := new_patch_embed(f32, 3, 64, context.temp_allocator)

	// Input: (1, 3, 256, 256) - typical after initial downsampling
	input_data := make([]f32, 1 * 3 * 256 * 256, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32()
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, 256, 256}, context.temp_allocator)

	// Forward pass
	output := forward_patch_embed(pe, input, context.temp_allocator)

	// Two conv layers with stride=2 each: 256 -> 128 -> 64
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")
	testing.expect(t, output.shape[1] == 64, "Output channels should be 64")
	testing.expect(t, output.shape[2] == 64, "Height should be 64 (256/4)")
	testing.expect(t, output.shape[3] == 64, "Width should be 64 (256/4)")

}

@(test)
test_mb_conv :: proc(t: ^testing.T) {
	// Test MBConv
	mb := new_mb_conv(f32, 64, 64, 4, context.temp_allocator)

	// Input: (1, 64, 32, 32)
	input_data := make([]f32, 1 * 64 * 32 * 32, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32()
	}
	input := tensor.new_with_init(input_data, []uint{1, 64, 32, 32}, context.temp_allocator)

	// Forward pass
	output := forward_mb_conv(mb, input, context.temp_allocator)

	// Should preserve shape due to residual connection
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")
	testing.expect(t, output.shape[1] == 64, "Channels should be preserved")
	testing.expect(t, output.shape[2] == 32, "Height should be preserved")
	testing.expect(t, output.shape[3] == 32, "Width should be preserved")

}

@(test)
test_attention :: proc(t: ^testing.T) {
	// Test Attention mechanism
	attn := new_attention(f32, 128, 32, 4, 1, [2]uint{7, 7}, context.temp_allocator)

	// Input: (1, 49, 128) - 7x7 patches with 128 channels
	input_data := make([]f32, 1 * 49 * 128, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32()
	}
	input := tensor.new_with_init(input_data, []uint{1, 49, 128}, context.temp_allocator)

	// Forward pass
	output := forward_attention(attn, input, context.temp_allocator)

	// Should preserve input shape
	testing.expect(t, len(output.shape) == 3, "Output should be 3D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")
	testing.expect(t, output.shape[1] == 49, "Sequence length should be preserved")
	testing.expect(t, output.shape[2] == 128, "Feature dimension should be preserved")

}

@(test)
test_tiny_vit_block :: proc(t: ^testing.T) {
	// Test TinyViT Block
	block := new_tiny_vit_block(f32, 128, [2]uint{16, 16}, 4, 7, context.temp_allocator)

	// Input: (1, 256, 128) - 16x16 patches with 128 channels
	input_data := make([]f32, 1 * 256 * 128, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32()
	}
	input := tensor.new_with_init(input_data, []uint{1, 256, 128}, context.temp_allocator)

	// Forward pass
	output := forward_tiny_vit_block(block, input, context.temp_allocator)

	// Should preserve input shape
	testing.expect(t, len(output.shape) == 3, "Output should be 3D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")
	testing.expect(t, output.shape[1] == 256, "Sequence length should be preserved")
	testing.expect(t, output.shape[2] == 128, "Feature dimension should be preserved")

}

@(test)
test_tiny_vit_5m_integration :: proc(t: ^testing.T) {
	// Full integration test of TinyViT-5M

	// Create model
	model := new_tiny_vit_5m(f32, 256, context.temp_allocator)

	// Create input: (1, 3, 256, 256)
	input_size := 1 * 3 * 256 * 256
	input_data := make([]f32, input_size, context.temp_allocator)

	// Initialize with small random values to avoid numerical issues
	for i in 0 ..< len(input_data) {
		input_data[i] = (rand.float32() - 0.5) * 0.1 // Small values [-0.05, 0.05]
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, 256, 256}, context.temp_allocator)


	// Step-by-step forward pass
	xs := forward_patch_embed(model.patch_embed, input, context.temp_allocator)

	xs = forward_conv_layer(model.layer0, xs, context.temp_allocator)

	for i in 0 ..< len(model.layers) {
		layer := model.layers[i]
		xs = forward_basic_layer(layer, xs, context.temp_allocator)
	}

	// Neck: reshape to 4D and apply convolutions
	b := xs.shape[0]
	c := xs.shape[2]

	// Calculate expected spatial dimensions based on the output
	spatial_size := xs.shape[1] // sequence length
	spatial_dim := uint(math.sqrt(f64(spatial_size))) // assume square

	// Reshape (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
	xs_4d := tensor.reshape(xs, []uint{b, spatial_dim, spatial_dim, c}, context.temp_allocator)

	xs_conv := tensor.permute(xs_4d, []uint{0, 3, 1, 2}, context.temp_allocator)

	// Apply neck convolutions with layer norms
	conv1_out := nn.forward_conv2d(model.neck_conv1, xs_conv, context.temp_allocator)

	ln1_out := nn.forward_layer_norm(model.neck_ln1, conv1_out, context.temp_allocator)

	conv2_out := nn.forward_conv2d(model.neck_conv2, ln1_out, context.temp_allocator)

	output := nn.forward_layer_norm(model.neck_ln2, conv2_out, context.temp_allocator)

	// Check output shape
	testing.expect(t, len(output.shape) == 4, "Output should be 4D")
	testing.expect(t, output.shape[0] == 1, "Batch dimension should be 1")

	// Check that output contains reasonable values (not NaN/Inf)
	has_valid_values := true
	sample_count := min(100, len(output.data))
	for i in 0 ..< sample_count {
		val := output.data[i]
		// Check for NaN (val != val) or if value is too extreme
		if val != val || val > 1e10 || val < -1e10 {
			has_valid_values = false
			break
		}
	}
	testing.expect(t, has_valid_values, "Output should contain valid float values")

	// Print some statistics
	min_val := output.data[0]
	max_val := output.data[0]
	sum_val: f32 = 0

	for val in output.data {
		if val < min_val do min_val = val
		if val > max_val do max_val = val
		sum_val += val
	}
	mean_val := sum_val / f32(len(output.data))
}

@(test)
test_component_shapes :: proc(t: ^testing.T) {
	// Test that all components handle shape transformations correctly

	// Test patch embedding: 1024x1024 -> 256x256 -> 64x64
	pe := new_patch_embed(f32, 3, 64, context.temp_allocator)

	// Start with smaller input for faster testing
	input_data := make([]f32, 1 * 3 * 256 * 256, context.temp_allocator)
	for i in 0 ..< len(input_data) {
		input_data[i] = rand.float32() * 0.1
	}
	input := tensor.new_with_init(input_data, []uint{1, 3, 256, 256}, context.temp_allocator)
	pe_out := forward_patch_embed(pe, input, context.temp_allocator)

	// Test conv layer
	conv_layer := new_conv_layer(f32, 64, 128, [2]uint{64, 64}, 2, true, 4, context.temp_allocator)
	conv_out := forward_conv_layer(conv_layer, pe_out, context.temp_allocator)

	// Test basic layer
	basic_layer := new_basic_layer(
		f32,
		128,
		160,
		[2]uint{32, 32},
		2,
		4,
		7,
		true,
		context.temp_allocator,
	)
	basic_out := forward_basic_layer(basic_layer, conv_out, context.temp_allocator)
}
