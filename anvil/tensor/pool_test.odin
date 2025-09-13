package tensor

import "core:slice"
import "core:testing"

@(test)
test_max_pool_2d :: proc(t: ^testing.T) {
	// Test 2x2 pooling with stride 2, no padding (backward compatibility)
	{
		input := new_with_init(
			[]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			[]uint{1, 1, 4, 4}, // BCHW
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{2, 2}, 2, 0, context.temp_allocator)

		expected := []f32{6, 8, 14, 16}
		testing.expect(t, slice.equal(output.data, expected), "2x2 max pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 2, 2}), "Output shape mismatch")
	}

	// Test 3x3 pooling with stride 1, no padding
	{
		input := new_with_init(
			[]f32 {
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9,
				10,
				11,
				12,
				13,
				14,
				15,
				16,
				17,
				18,
				19,
				20,
				21,
				22,
				23,
				24,
				25,
			},
			[]uint{1, 1, 5, 5},
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{3, 3}, 1, 0, context.temp_allocator)

		expected := []f32{13, 14, 15, 18, 19, 20, 23, 24, 25}
		testing.expect(t, slice.equal(output.data, expected), "3x3 max pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 1, 3, 3}), "Output shape mismatch")
	}

	// Test multi-channel pooling
	{
		input := new_with_init(
			[]f32 {
				// Channel 0
				1,
				2,
				3,
				4,
				5,
				6,
				7,
				8,
				9,
				10,
				11,
				12,
				13,
				14,
				15,
				16,
				// Channel 1
				16,
				15,
				14,
				13,
				12,
				11,
				10,
				9,
				8,
				7,
				6,
				5,
				4,
				3,
				2,
				1,
			},
			[]uint{1, 2, 4, 4},
			context.temp_allocator,
		)

		output := max_pool_2d(input, [2]uint{2, 2}, 2, 0, context.temp_allocator)

		expected := []f32{6, 8, 14, 16, 16, 14, 8, 6}
		testing.expect(t, slice.equal(output.data, expected), "Multi-channel pooling failed")
		testing.expect(t, slice.equal(output.shape, []uint{1, 2, 2, 2}), "Output shape mismatch")
	}

	// Test with padding=1
	{
		input := new_with_init([]f32{1, 2, 3, 4}, []uint{1, 1, 2, 2}, context.temp_allocator)

		output := max_pool_2d(input, [2]uint{2, 2}, 1, 1, context.temp_allocator)

		// With padding=1, the 2x2 input becomes effectively 4x4 padded with -inf
		// Output should be 3x3
		expected := []f32{1, 2, 2, 3, 4, 4, 3, 4, 4}
		testing.expect(t, slice.equal(output.data, expected), "Pooling with padding failed")
		testing.expect(
			t,
			slice.equal(output.shape, []uint{1, 1, 3, 3}),
			"Output shape mismatch with padding",
		)
	}
}
