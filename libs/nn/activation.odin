package nn

import "core:math"


// Gelu with tanh approximation
gelu :: #force_inline proc($T: typeid, x: T) -> T {
	sqrt_2_over_pi: T
	when T == f32 || T == f64 || T == f16 {
		sqrt_2_over_pi = math.sqrt(T(2.0) / math.PI)
	} else {
		#panic("GELU only supports f16, f32, f64")
	}

	inner := sqrt_2_over_pi * (x + 0.044715 * x * x * x)
	return T(0.5) * x * (T(1.0) + math.tanh(inner))
}

relu :: #force_inline proc($T: typeid, x: T) -> T {
	return math.max(x, T(0))
}


import "core:testing"

@(test)
test_gelu :: proc(t: ^testing.T) {
	// TODO
}

@(test)
test_relu :: proc(t: ^testing.T) {
	assert(relu(i32, -2) == 0)
	assert(relu(i32, 2) == 2)
	assert(relu(f32, -2) == 0)
	assert(relu(f32, 2) == 2)
}
