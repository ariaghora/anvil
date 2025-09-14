package simd_backend

// Platform-specific exp abstraction
when ODIN_OS == .Darwin {
	foreign import accelerate "system:Accelerate.framework"

	foreign accelerate {
		vvexpf :: proc(y: [^]f32, x: [^]f32, n: ^i32) ---
	}

	expf_4 :: proc(dst: ^#simd[4]f32, src: ^#simd[4]f32) {
		count := i32(4)
		vvexpf(cast([^]f32)dst, cast([^]f32)src, &count)
	}

	// Blast em all
	expf_bath :: proc(dst, src: []f32) {
		assert(len(dst) == len(src))
		n := i32(len(src))
		vvexpf(raw_data(dst), raw_data(src), &n)
	}
} else {
	// Fallback for other platforms
	expf_4 :: proc(dst: ^#simd[4]f32, src: ^#simd[4]f32) {
		dst^ = #simd[4]f32 {
			math.exp(simd.extract(src, 0)),
			math.exp(simd.extract(src, 1)),
			math.exp(simd.extract(src, 2)),
			math.exp(simd.extract(src, 3)),
		}
	}

	expf_batch :: proc(dst, src: []f32) {
		for i in 0 ..< len(src) {
			dst[i] = math.exp(src[i])
		}
	}
}
