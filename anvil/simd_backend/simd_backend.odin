package simd_backend

import "core:math"
import "core:simd"

when ODIN_ARCH == .amd64 {
	when #config(AVX512, false) {
		SIMD_LANES :: 16
	} else when #config(AVX2, false) {
		SIMD_LANES :: 8
	} else {
		SIMD_LANES :: 4
	}
} else when ODIN_ARCH == .arm64 {
	SIMD_LANES :: 4 // 128-bit NEON
} else {
	SIMD_LANES :: 4
}
SIMD_F32 :: #simd[SIMD_LANES]f32

when ODIN_OS == .Darwin {
	foreign import accelerate "system:Accelerate.framework"

	foreign accelerate {
		vvexpf :: proc(y: [^]f32, x: [^]f32, n: ^i32) ---
		vDSP_vsadd :: proc(A: [^]f32, stride_A: i32, B: ^f32, C: [^]f32, stride_C: i32, n: u32) ---
		vDSP_vadd :: proc(A: [^]f32, IA: i32, B: [^]f32, IB: i32, C: [^]f32, IC: i32, N: u32) ---
		vDSP_mtrans :: proc(A: [^]f32, IA: i32, C: [^]f32, IC: i32, M: u32, N: u32) ---
		vDSP_vmax :: proc(A: [^]f32, IA: i32, B: [^]f32, IB: i32, C: [^]f32, IC: i32, N: u32) ---
		vDSP_maximum :: proc(A: [^]f32) -> f32 ---
		vDSP_sve :: proc(A: [^]f32, stride_A: i32, C: ^f32, n: u32) ---
	}

	expf_4 :: proc(dst: ^#simd[4]f32, src: ^#simd[4]f32) {
		count := i32(4)
		vvexpf(cast([^]f32)dst, cast([^]f32)src, &count)
	}

	addf_batch :: proc(dst, a, b: []f32) {
		vDSP_vadd(raw_data(a), 1, raw_data(b), 1, raw_data(dst), 1, u32(len(a)))
	}

	vsaddf_batch :: proc(dst, a: []f32, b: ^f32) {
		vDSP_vsadd(raw_data(a), 1, b, raw_data(dst), 1, u32(len(a)))
	}

	expf_batch :: proc(dst, src: []f32) {
		assert(len(dst) == len(src))
		n := i32(len(src))
		vvexpf(raw_data(dst), raw_data(src), &n)
	}

	maxf_batch :: proc(a, b, out: []f32) {
		vDSP_vmax(raw_data(a), 1, raw_data(b), 1, raw_data(out), 1, u32(len(a)))
	}

	transposef :: proc(dst, src: []f32, rows, cols: uint) {
		vDSP_mtrans(raw_data(src), i32(cols), raw_data(dst), i32(rows), u32(rows), u32(cols))
	}

} else {
	// Fallback for other platforms
	expf_4 :: proc(dst: ^#simd[4]f32, src: ^#simd[4]f32) {
		dst^ = #simd[4]f32 {
			math.exp(simd.extract(src^, 0)),
			math.exp(simd.extract(src^, 1)),
			math.exp(simd.extract(src^, 2)),
			math.exp(simd.extract(src^, 3)),
		}
	}

	expf_batch :: proc(dst, src: []f32) {
		for i in 0 ..< len(src) {
			dst[i] = math.exp(src[i])
		}
	}
}

@(private = "file")
splat_f32x4 :: #force_inline proc(val: f32) -> #simd[4]f32 {
	return #simd[4]f32{val, val, val, val}
}
@(private = "file")
splat_f32x8 :: #force_inline proc(val: f32) -> #simd[8]f32 {
	return #simd[8]f32{val, val, val, val, val, val, val, val}
}

splat :: #force_inline proc($T: typeid, val: f32) -> T {
	when T == #simd[4]f32 {
		return splat_f32x4(val)
	} else when T == #simd[8]f32 {
		return splat_f32x8(val)
	} else when T == #simd[16]f32 {
		return splat_f32x16(val) // TODO
	} else {
		#panic("Unsupported SIMD type")
	}
}

max_f32 :: proc(va, vb: #simd[4]f32) -> #simd[4]f32 {
	// NOTE(Aria): simd.max(segfaults in WASM)
	// This is just a workaround
	when ODIN_OS == .Darwin || ODIN_OS == .Windows || ODIN_OS == .Linux {
		return simd.max(va, vb)
	} else {
		return #simd[4]f32 {
			max(simd.extract(va, 0), simd.extract(vb, 0)),
			max(simd.extract(va, 1), simd.extract(vb, 1)),
			max(simd.extract(va, 2), simd.extract(vb, 2)),
			max(simd.extract(va, 3), simd.extract(vb, 3)),
		}
	}
}
