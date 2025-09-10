package matmul

import "core:slice"

when ODIN_OS == .Darwin {
	foreign import blas "system:Accelerate.framework"
} else when ODIN_OS == .Linux {
	foreign import blas "system:openblas"
}

CBLAS_ORDER :: enum i32 {
	RowMajor = 101,
	ColMajor = 102,
}

CBLAS_TRANSPOSE :: enum i32 {
	NoTrans   = 111,
	Trans     = 112,
	ConjTrans = 113,
}

@(default_calling_convention = "c")
foreign blas {
	@(link_name = "cblas_sgemm")
	sgemm :: proc(order: CBLAS_ORDER, transa, transb: CBLAS_TRANSPOSE, m, n, k: i32, alpha: f32, a: [^]f32, lda: i32, b: [^]f32, ldb: i32, beta: f32, c: [^]f32, ldc: i32) ---

	@(link_name = "cblas_dgemm")
	dgemm :: proc(order: CBLAS_ORDER, transa, transb: CBLAS_TRANSPOSE, m, n, k: i32, alpha: f64, a: [^]f64, lda: i32, b: [^]f64, ldb: i32, beta: f64, c: [^]f64, ldc: i32) ---
}

matmul_2d :: proc(a: []$T, b: []T, m, n, k: uint, c: []T, allocator := context.allocator) {
	assert(len(a) == int(m * k), "A matrix size mismatch")
	assert(len(b) == int(k * n), "B matrix size mismatch")


	when T == f32 {
		sgemm(
			CBLAS_ORDER.RowMajor,
			CBLAS_TRANSPOSE.NoTrans,
			CBLAS_TRANSPOSE.NoTrans,
			i32(m),
			i32(n),
			i32(k),
			1.0,
			raw_data(a),
			i32(k),
			raw_data(b),
			i32(n),
			0.0,
			raw_data(c),
			i32(n),
		)
	} else when T == f64 {
		dgemm(
			CBLAS_ORDER.RowMajor,
			CBLAS_TRANSPOSE.NoTrans,
			CBLAS_TRANSPOSE.NoTrans,
			i32(m),
			i32(n),
			i32(k),
			1.0,
			raw_data(a),
			i32(k),
			raw_data(b),
			i32(n),
			0.0,
			raw_data(c),
			i32(n),
		)
	} else {
		#panic("matmul only supports f32 and f64")
	}
}
