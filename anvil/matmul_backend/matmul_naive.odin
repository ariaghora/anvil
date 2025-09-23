package matmul_backend

import "core:fmt"
import "core:simd"

gemm_generic :: proc(
	$T: typeid,
	order: CBLAS_ORDER,
	transa, transb: CBLAS_TRANSPOSE,
	m, n, k: i32,
	alpha: T,
	a: [^]T,
	lda: i32,
	b: [^]T,
	ldb: i32,
	beta: T,
	c: [^]T,
	ldc: i32,
) {
	assert(order == CBLAS_ORDER.RowMajor, "Only row-major order supported in naive implementation")

	if beta != 1.0 {
		for i in 0 ..< m {
			for j in 0 ..< n {
				c[i * ldc + j] *= beta
			}
		}
	}

	if transa == CBLAS_TRANSPOSE.NoTrans && transb == CBLAS_TRANSPOSE.NoTrans {
		for i in 0 ..< m {
			for j in 0 ..< n {
				sum: T = 0.0
				for l in 0 ..< k {
					sum += a[i * lda + l] * b[l * ldb + j]
				}
				c[i * ldc + j] += alpha * sum
			}
		}
	} else if transa == CBLAS_TRANSPOSE.NoTrans && transb == CBLAS_TRANSPOSE.Trans {
		for i in 0 ..< m {
			for j in 0 ..< n {
				sum: T = 0.0
				for l in 0 ..< k {
					sum += a[i * lda + l] * b[j * ldb + l]
				}
				c[i * ldc + j] += alpha * sum
			}
		}
	} else if transa == CBLAS_TRANSPOSE.Trans && transb == CBLAS_TRANSPOSE.NoTrans {
		for i in 0 ..< m {
			for j in 0 ..< n {
				sum: T = 0.0
				for l in 0 ..< k {
					sum += a[l * lda + i] * b[l * ldb + j]
				}
				c[i * ldc + j] += alpha * sum
			}
		}
	} else { 	// Both transposed
		for i in 0 ..< m {
			for j in 0 ..< n {
				sum: T = 0.0
				for l in 0 ..< k {
					sum += a[l * lda + i] * b[j * ldb + l]
				}
				c[i * ldc + j] += alpha * sum
			}
		}
	}
}

sgemm_simd :: proc(
	m, n, k: i32,
	alpha: f32,
	a: [^]f32,
	lda: i32,
	b: [^]f32,
	ldb: i32,
	beta: f32,
	c: [^]f32,
	ldc: i32,
) {
	if beta != 1.0 {
		for i in 0 ..< m {
			for j in 0 ..< n {
				c[i * ldc + j] *= beta
			}
		}
	}

	for i in 0 ..< m {
		j: i32 = 0
		for ; j + 4 <= n; j += 4 {
			sum_vec := #simd[4]f32{0, 0, 0, 0}

			for l in 0 ..< k {
				a_scalar := a[i * lda + l]
				a_vec := #simd[4]f32{a_scalar, a_scalar, a_scalar, a_scalar}

				b_base := l * ldb + j
				b_vec := #simd[4]f32{b[b_base], b[b_base + 1], b[b_base + 2], b[b_base + 3]}

				sum_vec += a_vec * b_vec
			}

			alpha_vec := #simd[4]f32{alpha, alpha, alpha, alpha}
			result_vec := alpha_vec * sum_vec

			c_base := i * ldc + j
			existing_c := (^#simd[4]f32)(&c[c_base])^
			final_c := existing_c + result_vec
			(^#simd[4]f32)(&c[c_base])^ = final_c
		}

		for ; j < n; j += 1 {
			sum: f32 = 0.0
			for l in 0 ..< k {
				sum += a[i * lda + l] * b[l * ldb + j]
			}
			c[i * ldc + j] += alpha * sum
		}
	}
}

sgemm_naive :: proc(
	order: CBLAS_ORDER,
	transa, transb: CBLAS_TRANSPOSE,
	m, n, k: i32,
	alpha: f32,
	a: [^]f32,
	lda: i32,
	b: [^]f32,
	ldb: i32,
	beta: f32,
	c: [^]f32,
	ldc: i32,
) {
	if transa == .NoTrans && transb == .NoTrans {
		sgemm_simd(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	gemm_generic(f32, order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

dgemm_naive :: proc(
	order: CBLAS_ORDER,
	transa, transb: CBLAS_TRANSPOSE,
	m, n, k: i32,
	alpha: f64,
	a: [^]f64,
	lda: i32,
	b: [^]f64,
	ldb: i32,
	beta: f64,
	c: [^]f64,
	ldc: i32,
) {
	if transa == .NoTrans && transb == .NoTrans {
		// TODO(Aria): implement dgemm_simd
	}
	gemm_generic(f64, order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
