package cholesky

import "../../tensor"
import "core:fmt"
import "core:math"
import "core:slice"

Decompose_Error :: enum {
	Cho_Nonsquare_Input,
	Cho_Non_PSD,
}

// Every symmetric PSD matrix X can be decomposed into a product
// of a unique lower triangular matrix and its transpose, X = LL'.
// We find L with cholesky decomposition.
cholesky_decompose :: proc(
	A: ^tensor.Tensor($T),
	allocator := context.allocator,
) -> (
	^tensor.Tensor(T),
	Decompose_Error,
) {
	n := A.shape[0]
	if n != A.shape[1] {
		return nil, .Cho_Nonsquare_Input // Not square
	}

	L := tensor.zeros(T, {n, n}, allocator)

	for i in 0 ..< n {
		for j in 0 ..= i {
			sum := T(0)

			// Sum L[i,k] * L[j,k] for k < j
			for k in 0 ..< j {
				sum += L.data[i * n + k] * L.data[j * n + k]
			}

			if i == j {
				// Diagonal element
				val := A.data[i * n + i] - sum
				if val <= 0 {
					// Matrix not positive definite
					tensor.free_tensor(L, allocator)
					return nil, .Cho_Non_PSD
				}
				L.data[i * n + i] = math.sqrt(val)
			} else {
				// Off-diagonal
				L.data[i * n + j] = (A.data[i * n + j] - sum) / L.data[j * n + j]
			}
		}
	}

	return L, nil
}

import "core:log"
import "core:testing"

@(test)
cholesky_decompose_test :: proc(t: ^testing.T) {
	context.allocator = context.temp_allocator

	// Test 1: Identity matrix
	iden := tensor.new_with_init([]f32{1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3})
	L, err_cho := cholesky_decompose(iden)
	testing.expect(t, err_cho == nil)
	LL_trans := tensor.matmul(L, tensor.matrix_transpose(L))
	all_same := slice.equal(LL_trans.data, iden.data)
	testing.expect(t, all_same)


	// Test 2: Covariance-like matrix (guaranteed positive semi-definite)
	X := tensor.randn(f32, {5, 3}, 0, 1) // 5 samples, 3 features
	K := tensor.matmul(tensor.transpose(X, 0, 1), X) // 3x3
	// Add jitter to diagonal for positive **semi-**definiteness
	for i in 0 ..< 3 do K.data[i * 3 + i] += 1e-3

	L, err_cho = cholesky_decompose(K)
	testing.expect(t, err_cho == nil)
	LL_trans = tensor.matmul(L, tensor.matrix_transpose(L))
	for _, i in LL_trans.data {
		testing.expect(t, abs(LL_trans.data[i] - K.data[i]) < 1e-5)
	}
}
