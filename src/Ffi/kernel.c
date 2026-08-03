/*
 * Minimal, dependency-free matmul kernel used by FlashAttention\Kernel\FfiKernel.
 *
 * Not a BLAS replacement -- just a tight, restrict-qualified C loop that
 * the C compiler can auto-vectorize (SSE2/AVX depending on -march), which
 * pure PHP cannot get anywhere near. Compiled to a shared library and
 * called from PHP via the FFI extension (no Zend extension build step
 * needed, which keeps the library composer-installable).
 *
 * Build: see build.sh in this directory.
 */

#include <stddef.h>

/* out[m x n] = a[m x k] * b[n x k]^T  (row-major) */
void fa_matmul_at_bt(
    const double *restrict a,
    const double *restrict b,
    double *restrict out,
    long m, long k, long n
) {
    for (long i = 0; i < m; i++) {
        const double *arow = a + i * k;
        double *orow = out + i * n;
        for (long j = 0; j < n; j++) {
            const double *brow = b + j * k;
            double sum = 0.0;
            for (long p = 0; p < k; p++) {
                sum += arow[p] * brow[p];
            }
            orow[j] = sum;
        }
    }
}

/* out[m x n] = a[m x k] * b[k x n]  (row-major) */
void fa_matmul(
    const double *restrict a,
    const double *restrict b,
    double *restrict out,
    long m, long k, long n
) {
    for (long i = 0; i < m; i++) {
        double *orow = out + i * n;
        for (long j = 0; j < n; j++) {
            orow[j] = 0.0;
        }
        const double *arow = a + i * k;
        for (long p = 0; p < k; p++) {
            double aval = arow[p];
            if (aval == 0.0) continue;
            const double *brow = b + p * n;
            for (long j = 0; j < n; j++) {
                orow[j] += aval * brow[j];
            }
        }
    }
}
