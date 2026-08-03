<?php

declare(strict_types=1);

namespace FlashAttention\Kernel;

use FlashAttention\Matrix;

/**
 * A Kernel provides the numeric primitives FlashAttention needs on each
 * tile. Swapping the kernel (pure PHP vs FFI/C) is how this library gets
 * "optimized" without changing the tiling/online-softmax algorithm itself.
 */
interface KernelInterface
{
    /** C = A * B^T, where A is (m x k) and B is (n x k), result is (m x n). */
    public function matMulTransposeB(Matrix $a, Matrix $b): Matrix;

    /** C = A * B, where A is (m x k) and B is (k x n), result is (m x n). */
    public function matMul(Matrix $a, Matrix $b): Matrix;
}
