<?php

declare(strict_types=1);

namespace FlashAttention\Kernel;

use FlashAttention\Matrix;

/**
 * FFI-backed kernel that calls into a small compiled C shared library
 * (src/Ffi/kernel.c, built by src/Ffi/build.sh) for the two matmuls
 * FlashAttention needs per tile. This is where the real wall-clock
 * speedup over PhpKernel comes from -- the tiling/online-softmax
 * algorithm in FlashAttention.php is unchanged either way.
 *
 * Falls back gracefully: construct with a $libraryPath; if the ext-ffi
 * extension isn't loaded or the .so isn't built yet, throw immediately
 * so callers can catch it and fall back to PhpKernel (see
 * FlashAttention::withBestAvailableKernel()).
 */
final class FfiKernel implements KernelInterface
{
    private \FFI $ffi;

    public function __construct(?string $libraryPath = null)
    {
        if (!extension_loaded('ffi')) {
            throw new \RuntimeException('The FFI extension is not loaded.');
        }

        $libraryPath ??= self::defaultLibraryPath();

        if (!is_file($libraryPath)) {
            throw new \RuntimeException(
                "FFI kernel library not found at {$libraryPath}. Run src/Ffi/build.sh first."
            );
        }

        $this->ffi = \FFI::cdef(
            <<<'CDEF'
                void fa_matmul_at_bt(const double *a, const double *b, double *out, long m, long k, long n);
                void fa_matmul(const double *a, const double *b, double *out, long m, long k, long n);
                CDEF,
            $libraryPath
        );
    }

    public static function defaultLibraryPath(): string
    {
        $base = __DIR__ . '/../Ffi/libflashkernel';
        return PHP_OS_FAMILY === 'Darwin' ? "{$base}.dylib" : "{$base}.so";
    }

    public static function isAvailable(?string $libraryPath = null): bool
    {
        if (!extension_loaded('ffi')) {
            return false;
        }
        $libraryPath ??= self::defaultLibraryPath();
        return is_file($libraryPath);
    }

    public function matMulTransposeB(Matrix $a, Matrix $b): Matrix
    {
        if ($b->cols !== $a->cols) {
            throw new \InvalidArgumentException(
                "Shape mismatch in matMulTransposeB: A is {$a->rows}x{$a->cols}, B is {$b->rows}x{$b->cols}"
            );
        }

        $m = $a->rows;
        $k = $a->cols;
        $n = $b->rows;

        [$aBuf, $bBuf, $outBuf] = $this->toCBuffers($a, $b, $m * $n);
        $this->ffi->fa_matmul_at_bt($aBuf, $bBuf, $outBuf, $m, $k, $n);

        return $this->fromCBuffer($outBuf, $m, $n);
    }

    public function matMul(Matrix $a, Matrix $b): Matrix
    {
        if ($b->rows !== $a->cols) {
            throw new \InvalidArgumentException(
                "Shape mismatch in matMul: A is {$a->rows}x{$a->cols}, B is {$b->rows}x{$b->cols}"
            );
        }

        $m = $a->rows;
        $k = $a->cols;
        $n = $b->cols;

        [$aBuf, $bBuf, $outBuf] = $this->toCBuffers($a, $b, $m * $n);
        $this->ffi->fa_matmul($aBuf, $bBuf, $outBuf, $m, $k, $n);

        return $this->fromCBuffer($outBuf, $m, $n);
    }

    /**
     * Allocate flat (1-D) C double buffers and copy the matrices in.
     * Flat buffers are used (rather than double[rows][cols]) because a
     * 2-D FFI CData does not decay to a pointer the way a real C array
     * does -- casting it to double* is undefined and segfaults. A flat
     * buffer with manual row*cols + col indexing is what the C side
     * (kernel.c) expects anyway, since it's already written in terms of
     * flat row-major pointers.
     *
     * @return array{0: \FFI\CData, 1: \FFI\CData, 2: \FFI\CData}
     */
    private function toCBuffers(Matrix $a, Matrix $b, int $outSize): array
    {
        $aSize = $a->rows * $a->cols;
        $aBuf = \FFI::new("double[{$aSize}]", false);
        for ($i = 0; $i < $aSize; $i++) {
            $aBuf[$i] = $a->data[$i];
        }

        $bSize = $b->rows * $b->cols;
        $bBuf = \FFI::new("double[{$bSize}]", false);
        for ($i = 0; $i < $bSize; $i++) {
            $bBuf[$i] = $b->data[$i];
        }

        $outBuf = \FFI::new("double[{$outSize}]", false);

        return [$aBuf, $bBuf, $outBuf];
    }

    private function fromCBuffer(\FFI\CData $buf, int $rows, int $cols): Matrix
    {
        $out = new Matrix($rows, $cols);
        $n = $rows * $cols;
        for ($i = 0; $i < $n; $i++) {
            $out->data[$i] = $buf[$i];
        }
        return $out;
    }
}
