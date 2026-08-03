<?php

declare(strict_types=1);

namespace FlashAttention\Kernel;

use FlashAttention\Matrix;

/**
 * Pure-PHP reference kernel. No extensions required.
 *
 * Optimizations applied (all measurable in PHP, unlike in a compiled
 * language where the compiler would do most of this automatically):
 *  - flat SplFixedArray storage instead of array-of-arrays (see Matrix.php)
 *  - local variable caching of ->data and dimensions to avoid repeated
 *    property lookups inside hot loops
 *  - loop order chosen so the innermost loop walks contiguous memory
 *    (row-major access) to keep OPcache/CPU cache behaviour reasonable
 *  - accumulate into a plain float local before writing back to the
 *    SplFixedArray once per element, instead of read-modify-write on it
 */
final class PhpKernel implements KernelInterface
{
    public function matMulTransposeB(Matrix $a, Matrix $b): Matrix
    {
        $m = $a->rows;
        $k = $a->cols;
        $n = $b->rows;

        if ($b->cols !== $k) {
            throw new \InvalidArgumentException(
                "Shape mismatch in matMulTransposeB: A is {$a->rows}x{$a->cols}, B is {$b->rows}x{$b->cols}"
            );
        }

        $out = new Matrix($m, $n);
        $ad = $a->data;
        $bd = $b->data;
        $od = $out->data;

        for ($i = 0; $i < $m; $i++) {
            $aRowBase = $i * $k;
            $outRowBase = $i * $n;
            for ($j = 0; $j < $n; $j++) {
                $bRowBase = $j * $k;
                $sum = 0.0;
                for ($p = 0; $p < $k; $p++) {
                    $sum += $ad[$aRowBase + $p] * $bd[$bRowBase + $p];
                }
                $od[$outRowBase + $j] = $sum;
            }
        }

        $out->data = $od;
        return $out;
    }

    public function matMul(Matrix $a, Matrix $b): Matrix
    {
        $m = $a->rows;
        $k = $a->cols;
        $n = $b->cols;

        if ($b->rows !== $k) {
            throw new \InvalidArgumentException(
                "Shape mismatch in matMul: A is {$a->rows}x{$a->cols}, B is {$b->rows}x{$b->cols}"
            );
        }

        $out = new Matrix($m, $n);
        $ad = $a->data;
        $bd = $b->data;
        $od = $out->data;

        // Initialize accumulator row buffer once, reused per i to avoid
        // repeated SplFixedArray offset writes during accumulation.
        $rowAcc = array_fill(0, $n, 0.0);

        for ($i = 0; $i < $m; $i++) {
            $aRowBase = $i * $k;
            for ($j = 0; $j < $n; $j++) {
                $rowAcc[$j] = 0.0;
            }
            for ($p = 0; $p < $k; $p++) {
                $aVal = $ad[$aRowBase + $p];
                if ($aVal === 0.0) {
                    continue;
                }
                $bRowBase = $p * $n;
                for ($j = 0; $j < $n; $j++) {
                    $rowAcc[$j] += $aVal * $bd[$bRowBase + $j];
                }
            }
            $outRowBase = $i * $n;
            for ($j = 0; $j < $n; $j++) {
                $od[$outRowBase + $j] = $rowAcc[$j];
            }
        }

        $out->data = $od;
        return $out;
    }
}
