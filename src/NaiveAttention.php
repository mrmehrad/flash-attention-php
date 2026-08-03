<?php

declare(strict_types=1);

namespace FlashAttention;

/**
 * Textbook O(N^2) attention: materializes the full (N x M) score matrix.
 * Used only as a correctness oracle for FlashAttention's tests/benchmarks
 * -- this is exactly the memory blow-up FlashAttention avoids.
 */
final class NaiveAttention
{
    public function __construct(
        private bool $causal = false,
        private ?float $scale = null,
    ) {
    }

    public function forward(Matrix $q, Matrix $k, Matrix $v): Matrix
    {
        $n = $q->rows;
        $m = $k->rows;
        $d = $q->cols;
        $dv = $v->cols;
        $scale = $this->scale ?? (1.0 / sqrt((float) $d));

        $scores = Matrix::zeros($n, $m);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $m; $j++) {
                if ($this->causal && $j > $i) {
                    $scores->set($i, $j, -INF);
                    continue;
                }
                $sum = 0.0;
                for ($p = 0; $p < $d; $p++) {
                    $sum += $q->get($i, $p) * $k->get($j, $p);
                }
                $scores->set($i, $j, $sum * $scale);
            }
        }

        $output = Matrix::zeros($n, $dv);
        for ($i = 0; $i < $n; $i++) {
            $rowMax = -INF;
            for ($j = 0; $j < $m; $j++) {
                $val = $scores->get($i, $j);
                if ($val > $rowMax) {
                    $rowMax = $val;
                }
            }
            $sum = 0.0;
            $weights = array_fill(0, $m, 0.0);
            for ($j = 0; $j < $m; $j++) {
                $val = $scores->get($i, $j);
                $e = $val === -INF ? 0.0 : exp($val - $rowMax);
                $weights[$j] = $e;
                $sum += $e;
            }
            if ($sum > 0.0) {
                for ($j = 0; $j < $m; $j++) {
                    $weights[$j] /= $sum;
                }
            }
            for ($p = 0; $p < $dv; $p++) {
                $acc = 0.0;
                for ($j = 0; $j < $m; $j++) {
                    $acc += $weights[$j] * $v->get($j, $p);
                }
                $output->set($i, $p, $acc);
            }
        }

        return $output;
    }
}
