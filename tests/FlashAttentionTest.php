<?php

declare(strict_types=1);

namespace FlashAttention\Tests;

use FlashAttention\FlashAttention;
use FlashAttention\Matrix;
use FlashAttention\NaiveAttention;
use PHPUnit\Framework\TestCase;

final class FlashAttentionTest extends TestCase
{
    private function randomMatrix(int $rows, int $cols, int $seed): Matrix
    {
        mt_srand($seed);
        $m = new Matrix($rows, $cols);
        for ($i = 0; $i < $rows * $cols; $i++) {
            $m->data[$i] = mt_rand(-1000, 1000) / 1000.0;
        }
        return $m;
    }

    private function maxAbsDiff(Matrix $a, Matrix $b): float
    {
        $this->assertSame($a->rows, $b->rows);
        $this->assertSame($a->cols, $b->cols);
        $max = 0.0;
        for ($i = 0, $n = $a->rows * $a->cols; $i < $n; $i++) {
            $max = max($max, abs($a->data[$i] - $b->data[$i]));
        }
        return $max;
    }

    #[\PHPUnit\Framework\Attributes\DataProvider("shapeProvider")]
    public function testMatchesNaiveAttention(int $n, int $m, int $d, int $dv, bool $causal, int $br, int $bc): void
    {
        $q = $this->randomMatrix($n, $d, 10 + $n + $d);
        $k = $this->randomMatrix($m, $d, 20 + $m + $d);
        $v = $this->randomMatrix($m, $dv, 30 + $m + $dv);
        $scale = 1.0 / sqrt((float) $d);

        $flash = new FlashAttention($br, $bc, $causal, $scale);
        $naive = new NaiveAttention($causal, $scale);

        $diff = $this->maxAbsDiff($flash->forward($q, $k, $v), $naive->forward($q, $k, $v));
        $this->assertLessThan(1e-9, $diff);
    }

    public static function shapeProvider(): array
    {
        return [
            'small non-causal, block aligned' => [8, 8, 4, 4, false, 4, 4],
            'ragged sequence lengths' => [17, 13, 8, 6, false, 5, 4],
            'causal, block aligned' => [32, 32, 16, 16, true, 8, 8],
            'causal, misaligned block sizes' => [30, 30, 8, 8, true, 7, 11],
            'causal, single token' => [1, 1, 4, 4, true, 4, 4],
            'block size larger than sequence' => [5, 5, 4, 4, true, 64, 64],
        ];
    }

    public function testMultiHeadMatchesPerHeadNaiveAttention(): void
    {
        $numHeads = 4;
        $headDim = 8;
        $n = 20;

        $q = $this->randomMatrix($n, $numHeads * $headDim, 1);
        $k = $this->randomMatrix($n, $numHeads * $headDim, 2);
        $v = $this->randomMatrix($n, $numHeads * $headDim, 3);

        $flash = new FlashAttention(blockSizeQ: 6, blockSizeKv: 6, causal: true);
        $multiHeadOut = $flash->forwardMultiHead($q, $k, $v, $numHeads);

        $naive = new NaiveAttention(causal: true);

        for ($h = 0; $h < $numHeads; $h++) {
            $qh = new Matrix($n, $headDim);
            $kh = new Matrix($n, $headDim);
            $vh = new Matrix($n, $headDim);
            for ($i = 0; $i < $n; $i++) {
                for ($c = 0; $c < $headDim; $c++) {
                    $qh->set($i, $c, $q->get($i, $h * $headDim + $c));
                    $kh->set($i, $c, $k->get($i, $h * $headDim + $c));
                    $vh->set($i, $c, $v->get($i, $h * $headDim + $c));
                }
            }

            $expected = $naive->forward($qh, $kh, $vh);

            for ($i = 0; $i < $n; $i++) {
                for ($c = 0; $c < $headDim; $c++) {
                    $this->assertEqualsWithDelta(
                        $expected->get($i, $c),
                        $multiHeadOut->get($i, $h * $headDim + $c),
                        1e-9
                    );
                }
            }
        }
    }

    public function testRejectsMismatchedHeadDimensions(): void
    {
        $this->expectException(\InvalidArgumentException::class);

        $flash = new FlashAttention();
        $flash->forward(Matrix::zeros(4, 8), Matrix::zeros(4, 6), Matrix::zeros(4, 6));
    }
}
