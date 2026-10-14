<?php

declare(strict_types=1);

namespace FlashAttention;

use FlashAttention\Kernel\FfiKernel;
use FlashAttention\Kernel\KernelInterface;
use FlashAttention\Kernel\PhpKernel;

/**
 * FlashAttention v2 (Dao, 2023) forward pass, implemented in PHP.
 *
 * The point of FlashAttention is *not* a different math result from
 * standard softmax(QK^T/sqrt(d))V attention -- it's the same result,
 * computed without ever materializing the full (N x M) attention matrix
 * in memory. It does this by:
 *
 *   1. Tiling Q into row-blocks of size Br and K/V into row-blocks of
 *      size Bc, so at any moment only a (Br x Bc) score tile exists.
 *   2. Using the "online softmax" trick: instead of computing a full
 *      row's softmax in one pass (which needs the whole row), it keeps a
 *      running max (m), running sum (l), and running weighted output
 *      accumulator (O) per query row, and *rescales* them every time a
 *      new K/V block shifts the running max, via the identity
 *          softmax(x)_k = exp(x_k - m) / sum(exp(x_i - m))
 *      which holds for *any* m, so shifting m by a correction factor
 *      exp(m_old - m_new) keeps everything mathematically exact.
 *
 * This PHP port keeps that exact algorithm (see `forwardSingleHead`).
 * What "optimized" means here, since PHP has no GPU kernels to write,
 * is: (a) flat-buffer matrices instead of nested arrays (see Matrix.php),
 * (b) an optional FFI-compiled-C kernel for the O(block^2 * d) matmuls,
 * which is where nearly all the FLOPs live, and (c) avoiding
 * re-allocating tiles more than necessary. The block-level algorithm
 * itself is what gives the O(N) memory (vs O(N^2) for naive attention),
 * independent of which kernel executes the matmuls.
 */
final class FlashAttention
{
    /** @var positive-int number of query rows processed per tile (Br in the paper) */
    private int $blockSizeQ;

    /** @var positive-int number of key/value rows processed per tile (Bc in the paper) */
    private int $blockSizeKv;

    private bool $causal;

    /** Explicit scale override; when null, 1/sqrt(headDim) is used per call. */
    private ?float $userScale;

    private KernelInterface $kernel;

    public function __construct(
        int $blockSizeQ = 64,
        int $blockSizeKv = 64,
        bool $causal = false,
        ?float $scale = null,
        ?KernelInterface $kernel = null,
    ) {
        if ($blockSizeQ < 1 || $blockSizeKv < 1) {
            throw new \InvalidArgumentException('Block sizes must be positive integers.');
        }

        $this->blockSizeQ = $blockSizeQ;
        $this->blockSizeKv = $blockSizeKv;
        $this->causal = $causal;
        $this->userScale = $scale;
        $this->kernel = $kernel ?? new PhpKernel();
    }

    /**
     * Convenience constructor that picks the FFI kernel when available
     * (extension loaded + shared library built) and falls back to the
     * pure-PHP kernel otherwise, so the library works out of the box on
     * any PHP install while still benefiting from the compiled kernel
     * when the consumer has run src/Ffi/build.sh.
     */
    public static function withBestAvailableKernel(
        int $blockSizeQ = 64,
        int $blockSizeKv = 64,
        bool $causal = false,
        ?float $scale = null,
    ): self {
        $kernel = FfiKernel::isAvailable()
            ? new FfiKernel()
            : new PhpKernel();

        return new self($blockSizeQ, $blockSizeKv, $causal, $scale, $kernel);
    }

    public function usingKernel(): string
    {
        return get_class($this->kernel);
    }

    /**
     * Single-head forward pass.
     *
     * @param Matrix $q (N x d)
     * @param Matrix $k (M x d)
     * @param Matrix $v (M x dv)
     * @return Matrix (N x dv)
     */
    public function forward(Matrix $q, Matrix $k, Matrix $v): Matrix
    {
        if ($q->cols !== $k->cols) {
            throw new \InvalidArgumentException(
                "Q and K must share head dim: Q is {$q->rows}x{$q->cols}, K is {$k->rows}x{$k->cols}"
            );
        }
        if ($k->rows !== $v->rows) {
            throw new \InvalidArgumentException(
                "K and V must share sequence length: K is {$k->rows}x{$k->cols}, V is {$v->rows}x{$v->cols}"
            );
        }

        $scale = $this->userScale ?? (1.0 / sqrt((float) $q->cols));

        return $this->forwardSingleHead($q, $k, $v, $scale);
    }

    /**
     * Multi-head forward pass over interleaved-by-head Q/K/V.
     *
     * @param Matrix $q (N x numHeads*headDim)
     * @param Matrix $k (M x numHeads*headDim)
     * @param Matrix $v (M x numHeads*headDimV)
     * @return Matrix (N x numHeads*headDimV)
     */
    public function forwardMultiHead(Matrix $q, Matrix $k, Matrix $v, int $numHeads): Matrix
    {
        if ($numHeads < 1) {
            throw new \InvalidArgumentException('numHeads must be >= 1.');
        }
        if ($q->cols % $numHeads !== 0 || $k->cols % $numHeads !== 0 || $v->cols % $numHeads !== 0) {
            throw new \InvalidArgumentException('Q/K/V column counts must be divisible by numHeads.');
        }

        $headDim = intdiv($q->cols, $numHeads);
        $headDimV = intdiv($v->cols, $numHeads);
        $n = $q->rows;
        $scale = $this->userScale ?? (1.0 / sqrt((float) $headDim));

        $output = Matrix::zeros($n, $numHeads * $headDimV);

        for ($h = 0; $h < $numHeads; $h++) {
            $qh = $this->extractHead($q, $h, $headDim);
            $kh = $this->extractHead($k, $h, $headDim);
            $vh = $this->extractHead($v, $h, $headDimV);

            $oh = $this->forwardSingleHead($qh, $kh, $vh, $scale);
            $this->writeHead($output, $oh, $h, $headDimV);
        }

        return $output;
    }

    private function extractHead(Matrix $m, int $headIndex, int $headDim): Matrix
    {
        $out = new Matrix($m->rows, $headDim);
        $colOffset = $headIndex * $headDim;
        for ($i = 0; $i < $m->rows; $i++) {
            $srcBase = $i * $m->cols + $colOffset;
            $dstBase = $i * $headDim;
            for ($j = 0; $j < $headDim; $j++) {
                $out->data[$dstBase + $j] = $m->data[$srcBase + $j];
            }
        }
        return $out;
    }

    private function writeHead(Matrix $dst, Matrix $headOutput, int $headIndex, int $headDim): void
    {
        $colOffset = $headIndex * $headDim;
        for ($i = 0; $i < $dst->rows; $i++) {
            $dstBase = $i * $dst->cols + $colOffset;
            $srcBase = $i * $headDim;
            for ($j = 0; $j < $headDim; $j++) {
                $dst->data[$dstBase + $j] = $headOutput->data[$srcBase + $j];
            }
        }
    }

    /**
     * The actual FlashAttention v2 tiling + online-softmax algorithm.
     * See the class docblock for the mathematical explanation.
     */
    private function forwardSingleHead(Matrix $q, Matrix $k, Matrix $v, float $scale): Matrix
    {
        $n = $q->rows;
        $m = $k->rows;
        $dv = $v->cols;

        $br = min($this->blockSizeQ, $n);
        $bc = min($this->blockSizeKv, $m);

        $tr = (int) ceil($n / $br); // number of Q row-blocks
        $tc = (int) ceil($m / $bc); // number of K/V row-blocks

        $output = Matrix::zeros($n, $dv);
        $outData = $output->data;

        for ($i = 0; $i < $tr; $i++) {
            $iStart = $i * $br;
            $iLen = min($br, $n - $iStart);
            $qi = $q->sliceRows($iStart, $iLen);

            // Local running softmax statistics for this query block
            $mi = array_fill(0, $iLen, -INF);
            $li = array_fill(0, $iLen, 0.0);

            // Local output accumulator for this query block
            $oiData = new \SplFixedArray($iLen * $dv);
            for ($idx = 0, $cnt = $iLen * $dv; $idx < $cnt; $idx++) {
                $oiData[$idx] = 0.0;
            }

            for ($j = 0; $j < $tc; $j++) {
                $jStart = $j * $bc;
                $jLen = min($bc, $m - $jStart);

                // Causal masking works at block granularity: skip this
                // (i, j) tile entirely if every key index in the block
                // is greater than every query index in the block (fully
                // masked).
                if ($this->causal && $jStart > $iStart + $iLen - 1) {
                    continue;
                }

                $kj = $k->sliceRows($jStart, $jLen);
                $vj = $v->sliceRows($jStart, $jLen);

                // Sij = scale * Qi @ Kj^T, shape (iLen x jLen)
                $sij = $this->kernel->matMulTransposeB($qi, $kj);
                $sd = $sij->data;
                for ($idx = 0, $cnt = $iLen * $jLen; $idx < $cnt; $idx++) {
                    $sd[$idx] *= $scale;
                }

                if ($this->causal && $jStart + $jLen - 1 > $iStart) {
                    // This tile straddles the diagonal: mask entries
                    // where global key index > global query index.
                    for ($r = 0; $r < $iLen; $r++) {
                        $globalI = $iStart + $r;
                        $rowBase = $r * $jLen;
                        for ($c = 0; $c < $jLen; $c++) {
                            $globalJ = $jStart + $c;
                            if ($globalJ > $globalI) {
                                $sd[$rowBase + $c] = -INF;
                            }
                        }
                    }
                }
                $sij->data = $sd;

                // Row-wise max and exp(S - max) for this tile only
                // (this is the "online" part: mij is a *local* max over
                // just this tile, not the global row max yet).
                $mij = array_fill(0, $iLen, -INF);
                for ($r = 0; $r < $iLen; $r++) {
                    $rowBase = $r * $jLen;
                    $rowMaxLocal = -INF;
                    for ($c = 0; $c < $jLen; $c++) {
                        $val = $sd[$rowBase + $c];
                        if ($val > $rowMaxLocal) {
                            $rowMaxLocal = $val;
                        }
                    }
                    $mij[$r] = $rowMaxLocal;
                }

                $pij = new Matrix($iLen, $jLen);
                $pd = $pij->data;
                $lij = array_fill(0, $iLen, 0.0);
                for ($r = 0; $r < $iLen; $r++) {
                    $rowBase = $r * $jLen;
                    $rm = $mij[$r];
                    $sum = 0.0;
                    if ($rm === -INF) {
                        // Entire row masked in this tile (fully causal-masked);
                        // contributes nothing.
                        for ($c = 0; $c < $jLen; $c++) {
                            $pd[$rowBase + $c] = 0.0;
                        }
                    } else {
                        for ($c = 0; $c < $jLen; $c++) {
                            $e = exp($sd[$rowBase + $c] - $rm);
                            $pd[$rowBase + $c] = $e;
                            $sum += $e;
                        }
                    }
                    $lij[$r] = $sum;
                }
                $pij->data = $pd;

                // Pij @ Vj, shape (iLen x dv) -- the other big matmul.
                $pv = $this->kernel->matMul($pij, $vj);
                $pvData = $pv->data;

                // Online-softmax merge of this tile's stats into the
                // running (mi, li, oi) for this query block.
                for ($r = 0; $r < $iLen; $r++) {
                    $mOld = $mi[$r];
                    $lOld = $li[$r];
                    $mTile = $mij[$r];
                    $mNew = $mOld > $mTile ? $mOld : $mTile;

                    // exp(-INF - finite) = 0 correctly in PHP/IEEE754, and
                    // exp(-INF - (-INF)) would be NAN, so guard that case
                    // (happens only when a row has been entirely masked
                    // by every tile seen so far, i.e. still at -INF).
                    $alpha = ($mOld === -INF && $mNew === -INF) ? 0.0 : exp($mOld - $mNew);
                    $beta = ($mTile === -INF && $mNew === -INF) ? 0.0 : exp($mTile - $mNew);

                    $li[$r] = $alpha * $lOld + $beta * $lij[$r];
                    $mi[$r] = $mNew;

                    $oldBase = $r * $dv;
                    for ($c = 0; $c < $dv; $c++) {
                        $oiData[$oldBase + $c] =
                            $alpha * $oiData[$oldBase + $c] + $beta * $pvData[$oldBase + $c];
                    }
                }
            }

            // Final normalization for this query block: divide every row's accumulator
            // by its final softmax denominator.
            for ($r = 0; $r < $iLen; $r++) {
                $l = $li[$r];
                $globalRow = $iStart + $r;
                $outBase = $globalRow * $dv;
                $localBase = $r * $dv;
                if ($l > 0.0) {
                    $invL = 1.0 / $l;
                    for ($c = 0; $c < $dv; $c++) {
                        $outData[$outBase + $c] = $oiData[$localBase + $c] * $invL;
                    }
                }
            }
        }

        $output->data = $outData;

        return $output;
    }
}
