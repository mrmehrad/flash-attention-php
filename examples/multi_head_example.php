<?php

declare(strict_types=1);

require __DIR__ . '/../bootstrap.php';

use FlashAttention\FlashAttention;
use FlashAttention\Matrix;

// 10 tokens, 4 heads, head dim 8 => 32 columns total (heads laid out
// contiguously per row: [head0_dims..., head1_dims..., ...]).
$numHeads = 4;
$headDim = 8;
$seqLen = 10;

function randomMatrix(int $rows, int $cols): Matrix
{
    $m = new Matrix($rows, $cols);
    for ($i = 0; $i < $rows * $cols; $i++) {
        $m->data[$i] = mt_rand(-1000, 1000) / 1000.0;
    }
    return $m;
}

mt_srand(7);
$q = randomMatrix($seqLen, $numHeads * $headDim);
$k = randomMatrix($seqLen, $numHeads * $headDim);
$v = randomMatrix($seqLen, $numHeads * $headDim);

$flash = FlashAttention::withBestAvailableKernel(blockSizeQ: 4, blockSizeKv: 4, causal: true);

$output = $flash->forwardMultiHead($q, $k, $v, $numHeads);

echo "Using kernel: {$flash->usingKernel()}\n";
printf("Multi-head output shape: %d x %d (numHeads=%d, headDim=%d)\n", $output->rows, $output->cols, $numHeads, $headDim);
