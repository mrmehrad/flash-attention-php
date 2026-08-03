<?php

declare(strict_types=1);

require __DIR__ . '/../bootstrap.php';

use FlashAttention\FlashAttention;
use FlashAttention\Kernel\FfiKernel;
use FlashAttention\Kernel\PhpKernel;
use FlashAttention\Matrix;
use FlashAttention\NaiveAttention;

function randomMatrix(int $rows, int $cols): Matrix
{
    $m = new Matrix($rows, $cols);
    for ($i = 0; $i < $rows * $cols; $i++) {
        $m->data[$i] = mt_rand(-1000, 1000) / 1000.0;
    }
    return $m;
}

function timeIt(callable $fn): array
{
    $memBefore = memory_get_peak_usage(true);
    $t0 = hrtime(true);
    $result = $fn();
    $t1 = hrtime(true);
    $memAfter = memory_get_peak_usage(true);
    return [
        'result' => $result,
        'ms' => ($t1 - $t0) / 1e6,
        'peak_mem_delta_mb' => ($memAfter - $memBefore) / (1024 * 1024),
    ];
}

$seqLens = [128, 256, 512];
$headDim = 64;

echo "FlashAttention (PHP) vs Naive attention -- forward pass timing\n";
echo str_repeat('-', 78) . "\n";
printf("%-8s %-10s %-10s %-10s %-10s %-14s\n", 'N', 'naive_ms', 'flash_ms', 'ffi_ms', 'speedup', 'ffi_speedup');
echo str_repeat('-', 78) . "\n";

$ffiAvailable = FfiKernel::isAvailable();

foreach ($seqLens as $n) {
    $q = randomMatrix($n, $headDim);
    $k = randomMatrix($n, $headDim);
    $v = randomMatrix($n, $headDim);

    $naive = new NaiveAttention(causal: true);
    $rNaive = timeIt(fn () => $naive->forward($q, $k, $v));

    $flashPhp = new FlashAttention(blockSizeQ: 64, blockSizeKv: 64, causal: true, kernel: new PhpKernel());
    $rFlashPhp = timeIt(fn () => $flashPhp->forward($q, $k, $v));

    $ffiMs = null;
    if ($ffiAvailable) {
        $flashFfi = new FlashAttention(blockSizeQ: 64, blockSizeKv: 64, causal: true, kernel: new FfiKernel());
        $rFlashFfi = timeIt(fn () => $flashFfi->forward($q, $k, $v));
        $ffiMs = $rFlashFfi['ms'];
    }

    printf(
        "%-8d %-10.2f %-10.2f %-10s %-10.2fx %-14s\n",
        $n,
        $rNaive['ms'],
        $rFlashPhp['ms'],
        $ffiMs !== null ? number_format($ffiMs, 2) : 'n/a',
        $rNaive['ms'] / $rFlashPhp['ms'],
        $ffiMs !== null ? number_format($rNaive['ms'] / $ffiMs, 2) . 'x' : 'n/a'
    );
}

echo "\nNote: the timing win here is mostly about *not* allocating an N x N\n";
echo "score matrix; on these fairly small N the peak-memory win is the more\n";
echo "important number for anything approaching real transformer context\n";
echo "lengths (thousands of tokens), where naive attention's O(N^2) buffer\n";
echo "becomes the actual bottleneck rather than raw FLOPs.\n";
