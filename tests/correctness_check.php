<?php

declare(strict_types=1);

require __DIR__ . '/../bootstrap.php';

use FlashAttention\FlashAttention;
use FlashAttention\Matrix;
use FlashAttention\NaiveAttention;

function randomMatrix(int $rows, int $cols, int $seed): Matrix
{
    mt_srand($seed);
    $m = new Matrix($rows, $cols);
    for ($i = 0; $i < $rows * $cols; $i++) {
        $m->data[$i] = (mt_rand(-1000, 1000) / 1000.0);
    }
    return $m;
}

function maxAbsDiff(Matrix $a, Matrix $b): float
{
    $max = 0.0;
    for ($i = 0, $n = $a->rows * $a->cols; $i < $n; $i++) {
        $d = abs($a->data[$i] - $b->data[$i]);
        if ($d > $max) {
            $max = $d;
        }
    }
    return $max;
}

$cases = [
    // [N, M, d, dv, causal, Br, Bc]
    [8, 8, 4, 4, false, 4, 4],
    [17, 13, 8, 6, false, 5, 4],   // non-divisible sequence lengths / block sizes
    [32, 32, 16, 16, true, 8, 8],  // causal, block-aligned
    [30, 30, 8, 8, true, 7, 11],   // causal, block sizes that straddle the diagonal oddly
    [50, 50, 4, 4, true, 6, 6],
    [1, 1, 4, 4, true, 4, 4],      // degenerate single-token case
];

$failures = 0;

foreach ($cases as $idx => [$n, $m, $d, $dv, $causal, $br, $bc]) {
    $q = randomMatrix($n, $d, 1000 + $idx);
    $k = randomMatrix($m, $d, 2000 + $idx);
    $v = randomMatrix($m, $dv, 3000 + $idx);

    $scale = 1.0 / sqrt((float) $d);

    $flash = new FlashAttention($br, $bc, $causal, $scale);
    $naive = new NaiveAttention($causal, $scale);

    $outFlash = $flash->forward($q, $k, $v);
    $outNaive = $naive->forward($q, $k, $v);

    $diff = maxAbsDiff($outFlash, $outNaive);
    $ok = $diff < 1e-9;
    $failures += $ok ? 0 : 1;

    printf(
        "case %d  N=%-3d M=%-3d d=%-3d dv=%-3d causal=%s Br=%-3d Bc=%-3d  max_abs_diff=%.3e  %s\n",
        $idx,
        $n,
        $m,
        $d,
        $dv,
        $causal ? 'yes' : 'no ',
        $br,
        $bc,
        $diff,
        $ok ? 'OK' : 'FAIL'
    );
}

// Also sanity-check the FFI kernel against the pure-PHP kernel, if built.
if (\FlashAttention\Kernel\FfiKernel::isAvailable()) {
    $q = randomMatrix(20, 8, 42);
    $k = randomMatrix(20, 8, 43);
    $v = randomMatrix(20, 6, 44);

    $flashPhp = new FlashAttention(6, 6, true, null, new \FlashAttention\Kernel\PhpKernel());
    $flashFfi = new FlashAttention(6, 6, true, null, new \FlashAttention\Kernel\FfiKernel());

    $diff = maxAbsDiff($flashPhp->forward($q, $k, $v), $flashFfi->forward($q, $k, $v));
    $ok = $diff < 1e-9;
    $failures += $ok ? 0 : 1;
    printf("FFI vs PHP kernel agreement:  max_abs_diff=%.3e  %s\n", $diff, $ok ? 'OK' : 'FAIL');
} else {
    echo "FFI kernel not built (run src/Ffi/build.sh) -- skipping FFI agreement check.\n";
}

echo $failures === 0 ? "\nAll correctness checks passed.\n" : "\n{$failures} check(s) FAILED.\n";
exit($failures === 0 ? 0 : 1);
