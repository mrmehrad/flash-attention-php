<?php

declare(strict_types=1);

require __DIR__ . '/../bootstrap.php';

use FlashAttention\FlashAttention;
use FlashAttention\Matrix;

// Toy sequence: 6 tokens, head dim 4.
$q = Matrix::fromArray([
    [0.1, 0.2, 0.0, 0.3],
    [0.4, 0.0, 0.1, 0.2],
    [0.2, 0.2, 0.2, 0.2],
    [0.9, 0.1, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.3, 0.3, 0.1, 0.1],
]);
$k = $q; // self-attention: K/V derived from the same sequence as Q
$v = Matrix::fromArray([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
]);

// withBestAvailableKernel() uses the FFI-compiled kernel automatically
// if it has been built (see src/Ffi/build.sh); otherwise pure PHP.
$flash = FlashAttention::withBestAvailableKernel(
    blockSizeQ: 2,
    blockSizeKv: 2,
    causal: true, // each token only attends to itself and earlier tokens
);

$output = $flash->forward($q, $k, $v);

echo "Using kernel: {$flash->usingKernel()}\n\n";
echo "Causal self-attention output (6 tokens x 4 dims):\n";
foreach ($output->toArray() as $i => $row) {
    printf("token %d: [%s]\n", $i, implode(', ', array_map(fn ($x) => number_format($x, 4), $row)));
}
