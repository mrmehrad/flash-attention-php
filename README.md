# flash-attention-php

> FlashAttention v2 (tiled, memory-efficient exact attention) implemented in pure PHP, with an optional FFI-accelerated C kernel.

FlashAttention v2 ([Dao, 2023](https://arxiv.org/abs/2307.08691)) — implemented in pure PHP, with an optional FFI-accelerated C kernel. No GPU, no PyTorch, no Python required.

> FlashAttention isn't a different attention formula — it's the *same* softmax(QK^T / √d)V result, computed without ever allocating the full N×N score matrix. It does this via tiling (processing Q/K/V in small row-blocks) and **online softmax**: a running max/sum per query row that gets mathematically exact rescaling every time a new block shifts the max. This repo ports that exact algorithm to PHP.

## Why this exists

PHP will never out-FLOP a CUDA kernel. What FlashAttention buys you even on a CPU, in any language, is **O(N) memory instead of O(N²)** for the attention matrix — which matters the moment you try to run inference against a few thousand tokens in a language that doesn't have a GPU backend at all. This library is meant for PHP backends that need to run small/medium transformer inference (e.g. a custom classifier, a small in-house model, or a Cloudflare Workers–style edge function) without shelling out to Python.

## Features

- Exact FlashAttention v2 forward pass: tiling + online softmax, verified bit-for-bit (to float precision) against a naive O(N²) reference implementation across multiple shapes, including ragged sequence lengths and block sizes that straddle the causal diagonal.
- Causal masking with **block-level skipping** — fully-masked (i, j) tiles are never even computed, not just zeroed out.
- Multi-head attention (`forwardMultiHead`) over interleaved-by-head Q/K/V.
- Pluggable compute kernel:
  - `PhpKernel` — pure PHP, zero dependencies, works everywhere.
  - `FfiKernel` — calls a small compiled C library via `ext-ffi` for the matmuls. ~10–20x faster in local benchmarks (see below), same numeric result.
- Flat, `SplFixedArray`-backed matrices instead of array-of-arrays, to avoid PHP's per-row hash-table overhead in hot numeric loops.

## Installation

```bash
composer require mehrad/flash-attention-php
```

Or clone directly and use the included `bootstrap.php` autoloader (no Composer required):

```php
require 'bootstrap.php';
```

### Optional: build the FFI kernel

```bash
php -m | grep -i ffi   # confirm ext-ffi is enabled
bash src/Ffi/build.sh  # compiles src/Ffi/kernel.c -> src/Ffi/libflashkernel.so
```

If the shared library isn't built (or `ext-ffi` isn't loaded), the library transparently falls back to `PhpKernel` — nothing breaks.

## Usage

```php
use FlashAttention\FlashAttention;
use FlashAttention\Matrix;

$q = Matrix::fromArray([...]); // (seq_len x head_dim)
$k = Matrix::fromArray([...]); // (seq_len x head_dim)
$v = Matrix::fromArray([...]); // (seq_len x head_dim_v)

// Picks the FFI kernel automatically if built, else pure PHP.
$flash = FlashAttention::withBestAvailableKernel(
    blockSizeQ: 64,   // Br in the paper
    blockSizeKv: 64,  // Bc in the paper
    causal: true,
);

$output = $flash->forward($q, $k, $v); // (seq_len x head_dim_v)
```

Multi-head, with Q/K/V laid out as `(seq_len, numHeads * headDim)`:

```php
$output = $flash->forwardMultiHead($q, $k, $v, numHeads: 8);
```

See `examples/basic_example.php` and `examples/multi_head_example.php` for runnable end-to-end examples.

## Verifying correctness

```bash
php tests/correctness_check.php
```

This runs FlashAttention against a naive reference implementation (`NaiveAttention`) across several shapes — including non-causal, causal, ragged (non-block-aligned) sequence lengths, block sizes that straddle the causal diagonal, and a degenerate single-token case — and asserts the max absolute difference is within floating-point tolerance (~1e-16 in practice). It also cross-checks the FFI kernel against the pure-PHP kernel if the shared library has been built.

With PHPUnit installed (`composer require --dev phpunit/phpunit`), the same cases run as `tests/FlashAttentionTest.php`.

## Benchmarking

```bash
php benchmarks/benchmark.php
```

Sample run on this machine (causal self-attention, head_dim=64):

| N (seq len) | Naive (ms) | Flash/PHP (ms) | Flash/FFI (ms) | PHP speedup | FFI speedup |
|---|---|---|---|---|---|
| 128 | 92.5 | 41.7 | 7.6 | 2.2x | 12.2x |
| 256 | 337.2 | 138.2 | 23.2 | 2.4x | 14.6x |
| 512 | 1387.3 | 494.5 | 74.2 | 2.8x | 18.7x |

Numbers will vary by machine and PHP build (JIT/opcache settings, `-march=native` availability, etc). The wall-clock win is real, but the more important number for anything approaching real transformer context lengths (thousands of tokens) is **peak memory**, not raw FLOPs — naive attention's O(N²) score matrix is what actually runs a process out of memory first.

## How the algorithm works (short version)

In FlashAttention v2, the loop order was swapped so that the outer loop iterates over blocks of queries `(Q_i)`, and the inner loop iterates over blocks of keys/values `(K_j, V_j)`.

For each block of queries `(Q_i)`:

1. Initialize local statistics and output accumulators for this query block.
2. For each block of keys/values `(K_j, V_j)`:
    1. Compute the small tile `S_ij = scale · Q_i K_j^T`.
    2. Take `S_ij`'s row-max `m_ij` (local to this tile) and `P_ij = exp(S_ij - m_ij)`.
    3. Merge into the running per-row statistics using the rescaling identity: since `softmax(x)_k = exp(x_k - m) / Σ exp(x_i - m)` holds for *any* `m`, shifting the running max from `m_old` to `m_new = max(m_old, m_ij)` just requires multiplying everything accumulated so far by `exp(m_old - m_new)`, and the new tile's contribution by `exp(m_ij - m_new)`.
    4. Accumulate `O_i += exp(m_old - m_new)·O_i + exp(m_ij - m_new)·(P_ij V_j)` and the softmax denominator `l_i` the same way.
3. After all key/value blocks are processed for `Q_i`, divide `O_i` by the final `l_i` and write it to the global output matrix.

Causal masking is applied at block granularity: a tile is skipped outright if every key index in it is greater than every query index in it (fully future), and masked element-wise only for the diagonal-straddling tiles.

See the docblocks in `src/FlashAttention.php` for the full annotated implementation.

## Project layout

```
src/
  Matrix.php              flat SplFixedArray-backed matrix
  FlashAttention.php       the algorithm (tiling + online softmax)
  NaiveAttention.php       O(N^2) reference implementation, for testing
  Kernel/
    KernelInterface.php
    PhpKernel.php          pure PHP matmul
    FfiKernel.php          FFI-backed matmul (calls compiled C)
  Ffi/
    kernel.c               the C side of FfiKernel
    build.sh               compiles kernel.c -> libflashkernel.so
tests/
  correctness_check.php    standalone script, no PHPUnit required
  FlashAttentionTest.php   PHPUnit test suite
benchmarks/
  benchmark.php
examples/
  basic_example.php
  multi_head_example.php
```

## Limitations

- This is a **forward-pass-only** implementation (no backward/gradient pass). It's aimed at inference, not training, in PHP.
- No GPU support — PHP has no CUDA bindings. The FFI kernel is a CPU C kernel, not a GPU one.
- FP64 (PHP's native float) throughout, not FP16/BF16.

## License

MIT — see [LICENSE](LICENSE).

---

## به فارسی

این مخزن پیاده‌سازی الگوریتم **FlashAttention نسخه ۲** (مقاله Dao، ۲۰۲۳) را به زبان PHP ارائه می‌دهد؛ بدون نیاز به GPU، PyTorch یا Python.

نکتهٔ کلیدی FlashAttention این است که فرمول توجه را تغییر نمی‌دهد — دقیقاً همان `softmax(QK^T/√d)V` را محاسبه می‌کند، اما بدون این‌که هرگز کل ماتریس امتیاز N×N را در حافظه بسازد. این کار با «تکه‌بندی» (tiling) ورودی‌ها و تکنیک **softmax آنلاین** انجام می‌شود: برای هر ردیف از Q یک بیشینه و مجموع در حال اجرا نگه داشته می‌شود که با هر بلوک جدید از K/V به‌صورت دقیق ریاضی بازمقیاس‌دهی می‌شود.

### چرا PHP؟

PHP هرگز از نظر سرعت خام با CUDA رقابت نمی‌کند، اما مزیت FlashAttention — یعنی حافظهٔ O(N) به‌جای O(N²) — در هر زبانی، از جمله PHP، معتبر است. این کتابخانه برای بک‌اندهای PHP طراحی شده که نیاز به اجرای inference یک مدل ترنسفورمر کوچک یا متوسط دارند، بدون فراخوانی Python.

### نصب

```bash
composer require mehrad/flash-attention-php
```

یا با کلون مستقیم مخزن و استفاده از `bootstrap.php` (بدون نیاز به Composer).

### ساخت هستهٔ FFI (اختیاری، برای سرعت بیشتر)

```bash
bash src/Ffi/build.sh
```

اگر کتابخانهٔ FFI ساخته نشود یا افزونهٔ `ext-ffi` فعال نباشد، کتابخانه به‌طور خودکار به هستهٔ خالص PHP برمی‌گردد.

### اجرای تست صحت

```bash
php tests/correctness_check.php
```

خروجی FlashAttention با یک پیاده‌سازی سادهٔ O(N²) مقایسه می‌شود و اختلاف باید در حد خطای اعشاری (~1e-16) باشد.

### محدودیت‌ها

فقط forward pass پیاده‌سازی شده (نه backward/gradient)، بدون پشتیبانی GPU، و محاسبات با دقت FP64 (نه FP16/BF16).
