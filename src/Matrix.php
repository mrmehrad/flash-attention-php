<?php

declare(strict_types=1);

namespace FlashAttention;

/**
 * A lightweight row-major flat matrix backed by SplFixedArray<float>.
 *
 * Using a flat, fixed-size, typed-ish buffer instead of nested PHP arrays
 * avoids the per-row array overhead (hash tables, refcounting, zvals for
 * every sub-array) that makes naive array-of-arrays matrices slow in PHP.
 * This is the single biggest lever for "optimized PHP" numeric code short
 * of dropping into FFI/C.
 */
final class Matrix
{
    public readonly int $rows;
    public readonly int $cols;
    public \SplFixedArray $data;

    public function __construct(int $rows, int $cols, ?\SplFixedArray $data = null)
    {
        $this->rows = $rows;
        $this->cols = $cols;
        $this->data = $data ?? new \SplFixedArray($rows * $cols);
    }

    public static function zeros(int $rows, int $cols): self
    {
        $m = new self($rows, $cols);
        for ($i = 0, $n = $rows * $cols; $i < $n; $i++) {
            $m->data[$i] = 0.0;
        }
        return $m;
    }

    public static function fill(int $rows, int $cols, float $value): self
    {
        $m = new self($rows, $cols);
        for ($i = 0, $n = $rows * $cols; $i < $n; $i++) {
            $m->data[$i] = $value;
        }
        return $m;
    }

    /** Build from a nested PHP array (rows of rows), e.g. user-provided Q/K/V. */
    public static function fromArray(array $rows): self
    {
        $r = count($rows);
        $c = $r > 0 ? count($rows[0]) : 0;
        $m = new self($r, $c);
        $idx = 0;
        foreach ($rows as $row) {
            foreach ($row as $v) {
                $m->data[$idx++] = (float) $v;
            }
        }
        return $m;
    }

    public function toArray(): array
    {
        $out = [];
        for ($i = 0; $i < $this->rows; $i++) {
            $row = [];
            $base = $i * $this->cols;
            for ($j = 0; $j < $this->cols; $j++) {
                $row[] = $this->data[$base + $j];
            }
            $out[] = $row;
        }
        return $out;
    }

    #[\Override]
    public function __toString(): string
    {
        return sprintf('Matrix(%dx%d)', $this->rows, $this->cols);
    }

    public function get(int $i, int $j): float
    {
        return $this->data[$i * $this->cols + $j];
    }

    public function set(int $i, int $j, float $value): void
    {
        $this->data[$i * $this->cols + $j] = $value;
    }

    /**
     * Extract a contiguous row-block [rowStart, rowStart+numRows) as a new Matrix.
     * Used to slice out Q_i / K_j / V_j tiles without copying the whole matrix.
     */
    public function sliceRows(int $rowStart, int $numRows): self
    {
        $cols = $this->cols;
        $out = new self($numRows, $cols);
        // Contiguous memcpy-like copy since rows are laid out sequentially.
        $srcBase = $rowStart * $cols;
        for ($k = 0, $n = $numRows * $cols; $k < $n; $k++) {
            $out->data[$k] = $this->data[$srcBase + $k];
        }
        return $out;
    }

    public function writeRows(int $rowStart, self $block): void
    {
        $cols = $this->cols;
        $dstBase = $rowStart * $cols;
        for ($k = 0, $n = $block->rows * $cols; $k < $n; $k++) {
            $this->data[$dstBase + $k] = $block->data[$k];
        }
    }
}
