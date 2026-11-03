#!/usr/bin/env python3
"""Bit-accurate reference model and deterministic test-vector generator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

N, D = 2, 4
DATA_FRAC = 4
SCALE_FRAC = 8
SCALE_MUL_Q = 128
WEIGHT_FRAC = 8

X = [[16, 0, 16, 0], [0, 16, 0, 16]]
IDENTITY_Q44 = [[16 if r == c else 0 for c in range(D)] for r in range(D)]
WQ = IDENTITY_Q44
WK = IDENTITY_Q44
WV = [
    [16, 8, 0, 0],
    [0, 16, 8, 0],
    [0, 0, 16, 8],
    [8, 0, 0, 16],
]


def round_shift(value: int, shift: int) -> int:
    if shift == 0:
        return value
    bias = 1 << (shift - 1)
    return (value + bias) >> shift if value >= 0 else -(((-value) + bias) >> shift)


def sat(value: int, width: int) -> int:
    return max(-(1 << (width - 1)), min((1 << (width - 1)) - 1, value))


def matmul_q44(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [sat(round_shift(sum(a[i][k] * b[k][j] for k in range(D)), DATA_FRAC), 16)
         for j in range(D)]
        for i in range(N)
    ]


def exp_lut(delta: int) -> int:
    half = 1 << (DATA_FRAC - 1)
    thresholds = [-(2 * i + 1) * half // 2 for i in range(12)]
    values = [256, 155, 94, 57, 35, 21, 13, 8, 5, 3, 2, 1]
    for threshold, value in zip(thresholds, values):
        if delta >= threshold:
            return value
    return 0


def model():
    q, k, v = (matmul_q44(X, w) for w in (WQ, WK, WV))
    score = []
    for i in range(N):
        row = []
        for j in range(N):
            dot = sum(q[i][d] * k[j][d] for d in range(D))
            row.append(sat(round_shift(dot * SCALE_MUL_Q, DATA_FRAC + SCALE_FRAC), 24))
        score.append(row)

    weights = []
    for row in score:
        exps = [exp_lut(x - max(row)) for x in row]
        total = sum(exps)
        weights.append([((x << WEIGHT_FRAC) + (total >> 1)) // total for x in exps])

    output = []
    for i in range(N):
        output.append([
            sat(round_shift(sum(weights[i][j] * v[j][d] for j in range(N)), WEIGHT_FRAC), 16)
            for d in range(D)
        ])
    return q, k, v, score, weights, output


def flatten(matrix):
    return [x for row in matrix for x in row]


def write_hex(path: Path, values, width: int):
    digits = (width + 3) // 4
    mask = (1 << width) - 1
    path.write_text("".join(f"{value & mask:0{digits}x}\n" for value in values))


def write_vectors(root: Path):
    data = root / "data"
    data.mkdir(exist_ok=True)
    for name, matrix in (("x", X), ("wq", WQ), ("wk", WK), ("wv", WV)):
        write_hex(data / f"{name}.hex", flatten(matrix), 8)
    output = model()[-1]
    write_hex(data / "expected_o.hex", flatten(output), 16)


def read_output(path: Path):
    return [[int(x) for x in line.split()] for line in path.read_text().splitlines() if line.strip()]


def print_report():
    q, k, v, score, weights, output = model()
    names = (("Q", q), ("K", k), ("V", v), ("Score", score),
             ("Softmax weights (Q0.8)", weights), ("Output", output))
    for name, matrix in names:
        print(f"{name}:")
        for row in matrix:
            print("  " + " ".join(f"{x:4d}" for x in row))
    # Also report error against ideal floating-point attention for this vector.
    qf = [[x / 16 for x in row] for row in q]
    kf = [[x / 16 for x in row] for row in k]
    vf = [[x / 16 for x in row] for row in v]
    sf = [[sum(qf[i][d] * kf[j][d] for d in range(D)) / math.sqrt(D)
           for j in range(N)] for i in range(N)]
    wf = [[math.exp(x - max(row)) / sum(math.exp(y - max(row)) for y in row)
           for x in row] for row in sf]
    of = [[sum(wf[i][j] * vf[j][d] for j in range(N)) for d in range(D)]
          for i in range(N)]
    oq = [[x / 16 for x in row] for row in output]
    max_err = max(abs(oq[i][d] - of[i][d]) for i in range(N) for d in range(D))
    print(f"Max absolute error vs ideal float: {max_err:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-vectors", action="store_true")
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.write_vectors:
        write_vectors(root)
    print_report()
    if args.check:
        expected = model()[-1]
        observed = read_output(args.check)
        if observed != expected:
            raise SystemExit(f"FAIL: HDL output {observed} != golden {expected}")
        print(f"PASS: {args.check} matches the bit-accurate Python model")


if __name__ == "__main__":
    main()
