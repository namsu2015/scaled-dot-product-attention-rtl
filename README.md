# Fixed-Point Scaled Dot-Product Attention in SystemVerilog

This project implements

```text
Q = XWQ
K = XWK
V = XWV
S = QK^T / sqrt(d_k)
A = softmax(S)
O = AV
```

in synthesizable SystemVerilog. The default test uses two tokens and an embedding dimension of four. Inputs and weights are signed INT8 values read from text files.

Two implementations are included:

- `attention_core.sv`: one reused datapath with low hardware cost
- `attention_core_pipelined.sv`: unrolled nine-stage pipeline with lower latency

Both implementations produce the same fixed-point result.

## Test result

The design was compiled and simulated with Icarus Verilog 12.0.

```text
Sequential core: 141 cycles, PASS
Pipelined core:    8 cycles, PASS
```

The pipelined core reduces compute latency by `141 / 8 = 17.625x` for the supplied test. This is not an area-normalized speedup; the pipelined version uses much more parallel hardware.

The maximum absolute output error relative to the floating-point calculation is `0.009471` for this example.

## Files

```text
rtl/attention_core.sv             Sequential RTL
rtl/attention_core_pipelined.sv   Pipelined RTL
tb/tb_attention.sv                Self-checking testbench
scripts/golden_model.py           Bit-accurate Python model
data/x.hex                        Input tokens
data/wq.hex                       Query weights
data/wk.hex                       Key weights
data/wv.hex                       Value weights
data/expected_o.hex               Expected fixed-point output
results/output.txt                Sequential output
results/output_pipelined.txt      Pipelined output
results/attention.vcd             Sequential waveform
results/attention_pipelined.vcd   Pipelined waveform
Makefile                          Build and test commands
```

## Running the test

Requirements:

- Python 3
- Icarus Verilog
- Make

Run both implementations:

```sh
make test
```

Run them separately:

```sh
make verify
make verify-pipelined
```

The expected messages are:

```text
LATENCY: 141 cycles from accepted start to done
PASS: all 8 output elements match bit-exact golden data
PASS: results/output.txt matches the bit-accurate Python model

LATENCY: 8 cycles from accepted start to done
PASS: all 8 output elements match bit-exact golden data
PASS: results/output_pipelined.txt matches the bit-accurate Python model
```

Waveforms can be opened with GTKWave:

```sh
gtkwave results/attention.vcd
gtkwave results/attention_pipelined.vcd
```

## Numerical example

The stored integer values are shown below. Divide Q, K, V, S, and O by 16. Divide A by 256.

```text
X = [[16,  0, 16,  0],
     [ 0, 16,  0, 16]]

WQ = WK = 16 * identity

WV = [[16,  8,  0,  0],
      [ 0, 16,  8,  0],
      [ 0,  0, 16,  8],
      [ 8,  0,  0, 16]]

Q = K = [[16,  0, 16,  0],
         [ 0, 16,  0, 16]]

V = [[16,  8, 16,  8],
     [ 8, 16,  8, 16]]

S = [[16,  0],
     [ 0, 16]]

A = [[187,  69],
     [ 69, 187]]

O = [[14, 10, 14, 10],
     [10, 14, 10, 14]]
```

The output text files therefore contain:

```text
14 10 14 10
10 14 10 14
```

## Fixed-point formats

| Value | Format | Real value |
|---|---:|---:|
| X and weights | signed 8-bit, 4 fractional bits | stored value / 16 |
| Q, K, V, O | signed 16-bit, 4 fractional bits | stored value / 16 |
| Scores | signed 24-bit, 4 fractional bits | stored value / 16 |
| Scale constant | unsigned Q0.8 | stored value / 256 |
| Exponential and attention weights | unsigned Q0.8 | stored value / 256 |
| Accumulators | signed 64-bit | full intermediate precision |

For `D=4`,

```text
1 / sqrt(D) = 0.5
SCALE_MUL_Q = round(0.5 * 256) = 128
```

Matrix products keep their full fractional precision during accumulation. Requantization uses symmetric round-to-nearest with ties away from zero. Q, K, V, scores, and outputs saturate at their destination width rather than wrapping.

The 64-bit accumulator is larger than required for the default dimensions. Its bound must be checked again if N, D, or the vector width is increased substantially.

## Softmax approximation

The softmax unit first subtracts the largest value in each row. This prevents the exponential input from becoming positive and guarantees that at least one LUT output is nonzero.

The exponential is approximated by a Q0.8 lookup table with 0.5-wide bins:

| Input | LUT output |
|---:|---:|
| 0.0 | 256 |
| -0.5 | 155 |
| -1.0 | 94 |
| -1.5 | 57 |
| -2.0 | 35 |
| -2.5 | 21 |
| -3.0 | 13 |
| -3.5 | 8 |
| -4.0 | 5 |
| -4.5 | 3 |
| -5.0 | 2 |
| -5.5 | 1 |

Smaller inputs map to zero. The weights are computed as

```text
A[i] = round(exp[i] * 256 / sum(exp))
```

The RTL uses the synthesizable `/` operator for normalization. A synthesis tool may infer a large combinational divider. A reciprocal LUT followed by a multiply, or a multi-cycle reciprocal unit, would be a better choice in a larger accelerator.

## Sequential architecture

The sequential core uses an FSM and reuses its arithmetic across matrix elements.

```text
load -> Q -> K -> V -> scores -> row max -> exp -> normalize -> output
```

One dot-product term is handled per clock. For square D matrices, the default schedule is approximately

```text
3*N*D*D + N*N*D + 3*N*N + N*N*D + 1
```

cycles. For `N=2, D=4`, the measured latency is 141 cycles.

This version is useful as a bit-accurate reference and as a low-area implementation.

## Pipelined architecture

The pipelined core separates the work into these registered stages:

```text
1. X*W products for Q, K, and V
2. Q/K/V reduction and requantization
3. Q*K products
4. Score reduction and scaling
5. Row maximum
6. Exponential lookup and row sum
7. Weight normalization
8. Attention-weight*V products
9. Output reduction and requantization
```

All matrix elements in a stage are spatially unrolled. For the default configuration, the RTL exposes the following multiply operations before synthesis optimization:

```text
Q/K/V stage:  3*N*D*D = 96
Score stage:  N*N*D   = 16
Output stage: N*D*N   = 16
Total:                    128
```

It also exposes four parallel normalization divisions. These are RTL operator counts, not final DSP or gate counts. Only a synthesis report can give reliable area, timing, and power numbers.

The current input interface loads one matrix and then starts one calculation, so it does not overlap separate input transactions. Ping-pong input buffers or a streaming interface would be needed to use the stage pipeline across multiple attention operations.

## Input and output format

Input files contain one hexadecimal two's-complement value per line in row-major order.

For signed 8-bit values with four fractional bits:

```text
1.0  -> 10
0.5  -> 08
-1.0 -> f0
```

X contains `N*D` entries. Each weight file contains `D*D` entries. The testbench reads these files with `$readmemh`, loads the core through its write interface, starts the calculation, checks every output, writes a decimal text matrix, and creates a VCD file.

## Parameter changes

The RTL parameters include N, D, bit widths, and fractional widths. When D changes, update

```text
SCALE_MUL_Q = round(256 / sqrt(D))
```

The supplied testbench and Python model use fixed `N=2, D=4` constants. They must be changed together when testing another shape.

## Verification

`golden_model.py` implements the same rounding, saturation, exponential LUT, and integer normalization as the RTL. The test flow performs three checks:

1. The SystemVerilog testbench compares every output against `expected_o.hex`.
2. Python reads each generated output text file and checks exact equality.
3. The Makefile checks that both VCD and output files are nonempty.

The sequential and pipelined output text files are also identical for the supplied test.

## Limitations

- Only one deterministic input set is currently included.
- The LUT error is measured for the supplied example, not across a full model dataset.
- No synthesis, place-and-route, frequency, area, or power result is claimed.
- The pipelined divider is still combinational.
- Larger dimensions need SRAM/BRAM mapping and a tiled dataflow rather than full unrolling.

## Related work

PEANO-ViT is relevant prior work on hardware approximation of Vision Transformer nonlinear functions. This repository does not implement or reproduce its datapath; it uses a separate row-max and LUT-based softmax.

M. E. Sadeghi, A. Fayyazi, S. Azizi, and M. Pedram, “PEANO-ViT: Power-Efficient Approximations of Non-Linearities in Vision Transformers,” ISLPED, 2024. [arXiv](https://arxiv.org/abs/2406.14854), [DOI](https://doi.org/10.1145/3665314.3670843).

## License

MIT
