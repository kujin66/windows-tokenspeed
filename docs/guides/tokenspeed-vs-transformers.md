# TokenSpeed versus Transformers: matched no-TokenSpeed baselines

See the [complete fresh experiment report](../validation/tokenspeed-vs-transformers-20261029/README.md), [machine-readable comparisons](../validation/tokenspeed-vs-transformers-20261029/results.json), and [all consistency flags](../validation/tokenspeed-vs-transformers-20261029/comparisons.csv).

- **windows: TokenSpeed FI eager is 2.99–3.32x stock HF generate**, and 1.57–1.94x the static eager control with raw FI RMSNorm. In decode-graph mode, it is **1.14–1.80x** the request-sized HF graph with FI RMSNorm.
  Selecting the fastest measured no-TokenSpeed graph control separately for each case gives 1.13–1.80x. This conservative comparison includes the math-flag variant; it does not establish a causal effect of that flag.
- **wsl2: TokenSpeed FI eager is 1.76–2.19x stock HF generate**, and 1.31–1.60x the static eager control with raw FI RMSNorm. In decode-graph mode, it is **1.11–1.65x** the request-sized HF graph with FI RMSNorm.
  Selecting the fastest measured no-TokenSpeed graph control separately for each case gives 1.11–1.65x. This conservative comparison includes the math-flag variant; it does not establish a causal effect of that flag.

| Platform | Input / batch | HF generate | HF static eager + FI RMSNorm | TokenSpeed Triton | TokenSpeed FI | TS FI / HF generate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| windows | 128 / 1 | 21.13 | 40.27 | 53.15 | 63.11 | 2.987x |
| windows | 128 / 4 | 84.65 | 140.32 | 205.84 | 272.00 | 3.213x |
| windows | 512 / 1 | 21.77 | 40.83 | 54.44 | 65.38 | 3.003x |
| windows | 512 / 4 | 68.47 | 138.66 | 202.39 | 227.31 | 3.320x |
| wsl2 | 128 / 1 | 31.95 | 42.98 | not measured | 56.21 | 1.759x |
| wsl2 | 128 / 4 | 117.74 | 151.14 | not measured | 226.33 | 1.922x |
| wsl2 | 512 / 1 | 30.99 | 43.46 | not measured | 62.95 | 2.031x |
| wsl2 | 512 / 4 | 110.82 | 152.13 | not measured | 242.82 | 2.191x |

| Platform | Input / batch | HF graph, 704, Torch norm | HF graph, fit KV, FI norm | HF fit KV, FI norm, math flag | TokenSpeed Triton | TokenSpeed FI | TS FI / HF fit KV + FI norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| windows | 128 / 1 | 98.64 | 123.81 | 125.24 | 143.44 | 141.60 | 1.144x |
| windows | 128 / 4 | 235.85 | 372.61 | 384.67 | 606.79 | 581.75 | 1.561x |
| windows | 512 / 1 | 87.95 | 108.25 | 111.60 | 130.66 | 126.10 | 1.165x |
| windows | 512 / 4 | 227.29 | 274.18 | 267.71 | 506.77 | 493.94 | 1.802x |
| wsl2 | 128 / 1 | 99.84 | 126.97 | 126.93 | not measured | 140.64 | 1.108x |
| wsl2 | 128 / 4 | 254.35 | 403.98 | 396.61 | not measured | 536.36 | 1.328x |
| wsl2 | 512 / 1 | 95.82 | 110.37 | 110.20 | not measured | 130.92 | 1.186x |
| wsl2 | 512 / 4 | 240.19 | 283.56 | 283.24 | not measured | 467.21 | 1.648x |

These ratios compare the complete measured stacks. Controlled graph, RMSNorm, KV-capacity and dispatcher checks and all limitations are documented in the full report. They do not establish an independent scheduler-only speedup.
