# TokenSpeed versus vLLM on native Windows ARM64

Two independent engines are compared; there is no TokenSpeed enable/disable switch in this installed vLLM. All ratios use fresh matched model-generation runs with BF16 weights and normal text-output APIs. Values below 1 indicate TokenSpeed has lower throughput.

| Execution mode | TokenSpeed Triton / vLLM | TokenSpeed FI / vLLM |
| --- | ---: | ---: |
| eager | 0.89–0.99x (-11.0% to -0.6%) | 1.07–1.19x (+6.8% to +19.5%) |
| graph | 0.96–1.05x (-4.0% to +4.5%) | 0.94–0.99x (-5.9% to -1.4%) |

**Eager versus vLLM:** triton: 0.6–11.0% lower; flashinfer: 6.8–19.5% higher.

**Graph versus vLLM:** triton: -4.0% to +4.5% relative throughput; flashinfer: 1.4–5.9% lower.

There is no general TokenSpeed throughput advantage over vLLM across modes/backends in this matrix. Small graph differences do not establish a stable ranking from five rounds.

Eager, aggregate output tokens/s:

| Input / batch | vLLM | TokenSpeed Triton | TokenSpeed FI | TS Triton / vLLM | TS FI / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 1 | 59.00 | 58.65 | 63.02 | 0.994x | 1.068x |
| 128 / 4 | 228.44 | 212.93 | 260.88 | 0.932x | 1.142x |
| 512 / 1 | 61.60 | 54.82 | 67.90 | 0.890x | 1.102x |
| 512 / 4 | 229.22 | 222.55 | 273.89 | 0.971x | 1.195x |

Non-eager, aggregate output tokens/s:

| Input / batch | vLLM | TokenSpeed Triton | TokenSpeed FI | TS Triton / vLLM | TS FI / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 1 | 144.75 | 144.22 | 139.61 | 0.996x | 0.965x |
| 128 / 4 | 582.45 | 608.83 | 564.28 | 1.045x | 0.969x |
| 512 / 1 | 133.66 | 128.36 | 125.84 | 0.960x | 0.941x |
| 512 / 4 | 509.51 | 515.84 | 502.62 | 1.012x | 0.986x |

See the [complete methods, actual graph policies and limitations](../validation/tokenspeed-vs-vllm-20261029/README.md), [raw derived comparisons](../validation/tokenspeed-vs-vllm-20261029/results.json), and [every consistency flag](../validation/tokenspeed-vs-vllm-20261029/comparisons.csv). The earlier HF 3x result is a supplementary framework reference and does not establish a gain over vLLM. Output/quality equivalence and production serving performance remain unproven.
