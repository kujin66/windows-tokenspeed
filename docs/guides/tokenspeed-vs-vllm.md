# TokenSpeed versus vLLM on native Windows ARM64

These are independent engines, with no TokenSpeed enable/disable switch in
the measured vLLM installation. Latest non-eager values use the original
vLLM/Triton retest and two-repeat pooled FlashInfer confirmation. The eager
table retains its separately measured earlier cohort; it was not rerun during
the confirmation. All values are aggregate output tokens/s.

## Latest non-eager summary

| Input / batch | vLLM | TS Triton | TS FlashInfer | TS Triton / vLLM | TS FI / vLLM |
| --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 1.042x | 1.028x |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 1.021x | 0.973x |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 0.965x | 0.971x |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 1.026x | 0.964x |

TS Triton/vLLM ranges from 0.965x to 1.042x; TS FlashInfer/vLLM ranges from
0.964x to 1.028x. There is no uniform winner. This summary explicitly combines
cohorts with matching recorded per-platform environments; it is not a new fully
paired experiment. See the [complete Windows/WSL2 retest](windows-wsl2-engine-backend-retest.md)
for WSL2 vLLM/Triton, both FI repetitions, actual graph policies, dependencies,
output-ID variability and raw evidence.

## Separately measured eager cohort

| Input / batch | vLLM | TS Triton | TS FlashInfer | TS Triton / vLLM | TS FI / vLLM |
| --- | --- | --- | --- | --- | --- |
| 128 / 1 | 59.00 | 58.65 | 63.02 | 0.994x | 1.068x |
| 128 / 4 | 228.44 | 212.93 | 260.88 | 0.932x | 1.142x |
| 512 / 1 | 61.60 | 54.82 | 67.90 | 0.890x | 1.102x |
| 512 / 4 | 229.22 | 222.55 | 273.89 | 0.971x | 1.195x |

In this earlier native eager cohort, TS Triton is 0.6-11.0% lower than vLLM;
TS FlashInfer is 6.8-19.5% higher. No WSL2 vLLM eager matrix was measured.
Eager and the latest graph values are not a controlled mode-only ablation.

The [earlier paired report](../validation/tokenspeed-vs-vllm-20261029/README.md),
[derived comparisons](../validation/tokenspeed-vs-vllm-20261029/results.json)
and [consistency flags](../validation/tokenspeed-vs-vllm-20261029/comparisons.csv)
retain that cohort's original eager and graph measurements. They are archival
references; their old graph values are superseded in the summary above.
The separate HF comparison does not measure a gain over vLLM. Throughput and
completion counts do not establish output/quality equivalence or production
serving performance.
