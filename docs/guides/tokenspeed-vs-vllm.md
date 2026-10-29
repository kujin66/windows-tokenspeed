# TokenSpeed versus vLLM: non-eager inference

TokenSpeed and vLLM are independent engines; the measured vLLM installation
has no TokenSpeed enable/disable switch. The model is Qwen3-0.6B BF16 on the
same RTX Spark N1X GPU.

The tables report the latest available non-eager measurement for each
configuration. FlashInfer combines two independent engine-process repetitions
per platform (ten measured calls per case); vLLM and Triton each use five
measured calls per case. These measurements come from separate cohorts with
verified per-platform environments. They are a cross-cohort summary, rather
than a single paired six-configuration experiment.

## Native Windows ARM64

| Input / batch | vLLM | TS Triton | TS FlashInfer | TS Triton / vLLM | TS FI / vLLM |
| --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 1.042x | 1.028x |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 1.021x | 0.973x |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 0.965x | 0.971x |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 1.026x | 0.964x |

Values are aggregate output tokens/s; ratios above 1 favor TokenSpeed.
TS Triton / vLLM ranges from 0.965x to 1.042x; TS FlashInfer / vLLM ranges
from 0.964x to 1.028x. There is no uniform winner.

## Windows and WSL2

| Input / batch | Win vLLM | Win TS Triton | Win TS FI | WSL2 vLLM | WSL2 TS Triton | WSL2 TS FI |
| --- | --- | --- | --- | --- | --- | --- |
| 128 / 1 | 137.79 | 143.62 | 141.63 | 143.02 | 141.98 | 138.85 |
| 128 / 4 | 585.17 | 597.74 | 569.60 | 549.26 | 571.50 | 564.50 |
| 512 / 1 | 134.78 | 130.01 | 130.94 | 134.46 | 125.93 | 132.60 |
| 512 / 4 | 509.22 | 522.49 | 490.88 | 496.45 | 499.86 | 501.26 |

The [complete comparison](windows-wsl2-engine-backend-retest.md) supplies
platform ratios, both FlashInfer repetitions, exact source/package identities,
generation settings, timing boundaries, graph policies, output-ID diagnostics
and raw measurement evidence. TokenSpeed captures decode graphs at sizes
[1,4] with prefill graphs disabled; vLLM uses its default Inductor and
FULL_AND_PIECEWISE policy. Software stacks differ across platforms.

Completion and throughput do not establish equivalent answers, production
serving performance or an isolated OS/compiler/scheduler effect. All model
runs record AOT inactive; these results are not a CuTe router speedup.
