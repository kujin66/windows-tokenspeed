# Windows ARM64 / WSL2 本轮验证（2026-10-28）

Windows 四个 wheel 已在此工作区重新构建、安装并通过逐文件/ARM64 架构审计。
WSL2 3.0.3 已安装，VirtualMachinePlatform 已启用，但需要重启 Windows；Ubuntu 注册返回 HCS_E_SERVICE_NOT_AVAILABLE。
Ubuntu ARM64 镜像和官方 TokenSpeed 固定基线源码已经备好。**WSL2 性能尚未测得，不能得出两端速度差异。**

本轮 Windows 数值为重启前基线。重启后应重跑 Windows，再顺序测量 WSL2，以统一系统虚拟化配置状态。

[完整复现命令](../../guides/windows-arm64-reproduction.md) · [机器可读结果](results.json) · [Windows 原始性能数据](windows-bf16-perf.json)

## 本次测试

| 分组 | 通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: |
| runtime | 160 | 0 | 0 |
| gpu | 26 | 0 | 0 |
| perf_tools | 16 | 0 | 0 |

另通过 CUDA 主机布局负/正探测与 4097 元素 GPU 向量加法；不重复计入 pytest 项数。
首次运行时回归为 156 通过、4 个临时目录 ACL 初始化错误；原始失败日志保留。使用项目内临时/缓存目录后重跑，最终数量只统计重跑结果。
本次验证 6 个项目 PE 均为 ARM64/linker 14.51；Python 和第三方依赖不要求也不声称由同一 linker 构建。

## Windows 性能基线

Qwen3-0.6B BF16，单 GPU RTX Spark N1X / SM12.1，Triton attention，eager，prefix cache 关闭。
每 case 3 次预热、10 次实测；固定输出 128 token。吞吐量包括 prefill/decode 与 IPC，不是独立 decode kernel 吞吐量。

| 输入 token | Batch | 中位延迟（秒） | P95（秒） | 输出 tok/s（聚合） | 重复 greedy IDs |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 1 | 2.2108 | 2.6518 | 57.58 | 一致 |
| 128 | 4 | 2.3616 | 2.6027 | 214.59 | 一致 |
| 512 | 1 | 2.4005 | 2.5730 | 53.71 | 一致 |
| 512 | 4 | 2.1431 | 2.5154 | 235.45 | 一致 |

共 100 个实测请求、12800 个实测输出 token；预热另计。每个请求输出长度严格校验；逐轮 IDs 保存在 JSON。

先前严格一致性试跑在 batch=4 观察到 greedy token ID 差异，原因尚未定位。最终吞吐使用显式 consistency-policy=record，保留差异；吞吐完成不代表模型输出数值等价。
先前 context=640 的试跑因 512-token 输入触发生成截断而失败；其日志和 passed=false JSON 保留，但未混入本轮统计。最终 context=704，预留一个 64-token page。

## 产物与依赖

四个 wheel 在仓库 dist/；六个原生二进制的位置、SHA256、编译身份和安装一致性见 [原生审计](native-artifacts-verification.json)。
关键 ARM64 wheel、工具链和官方归档实际存放于 .port，Git 忽略这些本机大文件；版本约束见 requirements/windows-arm64-tested.txt。
本机仍通过 local-runtime.pth 共享其他 Python 运行依赖，不是完整离线或干净机器安装测试。

## 证据

日志和 JUnit 已转为 UTF-8，并替换本机用户路径、主机名和索引地址。
原始日志保留在 .port；[哈希清单](evidence-sha256.json) 同时保存原件与公开副本 SHA256。
历史 2026-10-26/27 测试未累计到本轮。后续 WSL2 结果应保存为新记录，不覆盖本轮待重启的真实状态。
