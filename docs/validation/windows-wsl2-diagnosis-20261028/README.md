# Windows ARM64 与 WSL2 性能差异排查

原始 10 轮吞吐差异见 [配对报告](../windows-wsl2-perf-20261028-after-reboot/README.md)。本次没有修改运行时源码、安装包或该报告。

本次最强证据是同模型的无 profiler 对照：Windows 从 eager 改成 decode graph 后，batch 1/4 分别提高 2.58x/2.94x；WSL2 提高 1.61x/1.95x，Windows 在 graph 模式下反而更快。此前 Windows 劣势主要出现在 eager 执行路径，不能概括成 Windows 平台整体更慢。
结合小算子回放实验，优先归因于大量逐次内核发射、包装与算子路径的开销差异。两侧算子实现、PyTorch、CUDA 和 Triton 并未对齐，尚不能把原始差异精确分摊到某个组件或操作系统。

## 实际差异

- Windows PyTorch 为 2.14.0a0+git20b0f57 / CUDA 13.4；WSL2 为官方 2.14.0 / 实际 CUDA 13.0。两侧都是 Release、原生 ARM64，Windows 不是 x64 模拟。
- Windows 使用 triton-windows 3.8.0.post29；WSL2 使用 tokenspeed-triton 3.9.10.post20261008。
- Windows RMSNorm 与 SwiGLU 走 Triton；Linux 默认走 FlashInfer。Triton attention 相同这一配置不能说明所有算子相同。
- 原始基准 enforce_eager=True，关闭 decode/prefill graph，Qwen3-0.6B 的 28 层每步需要大量主机侧内核提交。

相关源码：[Triton 入口](../../../tokenspeed-kernel/python/tokenspeed_kernel/_triton.py)、[RMSNorm](../../../tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/__init__.py)、[SwiGLU](../../../tokenspeed-kernel/python/tokenspeed_kernel/ops/activation/__init__.py)、[原始基准参数](../../../tools/perf/benchmark_tokenspeed.py)。

## 相同脚本的算子测试

同一 GPU 顺序运行，预热 20 次、每项 7 轮；使用 CUDA event 和阻塞墙钟。下表均为每项墙钟中位数，单位微秒。graph 每次回放打包 32 次操作（64MiB copy 为 8 次），其结果包含摊销后的 graph 提交开销。

| 操作 | Windows eager µs | WSL2 eager µs | Windows graph µs | WSL2 graph µs |
| --- | ---: | ---: | ---: | ---: |
| torch_add_4096 | 24.46 | 29.30 | 1.25 | 1.21 |
| device_copy_64MiB | 546.90 | 589.05 | 546.20 | 551.75 |
| rmsnorm_facade_m1_n1024 | 27.61 | 20.78 | 1.14 | 0.95 |
| bf16_mm_m1_k1024_n4096 | 27.63 | 24.98 | 10.69 | 11.88 |
| bf16_mm_m1_k1024_n151936 | 1224.62 | 1228.58 | 1218.84 | 1224.23 |
| rmsnorm_facade_m4_n1024 | 32.65 | 19.48 | 1.41 | 1.24 |
| silu_facade_m4_n6144 | 27.44 | 20.91 | 1.10 | 1.49 |

关键观察：大词表 BF16 矩阵乘约 1.22ms，两侧接近；64MiB 拷贝中 Windows 没有劣势。部分归一化的 eager 调用 Windows 更慢，而 graph 回放时单次执行只约 1µs，说明发射/包装开销值得优先排查。其他算子也有 Windows 更快的情况，不能说所有 CUDA 调用都慢。
算子脚本通过 RMSNorm、SwiGLU、抽样 GEMM 数值验证；微基准使用预分配输出、固定数据，不能将各项差值直接相加当作模型总差值。

## 同步与线程交接排除实验

每项预热 20 次、7 轮，每轮 100 次；D2H 为 4096 个 FP32 元素，数据落地后校验。该简化模型不等价于 TokenSpeed 的 Rust/C++ 调度器。

| 操作 | Windows µs | WSL2 µs |
| --- | ---: | ---: |
| idle_device_synchronize | 2.51 | 2.97 |
| empty_event_record_synchronize | 3.45 | 3.41 |
| add_event_synchronize | 24.22 | 66.93 |
| add_pinned_d2h_event_synchronize | 37.09 | 81.49 |
| python_queue_thread_roundtrip | 16.39 | 62.53 |
| thread_add_pinned_d2h_event_synchronize | 143.14 | 195.95 |

本轮 Windows 在这些同步与简化线程交接测试中并未更慢，不支持“Windows 每次同步必然多几毫秒”或“Windows 线程唤醒一定更慢”的解释。

## 真实模型与 CUDA Graph 对照

复用相同 BF16 模型与固定随机输入：输入 128、输出 128、batch 1/4；每 case 3 次预热、5 次实测。eager 的计时完成后，另采集一份 profiler 请求，profile 请求不计入吞吐。graph 关闭 profiler，仍关闭 prefill graph，仅将 decode eager 改为 graph，capture sizes 为 [1,4]。

| Batch | Windows eager tok/s | WSL2 eager tok/s | Windows graph tok/s | WSL2 graph tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.08 | 72.73 | 139.28 | 117.10 |
| 4 | 200.66 | 239.80 | 590.04 | 468.49 |
Batch 1：Windows graph/eager=2.575x，WSL2=1.610x；各平台 eager/graph 参考 IDs 相同：Windows=True，WSL2=True。
Graph 模式下 Windows 相对 WSL2 吞吐 +18.94%；跨平台 graph 参考 IDs 相同：True。各模式重复输出一致性见 results.json。
Batch 4：Windows graph/eager=2.941x，WSL2=1.954x；各平台 eager/graph 参考 IDs 相同：Windows=True，WSL2=True。
Graph 模式下 Windows 相对 WSL2 吞吐 +25.95%；跨平台 graph 参考 IDs 相同：False。各模式重复输出一致性见 results.json。

追加测试再次观察到 Windows eager batch=4 的重复输出 ID 差异；本轮两个平台 graph 重复输出均一致，但 batch=4 跨平台参考 IDs 仍不同。原因未定位，吞吐提高不能当作跨平台数值等价证明。
本轮 5 轮试验和原始 10 轮报告分别保留，吞吐有运行间波动；没有锁定 GPU/CPU 时钟，也不是 ABBA 多次交替的统计因果实验。graph 数值一致性不能只凭输出长度判定，逐轮 IDs 均保留在 JSON。

## Profiler 能说明什么

两侧每个典型 decode forward 均有 326 个 CUDA runtime/driver launch 回调；回调可能嵌套，不能把它直接当作 326 个不同 GPU 内核。launch API 时间使用同线程区间并集，避免重复相加。

| 平台 / Batch | decode forward 中位 ms | launch API 区间并集中位 ms | GPU 内核活动 |
| --- | ---: | ---: | --- |
| wsl2-engine / trace-b1 | 18.208 | 3.344 | 未采集到 |
| wsl2-engine / trace-b4 | 20.186 | 3.810 | 未采集到 |
| windows-engine / trace-b1 | 26.425 | 4.743 | 有 |
| windows-engine / trace-b4 | 25.011 | 4.496 | 有 |

重要限制：WSL2 的 torch profiler 有 CPU/CUDA API、没有 GPU kernel activity；该 GPU 时间是缺失值，不是零。Windows 则采集到了 GPU kernel activity，CUPTI 工作量不同。profiler 也显著改变了执行速度，因此本表不能作为无 profiler 的两平台耗时分摊，更不能据此计算纯驱动开销比例。
原始压缩 trace 保留在 .port/perf-diagnosis，公开文件只提供汇总与原始 SHA256；trace-analysis.json 保留各阶段、API、内核名称和统计。

## 不应归因的因素与后续优先级

- 不是 x64 模拟、Debug 构建或模型文件不同；此前原生审计与模型 SHA256 已通过。
- 不能归因于 NCCL 缺失：本次 world_size=1，不能套用多 GPU 通信瓶颈解释。
- WSL2 也使用 Windows 宿主 NVIDIA 驱动。NVIDIA 官方说明 WSL GPU 支持 WDDM 模式，不能称为“Windows 是 WDDM、WSL 是裸 Linux TCC”。本机 GPU 的驱动模式确认为 WDDM。
- 优先评估本模型的 decode graph，并检查输出；随后在匹配的 PyTorch/CUDA/Triton 上做逐项替换与算子路径 A/B。只有一次改变一个因素，才能分摊根因。
- 本次没有更换驱动、关闭系统防护、修改全局设置或覆盖运行时实现；微基准 graph 能运行不等于任意模型和服务模式都支持 graph。

## 参考与证据

- [NVIDIA CUDA Graphs：短内核的提交开销与 graph 摊销](https://developer.nvidia.com/blog/cuda-graphs/)。这里只引用机制，不把文中其他硬件的数值套到本机。
- [NVIDIA CUDA on WSL 官方说明](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
- [Triton Windows 项目](https://github.com/woct0rdho/triton-windows)。
- results.json：本轮汇总；windows/wsl2-micro.json 和 *-sync.json：全部逐轮样本；*-engine/*.json：真实模型的逐轮 token IDs 与耗时。
- scripts/：本次实验脚本；local-traces.json：未发布的大文件 trace 位置与 SHA256；references.json：查询地址及本地下载快照哈希；evidence-sha256.json：公开副本校验。
