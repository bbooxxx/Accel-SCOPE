# Accel-SCOPE 四算子缓存探索

## 结论

本轮使用同一个 4-bank 配置评估 4 个仓库现有算子。没有只保留高收益
结果：FFN、预填充 GEMM、数据变换和 split-K 归约均列在下表。

| 算子 | latency 变化 | power 变化 | 访存能量变化 | 结论 |
|---|---:|---:|---:|---|
| batch-1 GEMM（FFN/decode 代理） | 降低 **14.85%** | 降低 **26.31%** | 降低 37.25% | 受计算时间限制 |
| small-M GEMM（prefill 代理） | 降低 **14.32%** | 降低 **57.62%** | 降低 63.69% | DRAM 已清零，仍受计算时间限制 |
| FP32→FP16 tensor transform（访存代理） | 降低 **56.64%** | 降低 **57.42%** | 降低 81.54% | 访存主导，收益最大 |
| split-K reduce | 降低 **5.66%** | **增加 12.10%** | 增加 5.76% | 基线 L2 已 100% 命中 |

延迟和功耗在所有算子上同时下降 60% 不可达。batch-1 GEMM 代理即使使用
零等待的理想内存，延迟也只能下降 21.23%。数据变换的理论上限是 64.99%，但在当前
SCOPE 器件能量下，为了让延迟和功耗同时超过 60%，还需要把其每算子访存
能量再降低至少约 13.4%。

## 算子级指标

表中数据均取第二次执行的稳态结果。`power` 是缓存层级与片外访存的算子
期间平均功耗，不是 Jetson Orin 整芯片功耗。

| 算子 | 配置 | latency | power | L1 命中率 | L2 命中率 | L3 命中率 | DRAM 请求 |
|---|---|---:|---:|---:|---:|---:|---:|
| batch-1 GEMM proxy | Orin SRAM | 249.470 μs | 14.682 W | 0.70% | 2.13% | — | 1,231,027 |
| batch-1 GEMM proxy | SCOPE 4-bank | 212.428 μs | 10.819 W | 0.67% | 2.35% | 99.93% | 834 |
| small-M GEMM proxy | Orin SRAM | 138.882 μs | 7.074 W | 88.06% | 25.57% | — | 246,007 |
| small-M GEMM proxy | SCOPE 4-bank | 118.991 μs | 2.998 W | 88.61% | 100.00% | — | 0 |
| FP32→FP16 transform | Orin SRAM | 143.557 μs | 11.672 W | 0.00% | 0.00% | — | 540,001 |
| FP32→FP16 transform | SCOPE 4-bank | 62.253 μs | 4.970 W | 0.00% | 100.00% | — | 0 |
| split-K reduce | Orin SRAM | 9.005 μs | 2.163 W | 23.78% | 100.00% | — | 0 |
| split-K reduce | SCOPE 4-bank | 8.495 μs | 2.425 W | 23.64% | 100.00% | — | 0 |

## Benchmark 来源

本报告使用的不是完整 Transformer 或具身智能模型，而是
[Baidu Research DeepBench](https://github.com/baidu-research/DeepBench/tree/da81ba7820739e2e506dc27f382d15be5479f98f)
的基础算子。DeepBench 的目标是比较深度学习中常见的 GEMM、卷积和 RNN 等底层操作，
它不运行端到端模型。因此，报告中的“FFN/decode”和“prefill”是根据矩阵形状与
访存特征做的代理解释，不是 DeepBench 或 trace 自带的 Transformer 语义。

来源链如下：

1. DeepBench 在
   [`gemm_problems.h`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/kernels/gemm_problems.h)
   中定义推理 GEMM 形状。
2. Accel-Sim 的
   [`gpu-app-collection`](https://github.com/accel-sim/gpu-app-collection/tree/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench)
   保留 DeepBench CUDA 实现；本仓库的
   [`define-all-apps.yml`](../../util/job_launching/apps/define-all-apps.yml)
   明确列出了本次两个命令行。
3. Accel-Sim 通过 NVBit tracer 生成 SASS trace。
   [`generate-volta-traces.sh`](../../util/tracer_nvbit/generate-volta-traces.sh)
   将 `Deepbench_nvidia_tencore` 和 `Deepbench_nvidia_normal` 打包为 `deepbench.tgz`；
   [`get-accel-sim-traces.py`](../../get-accel-sim-traces.py)
   定义了 Accel-Sim 1.1.0 trace archive 的下载位置。
4. 本次未重新编译 DeepBench，也未安装 CUDA；只读取服务器上已解压的
   `hw_run/deepbench/11.0/gemm_bench/<case>/traces` 文件。trace 未提供应用源码 commit，
   因此能够追溯到 DeepBench 实现和 Accel-Sim 生成流程，但无法证明当初生成
   archive 时使用的 `gpu-app-collection` 精确 SHA。

## GEMM 参数与工作集

DeepBench 命令行格式是
`gemm_bench-tencore inference half M N K A_transpose B_transpose`。
[`gemm_bench.cu`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/nvidia/gemm_bench.cu)
通过 `cublasGemmEx` 计算 `A × B = C`；本次两个 case 均为 FP16、NN（A/B 均不转置）。

| trace case | 原始参数 | 逻辑矩阵 | 纯 FP16 矩阵约占用 | 本报告的用途 |
|---|---|---|---:|---|
| `inference_half_7680_1_2560_0_0` | M=7680, N=1, K=2560, NN | A: 7680×2560; B: 2560×1; C: 7680×1 | 37.52 MiB | batch-1 GEMM/GEMV，作为单 token FFN/decode 投影代理 |
| `inference_half_35_1500_2560_0_0` | M=35, N=1500, K=2560, NN | A: 35×2560; B: 2560×1500; C: 35×1500 | 7.60 MiB | small-M 多输出 GEMM，作为多 token/prefill 投影代理 |

上表只计算 A/B/C，不包括 cuBLAS 的 split-K 临时空间。第一个 case 的逻辑工作集
大于 32 MiB L2，需要 L3 承接跨轮复用；第二个 case 的主矩阵可放入 32 MiB L2，
但无法放入 4 MiB baseline L2，这与仿真中第二轮 L2 命中率从 25.57% 升到
100% 的现象一致。

## 四个算子的真实身份

| 报告名称 | trace kernel | trace 中的实现 | launch 元数据 | 如何解读 |
|---|---|---|---|---|
| batch-1 GEMM proxy | kernel-7 / kernel-8 | `cutlass_70_tensorop_h884gemm_128x64_nn_align8` | grid 8×8×10; block 128; 24 KiB shared memory; 128 regs/thread | 两个 kernel 分别来自 `cublasGemmEx` 的 warm-up 和正式调用；取 kernel-8 作稳态数据。只是 FFN/decode 形状代理，不包括激活、归一化或残差。 |
| small-M GEMM proxy | kernel-7 / kernel-9 | `cutlass_70_tensorop_h884gemm_64x64_nn_align1` | grid 24×1×11; block 128; 16 KiB shared memory; 134 regs/thread | grid-z=11 表示该 cuBLAS 实现采用 split-K。kernel-7/9 是 warm-up/正式 GEMM，取 kernel-9。它不是 Attention QKᵀ、softmax 或 AV 的完整序列。 |
| FP32→FP16 transform proxy | kernel-5 重复两次 | Thrust `unary_transform`/`copy`，从 `float` 读取并写入 `uint16_t` | grid 7500×1×1; block 256; 0 shared memory; 12 regs/thread | 源自 [`tensor.h`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/nvidia/tensor.h) 中的 FP16 随机张量初始化，不是模型的真实 KV-cache 或激活算子。本报告只把它当作约 23.04 MB（21.97 MiB）连续读写工作集的访存代理。 |
| split-K reduce | kernel-8 / kernel-10 | cuBLAS `splitKreduce_kernel<half,...>` | grid 411×1×1; block 128; 0 shared memory; 32 regs/thread | 对上一行 GEMM 的 11 份 K 切片部分结果做归约。工作集较小，baseline L2 已能完全容纳，因而更换为 eDRAM 后功耗反而上升。 |

DeepBench 的 `time_gemm` 先执行一次 warm-up，再执行一次计时 GEMM。为获得算子粒度
指标，本报告没有执行原始 10-kernel 端到端序列，而是用独立 `kernelslist` 分别重放
GEMM、transform 和 reduce，并取第二条 `ACCEL_SCOPE_OPERATOR` 记录。所以这些数据是
“隔离算子的稳态缓存复用”，不是 DeepBench 端到端 latency，也不是冷启动结果。

## Trace 元数据与跨架构边界

| 字段 | 值 | 含义 |
|---|---|---|
| trace 目录 | `deepbench/11.0/gemm_bench/...` | `11.0` 表示 trace 集的 CUDA 工具链目录 |
| `binary version` | 70 | 被追踪 SASS 是 SM70/Volta 二进制，不是 Orin SM87 原生二进制 |
| `nvbit version` | 1.4 | 生成指令/地址 trace 的 NVBit 版本 |
| `accelsim tracer version` | 3 | Accel-Sim trace 格式版本 |
| 仿真配置 | Jetson Orin SM87 | 执行资源、时钟、缓存与内存时序由 Orin 配置提供 |

这是“SM70 指令与地址流 + SM87/Orin 微架构”的跨架构 trace-driven 比较。它适合在同一
trace 上比较缓存设计，但不能替代 Jetson Orin 实机标定，也不能代表 OpenVLA、
LLaMA 或其他具体模型的端到端 Attention/FFN 表现。

## 配置与执行序列

4-bank 配置在每个 Orin memory channel 下设置 4 个独立 cache subpartition，同时
缩小每个 bank，因此总 L2 仍为 32 MiB，总 L3 仍为 384 MiB。器件映射、latency
和每次访问能量与第一版一致，只改变 bank 并行度。配置见
[`scope-v8-banked4.config`](../../gpu-simulator/configs/accel-scope/scope-v8-banked4.config)。

| 算子 | DeepBench 现有 trace | 执行序列 |
|---|---|---|
| batch-1 GEMM proxy | `inference_half_7680_1_2560_0_0` | kernel-7 / kernel-8 |
| small-M GEMM proxy | `inference_half_35_1500_2560_0_0` | kernel-7 / kernel-9 |
| FP32→FP16 transform proxy | `inference_half_35_1500_2560_0_0` | kernel-5 重复两次 |
| split-K reduce | `inference_half_35_1500_2560_0_0` | kernel-8 / kernel-10 |

所有实验只读取仓库现有 trace，没有安装 CUDA，也没有生成新 trace。机器可读数据见
[`operator-suite.json`](operator-suite.json)。

## bank 探索

| tensor transform 配置 | latency | latency 下降 | power | power 下降 |
|---|---:|---:|---:|---:|
| Orin SRAM | 143.557 μs | — | 11.672 W | — |
| SCOPE 原始组织 | 68.825 μs | 52.06% | 4.497 W | **61.47%** |
| 2-bank/channel | 62.719 μs | 56.31% | 4.933 W | 57.74% |
| 4-bank/channel | 62.253 μs | **56.64%** | 4.970 W | 57.42% |
| 理想内存下界 | 50.261 μs | 64.99% | — | — |

把数据端口从 32 B/cycle 改为 128 B/cycle 没有改变结果，说明瓶颈不在单端口
宽度。4-bank 相比 2-bank 的收益已很小，继续增加 bank 不是有效方向。
