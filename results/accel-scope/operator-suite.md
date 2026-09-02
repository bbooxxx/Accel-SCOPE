# Accel-SCOPE 四算子缓存探索

## 结论

本轮使用同一个 4-bank 配置评估 4 个仓库现有算子。没有只保留高收益
结果：FFN、预填充 GEMM、数据变换和 split-K 归约均列在下表。

| 算子 | latency 变化 | power 变化 | 访存能量变化 | 结论 |
|---|---:|---:|---:|---|
| FFN decode-like GEMM | 降低 **14.85%** | 降低 **26.31%** | 降低 37.25% | 受计算时间限制 |
| prefill-like GEMM | 降低 **14.32%** | 降低 **57.62%** | 降低 63.69% | DRAM 已清零，仍受计算时间限制 |
| activation/KV transform proxy | 降低 **56.64%** | 降低 **57.42%** | 降低 81.54% | 访存主导，收益最大 |
| split-K reduce | 降低 **5.66%** | **增加 12.10%** | 增加 5.76% | 基线 L2 已 100% 命中 |

延迟和功耗在所有算子上同时下降 60% 不可达。FFN 即使使用零等待的理想
内存，延迟也只能下降 21.23%。数据变换的理论上限是 64.99%，但在当前
SCOPE 器件能量下，为了让延迟和功耗同时超过 60%，还需要把其每算子访存
能量再降低至少约 13.4%。

## 算子级指标

表中数据均取第二次执行的稳态结果。`power` 是缓存层级与片外访存的算子
期间平均功耗，不是 Jetson Orin 整芯片功耗。

| 算子 | 配置 | latency | power | L1 命中率 | L2 命中率 | L3 命中率 | DRAM 请求 |
|---|---|---:|---:|---:|---:|---:|---:|
| FFN GEMM | Orin SRAM | 249.470 μs | 14.682 W | 0.70% | 2.13% | — | 1,231,027 |
| FFN GEMM | SCOPE 4-bank | 212.428 μs | 10.819 W | 0.67% | 2.35% | 99.93% | 834 |
| prefill GEMM | Orin SRAM | 138.882 μs | 7.074 W | 88.06% | 25.57% | — | 246,007 |
| prefill GEMM | SCOPE 4-bank | 118.991 μs | 2.998 W | 88.61% | 100.00% | — | 0 |
| tensor transform | Orin SRAM | 143.557 μs | 11.672 W | 0.00% | 0.00% | — | 540,001 |
| tensor transform | SCOPE 4-bank | 62.253 μs | 4.970 W | 0.00% | 100.00% | — | 0 |
| split-K reduce | Orin SRAM | 9.005 μs | 2.163 W | 23.78% | 100.00% | — | 0 |
| split-K reduce | SCOPE 4-bank | 8.495 μs | 2.425 W | 23.64% | 100.00% | — | 0 |

## 配置与 trace

4-bank 配置在每个 Orin memory channel 下设置 4 个独立 cache subpartition，同时
缩小每个 bank，因此总 L2 仍为 32 MiB，总 L3 仍为 384 MiB。器件映射、latency
和每次访问能量与第一版一致，只改变 bank 并行度。配置见
[`scope-v8-banked4.config`](../../gpu-simulator/configs/accel-scope/scope-v8-banked4.config)。

| 算子 | DeepBench 现有 trace | 执行序列 |
|---|---|---|
| FFN GEMM | `inference_half_7680_1_2560_0_0` | kernel-7 / kernel-8 |
| prefill GEMM | `inference_half_35_1500_2560_0_0` | kernel-7 / kernel-9 |
| tensor transform | `inference_half_35_1500_2560_0_0` | kernel-5 重复两次 |
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
