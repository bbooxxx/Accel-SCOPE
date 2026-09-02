# Accel-SCOPE 三算子报告

## 1. 算子来源

三个算子均来自 [Baidu Research DeepBench](https://github.com/baidu-research/DeepBench/tree/da81ba7820739e2e506dc27f382d15be5479f98f)，
使用 Accel-Sim 仓库已有的 NVBit trace，没有重新编译 CUDA 程序或生成 trace。

| 算子 | DeepBench 来源 | 使用的 trace/kernel |
|---|---|---|
| batch-1 GEMM（FFN/decode 代理） | `gemm_bench-tencore inference half 7680 1 2560 0 0` | `inference_half_7680_1_2560_0_0`，kernel-7/8，`cutlass_70_tensorop_h884gemm_128x64_nn_align8` |
| small-M GEMM（prefill 代理） | `gemm_bench-tencore inference half 35 1500 2560 0 0` | `inference_half_35_1500_2560_0_0`，kernel-7/9，`cutlass_70_tensorop_h884gemm_64x64_nn_align1` |
| FP32→FP16 tensor transform（访存代理） | 上述 small-M DeepBench case 中的 FP16 随机输入生成 | kernel-5 重复两次，Thrust `copy`/`unary_transform` |

DeepBench 代码来源：[`gemm_bench.cu`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/nvidia/gemm_bench.cu)、
[`tensor.h`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/nvidia/tensor.h)。

## 2. 算子配置

| 算子 | 计算 | 输入 | 输出 | 数据规模 |
|---|---|---|---|---:|
| batch-1 GEMM | FP16 `C ← A×B+C`，M=7680, N=1, K=2560，A/B 均不转置 | A: `7680×2560` FP16；B: `2560×1` FP16；C: `7680×1` FP16 | C: `7680×1` FP16 | A/B/C 合计 37.52 MiB |
| small-M GEMM | FP16 `C ← A×B+C`，M=35, N=1500, K=2560，A/B 均不转置 | A: `35×2560` FP16；B: `2560×1500` FP16；C: `35×1500` FP16 | C: `35×1500` FP16 | A/B/C 合计 7.60 MiB |
| FP32→FP16 transform | 将 small-M GEMM 的 B 矩阵随机值从 FP32 转为 FP16 | `2560×1500` FP32，3,840,000 个元素 | `2560×1500` FP16，3,840,000 个元素 | 读 15.36 MB，写 7.68 MB，合计 23.04 MB |

对比配置为 Jetson Orin L1/L2 SRAM baseline（无 L3）与 SCOPE 4-bank 三级缓存：
L1 SRAM + 32 MiB L2 TFET-eDRAM + 384 MiB L3 OSFET-eDRAM。

## 3. latency、power 与缓存命中率

| 算子 | 缓存配置 | latency | power | L1 命中率 | L2 命中率 | L3 命中率 |
|---|---|---:|---:|---:|---:|---:|
| batch-1 GEMM | Orin SRAM baseline | 249.470 μs | 14.682 W | 0.70% | 2.13% | 无 L3 |
| batch-1 GEMM | SCOPE 4-bank | 212.428 μs（降低 **14.85%**） | 10.819 W（降低 **26.31%**） | 0.67% | 2.35% | 99.93% |
| small-M GEMM | Orin SRAM baseline | 138.882 μs | 7.074 W | 88.06% | 25.57% | 无 L3 |
| small-M GEMM | SCOPE 4-bank | 118.991 μs（降低 **14.32%**） | 2.998 W（降低 **57.62%**） | 88.61% | 100.00% | 未访问 |
| FP32→FP16 transform | Orin SRAM baseline | 143.557 μs | 11.672 W | 0.00% | 0.00% | 无 L3 |
| FP32→FP16 transform | SCOPE 4-bank | 62.253 μs（降低 **56.64%**） | 4.970 W（降低 **57.42%**） | 0.00% | 100.00% | 未访问 |

latency 降低主要来自 L2/L3 承接了原本需要访问片外 DRAM 的请求；
power 降低主要来自高能耗片外访存减少。表中 power 是缓存层级与片外存储的
算子期间平均功耗，不是 Jetson Orin 整芯片功耗。
