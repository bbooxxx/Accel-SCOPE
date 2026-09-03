# Accel-SCOPE 短算子仿真报告

## 1. 算子来源

3 个算子均来自 [Baidu Research DeepBench](https://github.com/baidu-research/DeepBench/tree/da81ba7820739e2e506dc27f382d15be5479f98f)，
使用 Accel-Sim 仓库已有的 NVBit trace；没有重新编译 CUDA 程序或生成 trace。

| 算子 | DeepBench benchmark | 使用的 trace/kernel |
|---|---|---|
| GEMM | `gemm_bench-tencore train half 2048 64 2048 1 0` | `train_half_2048_64_2048_1_0`，kernel-7/9，`cutlass_70_tensorop_h884gemm_128x128_tn_align8` |
| 小批量 GEMM（接近 GEMV） | `gemm_bench-tencore inference half 6144 4 2048 0 0` | `inference_half_6144_4_2048_0_0`，kernel-7/9，`cutlass_70_tensorop_h884gemm_64x64_nn_align8` |
| small-M GEMM | `gemm_bench-tencore inference half 35 1500 2560 0 0` | `inference_half_35_1500_2560_0_0`，kernel-7/9，`cutlass_70_tensorop_h884gemm_64x64_nn_align1` |

DeepBench 程序来源：[`gemm_bench.cu`](https://github.com/accel-sim/gpu-app-collection/blob/dad09cb0487845edc7524ded814c6cde9f0ef6a1/src/cuda/DeepBench/code/nvidia/gemm_bench.cu)。

## 2. 算子配置

| 算子 | 计算 | 输入 | 输出 | 数据规模 |
|---|---|---|---|---:|
| GEMM | FP16 `C ← Aᵀ×B+C`；M=2048，N=64，K=2048 | A: `2048×2048`；B: `2048×64`；C: `2048×64` | C: `2048×64` FP16 | 8.50 MiB |
| 小批量 GEMM（接近 GEMV） | FP16 `C ← A×B+C`；M=6144，N=4，K=2048 | A: `6144×2048`；B: `2048×4`；C: `6144×4` | C: `6144×4` FP16 | 24.06 MiB |
| small-M GEMM | FP16 `C ← A×B+C`；M=35，N=1500，K=2560 | A: `35×2560`；B: `2560×1500`；C: `35×1500` | C: `35×1500` FP16 | 7.60 MiB |

每个矩阵核连续执行两次，表中使用第二次的稳态结果。对比配置为：

- Orin SRAM baseline：原有 L1/L2 SRAM，无 L3；
- SCOPE 4-bank：L1 SRAM + 32 MiB L2 TFET-eDRAM + 384 MiB L3 OSFET-eDRAM。

## 3. latency、power 与缓存命中率

| 算子 | 缓存配置 | latency | power | L1 命中率 | L2 命中率 | L3 命中率 |
|---|---|---:|---:|---:|---:|---:|
| GEMM | Orin SRAM baseline | 103.606 μs | 9.050 W | 12.35% | 35.95% | 无 L3 |
| GEMM | SCOPE 4-bank | 91.465 μs（降低 **11.72%**） | 2.392 W（降低 **73.57%**） | 14.94% | 100.00% | 未访问 |
| 小批量 GEMM（接近 GEMV） | Orin SRAM baseline | 150.972 μs | 15.601 W | 4.51% | 1.28% | 无 L3 |
| 小批量 GEMM（接近 GEMV） | SCOPE 4-bank | 141.572 μs（降低 **6.23%**） | 2.487 W（降低 **84.06%**） | 4.53% | 100.00% | 未访问 |
| small-M GEMM | Orin SRAM baseline | 138.882 μs | 7.074 W | 88.06% | 25.57% | 无 L3 |
| small-M GEMM | SCOPE 4-bank | 118.991 μs（降低 **14.32%**） | 2.998 W（降低 **57.62%**） | 88.61% | 100.00% | 未访问 |

3 个工作集都能装入扩容后的 32 MiB L2，因此第二次执行的片外 DRAM 请求均降为 0，
latency 和 power 随之降低；请求全部在 L2 命中，所以没有继续访问 L3。
表中 power 是缓存层级与片外存储的算子期间平均功耗，不是 Jetson Orin 整芯片功耗。
