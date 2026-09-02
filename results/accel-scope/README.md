# Accel-SCOPE：Jetson Orin 三级异构缓存评估

## 结论

Accel-SCOPE 已在 Jetson Orin 的 trace-driven Accel-Sim 模型中加入真实的
`L2 miss → L3 → DRAM` 数据路径，并按算子输出 latency、power、各级命中率和
片外请求数。实验只读取仓库已有 trace；没有 Jetson Orin 实机，也没有安装
CUDA 或生成新 trace。

默认对比使用 DeepBench FP16 GEMM `inference_half_7680_1_2560_0_0` 的
kernel-7/kernel-8。二者是地址完全相同的 CUTLASS GEMM，接近 batch-1 FFN
投影；kernel-7 用于预热，表中取 kernel-8 的稳态算子结果。

| 配置 | latency | power¹ | L1 命中率 | L2 命中率 | L3 命中率 | DRAM 请求 |
|---|---:|---:|---:|---:|---:|---:|
| Orin 默认 L1/L2 SRAM | 249.470 μs | 14.682 W | 0.70% | 2.13% | — | 1,231,027 |
| SCOPE 最优异构映射 | 212.135 μs | 10.835 W | 0.69% | 2.33% | 99.93% | 889 |
| 变化 | **降低 14.97%** | **降低 26.21%** | -0.01 pp | +0.20 pp | — | **减少 99.93%** |

¹ `power` 是算子执行期间的缓存层级与片外访存平均功耗，不是未经实机标定的
Orin 整芯片功耗。总访存能量同时降低 37.25%。

收益来自稳态权重复用：L3 将约 123 万次片外请求降到 889 次，以 35-cycle
L3 访问替代 604-cycle DRAM 访问；同时 L3/DRAM 每次访问能量分别按
1428.444 pJ/2560 pJ 计入。

## 冷缓存结果

单独只跑 kernel-7 时没有 L3 复用，结果会变差。这是本次迭代选择连续推理
场景的原因，也说明结果不能外推到一次性流式 workload。

| 配置 | latency | power | L1 命中率 | L2 命中率 | L3 命中率 | DRAM 请求 |
|---|---:|---:|---:|---:|---:|---:|
| Orin 默认 L1/L2 SRAM | 249.592 μs | 14.676 W | 0.69% | 2.13% | — | 1,231,097 |
| SCOPE 最优异构映射 | 256.263 μs | 21.235 W | 0.74% | 2.27% | 0.00% | 1,228,559 |

冷缓存下延迟增加 2.67%，功耗增加 44.69%。因此 Accel-SCOPE 的收益条件是
工作集在连续算子或连续 token 间复用，而不是“增加 L3 必然更快”。

## 配置映射

| 层级 | Orin baseline | SCOPE 映射 | 可调参数 |
|---|---|---|---|
| L1 | SRAM，保留 Orin 192 KiB/SM | SRAM，保留 Orin 组织 | 原 `gpgpu_l1_latency`、访问能量、静态功耗 |
| L2 | SRAM，4 MiB 总容量 | TFET-eDRAM，32 MiB 总容量 | 容量/相连度/策略、额外 latency、能量、静态功耗 |
| L3 | 无 | OSFET-eDRAM，384 MiB 总容量 | 容量/相连度/策略、latency、能量、静态与刷新功耗 |

L2/L3 容量按 Orin 的 16 个 memory partition 分配；均为 128 B line、16-way、
LRU。SCOPE 的全局 L1 容量无法无歧义映射到 GPU 的每 SM L1，因此保留 Orin
L1，只映射其 SRAM 器件参数。默认功耗采用
`P_static + Σ(accesses × energy/access) / operator_latency`；还提供
`accel_scope_noncache_power_mw` 供后续整芯片标定。

## 复现

服务器使用现有 CUDA 12.8 头文件完成编译，但没有安装 CUDA，也没有运行
tracer。已有可执行文件和 DeepBench trace 时只需：

```bash
source ./gpu-simulator/setup_environment.sh release
./util/accel_scope/run_existing_trace.sh \
  /data/bxzheng/github/accel-sim-framework/hw_run/deepbench/11.0/gemm_bench/inference_half_7680_1_2560_0_0/traces
```

原始算子行格式为 `ACCEL_SCOPE_OPERATOR ...`，机器可读结果见
[`summary.json`](summary.json)。

## 边界

- 这是无实机条件下的相对比较，未做 Jetson Orin 绝对值校准。
- 现有 DeepBench trace 的 binary version 为 70；执行资源与内存时序使用 Orin
  SM87 配置，因此适合设计空间比较，不应作为 Orin 实测值。
- 该算子近似 FFN GEMM，不代表 softmax、KV-cache 等完整 Attention 行为。
- 标定来源：SCOPE `3fa4fdb`、Orin Accel-Sim `a10fbea`、底层 GPGPU-Sim
  `09c0fea`。

## 后续探索

第一版结果保持不变。后续 4-bank 配置和 4 算子比较见
[《Accel-SCOPE 四算子缓存探索》](operator-suite.md)。
