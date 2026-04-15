# Speculative Decoding Review Snapshot

## 核心改动说明

### 1. `JointInference` 流程扩展

保留文件：

- `core/testcasecontroller/algorithm/paradigm/joint_inference/joint_inference.py`
- `examples/resources/third_party/sedna-0.6.0.1-src/sedna/core/joint_inference/joint_inference.py`

为什么修改：

- 原始 `jointinference` 流程面向样本级推理，不能直接承载 token-level speculative decoding。
- 本次 benchmark 需要在同一样本内部进行多轮 drafter / verifier 协同，因此需要在框架层补充对应的执行流程。

怎么修改：

- 在 `JointInference` 中加入 speculative decoding 所需的多轮协同控制逻辑。
- 支持 `drafter` 提案、`verifier` 校验、接受 / 修正 / 停止等 token 级交互。
- 在统一入口下支持三种模式：`collaboration`、`cloud-only`、`edge-only`。

### 2. `drafter` / `verifier` 模块实现

保留文件：

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/ar/drafter.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/ar/verifier.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/base_drafter.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/base_verifier.py`

为什么修改：

- 需要把 speculative decoding 的职责从原有通用 `cloud` / `edge` 语义中拆出来，明确为 `drafter` 和 `verifier` 两个角色。
- 需要实现 token 级 proposal / verify / commit 逻辑，并维护 request 级状态与 cache。

怎么修改：

- 定义了 `BaseSpeculativeDrafter` 和 `BaseSpeculativeVerifier` 作为协议边界。
- 在 `drafter.py` 中实现 draft token 生成、session/cache 推进、edge-only 路径。
- 在 `verifier.py` 中实现 verify、接受率相关状态推进、cloud-only 路径。
- 清理了调试期冗余逻辑，使代码更接近最终评审版本。

### 3. Benchmark 配置与支撑代码

保留文件：

- `examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/data_processor.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/result_builder.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/common/*.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testenv/*.py`
- `examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml`

为什么修改：

- 需要一个完整 benchmark 示例，把框架改动真正落到可运行的 speculative decoding benchmark 中。
- 需要统一配置模型、模式、draft window、metrics 和 benchmark 输出。

怎么修改：

- 新增 benchmark 入口配置和 testenv 配置。
- 新增 request normalization、result construction、stop condition、timeline 等公共逻辑。
- 新增指标实现：`Time to First Token`、`Throughput`、`Internal Token Latency`、`End-to-End Latency`、`Acceptance Rate`。
