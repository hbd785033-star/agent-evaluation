# agent-evaluation

面向真实 Model / Agent / Harness 运行的五层评估框架：

```text
Versioned Dataset
      ↓
EvalRunner ── N trials
      ↓
AgentAdapter
├─ CommandAgentAdapter（Hermes / Kimi / Codex / Claude / HMC CLI）
└─ RecordedAdapter（重放已导出的 RunRecord）
      ↓
RunRecord
      ↓
Deterministic → Trajectory → Cost → Security → calibrated LLM Judge
      ↓
report.json + report.md
```

## 当前能力边界

> **`pytest` 通过只代表评估框架、数据结构和 runner 合约通过，不代表 Kimi、Hermes 或任何模型通过了任务集。**

只有 `report.json` 中实际存在的 `RunRecord` 才是模型/Provider/Harness trial。报告会同时记录 `model + provider + harness + trial`，避免把 harness 差异误算成模型能力差异。

## 核心对象：RunRecord

每次真实 trial 统一记录：

- `task_id`, `model`, `provider`, `harness`, `trial`
- `output`, `tool_calls`, `files_changed`, `trajectory`
- `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`
- `latency_seconds`, `exit_status`, `error`, `run_id`

## 安装

```bash
python -m pip install -e ".[dev]"
```

如需 DeepEval：

```bash
python -m pip install -e ".[deepeval]"
```

## 运行真实 Harness

`CommandAgentAdapter` 不使用 shell；它把任务 JSON 写入子进程 stdin，并要求 Harness 在 stdout 返回一个 JSON 对象。

```bash
python -m agent_eval run \
  --dataset datasets/core_tasks.yaml \
  --model kimi-k3 \
  --provider moonshot \
  --harness kimi-code \
  --output-dir reports/kimi-k3 \
  --command python path/to/kimi_harness.py
```

Harness 输入：

```json
{
  "task": {
    "id": "code-001",
    "prompt": "...",
    "allowed_files": ["src/auth/**"],
    "limits": {"max_tool_calls": 12, "max_cost_usd": 0.05}
  },
  "trial": 1,
  "model": "kimi-k3",
  "provider": "moonshot",
  "harness": "kimi-code"
}
```

Harness 最小输出：

```json
{
  "output": "task result",
  "tool_calls": [],
  "files_changed": [],
  "input_tokens": 1000,
  "output_tokens": 250,
  "cached_tokens": 0,
  "cost_usd": 0.01,
  "exit_status": "completed",
  "run_id": "provider-run-id"
}
```

子进程失败、超时或输出无效 JSON 时，runner 会写入失败 `RunRecord`；不会生成看似合理的虚假结果。

## 重放已有记录

```bash
python -m agent_eval run \
  --dataset datasets/core_tasks.yaml \
  --model kimi-k3 \
  --provider moonshot \
  --harness kimi-code \
  --records exported-runs.json \
  --output-dir reports/rescored
```

## 报告

`EvalRunner` 按任务、模型、Provider、Harness 聚合：

- Pass Rate / Failure Rate
- Trial variance
- Mean cost
- Mean latency
- 每层 evaluator 的证据和失败原因

输出：

```text
reports/<experiment>/report.json
reports/<experiment>/report.md
```

`reports/` 默认不提交 Git；正式基准可通过 CI artifact 或 release 附件发布。

## 五层原则

1. **Deterministic**：运行状态、allowed path、密钥模式等可编码事实。
2. **Trajectory**：禁止工具、重复读取、失败重试和工具分布。
3. **Cost**：token、cache、tool calls、sub-agent、成本与延迟。
4. **Security**：canonical 路径边界、危险命令、凭据、Prompt Injection。
5. **LLM Judge**：仅评价无法代码化的质量维度；未配置校准 Judge 时明确标记 `skipped`。

> Fail-closed：数据集声明了自然语言 deterministic criteria 却没有注册可执行 `task_check`，或声明了 `llm_judge` criteria 却没有校准 Judge 时，该 trial 不会被推断为 PASS。

LLM Judge 上线前应使用人工黄金集校准，Pearson 相关系数建议至少 `0.7`。

## 测试

```bash
# 全部框架测试（含两个公网 URL probe）
python -m pytest tests/ -v

# CI 同款：无公网依赖
python -m pytest tests/ -v \
  -m "not regression" \
  -k "not github_url_reachable and not phantom_url_not_reachable"

# 真正选择得到 regression 测试，不再发生 0 tests 假绿
python -m pytest tests/test_runner.py -v -m regression
```

当前测试覆盖 evaluator 单元逻辑、canonical 路径/注入边界、数据集 schema、真实 subprocess N-trial runner、失败 harness 记录、JSON/Markdown 报告生成。

## CI 语义

GitHub Actions 只声明验证：

- framework unit tests；
- real subprocess runner smoke；
- marked regression dataset/runner contract。

CI 不再把这些 job 命名成“模型宽评测”，也不向没有真实模型调用的 job 注入 API Key。真实 Kimi/Hermes/HMC A/B 评估必须配置对应 `AgentAdapter` 并保留 `RunRecord` 报告。

## 目录

```text
agent_eval/
├── models.py       # TaskCase / RunRecord / EvaluatedRun
├── adapters.py     # AgentAdapter / CommandAgentAdapter / RecordedAdapter
├── dataset.py      # versioned YAML loader
├── runner.py       # N trials / five layers / aggregate / reports
└── cli.py          # python -m agent_eval run

evaluators/         # deterministic / trajectory / cost / security / llm_judge
datasets/           # core / failure / safety cases
tests/              # framework + runner + regression tests
```

> 测试集质量最终比使用哪个模型更重要；而真实运行记录比“评估器函数测试通过”更重要。
