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

只有 `report.json` 中实际存在且通过完整性校验的 `RunRecord` 才是模型/Provider/Harness trial。`task_id + model + provider + harness + trial` 由 Runner 固定，Harness 不能覆盖；重复 `run_id`、负 telemetry、`cached_tokens > input_tokens` 或错误字段类型会将 trial 标记失败。

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

### 可重复 PASS 演示

以下命令使用仓库内的 controlled Perfect Agent、checker profile 和固定 Judge
fixture，不调用外部模型：

```bash
python -m agent_eval run \
  --dataset datasets/smoke_tasks.yaml \
  --model controlled-perfect \
  --provider fixture \
  --harness perfect-agent \
  --checker-profile profiles/checkers/smoke-v1.yaml \
  --judge-profile profiles/judges/controlled-v1.json \
  --source-cwd . \
  --workspace-root ../agent-eval-smoke-workspaces \
  --output-dir reports/smoke \
  --command python examples/perfect_agent.py
```

成功时 CLI 返回 `0`，并在 `report.json` 中记录每条 criterion evidence 与
Judge 的 `calibration_id`、prompt 和 rubric 版本。`controlled-v1` 仅用于测试
闭环，不代表生产 LLM Judge 已完成校准。

`dataset`、`checker_profile`、`judge_profile` 和 `trials` 可写入 `eval.yaml`，再运行
`python -m agent_eval run --config eval.yaml ...`；model/provider、workspace 与输入源仍由
CLI 显式提供，避免配置文件悄然改变执行边界。

`CommandAgentAdapter` 不使用 shell；它把任务 JSON 写入子进程 stdin，并要求 Harness 在 stdout 返回一个 JSON 对象。每个 task/trial 默认获得独立的临时 **workspace**；`--source-cwd` 指定的源树会被复制到各自 workspace，canonical path 检查绑定同一个可信根。workspace 复制不等于操作系统 sandbox，不能阻止恶意进程访问宿主机其他绝对路径。因此含 `allowed_files`、`forbidden_files` 或 `forbidden_actions` 的任务只有在控制面确认 `isolation_level=os` 时才能通过；普通 Command adapter 会报告 `isolation_level=workspace` 并对这类任务 fail-closed。

```bash
python -m agent_eval run \
  --dataset datasets/core_tasks.yaml \
  --model kimi-k3 \
  --provider moonshot \
  --harness kimi-code \
  --source-cwd path/to/source-tree \
  --workspace-root reports/workspaces \
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

子进程失败、超时、输出 `null`/无效 JSON 或返回 malformed telemetry 时，runner 会写入失败 `RunRecord`；不会生成看似合理的虚假结果。默认在评分后删除 workspace；需要审计工作区时使用 `--preserve-workspaces`。只要存在失败 trial 或聚合 pass rate 小于 100%，CLI 返回非零状态。

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

重放时，记录内自报的 `workspace_root`、`dataset_version`、`sandbox_id` 和 `isolation_level`
不被信任。没有控制面 authority 时，所有路径与隔离相关检查 fail-closed。

## 评估 AAO ExecutionRecord 0.1

```bash
python -m agent_eval evaluate execution-record.json \
  --dataset datasets/smoke_tasks.yaml \
  --checker-profile profiles/checkers/smoke-v1.yaml \
  --judge-profile profiles/judges/controlled-v1.json \
  --workspace-authority workspace-authority.json \
  --output-dir reports/aao
```

`workspace-authority.json` 是控制面生成的独立 artifact，按任务和 trial 精确绑定 workspace：

```json
{
  "schema_version": "0.1",
  "trusted_workspace_root": "/absolute/path/to/imported-workspaces",
  "isolation_level": "os",
  "workspaces": [
    {
      "task_id": "smoke-perfect-001",
      "trial": 1,
      "workspace_root": "/absolute/path/to/imported-workspaces/smoke-perfect-001-trial-1"
    }
  ]
}
```

输入必须精确匹配 `ExecutionRecord 0.1`。未知字段、缺失字段、重复 run ID 或
不支持的版本会被明确拒绝，不会静默忽略。外部记录不能选择可信根下的任意 sibling
workspace；只有 `--workspace-authority` 中与 `(task_id, trial)` 精确匹配的路径才会成为
checker authority。缺失、重复或越界 binding 会 fail-closed。

数据集中的 `file_exists` / `file_contains` 路径会 canonicalize 到可信 workspace；绝对路径、
`..` 与 symlink 逃逸会被拒绝。`command` / `pytest` 不接受数据集内联 argv，只能由调用者
显式选择的 checker profile 提供，因此不应加载来源不可信的 profile。

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
5. **LLM Judge**：仅评价无法代码化的质量维度；Judge 必须显式声明 `calibrated=True` 并返回严格 schema，否则 fail-closed。报告记录完整 calibration identity 及其 SHA-256 内容哈希。

> Fail-closed：数据集声明的 unknown criteria、未执行的 deterministic criteria、`forbidden_files`/`forbidden_actions`，或未配置校准 Judge 的 `llm_judge` criteria，均不会被推断为 PASS。任务检查器必须逐条覆盖声明的 criteria。

LLM Judge 上线前应使用人工黄金集校准，Pearson 相关系数建议至少 `0.7`。

## 测试

```bash
# 全部框架测试（含两个公网 URL probe）
python -m pytest tests/ -v

# CI 同款：无公网依赖
python -m pytest tests/ -v \
  -m "not regression and not network"

# 真正选择得到 regression 测试，不再发生 0 tests 假绿
python -m pytest tests/test_runner.py tests/test_review_regressions.py -v -m regression
```

当前测试覆盖 evaluator 单元逻辑、sandbox/canonical 路径/注入边界、数据集 schema、真实 subprocess N-trial runner、trial 污染隔离、身份与 telemetry 完整性、失败 harness 记录、JSON/Markdown 报告生成，以及 `failure_cases.yaml` 的真实评分。

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
