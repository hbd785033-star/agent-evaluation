# agent-evaluation

**五层 Agent 评估框架**：确定性检查 → 轨迹评估 → 成本评估 → 安全检查 → LLM Judge。

## 设计原则

1. **能用代码判断的，绝不先交给 LLM**
2. **检查 Outcome（环境状态），不只检查对话文本**
3. **同一任务跑多次 trial 取平均**，消除 LLM 输出的随机性
4. **LLM 评分器上线前必须用人工黄金集校准**（Pearson ≥ 0.7）
5. **测试集版本化，独立于 Agent 代码管理**

## 目录结构

```
agent-evaluation/
├── datasets/            # 测试集（版本化，独立 git tag）
│   ├── core_tasks.yaml      # 核心功能任务
│   ├── failure_cases.yaml   # 回归测试（生产故障复现）
│   └── safety_cases.yaml    # 安全测试
├── rubrics/             # LLM Judge 评分标准
│   ├── coding.md
│   ├── research.md
│   └── orchestration.md
├── evaluators/          # 评估器实现
│   ├── deterministic.py     # 第1层：确定性检查
│   ├── llm_judge.py         # 第2层：LLM 质量评分
│   ├── trajectory.py        # 第3层：执行轨迹分析
│   ├── cost.py              # 第4层：效率/成本评估
│   └── security.py          # 第5层：安全检查
├── tests/
│   └── test_agent.py        # pytest 测试套件
├── skills/
│   └── agent-evaluator/
│       └── SKILL.md         # 运行时评估 Skill
├── reports/             # 评估报告（gitignored）
└── .github/
    └── workflows/
        └── agent-evals.yml  # CI/CD 自动化
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行窄 evals（无需 API Key，CI 每次 commit 跑）
pytest tests/ -v -k "Deterministic or Trajectory or Cost or Security or Dataset"

# 运行全套
OPENAI_API_KEY=sk-... pytest tests/ -v

# 运行单个评估器
python evaluators/deterministic.py
python evaluators/trajectory.py
python evaluators/cost.py
python evaluators/security.py
python evaluators/llm_judge.py --provider openai
```

## CI 策略

| 触发时机 | 运行内容 | 需要 API Key |
|----------|---------|-------------|
| 每次 commit | 窄 evals（确定性+轨迹+安全+成本） | 否 |
| PR → main | 宽 evals（全套） | 是 |
| 推送 main | 回归测试（failure_cases） | 是 |

## 推荐工具栈

| 工具 | 用途 |
|------|------|
| [DeepEval](https://github.com/confident-ai/deepeval) ⭐17k | pytest 风格单元级 evals + 回归 |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) ⭐11k | OpenTelemetry Trace 存储 + 可观测性 |
| [Promptfoo](https://github.com/promptfoo/promptfoo) ⭐24k | 多供应商横向比较 + 红队测试 + CI |
| [AgentEvals](https://github.com/langchain-ai/agentevals) | 执行轨迹精细化评估 |

## 添加新测试案例

在 `datasets/core_tasks.yaml` 中追加，格式参考现有条目：

```yaml
- id: your-task-id
  category: coding|research|orchestration
  description: 任务描述
  prompt: |
    具体任务指令
  success_criteria:
    deterministic:
      - 检查项1
    llm_judge:
      - 质量要求1
  limits:
    max_agents: 1
    max_tool_calls: 10
    max_cost_usd: 0.05
  trials: 3
```

## 关于测试集版本化

```bash
# 发布新版本测试集时打 tag，与 Agent 代码解耦
git tag datasets-v1.1.0
git push origin datasets-v1.1.0
```

> 测试集的质量最终比你使用哪个模型更有价值。
