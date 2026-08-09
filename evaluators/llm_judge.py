"""
LLM Judge：对无法用代码判断的质量维度进行评分。
需要设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY。

运行方式：
  python evaluators/llm_judge.py \
    --task-id code-001 \
    --task-desc "修复登录接口错误" \
    --result-file reports/code-001-result.txt \
    --rubric rubrics/coding.md
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

JUDGE_SYSTEM_PROMPT = """你是一个严格的 Agent 任务评估员。
你的工作是评估 Agent 完成任务的质量。

规则：
1. 不根据 Agent 的自我陈述判定成功——检查证据。
2. 每项批评必须附证据（文件名、行号、具体内容）。
3. 先看确定性事实，再做主观判断。
4. 不得自行降低验收标准。
5. 评分标准：100=完美，80-99=合格，60-79=有问题，<60=不合格。

输出严格遵循 JSON 格式，不添加任何额外文字。"""

JUDGE_USER_TEMPLATE = """## 任务描述
{task_desc}

## 评分标准（Rubric）
{rubric}

## Agent 输出结果
{result}

## 确定性检查结果
{deterministic_checks}

请按以下 JSON 格式输出评分：
{{
  "passed": <true|false>,
  "score": <0-100>,
  "summary": "<一句话总结>",
  "failures": ["<具体失败原因，附证据>", ...],
  "evidence": ["<支持判断的证据>", ...],
  "strengths": ["<做得好的地方>", ...],
  "retry_instruction": "<如不合格，给出最小修复指令；合格则为空字符串>"
}}"""


def call_llm_judge(
    task_desc: str,
    rubric: str,
    result: str,
    deterministic_checks: dict,
    model: str = "gpt-4o-mini",
    provider: str = "openai"
) -> dict[str, Any]:
    """调用 LLM 进行评分，返回结构化结果。"""

    prompt = JUDGE_USER_TEMPLATE.format(
        task_desc=task_desc,
        rubric=rubric,
        result=result[:4000],   # 避免超出 context
        deterministic_checks=json.dumps(deterministic_checks, ensure_ascii=False, indent=2)
    )

    if provider == "openai":
        return _call_openai(prompt, model)
    elif provider == "anthropic":
        return _call_anthropic(prompt, model)
    else:
        raise ValueError(f"unknown provider: {provider}")


def _call_openai(prompt: str, model: str) -> dict[str, Any]:
    import json as _json
    import urllib.request
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set", "passed": False, "score": 0}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=_json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = _json.load(resp)
    content = data["choices"][0]["message"]["content"]
    result = _json.loads(content)
    result["model_used"] = model
    result["tokens"] = data.get("usage", {})
    return result


def _call_anthropic(prompt: str, model: str = "claude-haiku-4-5") -> dict[str, Any]:
    import json as _json
    import urllib.request
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set", "passed": False, "score": 0}

    full_prompt = prompt + "\n\nRespond with JSON only."
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": JUDGE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": full_prompt}]
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=_json.dumps(payload).encode(),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=60)
    data = _json.load(resp)
    content = data["content"][0]["text"]
    # strip markdown code blocks if present
    lines = content.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    content = "\n".join(lines).strip()
    result = _json.loads(content)
    result["model_used"] = model
    result["tokens"] = data.get("usage", {})
    return result


def calibrate_judge(
    judge_fn,
    golden_set: list[dict],
    threshold: float = 0.7
) -> dict:
    """
    用人工标注黄金集校准 LLM 评分器。
    golden_set: [{"task_desc":..., "result":..., "human_score": 85}, ...]
    返回 Pearson 相关系数，低于 threshold 时警告。
    """
    import math
    human_scores = [g["human_score"] for g in golden_set]
    llm_scores = []
    for g in golden_set:
        r = judge_fn(g["task_desc"], g.get("rubric",""), g["result"], {})
        llm_scores.append(r.get("score", 0))

    n = len(human_scores)
    if n < 2:
        return {"pearson": None, "warning": "样本太少，无法校准"}

    mean_human = sum(human_scores) / n
    mean_llm = sum(llm_scores) / n
    cov = sum(
        (human - mean_human) * (llm - mean_llm)
        for human, llm in zip(human_scores, llm_scores, strict=True)
    ) / n
    std_human = math.sqrt(sum((score - mean_human) ** 2 for score in human_scores) / n)
    std_llm = math.sqrt(sum((score - mean_llm) ** 2 for score in llm_scores) / n)
    pearson = 0.0 if std_human == 0 or std_llm == 0 else cov / (std_human * std_llm)
    warning = (
        ""
        if pearson >= threshold
        else f"相关性 {pearson:.2f} 低于阈值 {threshold}，评分器不可靠"
    )

    return {
        "pearson": round(pearson, 3),
        "calibrated": pearson >= threshold,
        "warning": warning,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-desc", default="修复登录接口的 401 错误")
    parser.add_argument("--result", default="我已修复了 login.py 中的错误并通过了测试。")
    parser.add_argument("--rubric-file", default="rubrics/coding.md")
    parser.add_argument("--provider", default="openai")
    args = parser.parse_args()

    rubric_path = Path(args.rubric_file)
    rubric = (
        rubric_path.read_text(encoding="utf-8")
        if rubric_path.exists()
        else "（无评分标准文件）"
    )

    result = call_llm_judge(
        task_desc=args.task_desc,
        rubric=rubric,
        result=args.result,
        deterministic_checks={"passed": True, "failed_count": 0},
        provider=args.provider
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
