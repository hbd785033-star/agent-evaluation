"""
效率/成本评估器：Token 消耗、API 成本、工具调用次数、重试次数。
"""
from __future__ import annotations
from dataclasses import dataclass


# 每1000 token 价格（USD），可按实际情况调整
PRICING = {
    "gpt-4o":             {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "claude-sonnet-4-5":  {"input": 0.003,  "output": 0.015},
    "claude-haiku-4-5":   {"input": 0.00025,"output": 0.00125},
    "deepseek-chat":      {"input": 0.00027,"output": 0.0011},
}

# 任务规模限制
TASK_LIMITS = {
    "simple": {
        "max_agents": 1, "max_tool_calls": 8,
        "max_retries": 1, "max_cost_usd": 0.05
    },
    "medium": {
        "max_agents": 2, "max_tool_calls": 20,
        "max_retries": 2, "max_cost_usd": 0.20
    },
    "complex": {
        "max_agents": 4, "max_tool_calls": 50,
        "max_retries": 3, "max_cost_usd": 1.00
    },
}


@dataclass
class CostReport:
    passed: bool
    estimated_cost_usd: float
    violations: list[str]
    stats: dict
    quality: str   # "excellent" / "acceptable" / "wasteful" / "over_budget"


def evaluate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    tool_calls: int = 0,
    retries: int = 0,
    sub_agents: int = 0,
    task_size: str = "medium",
    model: str = "claude-sonnet-4-5",
    duration_seconds: float = 0.0,
    cache_hit_rate: float | None = None,
) -> CostReport:
    """计算成本并检查是否超出任务限额。"""
    pricing = PRICING.get(model, {"input": 0.002, "output": 0.008})
    limits = TASK_LIMITS.get(task_size, TASK_LIMITS["medium"])

    # 缓存命中的 token 按半价计
    billable_input = input_tokens - cache_read_tokens + cache_read_tokens * 0.1
    cost = (billable_input / 1000 * pricing["input"]
            + output_tokens / 1000 * pricing["output"])

    violations = []
    if sub_agents > limits["max_agents"]:
        violations.append(f"子 Agent 数 {sub_agents} 超过限额 {limits['max_agents']}")
    if tool_calls > limits["max_tool_calls"]:
        violations.append(f"工具调用 {tool_calls} 超过限额 {limits['max_tool_calls']}")
    if retries > limits["max_retries"]:
        violations.append(f"重试次数 {retries} 超过限额 {limits['max_retries']}")
    if cost > limits["max_cost_usd"]:
        violations.append(f"估算成本 ${cost:.4f} 超过预算 ${limits['max_cost_usd']}")

    # 缓存命中率警告
    if cache_hit_rate is not None and cache_hit_rate < 0.3:
        violations.append(f"缓存命中率 {cache_hit_rate:.0%} 过低（<30%），可能存在 prompt 重写问题")

    # 质量评级
    budget = limits["max_cost_usd"]
    if len(violations) > 0:
        quality = "over_budget" if cost > budget else "wasteful"
    elif cost < budget * 0.4:
        quality = "excellent"
    else:
        quality = "acceptable"

    return CostReport(
        passed=len(violations) == 0,
        estimated_cost_usd=round(cost, 6),
        violations=violations,
        stats={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_hit_rate": cache_hit_rate,
            "tool_calls": tool_calls,
            "retries": retries,
            "sub_agents": sub_agents,
            "duration_seconds": duration_seconds,
            "model": model,
            "task_size": task_size,
        },
        quality=quality
    )


if __name__ == "__main__":
    import json
    r = evaluate_cost(
        input_tokens=8500, output_tokens=1200, cache_read_tokens=6000,
        tool_calls=14, retries=1, sub_agents=1,
        task_size="medium", model="claude-sonnet-4-5",
        duration_seconds=45.2, cache_hit_rate=0.71
    )
    print(json.dumps({
        "passed": r.passed, "cost_usd": r.estimated_cost_usd,
        "quality": r.quality, "violations": r.violations, "stats": r.stats
    }, indent=2))
