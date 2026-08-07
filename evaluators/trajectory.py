"""
轨迹评估器：检查 Agent 执行过程是否合理，不只看最终结果。
支持：工具调用检查、参数检查、重复操作检测、步骤顺序验证。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryStep:
    step_id: int
    tool_name: str
    arguments: dict
    result_summary: str = ""
    success: bool = True


@dataclass
class TrajectoryReport:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def check_trajectory(
    steps: list[TrajectoryStep],
    expected_tools: list[str] | None = None,
    forbidden_tools: list[str] | None = None,
    max_repeated_reads: int = 3,
    max_retry_same_tool: int = 2,
) -> TrajectoryReport:
    """
    分析 Agent 执行轨迹，检测常见问题。

    检测项：
    - 是否调用了禁止工具
    - 是否遗漏必要工具
    - 是否反复读取相同文件
    - 是否在相同参数下重复重试
    - 是否存在无限循环风险
    """
    violations = []
    warnings = []

    tool_call_counts: dict[str, int] = {}
    read_counts: dict[str, int] = {}
    consecutive_failures = 0
    last_tool = None

    for step in steps:
        tool = step.tool_name
        tool_call_counts[tool] = tool_call_counts.get(tool, 0) + 1

        # 检测禁止工具
        if forbidden_tools and tool in forbidden_tools:
            violations.append(f"step {step.step_id}: 调用了禁止工具 '{tool}'")

        # 检测重复读取文件
        if tool in ("read_file", "open_file") and "path" in step.arguments:
            fp = step.arguments["path"]
            read_counts[fp] = read_counts.get(fp, 0) + 1
            if read_counts[fp] > max_repeated_reads:
                violations.append(
                    f"step {step.step_id}: 文件 '{fp}' 被读取 {read_counts[fp]} 次（超过上限 {max_repeated_reads}）"
                )

        # 检测连续失败重试
        if not step.success:
            consecutive_failures += 1
            if consecutive_failures > max_retry_same_tool and tool == last_tool:
                violations.append(
                    f"step {step.step_id}: 工具 '{tool}' 连续失败 {consecutive_failures} 次未换策略"
                )
        else:
            consecutive_failures = 0

        last_tool = tool

    # 检测遗漏必要工具
    if expected_tools:
        used_tools = set(tool_call_counts.keys())
        missing = [t for t in expected_tools if t not in used_tools]
        if missing:
            violations.append(f"遗漏必要工具: {missing}")

    # 警告：工具调用过于集中
    total_calls = sum(tool_call_counts.values())
    for t, cnt in tool_call_counts.items():
        if total_calls > 5 and cnt / total_calls > 0.6:
            warnings.append(f"工具 '{t}' 占总调用的 {cnt/total_calls:.0%}，可能过度依赖单一工具")

    return TrajectoryReport(
        passed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        stats={
            "total_steps": len(steps),
            "tool_distribution": tool_call_counts,
            "unique_files_read": len(read_counts),
            "files_read_multiple_times": {k: v for k, v in read_counts.items() if v > 1},
        }
    )


def parse_hermes_trace(trace_json: dict) -> list[TrajectoryStep]:
    """从 Hermes/OpenTelemetry trace JSON 解析轨迹步骤。"""
    steps = []
    for i, span in enumerate(trace_json.get("spans", [])):
        attrs = span.get("attributes", {})
        tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool_name") or span.get("name", "unknown")
        args_raw = attrs.get("gen_ai.tool.call.arguments") or attrs.get("arguments") or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {}

        steps.append(TrajectoryStep(
            step_id=i,
            tool_name=tool_name,
            arguments=args,
            result_summary=attrs.get("gen_ai.tool.result", "")[:200],
            success=span.get("status", {}).get("code", "OK") != "ERROR"
        ))
    return steps


if __name__ == "__main__":
    # 示例：检查一段虚构轨迹
    demo_steps = [
        TrajectoryStep(0, "read_file", {"path": "src/auth.py"}),
        TrajectoryStep(1, "read_file", {"path": "src/auth.py"}),
        TrajectoryStep(2, "read_file", {"path": "src/auth.py"}),
        TrajectoryStep(3, "read_file", {"path": "src/auth.py"}),  # 第4次，超限
        TrajectoryStep(4, "write_file", {"path": "src/auth.py", "content": "..."}),
        TrajectoryStep(5, "run_tests", {}, success=False),
        TrajectoryStep(6, "run_tests", {}, success=False),
        TrajectoryStep(7, "run_tests", {}, success=False),  # 连续失败
    ]
    report = check_trajectory(
        demo_steps,
        expected_tools=["run_tests"],
        forbidden_tools=["git_push"],
        max_repeated_reads=3
    )
    print(json.dumps({
        "passed": report.passed,
        "violations": report.violations,
        "warnings": report.warnings,
        "stats": report.stats
    }, indent=2, ensure_ascii=False))
