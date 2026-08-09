"""Execute N real trials, score five layers, and emit aggregate reports."""
from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluators.cost import evaluate_cost
from evaluators.deterministic import (
    CheckResult,
    check_no_api_key_leak,
    check_no_forbidden_files_modified,
    run_checks,
)
from evaluators.security import run_security_suite
from evaluators.trajectory import TrajectoryStep, check_trajectory

from .adapters import AgentAdapter
from .models import EvaluatedRun, RunRecord, TaskCase

Judge = Callable[[TaskCase, RunRecord, dict[str, Any]], dict[str, Any]]
TaskCheck = Callable[[TaskCase, RunRecord], list[CheckResult]]


def _allowed_roots(patterns: list[str]) -> list[str]:
    roots: list[str] = []
    for pattern in patterns:
        prefix = pattern.replace("\\", "/").split("*", 1)[0].rstrip("/")
        if prefix:
            roots.append(prefix)
    return roots or ["."]


def _trajectory_steps(record: RunRecord) -> list[TrajectoryStep]:
    source = record.trajectory or record.tool_calls
    steps: list[TrajectoryStep] = []
    for index, raw in enumerate(source):
        arguments = raw.get("arguments", raw.get("args", {}))
        steps.append(TrajectoryStep(
            step_id=int(raw.get("step_id", index)),
            tool_name=str(raw.get("tool_name", raw.get("tool", raw.get("name", "unknown")))),
            arguments=arguments if isinstance(arguments, dict) else {},
            result_summary=str(raw.get("result_summary", "")),
            success=bool(raw.get("success", True)),
        ))
    return steps


def _commands_and_paths(record: RunRecord) -> tuple[list[str], list[str]]:
    commands: list[str] = []
    paths: list[str] = []
    for call in record.tool_calls:
        args = call.get("arguments", call.get("args", {}))
        if not isinstance(args, dict):
            continue
        for key in ("command", "cmd"):
            if key in args:
                commands.append(str(args[key]))
        for key in ("path", "file", "filepath"):
            if key in args:
                paths.append(str(args[key]))
    paths.extend(record.files_changed)
    return commands, paths


def evaluate_run(
    task: TaskCase,
    record: RunRecord,
    judge: Judge | None = None,
    task_check: TaskCheck | None = None,
) -> EvaluatedRun:
    deterministic_checks: list[CheckResult] = [
        CheckResult(
            passed=record.exit_status == "completed",
            check_name="run_completed",
            detail=record.error or record.exit_status,
        ),
        check_no_api_key_leak(record.output),
    ]
    if task.allowed_files:
        deterministic_checks.append(
            check_no_forbidden_files_modified(task.allowed_files, record.files_changed)
        )
    required_deterministic = task.success_criteria.get("deterministic", [])
    if required_deterministic:
        if task_check is None:
            deterministic_checks.append(CheckResult(
                passed=False,
                check_name="task_specific_deterministic_checks",
                detail=(
                    f"{len(required_deterministic)} criteria have no executable checker; "
                    "refusing to infer PASS from free-form text"
                ),
                evidence=list(required_deterministic),
            ))
        else:
            deterministic_checks.extend(task_check(task, record))
    deterministic = run_checks(deterministic_checks)

    trajectory_report = check_trajectory(_trajectory_steps(record))
    trajectory = {
        "passed": trajectory_report.passed,
        "violations": trajectory_report.violations,
        "warnings": trajectory_report.warnings,
        "stats": trajectory_report.stats,
    }

    estimated = evaluate_cost(
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cache_read_tokens=record.cached_tokens,
        tool_calls=len(record.tool_calls),
        retries=int(record.metadata.get("retries", 0)),
        sub_agents=int(record.metadata.get("sub_agents", 0)),
        model=record.model,
        duration_seconds=record.latency_seconds,
    )
    effective_cost = record.cost_usd if record.cost_usd > 0 else estimated.estimated_cost_usd
    cost_violations: list[str] = []
    max_cost = task.limits.get("max_cost_usd")
    max_tools = task.limits.get("max_tool_calls")
    max_agents = task.limits.get("max_agents")
    if max_cost is not None and effective_cost > float(max_cost):
        cost_violations.append(f"cost ${effective_cost:.6f} exceeds ${float(max_cost):.6f}")
    if max_tools is not None and len(record.tool_calls) > int(max_tools):
        cost_violations.append(f"tool calls {len(record.tool_calls)} exceed {int(max_tools)}")
    sub_agents = int(record.metadata.get("sub_agents", 0))
    if max_agents is not None and sub_agents > int(max_agents):
        cost_violations.append(f"sub-agents {sub_agents} exceed {int(max_agents)}")
    cost = {
        "passed": not cost_violations,
        "estimated_cost_usd": effective_cost,
        "violations": cost_violations,
        "stats": {
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cached_tokens": record.cached_tokens,
            "tool_calls": len(record.tool_calls),
            "latency_seconds": record.latency_seconds,
        },
    }

    commands, paths = _commands_and_paths(record)
    security = run_security_suite(
        output=record.output,
        commands=commands,
        paths_accessed=paths,
        allowed_roots=_allowed_roots(task.allowed_files),
        user_content=task.prompt,
        agent_response=record.output,
    )

    required_judge = task.success_criteria.get("llm_judge", [])
    if judge is not None:
        judge_result = judge(task, record, deterministic)
    elif required_judge:
        judge_result = {
            "passed": False,
            "skipped": True,
            "reason": (
                f"{len(required_judge)} rubric criteria require a calibrated judge; "
                "refusing to infer PASS"
            ),
        }
    else:
        judge_result = {
            "passed": True,
            "skipped": True,
            "reason": "task has no LLM-judge criteria",
        }
    layers = {
        "deterministic": deterministic,
        "trajectory": trajectory,
        "cost": cost,
        "security": security,
        "judge": judge_result,
    }
    passed = all(bool(layer.get("passed", False)) for layer in layers.values())
    return EvaluatedRun(record=record, passed=passed, layers=layers)


class EvalRunner:
    def __init__(
        self,
        adapter: AgentAdapter,
        judge: Judge | None = None,
        task_checks: dict[str, TaskCheck] | None = None,
    ) -> None:
        self.adapter = adapter
        self.judge = judge
        self.task_checks = task_checks or {}

    def run(
        self,
        tasks: list[TaskCase],
        *,
        trials_override: int | None = None,
    ) -> dict[str, Any]:
        runs: list[EvaluatedRun] = []
        for task in tasks:
            trial_count = trials_override if trials_override is not None else task.trials
            if trial_count < 1:
                raise ValueError("trial count must be positive")
            for trial in range(1, trial_count + 1):
                record = self.adapter.run(task, trial)
                runs.append(evaluate_run(
                    task,
                    record,
                    self.judge,
                    self.task_checks.get(task.id),
                ))
        return build_report(runs)


def build_report(runs: list[EvaluatedRun]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[EvaluatedRun]] = {}
    for run in runs:
        record = run.record
        key = (record.task_id, record.model, record.provider, record.harness)
        groups.setdefault(key, []).append(run)

    aggregates: list[dict[str, Any]] = []
    for (task_id, model, provider, harness), group in sorted(groups.items()):
        pass_values = [1.0 if run.passed else 0.0 for run in group]
        costs = [run.record.cost_usd or run.layers["cost"]["estimated_cost_usd"] for run in group]
        latencies = [run.record.latency_seconds for run in group]
        aggregates.append({
            "task_id": task_id,
            "model": model,
            "provider": provider,
            "harness": harness,
            "trials": len(group),
            "pass_rate": statistics.fmean(pass_values),
            "pass_variance": statistics.pvariance(pass_values),
            "mean_cost_usd": statistics.fmean(costs),
            "mean_latency_seconds": statistics.fmean(latencies),
            "failure_rate": 1.0 - statistics.fmean(pass_values),
        })

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
        "aggregates": aggregates,
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "report.json"
    md_path = target / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Agent Evaluation Report",
        "",
        f"Runs: **{report['run_count']}**",
        "",
        "| Task | Model | Provider | Harness | Trials | Pass rate | Mean cost | Mean latency |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        lines.append(
            f"| {row['task_id']} | {row['model']} | {row['provider']} | {row['harness']} "
            f"| {row['trials']} | {row['pass_rate']:.1%} | ${row['mean_cost_usd']:.6f} "
            f"| {row['mean_latency_seconds']:.3f}s |"
        )
    lines.extend([
        "",
        (
            "> A passing framework test does not imply that any model passed this dataset. "
            "Only records listed above are model/harness trials."
        ),
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
