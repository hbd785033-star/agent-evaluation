"""Execute N real trials, score five layers, and emit aggregate reports."""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import posixpath
import re
import statistics
import uuid
from collections import Counter
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
from .checkers import CheckerProfile, TaskCheckRegistry
from .models import EvaluatedRun, RunRecord, TaskCase

Judge = Callable[[TaskCase, RunRecord, dict[str, Any]], dict[str, Any]]
TaskCheck = Callable[[TaskCase, RunRecord], list[CheckResult]]


def _criterion_key(value: str) -> str:
    """Normalize cosmetic separators without equating unrelated criteria."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _allowed_roots(patterns: list[str]) -> list[str]:
    roots: list[str] = []
    for pattern in patterns:
        prefix = pattern.replace("\\", "/").split("*", 1)[0].rstrip("/")
        if prefix:
            roots.append(prefix)
    return roots or ["."]


def _normalized_path(value: str, workspace: Path | None = None) -> str:
    text = str(value).replace("\\", "/")
    candidate = Path(value)
    if workspace is not None and candidate.is_absolute():
        try:
            text = candidate.resolve(strict=False).relative_to(workspace).as_posix()
        except ValueError:
            text = candidate.resolve(strict=False).as_posix()
    return posixpath.normpath(text)


def _normalized_action(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _trajectory_steps(record: RunRecord) -> list[TrajectoryStep]:
    source = record.trajectory or record.tool_calls
    steps: list[TrajectoryStep] = []
    for index, raw in enumerate(source):
        arguments = raw.get("arguments", raw.get("args", {}))
        try:
            step_id = int(raw.get("step_id", index))
        except (TypeError, ValueError, OverflowError):
            step_id = index
        steps.append(
            TrajectoryStep(
                step_id=step_id,
                tool_name=str(raw.get("tool_name", raw.get("tool", raw.get("name", "unknown")))),
                arguments=arguments if isinstance(arguments, dict) else {},
                result_summary=str(raw.get("result_summary", "")),
                success=bool(raw.get("success", True)),
            )
        )
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
    *,
    authority_verified: bool = False,
    checker_profile: CheckerProfile | None = None,
) -> EvaluatedRun:
    integrity_errors = record.integrity_errors()
    deterministic_checks: list[CheckResult] = [
        CheckResult(
            passed=record.exit_status == "completed",
            check_name="run_completed",
            detail=record.error or record.exit_status,
        ),
        check_no_api_key_leak(record.output),
        CheckResult(
            passed=not integrity_errors,
            check_name="run_record_integrity",
            detail="ok" if not integrity_errors else "; ".join(integrity_errors),
            evidence=integrity_errors,
        ),
    ]
    workspace_raw = record.workspace_root
    workspace = Path(workspace_raw).resolve() if workspace_raw else None
    commands, paths = _commands_and_paths(record)
    if paths and workspace is None:
        deterministic_checks.append(
            CheckResult(
                False,
                "trusted_workspace_present",
                "paths were reported without a trusted workspace_root",
                paths,
            )
        )
    if task.allowed_files:
        deterministic_checks.append(
            check_no_forbidden_files_modified(
                task.allowed_files,
                record.files_changed,
                repo_root=workspace or Path.cwd(),
            )
        )
    normalized_paths = [(_normalized_path(path, workspace), path) for path in paths]
    forbidden_path_hits = [
        original
        for normalized, original in normalized_paths
        if any(
            fnmatch.fnmatch(normalized, _normalized_path(pattern))
            for pattern in task.forbidden_files
        )
    ]
    deterministic_checks.append(
        CheckResult(
            passed=not forbidden_path_hits,
            check_name="forbidden_files",
            detail="ok" if not forbidden_path_hits else f"forbidden paths: {forbidden_path_hits}",
            evidence=forbidden_path_hits,
        )
    )
    action_text = _normalized_action(
        "\n".join([*commands, *(step.tool_name for step in _trajectory_steps(record))])
    )
    forbidden_action_hits = [
        action for action in task.forbidden_actions if _normalized_action(action) in action_text
    ]
    deterministic_checks.append(
        CheckResult(
            passed=not forbidden_action_hits,
            check_name="forbidden_actions",
            detail=(
                "ok"
                if not forbidden_action_hits
                else f"forbidden actions observed: {forbidden_action_hits}"
            ),
            evidence=forbidden_action_hits,
        )
    )
    constrained = bool(task.allowed_files or task.forbidden_files or task.forbidden_actions)
    authority_sensitive = bool(
        constrained
        or record.workspace_root
        or record.isolation_level != "none"
        or record.sandbox_id
        or record.dataset_version
    )
    deterministic_checks.append(
        CheckResult(
            passed=authority_verified or not authority_sensitive,
            check_name="control_plane_authority_verified",
            detail=(
                "ok"
                if authority_verified or not authority_sensitive
                else "authority-sensitive record requires EvalRunner canonicalization"
            ),
        )
    )
    deterministic_checks.append(
        CheckResult(
            passed=not constrained or record.isolation_level == "os",
            check_name="os_sandbox_enforced",
            detail=(
                "ok"
                if not constrained or record.isolation_level == "os"
                else (
                    f"policy-constrained task requires os isolation; got {record.isolation_level!r}"
                )
            ),
        )
    )
    supported_criteria = {"deterministic", "llm_judge"}
    unknown_criteria = sorted(set(task.success_criteria) - supported_criteria)
    deterministic_checks.append(
        CheckResult(
            passed=not unknown_criteria,
            check_name="known_success_criteria",
            detail="ok" if not unknown_criteria else f"unsupported criteria: {unknown_criteria}",
            evidence=unknown_criteria,
        )
    )
    required_deterministic = task.success_criteria.get("deterministic", [])
    criterion_checks: list[CheckResult] = []
    if required_deterministic:
        if task_check is None:
            deterministic_checks.append(
                CheckResult(
                    passed=False,
                    check_name="task_specific_deterministic_checks",
                    detail=(
                        f"{len(required_deterministic)} criteria have no executable checker; "
                        "refusing to infer PASS from free-form text"
                    ),
                    evidence=list(required_deterministic),
                )
            )
        else:
            try:
                task_check_results = task_check(task, record)
            except Exception as exc:
                task_check_results = [
                    CheckResult(
                        False,
                        "task_specific_criteria_execution",
                        f"task checker failed: {type(exc).__name__}",
                    )
                ]
            criterion_checks = task_check_results
            deterministic_checks.extend(task_check_results)
            returned_names = [result.check_name for result in task_check_results]
            returned_keys = Counter(_criterion_key(name) for name in returned_names)
            required_keys = Counter(_criterion_key(name) for name in required_deterministic)
            if returned_keys != required_keys:
                deterministic_checks.append(
                    CheckResult(
                        False,
                        "task_specific_criteria_coverage",
                        (
                            f"checker returned criteria {returned_names!r}; expected "
                            f"{list(required_deterministic)!r}"
                        ),
                        list(required_deterministic),
                    )
                )
    deterministic = run_checks(deterministic_checks)
    deterministic["criterion_checks"] = [
        {
            "criterion_id": result.check_name,
            "passed": result.passed,
            "detail": result.detail,
            "evidence": result.evidence,
            **({"failure": result.detail} if not result.passed else {}),
        }
        for result in criterion_checks
    ]

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

    security = run_security_suite(
        output=record.output,
        commands=commands,
        paths_accessed=paths,
        allowed_roots=_allowed_roots(task.allowed_files),
        user_content=task.prompt,
        agent_response=record.output,
        base_dir=workspace or Path.cwd(),
    )

    required_judge = task.success_criteria.get("llm_judge", [])
    if judge is not None and getattr(judge, "calibrated", False) is True:
        try:
            judge_result = judge(task, record, deterministic)
        except Exception as exc:
            judge_result = {"passed": False, "error": f"judge failed: {type(exc).__name__}"}
        if not isinstance(judge_result, dict) or not isinstance(judge_result.get("passed"), bool):
            judge_result = {
                "passed": False,
                "error": "judge result must be an object with boolean passed",
            }
        score = judge_result.get("score")
        if score is not None and (
            not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 100
        ):
            judge_result = {"passed": False, "error": "judge score must be between 0 and 100"}
    elif judge is not None:
        judge_result = {
            "passed": False,
            "error": "judge is not marked calibrated",
        }
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
    if criterion_checks:
        layers["deterministic"]["criterion_checks"] = [
            {
                "criterion_id": result.check_name,
                "passed": result.passed,
                "detail": result.detail,
                "evidence": result.evidence,
                **({"failure": result.detail} if not result.passed else {}),
            }
            for result in criterion_checks
        ]
    passed = all(layer.get("passed") is True for layer in layers.values())
    return EvaluatedRun(record=record, passed=passed, layers=layers)


class EvalRunner:
    def __init__(
        self,
        adapter: AgentAdapter,
        judge: Judge | None = None,
        task_checks: dict[str, TaskCheck] | None = None,
        *,
        dataset_version: str = "",
        checker_registry: TaskCheckRegistry | None = None,
        checker_profile: CheckerProfile | None = None,
    ) -> None:
        self.adapter = adapter
        self.judge = judge
        self.task_checks = task_checks or {}
        self.checker_registry = checker_registry
        self.checker_profile = checker_profile
        self.dataset_version = str(dataset_version)
        self._expected_identity = {
            "model": str(adapter.model),
            "provider": str(adapter.provider),
            "harness": str(adapter.harness),
        }
        isolation_level = getattr(adapter, "isolation_level", "none")
        self._isolation_level = (
            isolation_level if isolation_level in {"none", "workspace", "os"} else "none"
        )
        workspace_root = getattr(adapter, "workspace_root", None)
        self._trusted_workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )

    def _canonicalize_record(
        self,
        task: TaskCase,
        trial: int,
        record: RunRecord,
        seen_run_ids: set[str],
    ) -> RunRecord:
        errors = record.integrity_errors()
        expected = {
            "task_id": task.id,
            "trial": trial,
            **self._expected_identity,
        }
        for field_name, expected_value in expected.items():
            if getattr(record, field_name) != expected_value:
                errors.append(
                    f"{field_name} mismatch: {getattr(record, field_name)!r} != {expected_value!r}"
                )
            setattr(record, field_name, expected_value)
        if not isinstance(record.run_id, str) or not record.run_id.strip():
            record.run_id = f"invalid-{uuid.uuid4().hex}"
        elif record.run_id in seen_run_ids:
            errors.append(f"duplicate run_id: {record.run_id}")
        seen_run_ids.add(record.run_id)
        if not isinstance(record.output, str):
            record.output = ""
        if record.error is not None and not isinstance(record.error, str):
            record.error = "invalid non-string harness error"
        if not isinstance(record.tool_calls, list):
            record.tool_calls = []
        if not isinstance(record.files_changed, list):
            record.files_changed = []
        if not isinstance(record.trajectory, list):
            record.trajectory = []
        if not isinstance(record.metadata, dict):
            record.metadata = {}
        for reserved in ("workspace_root", "dataset_version", "sandbox_id", "isolation_level"):
            if reserved in record.metadata:
                errors.append(f"reserved metadata field supplied by harness: {reserved}")
                record.metadata.pop(reserved, None)

        claimed_workspace = record.workspace_root
        trusted_workspace: Path | None = None
        if self._trusted_workspace_root is None:
            if claimed_workspace:
                errors.append("workspace_root supplied without control-plane trust root")
        elif not isinstance(claimed_workspace, str) or not claimed_workspace:
            errors.append("trusted adapter did not provide a workspace_root")
        else:
            candidate = Path(claimed_workspace).resolve(strict=False)
            try:
                candidate.relative_to(self._trusted_workspace_root)
            except ValueError:
                errors.append("workspace_root escapes control-plane trust root")
            else:
                trusted_workspace = candidate
        record.workspace_root = str(trusted_workspace) if trusted_workspace is not None else None
        expected_sandbox_id = (
            trusted_workspace.name
            if trusted_workspace is not None
            else f"untrusted-{task.id}-{trial}"
        )
        if record.sandbox_id and record.sandbox_id != expected_sandbox_id:
            errors.append("sandbox_id mismatch")
        record.sandbox_id = expected_sandbox_id
        if record.dataset_version and record.dataset_version != self.dataset_version:
            errors.append("dataset_version mismatch")
        record.dataset_version = self.dataset_version
        if record.isolation_level != self._isolation_level:
            errors.append("isolation_level mismatch")
        record.isolation_level = self._isolation_level
        for field_name in ("input_tokens", "output_tokens", "cached_tokens"):
            value = getattr(record, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                setattr(record, field_name, 0)
        if record.cached_tokens > record.input_tokens:
            record.cached_tokens = 0
        for field_name in ("cost_usd", "latency_seconds"):
            value = getattr(record, field_name)
            if not _is_finite_number(value) or value < 0:
                setattr(record, field_name, 0.0)
        for field_name in ("retries", "sub_agents"):
            value = record.metadata.get(field_name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                record.metadata[field_name] = 0
        if errors:
            record.exit_status = "failed"
            record.error = "run record integrity failure: " + "; ".join(errors)
            record.metadata["integrity_errors"] = errors
        return record

    def run(
        self,
        tasks: list[TaskCase],
        *,
        trials_override: int | None = None,
    ) -> dict[str, Any]:
        runs: list[EvaluatedRun] = []
        seen_run_ids: set[str] = set()
        for source_task in tasks:
            task = copy.deepcopy(source_task)
            trial_count = trials_override if trials_override is not None else task.trials
            if trial_count < 1:
                raise ValueError("trial count must be positive")
            for trial in range(1, trial_count + 1):
                adapter_task = copy.deepcopy(task)
                try:
                    record = self.adapter.run(adapter_task, trial)
                except Exception:  # noqa: BLE001 - adapter process boundary
                    record = RunRecord(
                        task.id,
                        self._expected_identity["model"],
                        self._expected_identity["provider"],
                        self._expected_identity["harness"],
                        trial,
                        exit_status="failed",
                        error="adapter execution failed",
                        run_id=f"invalid-{uuid.uuid4().hex}",
                    )
                if not isinstance(record, RunRecord):
                    record = RunRecord(
                        task.id,
                        self._expected_identity["model"],
                        self._expected_identity["provider"],
                        self._expected_identity["harness"],
                        trial,
                        exit_status="failed",
                        error="adapter returned a non-RunRecord value",
                        run_id=f"invalid-{uuid.uuid4().hex}",
                    )
                record = self._canonicalize_record(task, trial, record, seen_run_ids)
                task_check = self.task_checks.get(task.id)
                if task_check is None and self.checker_registry is not None:
                    registry = self.checker_registry
                    profile = self.checker_profile

                    def task_check(
                        checked_task,
                        checked_record,
                        registry=registry,
                        profile=profile,
                    ):
                        return registry.check(checked_task, checked_record, profile)

                try:
                    runs.append(
                        evaluate_run(
                            task,
                            record,
                            judge=self.judge,
                            task_check=task_check,
                            authority_verified=True,
                            checker_profile=self.checker_profile,
                        )
                    )
                finally:
                    cleanup = getattr(self.adapter, "cleanup", None)
                    if callable(cleanup):
                        cleanup(record)
        report = build_report(runs)
        report["dataset_version"] = self.dataset_version
        report["checker_profile"] = (
            self.checker_profile.identity() if self.checker_profile is not None else None
        )
        return report


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
        aggregates.append(
            {
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
            }
        )

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
    lines.extend(
        [
            "",
            (
                "> A passing framework test does not imply that any model passed this dataset. "
                "Only records listed above are model/harness trials."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
