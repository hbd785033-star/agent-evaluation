"""Execute N real trials, score five layers, and emit aggregate reports."""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import posixpath
import re
import statistics
import subprocess
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
from .models import EvaluatedRun, RunRecord, SuccessCriterion, TaskCase

Judge = Callable[[TaskCase, RunRecord, dict[str, Any]], dict[str, Any]]
TaskCheck = Callable[[TaskCase, RunRecord], list[CheckResult]]


def _criterion_key(value: str) -> str:
    """Normalize cosmetic separators without equating unrelated criteria."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _criterion_id(value: SuccessCriterion | str) -> str:
    """Return the stable ID while retaining legacy string criteria support."""
    return value.id if isinstance(value, SuccessCriterion) else value


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


def _trajectory_source(record: RunRecord) -> tuple[list[dict[str, Any]] | None, str]:
    if isinstance(record.trajectory, list):
        return record.trajectory, "complete"
    if isinstance(record.tool_calls, list):
        return record.tool_calls, "partial"
    return None, "unavailable"


def _trajectory_steps(source: list[dict[str, Any]]) -> list[TrajectoryStep]:
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
    tool_calls = record.tool_calls if isinstance(record.tool_calls, list) else []
    for call in tool_calls:
        args = call.get("arguments", call.get("args", {}))
        if not isinstance(args, dict):
            continue
        for key in ("command", "cmd"):
            if key in args:
                commands.append(str(args[key]))
        for key in ("path", "file", "filepath"):
            if key in args:
                paths.append(str(args[key]))
    if isinstance(record.files_changed, list):
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
    canonicalization_errors = (
        list(record.canonicalization_errors)
        if isinstance(record.canonicalization_errors, tuple)
        else []
    )
    integrity_errors = list(
        dict.fromkeys([*record.integrity_errors(), *canonicalization_errors])
    )
    deterministic_checks: list[CheckResult] = [
        CheckResult(
            passed=record.exit_status == "completed",
            check_name="run_completed",
            detail=record.error or record.exit_status,
        ),
        *([check_no_api_key_leak(record.output)] if isinstance(record.output, str) else []),
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
    file_constraints = bool(task.allowed_files or task.forbidden_files)
    if file_constraints and record.files_changed is None:
        deterministic_checks.append(
            CheckResult(
                False,
                "files_changed_evidence_available",
                "file constraints require files_changed evidence; source reported unavailable",
                [],
            )
        )
    action_evidence_available = record.tool_calls is not None or record.trajectory is not None
    if task.forbidden_actions and not action_evidence_available:
        deterministic_checks.append(
            CheckResult(
                False,
                "action_evidence_available",
                "forbidden-action policy requires tool or trajectory evidence",
                [],
            )
        )

    if paths and workspace is None:
        deterministic_checks.append(
            CheckResult(
                False,
                "trusted_workspace_present",
                "paths were reported without a trusted workspace_root",
                paths,
            )
        )
    if task.allowed_files and isinstance(record.files_changed, list):
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
    trajectory_source, trajectory_completeness = _trajectory_source(record)
    trajectory_steps = _trajectory_steps(trajectory_source) if trajectory_source is not None else []
    action_text = _normalized_action(
        "\n".join([*commands, *(step.tool_name for step in trajectory_steps)])
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
    required_ids = [_criterion_id(criterion) for criterion in required_deterministic]
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
                    evidence=required_ids,
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
            required_keys = Counter(_criterion_key(name) for name in required_ids)
            if returned_keys != required_keys:
                deterministic_checks.append(
                    CheckResult(
                        False,
                        "task_specific_criteria_coverage",
                        (
                            f"checker returned criteria {returned_names!r}; expected "
                            f"{required_ids!r}"
                        ),
                        required_ids,
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

    if trajectory_source is None:
        trajectory_required = bool(task.forbidden_actions)
        trajectory = {
            "passed": not trajectory_required,
            "status": "unavailable",
            "evaluated": False,
            "violations": (
                ["trajectory evidence unavailable for required action policy"]
                if trajectory_required
                else []
            ),
            "warnings": [],
            "stats": None,
        }
    else:
        trajectory_report = check_trajectory(trajectory_steps)
        trajectory = {
            "passed": trajectory_report.passed,
            "status": trajectory_completeness,
            "evaluated": True,
            "violations": trajectory_report.violations,
            "warnings": trajectory_report.warnings,
            "stats": trajectory_report.stats,
        }

    token_values_known = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (record.input_tokens, record.output_tokens, record.cached_tokens)
    )
    tool_count = len(record.tool_calls) if isinstance(record.tool_calls, list) else None
    retries_raw = record.metadata.get("retries") if isinstance(record.metadata, dict) else None
    sub_agents_raw = (
        record.metadata.get("sub_agents") if isinstance(record.metadata, dict) else None
    )
    retries = (
        retries_raw
        if isinstance(retries_raw, int) and not isinstance(retries_raw, bool) and retries_raw >= 0
        else None
    )
    sub_agents = (
        sub_agents_raw
        if isinstance(sub_agents_raw, int)
        and not isinstance(sub_agents_raw, bool)
        and sub_agents_raw >= 0
        else None
    )
    derived_estimate = None
    if (
        token_values_known
        and tool_count is not None
        and retries is not None
        and sub_agents is not None
        and isinstance(record.observed_model, str)
        and record.observed_model.strip()
        and _is_finite_number(record.latency_seconds)
        and record.latency_seconds >= 0
    ):
        derived_estimate = evaluate_cost(
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_tokens=record.cached_tokens,
            tool_calls=tool_count,
            retries=retries,
            sub_agents=sub_agents,
            model=record.observed_model,
            duration_seconds=record.latency_seconds,
        ).estimated_cost_usd
    source_cost_valid = (
        record.cost_usd is not None
        and _is_finite_number(record.cost_usd)
        and record.cost_usd >= 0
    )
    reported_cost = float(record.cost_usd) if source_cost_valid else None
    cost_basis = reported_cost if reported_cost is not None else derived_estimate
    cost_basis_semantics = (
        record.cost_semantics or "reported"
        if reported_cost is not None
        else "ae_derived_estimate"
        if derived_estimate is not None
        else None
    )
    cost_violations: list[str] = []
    max_cost = task.limits.get("max_cost_usd")
    max_tools = task.limits.get("max_tool_calls")
    max_agents = task.limits.get("max_agents")
    if max_cost is not None:
        if cost_basis is None:
            cost_violations.append("cost evidence unavailable for max_cost_usd")
        elif cost_basis > float(max_cost):
            cost_violations.append(f"cost ${cost_basis:.6f} exceeds ${float(max_cost):.6f}")
    if max_tools is not None:
        if tool_count is None:
            cost_violations.append("tool-call evidence unavailable for max_tool_calls")
        elif tool_count > int(max_tools):
            cost_violations.append(f"tool calls {tool_count} exceed {int(max_tools)}")
    if max_agents is not None:
        if sub_agents is None:
            cost_violations.append("sub-agent evidence unavailable for max_agents")
        elif sub_agents > int(max_agents):
            cost_violations.append(f"sub-agents {sub_agents} exceed {int(max_agents)}")
    cost_status = (
        "invalid"
        if record.cost_usd is not None and not source_cost_valid
        else "available"
        if cost_basis is not None
        else "unavailable"
    )
    cost = {
        "passed": not cost_violations and cost_status != "invalid",
        "status": cost_status,
        "source_cost_usd": record.cost_usd,
        "reported_cost_usd": reported_cost,
        "reported_cost_semantics": record.cost_semantics if reported_cost is not None else None,
        "derived_estimated_cost_usd": derived_estimate,
        "estimated_cost_usd": derived_estimate,
        "limit_basis_usd": cost_basis,
        "limit_basis_semantics": cost_basis_semantics,
        "violations": cost_violations,
        "stats": {
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cached_tokens": record.cached_tokens,
            "tool_calls": tool_count,
            "latency_seconds": record.latency_seconds,
        },
    }

    output_known = isinstance(record.output, str)
    security = run_security_suite(
        output=record.output if output_known else None,
        commands=commands,
        paths_accessed=paths,
        allowed_roots=_allowed_roots(task.allowed_files),
        user_content=task.prompt,
        agent_response=record.output if output_known else None,
        base_dir=workspace or Path.cwd(),
    )
    security["completeness"] = (
        "complete"
        if output_known and record.tool_calls is not None and record.files_changed is not None
        else "partial"
        if output_known or record.tool_calls is not None or record.files_changed is not None
        else "unavailable"
    )
    security["output_evidence"] = "available" if output_known else "unavailable"

    required_judge = task.success_criteria.get("llm_judge", [])
    if not output_known and (judge is not None or required_judge):
        judge_result = {
            "passed": False,
            "skipped": True,
            "reason": "output evidence unavailable; refusing to judge invented empty output",
        }
    elif judge is not None and getattr(judge, "calibrated", False) is True:
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
        self._source_identity_is_observed = bool(
            getattr(adapter, "identity_fields_are_observed", False)
        )
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
        if self._source_identity_is_observed:
            for source_name, observed_name in (
                ("model", "observed_model"),
                ("provider", "observed_provider"),
                ("harness", "observed_harness"),
            ):
                source_value = getattr(record, source_name)
                if getattr(record, observed_name) is None and (
                    isinstance(source_value, str) and source_value.strip()
                ):
                    setattr(record, observed_name, source_value)
        expected = {
            "task_id": task.id,
            "trial": trial,
            **self._expected_identity,
        }
        for field_name, expected_value in expected.items():
            if getattr(record, field_name) != expected_value and not (
                self._source_identity_is_observed
                and field_name in {"model", "provider", "harness"}
            ):
                errors.append(
                    f"{field_name} mismatch: {getattr(record, field_name)!r} != {expected_value!r}"
                )
            setattr(record, field_name, expected_value)
        if isinstance(record.run_id, str) and record.run_id.strip():
            if record.run_id in seen_run_ids:
                errors.append(f"duplicate run_id: {record.run_id}")
            seen_run_ids.add(record.run_id)
        elif record.run_id is not None:
            errors.append("run_id must be a non-empty string or null")
        if not isinstance(record.metadata, dict):
            errors.append("metadata must be an object")
            record.metadata = {"invalid_metadata_type": type(record.metadata).__name__}
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
        record.canonicalization_errors = tuple(errors)
        if errors:
            record.metadata["integrity_errors"] = list(errors)
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
                except (subprocess.TimeoutExpired, TimeoutError):
                    record = RunRecord(
                        task.id,
                        self._expected_identity["model"],
                        self._expected_identity["provider"],
                        self._expected_identity["harness"],
                        trial,
                        exit_status="timeout",
                        error="adapter execution timed out",
                        run_id=None,
                    )
                except Exception:  # noqa: BLE001 - adapter process boundary
                    record = RunRecord(
                        task.id,
                        self._expected_identity["model"],
                        self._expected_identity["provider"],
                        self._expected_identity["harness"],
                        trial,
                        exit_status="failed",
                        error="adapter execution failed",
                        run_id=None,
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
                        run_id=None,
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
        cost_populations: dict[str, list[float]] = {}
        for run in group:
            cost = run.record.cost_usd
            if cost is None or not _is_finite_number(cost) or cost < 0:
                continue
            semantics = run.record.cost_semantics or "reported"
            cost_populations.setdefault(semantics, []).append(float(cost))
        cost_available = sum(len(values) for values in cost_populations.values())
        serialized_populations = {
            semantics: {
                "sample_count": len(values),
                "mean_cost_usd": statistics.fmean(values),
            }
            for semantics, values in sorted(cost_populations.items())
        }
        if len(cost_populations) == 1:
            mean_cost_semantics, available_costs = next(iter(cost_populations.items()))
            mean_cost = statistics.fmean(available_costs)
        else:
            mean_cost_semantics = None
            mean_cost = None
        latencies = [
            float(run.record.latency_seconds)
            for run in group
            if _is_finite_number(run.record.latency_seconds)
            and run.record.latency_seconds >= 0
        ]
        aggregates.append(
            {
                "task_id": task_id,
                "model": model,
                "provider": provider,
                "harness": harness,
                "experiment_model": model,
                "experiment_provider": provider,
                "experiment_harness": harness,
                "identity_semantics": "experiment_labels",
                "trials": len(group),
                "pass_rate": statistics.fmean(pass_values),
                "pass_variance": statistics.pvariance(pass_values),
                "cost_available_samples": cost_available,
                "cost_missing_samples": len(group) - cost_available,
                "mean_cost_usd": mean_cost,
                "mean_cost_semantics": mean_cost_semantics,
                "cost_populations": serialized_populations,
                "latency_available_samples": len(latencies),
                "latency_missing_samples": len(group) - len(latencies),
                "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
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
        (
            "| Task | Experiment model | Experiment provider | Experiment harness | Trials "
            "| Pass rate | Available cost | Cost samples | Missing cost | Mean latency |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        mean_cost = (
            f"${row['mean_cost_usd']:.6f} ({row['mean_cost_semantics']})"
            if row["mean_cost_usd"] is not None
            else "N/A"
        )
        mean_latency = (
            f"{row['mean_latency_seconds']:.3f}s"
            if row["mean_latency_seconds"] is not None
            else "N/A"
        )
        lines.append(
            f"| {row['task_id']} | {row['experiment_model']} | {row['experiment_provider']} "
            f"| {row['experiment_harness']} | {row['trials']} | {row['pass_rate']:.1%} "
            f"| {mean_cost} | {row['cost_available_samples']} "
            f"| {row['cost_missing_samples']} | {mean_latency} |"
        )
    lines.extend(
        [
            "",
            (
                "> Experiment labels are grouping identity, not observed execution provenance. "
                "Missing evidence is excluded from numeric aggregates."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
