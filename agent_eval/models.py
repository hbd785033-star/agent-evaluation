"""Stable contracts for real agent evaluation runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """One stable criterion identity with optional executable checker config."""

    id: str
    description: str
    checker: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    legacy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise TypeError("success criterion id must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("success criterion description must be a string")
        if self.checker is not None and (
            not isinstance(self.checker, str) or not self.checker.strip()
        ):
            raise TypeError("success criterion checker must be a non-empty string or null")
        if not isinstance(self.config, dict):
            raise TypeError("success criterion config must be an object")

    def __str__(self) -> str:
        return self.id

    @classmethod
    def from_value(cls, raw: Any) -> SuccessCriterion:
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            return cls(id=raw, description=raw, legacy=True)
        if not isinstance(raw, dict):
            raise TypeError("success criteria entries must be strings or objects")
        criterion_id = raw.get("id")
        description = raw.get("description", criterion_id)
        checker = raw.get("checker", raw.get("type"))
        config = raw.get("config", raw.get("params", raw.get("with", {})))
        known = {"id", "description", "checker", "type", "config", "params", "with"}
        if not isinstance(config, dict):
            raise TypeError("success criterion config must be an object")
        merged_config = dict(config)
        merged_config.update({key: value for key, value in raw.items() if key not in known})
        return cls(
            id=criterion_id,
            description=description,
            checker=checker,
            config=merged_config,
        )

    def to_value(self) -> str | dict[str, Any]:
        if self.legacy:
            return self.id
        return {
            "id": self.id,
            "description": self.description,
            "checker": self.checker,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class TaskCase:
    id: str
    prompt: str
    category: str = "general"
    description: str = ""
    allowed_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    success_criteria: dict[str, list[SuccessCriterion | str | dict[str, Any]]] = field(
        default_factory=dict
    )
    limits: dict[str, float | int] = field(default_factory=dict)
    trials: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.success_criteria, dict):
            raise TypeError("task.success_criteria must be an object")
        normalized: dict[str, list[SuccessCriterion]] = {}
        for layer, criteria in self.success_criteria.items():
            if not isinstance(layer, str) or not isinstance(criteria, list):
                raise TypeError("task.success_criteria must map strings to lists")
            normalized[layer] = [SuccessCriterion.from_value(item) for item in criteria]
        self.success_criteria = normalized

    def success_criteria_mapping(self) -> dict[str, list[str | dict[str, Any]]]:
        return {
            layer: [criterion.to_value() for criterion in criteria]
            for layer, criteria in self.success_criteria.items()
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TaskCase:
        if not isinstance(raw, dict):
            raise TypeError("task must be an object")
        if not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise TypeError("task.id must be a non-empty string")

        def string_field(name: str, default: str) -> str:
            value = raw.get(name, default)
            if not isinstance(value, str):
                raise TypeError(f"task.{name} must be a string")
            return value

        def string_list(name: str) -> list[str]:
            value = raw.get(name, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise TypeError(f"task.{name} must be a list of strings")
            return list(value)

        success_criteria = raw.get("success_criteria", {})
        if not isinstance(success_criteria, dict) or not all(
            isinstance(key, str) and isinstance(value, list)
            for key, value in success_criteria.items()
        ):
            raise TypeError("task.success_criteria must map strings to lists")
        limits = raw.get("limits", {})
        if not isinstance(limits, dict) or not all(
            isinstance(key, str) and _is_finite_number(value) and value >= 0
            for key, value in limits.items()
        ):
            raise TypeError("task.limits must map strings to non-negative numbers")
        trials = raw.get("trials", 1)
        if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
            raise TypeError("task.trials must be a positive integer")
        known = {
            "id",
            "prompt",
            "category",
            "description",
            "allowed_files",
            "forbidden_files",
            "forbidden_actions",
            "success_criteria",
            "limits",
            "trials",
        }
        return cls(
            id=raw["id"],
            prompt=string_field("prompt", ""),
            category=string_field("category", "general"),
            description=string_field("description", ""),
            allowed_files=string_list("allowed_files"),
            forbidden_files=string_list("forbidden_files"),
            forbidden_actions=string_list("forbidden_actions"),
            success_criteria={key: list(value) for key, value in success_criteria.items()},
            limits=dict(limits),
            trials=trials,
            metadata={key: value for key, value in raw.items() if key not in known},
        )


@dataclass(slots=True)
class RunRecord:
    """One evaluation trial with experiment labels separate from execution provenance."""

    task_id: str
    model: str
    provider: str
    harness: str
    trial: int
    output: str | None = ""
    tool_calls: list[dict[str, Any]] | None = field(default_factory=list)
    files_changed: list[str] | None = field(default_factory=list)
    trajectory: list[dict[str, Any]] | None = field(default_factory=list)
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cached_tokens: int | None = 0
    cost_usd: float | None = 0.0
    latency_seconds: float = 0.0
    exit_status: str = "completed"
    error: str | None = None
    run_id: str | None = None
    observed_model: str | None = None
    observed_provider: str | None = None
    observed_harness: str | None = None
    planned_runtime: str | None = None
    selected_runtime: str | None = None
    observed_runtime: str | None = None
    cost_semantics: str | None = "reported"
    dataset_version: str = ""
    sandbox_id: str = ""
    workspace_root: str | None = None
    isolation_level: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    def integrity_errors(self) -> list[str]:
        """Return schema and accounting invariant violations without changing evidence."""
        errors: list[str] = []
        if self.output is not None and not isinstance(self.output, str):
            errors.append("output must be a string or null")
        if self.error is not None and not isinstance(self.error, str):
            errors.append("error must be a string or null")
        if self.run_id is not None and not isinstance(self.run_id, str):
            errors.append("run_id must be a string or null")
        for name in ("task_id", "model", "provider", "harness"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{name} must be a non-empty experiment label")
        for name in (
            "observed_model",
            "observed_provider",
            "observed_harness",
            "planned_runtime",
            "selected_runtime",
            "observed_runtime",
            "cost_semantics",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"{name} must be a non-empty string or null")
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            errors.append("trial must be a positive integer")
        if not isinstance(self.dataset_version, str):
            errors.append("dataset_version must be a string")
        if not isinstance(self.sandbox_id, str):
            errors.append("sandbox_id must be a string")
        if self.workspace_root is not None and not isinstance(self.workspace_root, str):
            errors.append("workspace_root must be a string or null")
        if self.isolation_level not in {"none", "workspace", "os"}:
            errors.append("isolation_level is invalid")
        if self.tool_calls is not None:
            if not isinstance(self.tool_calls, list) or not all(
                isinstance(item, dict) for item in self.tool_calls
            ):
                errors.append("tool_calls must be a list of objects or null")
            elif any(
                key in item and not isinstance(item[key], dict)
                for item in self.tool_calls
                for key in ("arguments", "args")
            ):
                errors.append("tool call arguments must be objects")
        if self.files_changed is not None:
            if not isinstance(self.files_changed, list) or not all(
                isinstance(item, str) for item in self.files_changed
            ):
                errors.append("files_changed must be a list of strings or null")
            elif any("\x00" in item for item in self.files_changed):
                errors.append("files_changed paths must not contain NUL")
        if self.trajectory is not None and (
            not isinstance(self.trajectory, list)
            or not all(isinstance(item, dict) for item in self.trajectory)
        ):
            errors.append("trajectory must be a list of objects or null")
        if not isinstance(self.metadata, dict):
            errors.append("metadata must be an object")
        for name in ("input_tokens", "output_tokens", "cached_tokens"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                errors.append(f"{name} must be a non-negative integer or null")
        if self.cost_usd is not None and (
            not _is_finite_number(self.cost_usd) or self.cost_usd < 0
        ):
            errors.append("cost_usd must be a finite non-negative number or null")
        if not _is_finite_number(self.latency_seconds) or self.latency_seconds < 0:
            errors.append("latency_seconds must be a finite non-negative number")
        if (
            isinstance(self.cached_tokens, int)
            and not isinstance(self.cached_tokens, bool)
            and isinstance(self.input_tokens, int)
            and not isinstance(self.input_tokens, bool)
            and self.cached_tokens > self.input_tokens
        ):
            errors.append("cached_tokens cannot exceed input_tokens")
        if self.exit_status not in {"completed", "failed", "cancelled", "timeout"}:
            errors.append("exit_status is invalid")
        if isinstance(self.metadata, dict):
            for name in ("retries", "sub_agents"):
                if name not in self.metadata or self.metadata[name] is None:
                    continue
                value = self.metadata[name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"metadata.{name} must be a non-negative integer or null")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunRecord:
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in fields})


@dataclass(slots=True)
class EvaluatedRun:
    record: RunRecord
    passed: bool
    layers: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record": self.record.to_dict(),
            "passed": self.passed,
            "layers": self.layers,
        }
