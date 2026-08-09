"""Stable contracts for real agent evaluation runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskCase:
    id: str
    prompt: str
    category: str = "general"
    description: str = ""
    allowed_files: list[str] = field(default_factory=list)
    forbidden_files: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    success_criteria: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, float | int] = field(default_factory=dict)
    trials: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> TaskCase:
        known = {
            "id", "prompt", "category", "description", "allowed_files",
            "forbidden_files", "forbidden_actions", "success_criteria",
            "limits", "trials",
        }
        return cls(
            id=str(raw["id"]),
            prompt=str(raw.get("prompt", "")),
            category=str(raw.get("category", "general")),
            description=str(raw.get("description", "")),
            allowed_files=list(raw.get("allowed_files", [])),
            forbidden_files=list(raw.get("forbidden_files", [])),
            forbidden_actions=list(raw.get("forbidden_actions", [])),
            success_criteria=dict(raw.get("success_criteria", {})),
            limits=dict(raw.get("limits", {})),
            trials=max(1, int(raw.get("trials", 1))),
            metadata={key: value for key, value in raw.items() if key not in known},
        )


@dataclass(slots=True)
class RunRecord:
    """Provider-neutral record of one real model/harness trial."""

    task_id: str
    model: str
    provider: str
    harness: str
    trial: int
    output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    exit_status: str = "completed"
    error: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
