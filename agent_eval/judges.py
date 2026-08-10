"""Versioned calibrated judge adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import RunRecord, TaskCase


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    schema_version: str
    calibration_id: str
    judge_model: str
    prompt_version: str
    rubric_version: str
    golden_dataset_version: str
    sample_count: int
    agreement_score: float
    calibrated: bool

    @classmethod
    def from_mapping(cls, raw: Any) -> CalibrationArtifact:
        if not isinstance(raw, dict):
            raise TypeError("calibration artifact must be an object")
        expected = {
            "schema_version",
            "calibration_id",
            "judge_model",
            "prompt_version",
            "rubric_version",
            "golden_dataset_version",
            "sample_count",
            "agreement_score",
            "calibrated",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown or missing:
            raise ValueError(
                f"invalid calibration artifact fields: missing={missing}, unknown={unknown}"
            )
        if raw["schema_version"] != "0.1":
            raise ValueError("calibration artifact schema_version must be 0.1")
        for name in (
            "calibration_id",
            "judge_model",
            "prompt_version",
            "rubric_version",
            "golden_dataset_version",
        ):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise TypeError(f"calibration artifact {name} must be a non-empty string")
        if (
            not isinstance(raw["sample_count"], int)
            or isinstance(raw["sample_count"], bool)
            or raw["sample_count"] < 1
        ):
            raise TypeError("calibration artifact sample_count must be positive")
        score = raw["agreement_score"]
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0 <= score <= 1:
            raise TypeError("calibration artifact agreement_score must be between 0 and 1")
        if not isinstance(raw["calibrated"], bool):
            raise TypeError("calibration artifact calibrated must be boolean")
        return cls(**raw)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationArtifact:
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    def identity(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "judge_model": self.judge_model,
            "prompt_version": self.prompt_version,
            "rubric_version": self.rubric_version,
            "golden_dataset_version": self.golden_dataset_version,
            "sample_count": self.sample_count,
            "agreement_score": self.agreement_score,
        }


@runtime_checkable
class JudgeAdapter(Protocol):
    calibrated: bool
    calibration_id: str

    def evaluate(
        self, task: TaskCase, record: RunRecord, deterministic: dict[str, Any]
    ) -> dict[str, Any]: ...


class ProfileJudgeAdapter:
    def __init__(
        self,
        artifact: CalibrationArtifact,
        evaluator: Callable[[TaskCase, RunRecord, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.artifact = artifact
        self.calibrated = artifact.calibrated
        self.calibration_id = artifact.calibration_id
        self.model = artifact.judge_model
        self.prompt_version = artifact.prompt_version
        self.rubric_version = artifact.rubric_version
        self._evaluator = evaluator

    def __call__(
        self, task: TaskCase, record: RunRecord, deterministic: dict[str, Any]
    ) -> dict[str, Any]:
        result = self.evaluate(task, record, deterministic)
        return {**result, **self.artifact.identity()}

    def evaluate(
        self, task: TaskCase, record: RunRecord, deterministic: dict[str, Any]
    ) -> dict[str, Any]:
        result = self._evaluator(task, record, deterministic)
        if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
            return {"passed": False, "error": "judge result requires boolean passed"}
        evidence = result.get("evidence", [])
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            return {"passed": False, "error": "judge evidence must be a list of strings"}
        if result["passed"] and not evidence:
            return {"passed": False, "error": "passing judge requires evidence"}
        return dict(result)


def controlled_profile_judge(path: str | Path) -> ProfileJudgeAdapter:
    """Load the deterministic smoke fixture; not a production LLM judge."""
    artifact = CalibrationArtifact.load(path)

    def evaluate(
        _task: TaskCase, record: RunRecord, _deterministic: dict[str, Any]
    ) -> dict[str, Any]:
        marker = "PERFECT_AGENT_EVIDENCE"
        passed = marker in record.output
        return {
            "passed": passed,
            "score": 100 if passed else 0,
            "summary": "controlled fixture evidence found"
            if passed
            else "fixture evidence missing",
            "evidence": [f"output_marker={marker}"] if passed else [],
        }

    return ProfileJudgeAdapter(artifact, evaluate)
