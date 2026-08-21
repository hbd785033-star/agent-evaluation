"""Fail-closed deterministic criterion checker registry."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluators.deterministic import CheckResult

Checker = Callable[[Any, Any, dict[str, Any]], CheckResult]


@dataclass(frozen=True, slots=True)
class CheckerProfile:
    profile_id: str
    schema_version: str = "0.1"
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CheckerProfile:
        if not isinstance(raw, dict) or raw.get("schema_version", "0.1") != "0.1":
            raise ValueError("checker profile schema_version must be 0.1")
        profile_id = raw.get("profile_id")
        checks = raw.get("checks", {})
        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or not isinstance(checks, dict)
        ):
            raise TypeError("checker profile requires profile_id and checks object")
        return cls(profile_id, "0.1", {str(k): dict(v) for k, v in checks.items()})

    def identity(self) -> dict[str, str]:
        return {"profile_id": self.profile_id, "schema_version": self.schema_version}


class TaskCheckRegistry:
    def __init__(self) -> None:
        self._custom: dict[str, Callable[..., CheckResult]] = {}

    def register(self, name: str, checker: Callable[..., CheckResult]) -> None:
        if not name.strip() or not callable(checker):
            raise TypeError("checker name and callable are required")
        self._custom[name] = checker

    def _run_one(
        self, criterion: Any, task: Any, record: Any, profile: CheckerProfile | None
    ) -> CheckResult:
        config = dict(criterion.config)
        checker_name = criterion.checker
        configured = profile.checks.get(criterion.id, {}) if profile is not None else {}
        if checker_name is None and configured:
            checker_name = configured.get("checker")
            config = {**{k: v for k, v in configured.items() if k != "checker"}, **config}
        if checker_name is None:
            return CheckResult(
                False, criterion.id, "no executable checker", ["criterion declaration"]
            )
        try:
            if checker_name == "file_exists":
                path = self._safe_path(record.workspace_root, config["path"])
                passed = path.is_file()
                return CheckResult(passed, criterion.id, str(path), [str(path)])
            if checker_name == "file_contains":
                path = self._safe_path(record.workspace_root, config["path"])
                needle = str(config["contains"])
                text = path.read_text(encoding="utf-8") if path.is_file() else ""
                return CheckResult(
                    path.is_file() and needle in text,
                    criterion.id,
                    str(path),
                    [str(path), f"contains={needle!r}"],
                )
            if checker_name == "command":
                if configured.get("checker") != "command":
                    return CheckResult(
                        False,
                        criterion.id,
                        "command requires a trusted checker profile",
                        [criterion.id],
                    )
                command = configured.get("command")
                if not isinstance(command, (list, tuple)) or not command:
                    return CheckResult(
                        False, criterion.id, "command must be a non-empty argv list", []
                    )
                result = subprocess.run(
                    list(map(str, command)),
                    cwd=record.workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                evidence = [
                    f"exit_code={result.returncode}",
                    result.stdout[-500:],
                    result.stderr[-500:],
                ]
                return CheckResult(
                    result.returncode == 0,
                    criterion.id,
                    "command executed",
                    [x for x in evidence if x],
                )
            if checker_name == "pytest":
                if configured.get("checker") != "pytest":
                    return CheckResult(
                        False,
                        criterion.id,
                        "pytest requires a trusted checker profile",
                        [criterion.id],
                    )
                path = self._safe_path(
                    record.workspace_root,
                    configured.get("path", config.get("path", ".")),
                )
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(path), "-q"],
                    cwd=record.workspace_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return CheckResult(
                    result.returncode == 0,
                    criterion.id,
                    "pytest executed",
                    [f"exit_code={result.returncode}", result.stdout[-500:], result.stderr[-500:]],
                )
            if checker_name == "custom":
                checker_name = str(config.get("name", ""))
            if checker_name in self._custom:
                fn = self._custom[checker_name]
                try:
                    result = fn(task, record, config)
                except TypeError:
                    result = fn(task, record)
                return (
                    result
                    if isinstance(result, CheckResult)
                    else CheckResult(
                        False, criterion.id, "custom checker returned invalid result", []
                    )
                )
            return CheckResult(
                False, criterion.id, f"unsupported checker: {checker_name}", [str(checker_name)]
            )
        except Exception as exc:  # checker boundary is fail-closed
            return CheckResult(
                False, criterion.id, f"checker failed: {type(exc).__name__}", [criterion.id]
            )

    @staticmethod
    def _safe_path(workspace_root: str | None, raw_path: Any) -> Path:
        if not workspace_root:
            raise ValueError("checker requires a trusted workspace")
        root = Path(workspace_root).resolve()
        raw = Path(str(raw_path))
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError("checker path escapes workspace")
        candidate = (root / raw).resolve(strict=False)
        candidate.relative_to(root)
        return candidate

    def check(
        self, task: Any, record: Any, profile: CheckerProfile | None = None
    ) -> list[CheckResult]:
        results = []
        for criterion in task.success_criteria.get("deterministic", []):
            result = self._run_one(criterion, task, record, profile)
            if result.passed and not result.evidence:
                result = CheckResult(
                    False, result.check_name, "passing checker returned no evidence", []
                )
            results.append(result)
        return results
