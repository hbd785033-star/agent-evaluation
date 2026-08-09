"""Adapters that turn real harness executions into RunRecord objects."""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import RunRecord, TaskCase


@runtime_checkable
class AgentAdapter(Protocol):
    model: str
    provider: str
    harness: str
    isolation_level: str
    workspace_root: Path | None

    def run(self, task: TaskCase, trial: int) -> RunRecord: ...


class CommandAgentAdapter:
    """Run an agent harness as a JSON-over-stdin subprocess.

    A fresh copied workspace provides trial-local cwd isolation, but it is not an
    operating-system sandbox. The authoritative ``isolation_level`` is therefore
    ``workspace`` and policy-constrained evaluations fail closed unless records
    come from a control-plane-verified OS sandbox.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        model: str,
        provider: str,
        harness: str = "command",
        timeout_seconds: float = 600,
        cwd: str | Path | None = None,
        workspace_root: str | Path = ".agent-eval-workspaces",
        preserve_workspaces: bool = False,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = list(command)
        self.model = model
        self.provider = provider
        self.harness = harness
        self.timeout_seconds = timeout_seconds
        self.cwd = Path(cwd).resolve() if cwd is not None else None
        self.workspace_root = Path(workspace_root).resolve()
        self.preserve_workspaces = preserve_workspaces
        self.isolation_level = "workspace"
        if self.cwd is not None:
            try:
                self.workspace_root.relative_to(self.cwd)
            except ValueError:
                pass
            else:
                raise ValueError("workspace_root must not be inside source cwd")

    def _prepare_workspace(self, task: TaskCase, trial: int) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", task.id).strip("-") or "task"
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = self.workspace_root / f"{safe_id}-trial-{trial}-{uuid.uuid4().hex[:8]}"
        if self.cwd is None:
            workspace.mkdir()
        else:
            shutil.copytree(
                self.cwd,
                workspace,
                symlinks=True,
                ignore=shutil.ignore_patterns(
                    ".agent-eval-workspaces", "__pycache__", ".pytest_cache"
                ),
            )
        return workspace

    def _authority(self, workspace: Path) -> dict[str, object]:
        return {
            "sandbox_id": workspace.name,
            "workspace_root": str(workspace),
            "isolation_level": self.isolation_level,
        }

    def _failed_record(
        self,
        task: TaskCase,
        trial: int,
        workspace: Path,
        started: float,
        error: str,
        *,
        output: str = "",
    ) -> RunRecord:
        return RunRecord(
            task_id=task.id,
            model=self.model,
            provider=self.provider,
            harness=self.harness,
            trial=trial,
            output=output,
            latency_seconds=time.perf_counter() - started,
            exit_status="failed",
            error=error,
            run_id=f"runner-{uuid.uuid4().hex}",
            **self._authority(workspace),
        )

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        workspace = self._prepare_workspace(task, trial)
        payload = {
            "task": {
                "id": task.id,
                "prompt": task.prompt,
                "category": task.category,
                "description": task.description,
                "allowed_files": task.allowed_files,
                "forbidden_files": task.forbidden_files,
                "forbidden_actions": task.forbidden_actions,
                "success_criteria": task.success_criteria,
                "limits": task.limits,
            },
            "trial": trial,
            "model": self.model,
            "provider": self.provider,
            "harness": self.harness,
            "workspace": str(workspace),
            "isolation_level": self.isolation_level,
        }
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=workspace,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._failed_record(task, trial, workspace, started, str(exc))

        if completed.returncode != 0:
            return self._failed_record(
                task,
                trial,
                workspace,
                started,
                completed.stderr[-2000:] or f"harness exit code {completed.returncode}",
                output=completed.stdout,
            )

        try:
            raw = json.loads(completed.stdout)
            if not isinstance(raw, dict):
                raise TypeError("harness output must be a JSON object")
            usage = raw.get("usage", {})
            if not isinstance(usage, dict):
                raise TypeError("usage must be an object")
            metadata_raw = raw.get("metadata", {})
            if not isinstance(metadata_raw, dict):
                raise TypeError("metadata must be an object")
            metadata = dict(metadata_raw)
            for reserved in (
                "workspace_root",
                "dataset_version",
                "sandbox_id",
                "isolation_level",
            ):
                metadata.pop(reserved, None)
            candidate = RunRecord(
                task_id=task.id,
                model=self.model,
                provider=self.provider,
                harness=self.harness,
                trial=trial,
                output=raw.get("output", ""),
                tool_calls=raw.get("tool_calls", []),
                files_changed=raw.get("files_changed", []),
                trajectory=raw.get("trajectory", []),
                input_tokens=raw.get("input_tokens", usage.get("input_tokens", 0)),
                output_tokens=raw.get("output_tokens", usage.get("output_tokens", 0)),
                cached_tokens=raw.get("cached_tokens", usage.get("cached_tokens", 0)),
                cost_usd=raw.get("cost_usd", usage.get("cost_usd", 0.0)),
                latency_seconds=time.perf_counter() - started,
                exit_status=raw.get("exit_status", "completed"),
                error=raw.get("error"),
                run_id=raw.get("run_id") or f"runner-{uuid.uuid4().hex}",
                metadata=metadata,
                **self._authority(workspace),
            )
            errors = candidate.integrity_errors()
            if errors:
                raise TypeError("; ".join(errors))
            return candidate
        except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as exc:
            return self._failed_record(
                task,
                trial,
                workspace,
                started,
                f"invalid harness fields: {exc}",
                output=completed.stdout,
            )

    def cleanup(self, record: RunRecord) -> None:
        if self.preserve_workspaces:
            return
        if record.workspace_root:
            shutil.rmtree(record.workspace_root, ignore_errors=True)


class RecordedAdapter:
    """Replay exported records with optional control-plane trust context."""

    def __init__(
        self,
        records: Sequence[RunRecord],
        *,
        model: str | None = None,
        provider: str | None = None,
        harness: str | None = None,
        workspace_root: str | Path | None = None,
        isolation_level: str = "none",
    ) -> None:
        self._records = {(record.task_id, record.trial): record for record in records}
        first = records[0] if records else None
        self.model = model or (first.model if first else "recorded")
        self.provider = provider or (first.provider if first else "recorded")
        self.harness = harness or (first.harness if first else "recorded")
        self.workspace_root = (
            Path(workspace_root).resolve() if workspace_root is not None else None
        )
        if isolation_level not in {"none", "workspace", "os"}:
            raise ValueError("isolation_level must be none, workspace, or os")
        self.isolation_level = isolation_level

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        try:
            return copy.deepcopy(self._records[(task.id, trial)])
        except KeyError as exc:
            raise KeyError(f"missing recorded run for {task.id} trial {trial}") from exc

    def cleanup(self, record: RunRecord) -> None:
        return None
