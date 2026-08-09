"""Adapters that turn real harness executions into RunRecord objects."""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import RunRecord, TaskCase


@runtime_checkable
class AgentAdapter(Protocol):
    model: str
    provider: str
    harness: str

    def run(self, task: TaskCase, trial: int) -> RunRecord: ...


class CommandAgentAdapter:
    """Run any agent harness as a JSON-over-stdin subprocess.

    The command receives a TaskCase JSON object on stdin and must write one JSON
    object to stdout. Provider/model/harness identity is supplied by the caller,
    not trusted from subprocess output.
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
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = list(command)
        self.model = model
        self.provider = provider
        self.harness = harness
        self.timeout_seconds = timeout_seconds
        self.cwd = str(cwd) if cwd is not None else None

    def run(self, task: TaskCase, trial: int) -> RunRecord:
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
        }
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=self.cwd,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return RunRecord(
                task_id=task.id,
                model=self.model,
                provider=self.provider,
                harness=self.harness,
                trial=trial,
                latency_seconds=time.perf_counter() - started,
                exit_status="failed",
                error=str(exc),
            )

        latency = time.perf_counter() - started
        if completed.returncode != 0:
            return RunRecord(
                task_id=task.id,
                model=self.model,
                provider=self.provider,
                harness=self.harness,
                trial=trial,
                output=completed.stdout,
                latency_seconds=latency,
                exit_status="failed",
                error=completed.stderr[-2000:] or f"harness exit code {completed.returncode}",
            )

        try:
            raw = json.loads(completed.stdout)
            if not isinstance(raw, dict):
                raise TypeError("harness output must be a JSON object")
        except (json.JSONDecodeError, TypeError) as exc:
            return RunRecord(
                task_id=task.id,
                model=self.model,
                provider=self.provider,
                harness=self.harness,
                trial=trial,
                output=completed.stdout,
                latency_seconds=latency,
                exit_status="failed",
                error=f"invalid harness JSON: {exc}",
            )

        usage = raw.get("usage", {}) if isinstance(raw.get("usage", {}), dict) else {}
        return RunRecord(
            task_id=task.id,
            model=self.model,
            provider=self.provider,
            harness=self.harness,
            trial=trial,
            output=str(raw.get("output", "")),
            tool_calls=list(raw.get("tool_calls", [])),
            files_changed=list(raw.get("files_changed", [])),
            trajectory=list(raw.get("trajectory", [])),
            input_tokens=int(raw.get("input_tokens", usage.get("input_tokens", 0))),
            output_tokens=int(raw.get("output_tokens", usage.get("output_tokens", 0))),
            cached_tokens=int(raw.get("cached_tokens", usage.get("cached_tokens", 0))),
            cost_usd=float(raw.get("cost_usd", usage.get("cost_usd", 0.0))),
            latency_seconds=latency,
            exit_status=str(raw.get("exit_status", "completed")),
            error=raw.get("error"),
            run_id=raw.get("run_id"),
            metadata=dict(raw.get("metadata", {})),
        )


class RecordedAdapter:
    """Replay committed or exported RunRecord JSON for reproducible scoring."""

    def __init__(self, records: Sequence[RunRecord]) -> None:
        self._records = {(record.task_id, record.trial): record for record in records}
        first = records[0] if records else None
        self.model = first.model if first else "recorded"
        self.provider = first.provider if first else "recorded"
        self.harness = first.harness if first else "recorded"

    def run(self, task: TaskCase, trial: int) -> RunRecord:
        try:
            return self._records[(task.id, trial)]
        except KeyError as exc:
            raise KeyError(f"missing recorded run for {task.id} trial {trial}") from exc
