"""AE-1B evidence-truth regressions for imported ExecutionRecord 0.1."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_eval.adapters import CommandAgentAdapter, RecordedAdapter
from agent_eval.execution_record import ExecutionRecordAdapter, load_execution_records
from agent_eval.models import EvaluatedRun, RunRecord, TaskCase
from agent_eval.runner import EvalRunner, build_report, evaluate_run


def _execution_record(**updates):
    raw = {
        "schema_version": "0.1",
        "task_id": "truth-task",
        "run_id": "runtime-run-1",
        "model": "observed-model",
        "provider": "observed-provider",
        "harness": "adaptive-agent-orchestrator",
        "status": "completed",
        "started_at": "2026-08-21T00:00:00Z",
        "finished_at": "2026-08-21T00:00:01Z",
        "latency_seconds": 1.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "cached_tokens": 0,
        "cost_usd": 0.01,
        "tool_calls": [],
        "files_changed": [],
        "output": "done",
        "workspace_root": None,
        "isolation_level": "workspace",
        "metadata": {"trial": 1},
    }
    raw.update(updates)
    return raw


def _load(tmp_path: Path, rows) -> list:
    path = tmp_path / "records.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return load_execution_records(path)


def test_full_execution_record_preserves_observed_provenance(tmp_path):
    raw = _execution_record(
        metadata={
            "trial": 1,
            "planned": {
                "runtime_plan": {"executor": "planned-runtime"},
                "runtime_selection": {"selected_runtime": "selected-runtime"},
            },
            "observed": {
                "runtime_adapter_invoked": True,
                "runtime_adapter": "observed-runtime",
            },
        }
    )

    converted = ExecutionRecordAdapter(_load(tmp_path, raw)).run(
        TaskCase(id="truth-task", prompt=""), 1
    )

    assert converted.model == "execution-record"
    assert converted.provider == "execution-record"
    assert converted.harness == "execution-record"
    assert converted.observed_model == "observed-model"
    assert converted.observed_provider == "observed-provider"
    assert converted.observed_harness == "adaptive-agent-orchestrator"
    assert converted.planned_runtime == "planned-runtime"
    assert converted.selected_runtime == "selected-runtime"
    assert converted.observed_runtime == "observed-runtime"


def test_all_producer_nullable_observations_remain_none(tmp_path):
    raw = _execution_record(
        status="failed",
        run_id=None,
        model=None,
        provider=None,
        input_tokens=None,
        output_tokens=None,
        cached_tokens=None,
        cost_usd=None,
        tool_calls=None,
        files_changed=None,
        output=None,
        workspace_root=None,
        isolation_level=None,
        metadata={"trial": 1},
    )

    record = ExecutionRecordAdapter(_load(tmp_path, raw)).run(
        TaskCase(id="truth-task", prompt=""), 1
    )

    assert record.run_id is None
    assert record.observed_model is None
    assert record.observed_provider is None
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cached_tokens is None
    assert record.cost_usd is None
    assert record.tool_calls is None
    assert record.files_changed is None
    assert record.output is None
    assert record.trajectory is None
    assert record.metadata["claimed_isolation_level"] is None


def test_observed_zero_and_empty_remain_observed(tmp_path):
    raw = _execution_record(
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        cost_usd=0.0,
        tool_calls=[],
        files_changed=[],
        output="",
    )
    record = ExecutionRecordAdapter(_load(tmp_path, raw)).run(
        TaskCase(id="truth-task", prompt=""), 1
    )

    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.cached_tokens == 0
    assert record.cost_usd == 0.0
    assert record.tool_calls == []
    assert record.files_changed == []
    assert record.output == ""


def test_cached_constraint_applies_only_when_values_are_known(tmp_path):
    _load(tmp_path, _execution_record(input_tokens=None, cached_tokens=9))
    _load(tmp_path, _execution_record(input_tokens=9, cached_tokens=None))

    with pytest.raises(ValueError, match="cached_tokens cannot exceed input_tokens"):
        _load(tmp_path, _execution_record(input_tokens=1, cached_tokens=2))


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "timeout"])
def test_terminal_status_is_preserved(tmp_path, status):
    raw = _execution_record(status=status, run_id=None if status != "completed" else "run-1")
    record = ExecutionRecordAdapter(_load(tmp_path, raw)).run(
        TaskCase(id="truth-task", prompt=""), 1
    )
    assert record.exit_status == status


def test_missing_run_id_is_ingested_and_never_synthesized(tmp_path):
    raw = _execution_record(run_id=None)
    adapter = ExecutionRecordAdapter(_load(tmp_path, raw))

    report = EvalRunner(adapter).run([TaskCase(id="truth-task", prompt="")])

    assert report["runs"][0]["record"]["run_id"] is None
    assert "invalid-" not in json.dumps(report)


def test_duplicate_real_run_ids_rejected_but_multiple_missing_ids_allowed(tmp_path):
    missing = [
        _execution_record(task_id="task-1", run_id=None),
        _execution_record(task_id="task-2", run_id=None),
    ]
    assert len(_load(tmp_path, missing)) == 2

    duplicate = [
        _execution_record(task_id="task-1", run_id="same"),
        _execution_record(task_id="task-2", run_id="same"),
    ]
    with pytest.raises(ValueError, match="duplicate ExecutionRecord run_id"):
        _load(tmp_path, duplicate)


def test_missing_trial_cannot_collapse_multiple_records_for_one_task(tmp_path):
    rows = [_execution_record(run_id=None), _execution_record(run_id=None)]
    with pytest.raises(ValueError, match="duplicate ExecutionRecord task_id/trial"):
        _load(tmp_path, rows)


def test_missing_observed_runtime_is_not_inferred_from_plan(tmp_path):
    raw = _execution_record(
        metadata={
            "planned": {
                "runtime_plan": {"executor": "planned"},
                "runtime_selection": {"selected_runtime": "selected"},
            },
            "observed": {
                "runtime_adapter_invoked": False,
                "runtime_adapter": "configured-but-not-invoked",
            },
        }
    )
    record = ExecutionRecordAdapter(_load(tmp_path, raw)).run(
        TaskCase(id="truth-task", prompt=""), 1
    )

    assert record.planned_runtime == "planned"
    assert record.selected_runtime == "selected"
    assert record.observed_runtime is None


def test_aao_verification_does_not_promote_ae_result(tmp_path):
    raw = _execution_record(
        status="failed",
        run_id=None,
        metadata={"trial": 1, "verification_status": "pass"},
    )
    report = EvalRunner(ExecutionRecordAdapter(_load(tmp_path, raw))).run(
        [TaskCase(id="truth-task", prompt="")]
    )

    assert report["runs"][0]["record"]["metadata"]["verification_status"] == "pass"
    assert report["runs"][0]["passed"] is False


def test_unavailable_and_observed_empty_trajectory_are_distinct():
    task = TaskCase(id="t", prompt="")
    unavailable = RunRecord("t", "m", "p", "h", 1, run_id="u", trajectory=None, tool_calls=None)
    observed_empty = RunRecord("t", "m", "p", "h", 1, run_id="e", trajectory=[])

    unavailable_result = evaluate_run(task, unavailable)
    empty_result = evaluate_run(task, observed_empty)

    assert unavailable_result.layers["trajectory"]["status"] == "unavailable"
    assert unavailable_result.layers["trajectory"]["evaluated"] is False
    assert unavailable_result.layers["trajectory"]["stats"] is None
    assert empty_result.layers["trajectory"]["status"] == "complete"
    assert empty_result.layers["trajectory"]["evaluated"] is True
    assert empty_result.layers["trajectory"]["stats"]["total_steps"] == 0


def test_missing_file_and_action_evidence_fail_closed():
    record = RunRecord(
        "t",
        "m",
        "p",
        "h",
        1,
        run_id="r",
        files_changed=None,
        tool_calls=None,
        trajectory=None,
    )
    task = TaskCase(
        id="t",
        prompt="",
        allowed_files=["src/**"],
        forbidden_actions=["git push"],
    )

    evaluated = evaluate_run(task, record)
    failures = {
        item["check"] for item in evaluated.layers["deterministic"]["failures"]
    }

    assert "files_changed_evidence_available" in failures
    assert "action_evidence_available" in failures
    assert evaluated.passed is False


def test_missing_output_remains_none_and_security_is_partial():
    record = RunRecord("t", "m", "p", "h", 1, run_id="r", output=None)

    evaluated = evaluate_run(TaskCase(id="t", prompt=""), record)

    assert evaluated.record.output is None
    assert evaluated.layers["security"]["completeness"] != "complete"


def test_unknown_cost_is_excluded_and_observed_zero_is_included():
    runs = [
        EvaluatedRun(RunRecord("t", "m", "p", "h", 1, cost_usd=None), False, {}),
        EvaluatedRun(
            RunRecord("t", "m", "p", "h", 2, cost_usd=0.0, cost_semantics="reported"),
            False,
            {},
        ),
        EvaluatedRun(
            RunRecord("t", "m", "p", "h", 3, cost_usd=2.0, cost_semantics="reported"),
            False,
            {},
        ),
    ]

    aggregate = build_report(runs)["aggregates"][0]

    assert aggregate["trials"] == 3
    assert aggregate["cost_available_samples"] == 2
    assert aggregate["cost_missing_samples"] == 1
    assert aggregate["mean_cost_usd"] == 1.0
    assert aggregate["mean_cost_semantics"] == "reported"


def test_no_available_cost_produces_null_mean():
    run = EvaluatedRun(RunRecord("t", "m", "p", "h", 1, cost_usd=None), False, {})
    aggregate = build_report([run])["aggregates"][0]

    assert aggregate["cost_available_samples"] == 0
    assert aggregate["cost_missing_samples"] == 1
    assert aggregate["mean_cost_usd"] is None


def test_negative_telemetry_fails_without_becoming_zero():
    record = RunRecord("t", "m", "p", "h", 1, cost_usd=-1.0, input_tokens=-1)
    report = EvalRunner(RecordedAdapter([record])).run([TaskCase(id="t", prompt="")])
    stored = report["runs"][0]["record"]

    assert report["runs"][0]["passed"] is False
    assert stored["cost_usd"] == -1.0
    assert stored["input_tokens"] == -1


def test_adapter_exception_has_no_fake_execution_run_id():
    class ExplodingAdapter:
        model = "m"
        provider = "p"
        harness = "h"
        isolation_level = "none"
        workspace_root = None

        def run(self, _task, _trial):
            raise OSError("boom")

        def cleanup(self, _record):
            return None

    report = EvalRunner(ExplodingAdapter()).run([TaskCase(id="t", prompt="")])
    assert report["runs"][0]["record"]["run_id"] is None


def test_recorded_adapter_rejects_duplicate_task_trial():
    records = [
        RunRecord("t", "m", "p", "h", 1, run_id="one"),
        RunRecord("t", "m", "p", "h", 1, run_id="two"),
    ]
    with pytest.raises(ValueError, match="duplicate RecordedAdapter task_id/trial"):
        RecordedAdapter(records)


def test_omitted_legacy_replay_evidence_defaults_to_none():
    record = RunRecord.from_dict(
        {"task_id": "t", "model": "m", "provider": "p", "harness": "h", "trial": 1}
    )

    assert record.output is None
    assert record.tool_calls is None
    assert record.files_changed is None
    assert record.trajectory is None
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cached_tokens is None
    assert record.cost_usd is None
    assert record.latency_seconds is None
    assert record.cost_semantics is None


def test_omitted_replay_evidence_cannot_pass_file_or_action_constraints(tmp_path):
    record = RunRecord.from_dict(
        {
            "task_id": "t",
            "model": "observed-m",
            "provider": "observed-p",
            "harness": "observed-h",
            "trial": 1,
            "workspace_root": str(tmp_path),
            "isolation_level": "os",
        }
    )
    report = EvalRunner(
        RecordedAdapter(
            [record],
            model="experiment-m",
            provider="experiment-p",
            harness="experiment-h",
            workspace_root=tmp_path,
            isolation_level="os",
        )
    ).run(
        [
            TaskCase(
                id="t",
                prompt="",
                allowed_files=["src/**"],
                forbidden_actions=["git push"],
            )
        ]
    )
    failures = {
        item["check"] for item in report["runs"][0]["layers"]["deterministic"]["failures"]
    }

    assert report["runs"][0]["passed"] is False
    assert "files_changed_evidence_available" in failures
    assert "action_evidence_available" in failures
    assert report["aggregates"][0]["cost_available_samples"] == 0


def test_recorded_adapter_preserves_legacy_observed_identity_before_override():
    source = RunRecord("t", "observed-m", "observed-p", "observed-h", 1)
    report = EvalRunner(
        RecordedAdapter(
            [source],
            model="experiment-m",
            provider="experiment-p",
            harness="experiment-h",
        )
    ).run([TaskCase(id="t", prompt="")])
    stored = report["runs"][0]["record"]

    assert stored["model"] == "experiment-m"
    assert stored["provider"] == "experiment-p"
    assert stored["harness"] == "experiment-h"
    assert stored["observed_model"] == "observed-m"
    assert stored["observed_provider"] == "observed-p"
    assert stored["observed_harness"] == "observed-h"


def test_command_adapter_omitted_observations_remain_none(tmp_path):
    harness = tmp_path / "unknown.py"
    harness.write_text("import json; print(json.dumps({}))", encoding="utf-8")
    adapter = CommandAgentAdapter(
        [sys.executable, str(harness)],
        model="m",
        provider="p",
        workspace_root=tmp_path / "workspaces",
    )

    record = adapter.run(TaskCase(id="t", prompt=""), 1)

    assert record.output is None
    assert record.tool_calls is None
    assert record.files_changed is None
    assert record.trajectory is None
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.cached_tokens is None
    assert record.cost_usd is None
    assert record.run_id is None
    adapter.cleanup(record)


def test_command_adapter_timeout_is_preserved(monkeypatch, tmp_path):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["agent"], 1)

    monkeypatch.setattr("agent_eval.adapters.subprocess.run", timeout)
    adapter = CommandAgentAdapter(
        ["agent"], model="m", provider="p", workspace_root=tmp_path / "workspaces"
    )
    record = adapter.run(TaskCase(id="t", prompt=""), 1)

    assert record.exit_status == "timeout"
    assert record.run_id is None
    assert record.output is None
    adapter.cleanup(record)


def test_generic_adapter_timeout_is_preserved():
    class TimeoutAdapter:
        model = "m"
        provider = "p"
        harness = "h"
        isolation_level = "none"
        workspace_root = None

        def run(self, _task, _trial):
            raise TimeoutError("timed out")

        def cleanup(self, _record):
            return None

    report = EvalRunner(TimeoutAdapter()).run([TaskCase(id="t", prompt="")])
    stored = report["runs"][0]["record"]

    assert stored["exit_status"] == "timeout"
    assert stored["run_id"] is None
    assert stored["output"] is None


def test_missing_output_security_is_unavailable_and_blocks_overall_pass():
    record = RunRecord(
        "t", "m", "p", "h", 1, output=None, tool_calls=[], files_changed=[], trajectory=[]
    )
    evaluated = evaluate_run(TaskCase(id="t", prompt=""), record)

    assert evaluated.layers["security"]["passed"] is False
    assert evaluated.layers["security"]["output_evidence"] == "unavailable"
    assert "output evidence unavailable" in evaluated.layers["security"]["violations"][0]
    assert evaluated.passed is False


def test_observed_empty_output_remains_evaluable_empty_output():
    record = RunRecord(
        "t", "m", "p", "h", 1, output="", tool_calls=[], files_changed=[], trajectory=[]
    )
    evaluated = evaluate_run(TaskCase(id="t", prompt=""), record)

    assert evaluated.record.output == ""
    assert evaluated.layers["security"]["output_evidence"] == "available"
    assert evaluated.layers["security"]["passed"] is True


def test_adapter_exception_marks_every_unobserved_field_none():
    class ExplodingAdapter:
        model = "m"
        provider = "p"
        harness = "h"
        isolation_level = "none"
        workspace_root = None

        def run(self, _task, _trial):
            raise OSError("boom")

        def cleanup(self, _record):
            return None

    report = EvalRunner(ExplodingAdapter()).run([TaskCase(id="t", prompt="")])
    stored = report["runs"][0]["record"]
    for name in (
        "output",
        "tool_calls",
        "files_changed",
        "trajectory",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cost_usd",
        "latency_seconds",
        "run_id",
    ):
        assert stored[name] is None
