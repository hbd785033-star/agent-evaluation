"""Milestone R2 vertical-slice contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_eval.adapters import RecordedAdapter
from agent_eval.checkers import CheckerProfile, TaskCheckRegistry
from agent_eval.cli import main
from agent_eval.execution_record import ExecutionRecordAdapter, load_execution_records
from agent_eval.judges import CalibrationArtifact, ProfileJudgeAdapter
from agent_eval.models import RunRecord, SuccessCriterion, TaskCase
from agent_eval.runner import EvalRunner
from evaluators.deterministic import CheckResult


def test_success_criterion_accepts_legacy_and_structured_entries():
    task = TaskCase.from_mapping(
        {
            "id": "criterion-contract",
            "success_criteria": {
                "deterministic": [
                    "legacy tests pass",
                    {
                        "id": "artifact_contains_answer",
                        "description": "artifact contains the expected answer",
                        "checker": "file_contains",
                        "config": {"path": "answer.txt", "contains": "42"},
                    },
                ]
            },
        }
    )
    legacy, structured = task.success_criteria["deterministic"]
    assert isinstance(legacy, SuccessCriterion)
    assert legacy.id == legacy.description == "legacy tests pass"
    assert legacy.checker is None
    assert structured.id == "artifact_contains_answer"
    assert structured.checker == "file_contains"
    assert structured.config == {"path": "answer.txt", "contains": "42"}
    assert task.success_criteria_mapping()["deterministic"] == [
        "legacy tests pass",
        {
            "id": "artifact_contains_answer",
            "description": "artifact contains the expected answer",
            "checker": "file_contains",
            "config": {"path": "answer.txt", "contains": "42"},
        },
    ]


def test_checker_registry_runs_builtins_profile_and_registered_custom(tmp_path):
    (tmp_path / "answer.txt").write_text("the answer is 42", encoding="utf-8")
    (tmp_path / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    task = TaskCase.from_mapping(
        {
            "id": "checker-contract",
            "success_criteria": {
                "deterministic": [
                    {"id": "exists", "checker": "file_exists", "path": "answer.txt"},
                    {
                        "id": "contains",
                        "checker": "file_contains",
                        "path": "answer.txt",
                        "contains": "42",
                    },
                    {
                        "id": "command",
                        "checker": "command",
                        "command": [sys.executable, "-c", "print('ok')"],
                    },
                    {"id": "pytest", "checker": "pytest", "path": "test_demo.py"},
                    "custom proof",
                ]
            },
        }
    )
    profile = CheckerProfile.from_mapping(
        {
            "schema_version": "0.1",
            "profile_id": "r2-checkers",
            "checks": {
                "custom proof": {"checker": "custom", "name": "proof"},
            },
        }
    )
    registry = TaskCheckRegistry()
    registry.register(
        "proof",
        lambda *_args: CheckResult(True, "custom proof", "registered proof", ["proof=registered"]),
    )
    record = RunRecord(
        "checker-contract",
        "m",
        "p",
        "h",
        1,
        run_id="checker-run",
        workspace_root=str(tmp_path),
    )

    report = EvalRunner(
        RecordedAdapter([record], workspace_root=tmp_path),
        checker_registry=registry,
        checker_profile=profile,
    ).run([task])

    assert report["runs"][0]["passed"] is True
    checks = report["runs"][0]["layers"]["deterministic"]["criterion_checks"]
    assert [check["criterion_id"] for check in checks] == [
        "exists",
        "contains",
        "command",
        "pytest",
        "custom proof",
    ]
    assert all(check["evidence"] for check in checks)
    assert report["checker_profile"] == {
        "profile_id": "r2-checkers",
        "schema_version": "0.1",
    }


def test_checker_registry_rejects_passing_result_without_evidence(tmp_path):
    task = TaskCase.from_mapping(
        {
            "id": "no-evidence",
            "success_criteria": {
                "deterministic": [{"id": "custom", "checker": "custom", "name": "empty"}]
            },
        }
    )
    registry = TaskCheckRegistry()
    registry.register("empty", lambda *_args: CheckResult(True, "custom", "trust me"))
    record = RunRecord("no-evidence", "m", "p", "h", 1, run_id="no-evidence-run")

    report = EvalRunner(RecordedAdapter([record]), checker_registry=registry).run([task])

    assert report["runs"][0]["passed"] is False
    criterion_check = report["runs"][0]["layers"]["deterministic"]["criterion_checks"][0]
    assert criterion_check["passed"] is False
    assert criterion_check["failure"] == "passing checker returned no evidence"


def test_calibrated_judge_records_versioned_identity():
    artifact = CalibrationArtifact.from_mapping(
        {
            "schema_version": "0.1",
            "calibration_id": "judge-cal-v1",
            "judge_model": "controlled-fake",
            "prompt_version": "judge-v1",
            "rubric_version": "core-v1",
            "golden_dataset_version": "judge-golden-v1",
            "sample_count": 30,
            "agreement_score": 0.86,
            "calibrated": True,
        }
    )
    judge = ProfileJudgeAdapter(
        artifact,
        lambda *_args: {
            "passed": True,
            "score": 100,
            "summary": "controlled evidence accepted",
            "evidence": ["fixture=perfect"],
        },
    )
    task = TaskCase.from_mapping(
        {
            "id": "judge-task",
            "success_criteria": {"llm_judge": ["minimal change"]},
        }
    )
    record = RunRecord("judge-task", "m", "p", "h", 1, run_id="judge-run")

    report = EvalRunner(RecordedAdapter([record]), judge=judge).run([task])

    layer = report["runs"][0]["layers"]["judge"]
    assert report["runs"][0]["passed"] is True
    assert layer["calibration_id"] == "judge-cal-v1"
    assert layer["prompt_version"] == "judge-v1"
    assert layer["rubric_version"] == "core-v1"
    assert layer["evidence"] == ["fixture=perfect"]


def test_uncalibrated_judge_fails_closed():
    artifact = CalibrationArtifact.from_mapping(
        {
            "schema_version": "0.1",
            "calibration_id": "draft",
            "judge_model": "controlled-fake",
            "prompt_version": "judge-v1",
            "rubric_version": "core-v1",
            "golden_dataset_version": "judge-golden-v1",
            "sample_count": 2,
            "agreement_score": 0.5,
            "calibrated": False,
        }
    )
    judge = ProfileJudgeAdapter(artifact, lambda *_args: {"passed": True, "score": 100})
    task = TaskCase.from_mapping(
        {
            "id": "judge-task",
            "success_criteria": {"llm_judge": ["quality"]},
        }
    )
    record = RunRecord("judge-task", "m", "p", "h", 1, run_id="judge-run")

    report = EvalRunner(RecordedAdapter([record]), judge=judge).run([task])

    assert report["runs"][0]["passed"] is False
    assert "not marked calibrated" in report["runs"][0]["layers"]["judge"]["error"]


def _execution_record(**updates):
    record = {
        "schema_version": "0.1",
        "task_id": "execution-task",
        "run_id": "run-001",
        "model": "model-a",
        "provider": "provider-a",
        "harness": "adaptive-agent-orchestrator",
        "status": "completed",
        "started_at": "2026-08-10T10:00:00Z",
        "finished_at": "2026-08-10T10:00:01Z",
        "latency_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 0,
        "cost_usd": 0.01,
        "tool_calls": [],
        "files_changed": [],
        "output": "done",
        "workspace_root": None,
        "isolation_level": "none",
        "metadata": {"trial": 1},
    }
    record.update(updates)
    return record


def test_execution_record_01_converts_without_silent_fields(tmp_path):
    path = tmp_path / "execution-record.json"
    path.write_text(json.dumps(_execution_record()), encoding="utf-8")

    records = load_execution_records(path)
    adapter = ExecutionRecordAdapter(records)
    converted = adapter.run(TaskCase(id="execution-task", prompt=""), 1)

    assert converted.run_id == "run-001"
    assert converted.metadata["execution_schema_version"] == "0.1"
    assert converted.output == "done"


def test_execution_record_rejects_unsupported_and_unknown_fields(tmp_path):
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(json.dumps(_execution_record(schema_version="0.2")), encoding="utf-8")
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(_execution_record(authority_graph={})), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported ExecutionRecord schema_version"):
        load_execution_records(unsupported)
    with pytest.raises(ValueError, match="unknown ExecutionRecord fields"):
        load_execution_records(unknown)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        json.dumps(_execution_record(latency_seconds=float("nan"))), encoding="utf-8"
    )
    with pytest.raises(TypeError, match="latency_seconds"):
        load_execution_records(nonfinite)


def test_aao_execution_record_fixture_is_accepted(tmp_path):
    aao_root = Path(__file__).parents[2] / "adaptive-agent-orchestrator"
    python = aao_root / ".venv" / "Scripts" / "python.exe"
    output = tmp_path / "aao-record.json"
    script = (
        "from contracts.execution import ExecutionRecord; "
        "ExecutionRecord(task_id='cross-repo-1',run_id='run-1',model='m',provider='p',"
        "status='completed',started_at='2026-08-10T10:00:00Z',"
        "finished_at='2026-08-10T10:00:01Z',latency_seconds=1,input_tokens=1,"
        "output_tokens=1,tool_calls=[],files_changed=[],output='ok',"
        "workspace_root=None,isolation_level='none',metadata={'trial':1}).export(r'"
        + str(output)
        + "')"
    )
    subprocess.run([str(python), "-c", script], cwd=aao_root, check=True)

    records = load_execution_records(output)

    assert records[0].raw["harness"] == "adaptive-agent-orchestrator"


def test_perfect_agent_cli_exits_zero_and_writes_evidence(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "report"

    exit_code = main(
        [
            "run",
            "--dataset",
            str(root / "datasets" / "smoke_tasks.yaml"),
            "--model",
            "controlled-perfect",
            "--provider",
            "fixture",
            "--harness",
            "perfect-agent",
            "--checker-profile",
            str(root / "profiles" / "checkers" / "smoke-v1.yaml"),
            "--judge-profile",
            str(root / "profiles" / "judges" / "controlled-v1.json"),
            "--source-cwd",
            str(root),
            "--workspace-root",
            str(tmp_path / "workspaces"),
            "--output-dir",
            str(output),
            "--command",
            sys.executable,
            str(root / "examples" / "perfect_agent.py"),
        ]
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["runs"][0]["passed"] is True
    criteria = report["runs"][0]["layers"]["deterministic"]["criterion_checks"]
    assert all(item["evidence"] for item in criteria)


def test_config_supplies_dataset_and_profiles(tmp_path):
    root = Path(__file__).parents[1]
    config = tmp_path / "eval.yaml"
    config.write_text(
        "dataset: " + str(root / "datasets" / "smoke_tasks.yaml").replace("\\", "/") + "\n"
        "checker_profile: "
        + str(root / "profiles" / "checkers" / "smoke-v1.yaml").replace("\\", "/")
        + "\n"
        "judge_profile: "
        + str(root / "profiles" / "judges" / "controlled-v1.json").replace("\\", "/")
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--config",
            str(config),
            "--model",
            "controlled-perfect",
            "--provider",
            "fixture",
            "--workspace-root",
            str(tmp_path / "workspaces"),
            "--source-cwd",
            str(root),
            "--output-dir",
            str(tmp_path / "report"),
            "--command",
            sys.executable,
            "examples/perfect_agent.py",
        ]
    )

    assert exit_code == 0
