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
from agent_eval.execution_record import (
    ExecutionRecordAdapter,
    WorkspaceAuthority,
    load_execution_records,
)
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
                "command": {
                    "checker": "command",
                    "command": [sys.executable, "-c", "print('ok')"],
                },
                "pytest": {"checker": "pytest", "path": "test_demo.py"},
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
        output="",
        tool_calls=[],
        files_changed=[],
        trajectory=[],
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


def test_checker_paths_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("SECRET", encoding="utf-8")
    task = TaskCase.from_mapping(
        {
            "id": "escape",
            "success_criteria": {
                "deterministic": [
                    {
                        "id": "escape",
                        "checker": "file_contains",
                        "path": "../secret.txt",
                        "contains": "SECRET",
                    }
                ]
            },
        }
    )
    record = RunRecord(
        "escape",
        "m",
        "p",
        "h",
        1,
        run_id="escape-run",
        workspace_root=str(workspace),
    )

    result = TaskCheckRegistry().check(task, record)[0]

    assert result.passed is False
    assert "checker failed" in result.detail


def test_inline_command_checker_is_not_trusted(tmp_path):
    task = TaskCase.from_mapping(
        {
            "id": "command",
            "success_criteria": {
                "deterministic": [
                    {
                        "id": "command",
                        "checker": "command",
                        "command": [sys.executable, "-c", "print('unsafe')"],
                    }
                ]
            },
        }
    )
    record = RunRecord(
        "command",
        "m",
        "p",
        "h",
        1,
        run_id="command-run",
        workspace_root=str(tmp_path),
    )

    result = TaskCheckRegistry().check(task, record)[0]

    assert result.passed is False
    assert "trusted checker profile" in result.detail


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
    record = RunRecord("judge-task", "m", "p", "h", 1, output="quality", run_id="judge-run")

    report = EvalRunner(RecordedAdapter([record]), judge=judge).run([task])

    layer = report["runs"][0]["layers"]["judge"]
    assert report["runs"][0]["passed"] is True
    assert layer["calibration_id"] == "judge-cal-v1"
    assert layer["prompt_version"] == "judge-v1"
    assert layer["rubric_version"] == "core-v1"
    assert layer["schema_version"] == "0.1"
    assert layer["calibrated"] is True
    assert len(layer["artifact_sha256"]) == 64
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
    record = RunRecord("judge-task", "m", "p", "h", 1, output="quality", run_id="judge-run")

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

    invalid_time = tmp_path / "invalid-time.json"
    invalid_time.write_text(
        json.dumps(_execution_record(started_at="not-a-date")), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="started_at"):
        load_execution_records(invalid_time)

    reversed_time = tmp_path / "reversed-time.json"
    reversed_time.write_text(
        json.dumps(
            _execution_record(
                started_at="2026-08-10T10:00:02Z",
                finished_at="2026-08-10T10:00:01Z",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finished_at"):
        load_execution_records(reversed_time)


def test_execution_record_authority_requires_explicit_trust(tmp_path):
    assigned = tmp_path / "assigned"
    assigned.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "result.txt").write_text("PASS\n", encoding="utf-8")
    path = tmp_path / "claimed.json"
    path.write_text(
        json.dumps(_execution_record(workspace_root=str(victim), isolation_level="os")),
        encoding="utf-8",
    )
    authority = WorkspaceAuthority.from_mapping(
        {
            "schema_version": "0.1",
            "trusted_workspace_root": str(tmp_path),
            "isolation_level": "os",
            "workspaces": [
                {
                    "task_id": "execution-task",
                    "trial": 1,
                    "workspace_root": str(assigned),
                }
            ],
        }
    )

    untrusted = ExecutionRecordAdapter(load_execution_records(path))
    trusted = ExecutionRecordAdapter(load_execution_records(path), authority=authority)
    converted = trusted.run(TaskCase(id="execution-task", prompt=""), 1)

    assert untrusted.workspace_root is None
    assert untrusted.isolation_level == "none"
    assert trusted.workspace_root == tmp_path.resolve()
    assert trusted.isolation_level == "os"
    assert converted.workspace_root == str(assigned.resolve())
    assert converted.metadata["claimed_workspace_root"] == str(victim)


def test_execution_record_cannot_substitute_sibling_workspace(tmp_path):
    assigned = tmp_path / "assigned"
    assigned.mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "result.txt").write_text("PASS\n", encoding="utf-8")
    path = tmp_path / "record.json"
    path.write_text(
        json.dumps(_execution_record(workspace_root=str(victim), isolation_level="os")),
        encoding="utf-8",
    )
    authority = WorkspaceAuthority.from_mapping(
        {
            "schema_version": "0.1",
            "trusted_workspace_root": str(tmp_path),
            "isolation_level": "os",
            "workspaces": [
                {
                    "task_id": "execution-task",
                    "trial": 1,
                    "workspace_root": str(assigned),
                }
            ],
        }
    )
    task = TaskCase.from_mapping(
        {
            "id": "execution-task",
            "success_criteria": {
                "deterministic": [
                    {
                        "id": "result",
                        "checker": "file_contains",
                        "path": "result.txt",
                        "contains": "PASS",
                    }
                ]
            },
        }
    )
    adapter = ExecutionRecordAdapter(load_execution_records(path), authority=authority)
    record = adapter.run(task, 1)

    result = TaskCheckRegistry().check(task, record)[0]

    assert record.workspace_root == str(assigned.resolve())
    assert result.passed is False


def test_aao_execution_record_fixture_is_accepted(tmp_path):
    aao_root = Path(__file__).parents[2] / "adaptive-agent-orchestrator"
    python = aao_root / ".venv" / "Scripts" / "python.exe"
    output = tmp_path / "aao-record.json"
    if aao_root.is_dir() and python.is_file():
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
    else:
        output.write_text(
            json.dumps(
                _execution_record(
                    task_id="cross-repo-1",
                    run_id="run-1",
                    model="m",
                    provider="p",
                    harness="adaptive-agent-orchestrator",
                    input_tokens=1,
                    output_tokens=1,
                    cost_usd=0.0,
                    output="ok",
                )
            ),
            encoding="utf-8",
        )

    records = load_execution_records(output)

    assert records[0].raw["harness"] == "adaptive-agent-orchestrator"


def test_evaluate_cli_does_not_trust_execution_record_authority_claims(tmp_path):
    root = Path(__file__).parents[1]
    assigned = tmp_path / "assigned-workspace"
    assigned.mkdir()
    victim = tmp_path / "victim-workspace"
    victim.mkdir()
    (victim / "result.txt").write_text("PASS\n", encoding="utf-8")
    record = tmp_path / "execution-record.json"
    record.write_text(
        json.dumps(
            _execution_record(
                task_id="smoke-perfect-001",
                output="PERFECT_AGENT_EVIDENCE",
                workspace_root=str(victim),
                isolation_level="os",
            )
        ),
        encoding="utf-8",
    )
    common = [
        "evaluate",
        str(record),
        "--dataset",
        str(root / "datasets" / "smoke_tasks.yaml"),
        "--checker-profile",
        str(root / "profiles" / "checkers" / "smoke-v1.yaml"),
        "--judge-profile",
        str(root / "profiles" / "judges" / "controlled-v1.json"),
    ]
    untrusted_output = tmp_path / "untrusted-report"
    substituted_output = tmp_path / "substituted-report"
    trusted_output = tmp_path / "trusted-report"
    authority = tmp_path / "workspace-authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "trusted_workspace_root": str(tmp_path),
                "isolation_level": "os",
                "workspaces": [
                    {
                        "task_id": "smoke-perfect-001",
                        "trial": 1,
                        "workspace_root": str(assigned),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    untrusted_exit = main([*common, "--output-dir", str(untrusted_output)])
    substituted_exit = main(
        [
            *common,
            "--output-dir",
            str(substituted_output),
            "--workspace-authority",
            str(authority),
        ]
    )
    (assigned / "result.txt").write_text("PASS\n", encoding="utf-8")
    trusted_exit = main(
        [*common, "--output-dir", str(trusted_output), "--workspace-authority", str(authority)]
    )

    untrusted_report = json.loads((untrusted_output / "report.json").read_text(encoding="utf-8"))
    assert untrusted_exit == 1
    assert untrusted_report["runs"][0]["passed"] is False
    assert untrusted_report["runs"][0]["record"]["workspace_root"] is None
    assert untrusted_report["runs"][0]["record"]["isolation_level"] == "none"
    assert substituted_exit == 1
    assert trusted_exit == 0


def test_evaluate_cli_writes_structured_checker_report_without_profile(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "result.txt").write_text("PASS\n", encoding="utf-8")
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        """version: r2-e2e
tasks:
  - id: imported-task
    prompt: Verify result
    success_criteria:
      deterministic:
        - id: result_exists
          checker: file_exists
          path: result.txt
        - id: result_contains_pass
          checker: file_contains
          path: result.txt
          contains: PASS
""",
        encoding="utf-8",
    )
    record = tmp_path / "execution-record.json"
    record.write_text(
        json.dumps(_execution_record(task_id="imported-task", workspace_root=str(workspace))),
        encoding="utf-8",
    )
    authority = tmp_path / "authority.json"
    authority.write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "trusted_workspace_root": str(tmp_path),
                "isolation_level": "workspace",
                "workspaces": [
                    {
                        "task_id": "imported-task",
                        "trial": 1,
                        "workspace_root": str(workspace),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"

    exit_code = main(
        [
            "evaluate",
            str(record),
            "--dataset",
            str(dataset),
            "--workspace-authority",
            str(authority),
            "--output-dir",
            str(output),
        ]
    )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["runs"][0]["passed"] is True
    criteria = report["runs"][0]["layers"]["deterministic"]["criterion_checks"]
    assert [item["criterion_id"] for item in criteria] == [
        "result_exists",
        "result_contains_pass",
    ]


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
