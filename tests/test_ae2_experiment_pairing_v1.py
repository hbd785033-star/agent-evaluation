"""AE-2 V1 experiment identity, ingestion, and pair-status tests."""

from __future__ import annotations

import copy

import pytest

from agent_eval.experiment_v1 import (
    ComparabilityStatus,
    ExperimentContractError,
    build_experiment_pair_report,
    compare_experiment_pair,
    parse_aao_experiment_record,
)


def _sha(label: str) -> str:
    return f"sha256:{label:0<64}"[:71]


def _namespace(
    *,
    arm: str = "A",
    task_hash: str = "task-a",
    starting_revision: str = "a" * 40,
    budget_id: str = "budget-a",
    completeness: str = "complete",
    effective_profile_id: str | None = "effective-a",
    observed_runtime: str | None = "runtime-a",
    runtime_run_id: str | None = "runtime-run-a",
    submission_attempted: bool = True,
    verification: str = "pass",
):
    return {
        "contract_version": "1.0",
        "experiment": {
            "experiment_id": "experiment-1",
            "experiment_definition_revision": "revision-1",
            "experiment_definition_sha256": _sha("experiment"),
            "comparison_kind": "runtime_comparison",
            "pair_id": "pair-1",
            "trial_id": 1,
            "arm_id": arm,
        },
        "task": {
            "task_definition_id": "task-definition-1",
            "task_definition_revision": "task-revision-1",
            "task_contract_sha256": _sha(task_hash),
            "prompt_sha256": _sha("prompt"),
            "success_criteria_sha256": _sha("criteria"),
            "dataset_or_fixture_revision": "fixture-1",
        },
        "configured_profile": {
            "runtime": observed_runtime,
            "harness": "adaptive-agent-orchestrator",
            "runtime_version": None,
            "model": None,
            "provider": None,
            "execution_mode": "direct",
            "tools_config_sha256": _sha("tools"),
            "policy_config_sha256": _sha("policy"),
            "reasoning_config_sha256": None,
            "environment_config_sha256": _sha("environment"),
            "workspace_contract": {
                "starting_revision": starting_revision,
                "isolation_mode": "workspace",
                "fixture_revision": "fixture-1",
            },
            "network_policy_identity": None,
            "sandbox_policy_identity": None,
            "approval_policy_identity": "approval-v1",
            "budget_id": budget_id,
        },
        "observed_profile": {
            "runtime": observed_runtime,
            "runtime_version": None,
            "model": None,
            "provider": None,
            "effective_workspace_revision": starting_revision if observed_runtime else None,
            "effective_workspace_root": "C:/Temp/non-semantic-root",
            "observed_isolation_level": "workspace" if observed_runtime else None,
            "observed_network_mode": None,
            "observed_sandbox_mode": None,
            "observed_approval_behavior": None,
            "tool_evidence_completeness": "complete" if observed_runtime else None,
            "file_evidence_completeness": "complete" if observed_runtime else None,
        },
        "profile_identity": {
            "configured_profile_id": "configured-a" if observed_runtime else None,
            "effective_profile_id": effective_profile_id,
            "completeness": completeness,
            "incompleteness_reasons": []
            if completeness == "complete"
            else ["observed_profile.runtime"],
        },
        "budget": {
            "configured_budget": {
                "max_children": 2,
                "max_depth": 1,
                "max_retries": 1,
                "max_total_calls": 8,
                "require_approval_above_calls": 5,
                "budget_id": budget_id,
            },
            "enforced_budget": {
                "calls_used": 1 if submission_attempted else 0,
                "calls_reserved": 0,
                "retries_used": 0,
                "children_used": 0,
                "depth_used": 0,
                "submission_prevented": not submission_attempted,
                "retry_prevented": False,
                "approval_required": False,
                "approval_granted": None,
            },
            "observed_usage": None,
            "estimated_cost": {
                "amount_usd": None,
                "provenance": None,
                "price_model_identity": None,
            },
            "billed_cost": {"amount_usd": None, "provenance": None},
        },
        "execution_lifecycle": {
            "submission_attempted": submission_attempted,
            "runtime_adapter_invoked": submission_attempted,
            "runtime_run_id": runtime_run_id,
            "terminal_status": "completed" if submission_attempted else None,
            "failure_phase": None if submission_attempted else "pre_submission",
            "failure_reason": None if submission_attempted else "approval denied",
        },
        "verification_status": verification,
    }


def _record(**kwargs):
    ns = _namespace(**kwargs)
    return {
        "schema_version": "0.1",
        "task_id": f"execution-{kwargs.get('arm', 'A')}",
        "run_id": ns["execution_lifecycle"]["runtime_run_id"],
        "model": None,
        "provider": None,
        "harness": "adaptive-agent-orchestrator",
        "status": "completed" if kwargs.get("submission_attempted", True) else "failed",
        "started_at": "2026-08-25T00:00:00Z",
        "finished_at": "2026-08-25T00:00:01Z",
        "latency_seconds": 1.0,
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
        "cost_usd": None,
        "tool_calls": None,
        "files_changed": None,
        "output": None,
        "workspace_root": None,
        "isolation_level": None,
        "metadata": {
            "trial": 1,
            "verification_status": ns.pop("verification_status"),
            "aao_experiment_v1": ns,
        },
    }


def test_same_trial_ab_arms_import_and_are_comparable():
    arm_a = parse_aao_experiment_record(_record(arm="A"))
    arm_b = parse_aao_experiment_record(_record(arm="B", observed_runtime="runtime-b"))

    result = compare_experiment_pair([arm_a, arm_b])

    assert arm_a.logical_key == ("experiment-1", "pair-1", 1, "A")
    assert arm_b.logical_key == ("experiment-1", "pair-1", 1, "B")
    assert result.status is ComparabilityStatus.COMPARABLE
    assert result.aao_verification == {"A": "pass", "B": "pass"}
    assert result.ae_judge is None


def test_public_comparability_status_set_has_exactly_four_states():
    assert {status.name for status in ComparabilityStatus} == {
        "COMPARABLE",
        "INCOMPARABLE",
        "INCOMPLETE",
        "INVALID",
    }
    assert {status.value for status in ComparabilityStatus} == {
        "comparable",
        "incomparable",
        "incomplete",
        "invalid",
    }
    assert not hasattr(ComparabilityStatus, "MISSING_COUNTERPART")


def test_missing_counterpart_is_incomplete_without_legacy_status():
    arm_a = parse_aao_experiment_record(_record(arm="A"))

    result = compare_experiment_pair([arm_a])

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert result.status.value != "missing_counterpart"
    assert result.reasons == ("exactly two arms are required",)


def test_material_profile_evidence_missing_is_incomplete():
    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(
                _record(
                    arm="B",
                    completeness="incomplete",
                    effective_profile_id=None,
                    observed_runtime=None,
                    runtime_run_id=None,
                    submission_attempted=False,
                )
            ),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert "effective profile is incomplete" in result.reasons


def test_required_unknown_evidence_is_incomplete_not_matched_or_mismatched():
    raw_a = _record(arm="A")
    raw_b = _record(arm="B", observed_runtime="runtime-b")
    raw_a["metadata"]["aao_experiment_v1"]["configured_profile"]["workspace_contract"][
        "starting_revision"
    ] = None
    raw_b["metadata"]["aao_experiment_v1"]["configured_profile"]["workspace_contract"][
        "starting_revision"
    ] = None

    result = compare_experiment_pair(
        [parse_aao_experiment_record(raw_a), parse_aao_experiment_record(raw_b)]
    )

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert any("workspace" in reason and "unavailable" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "field",
    ("tool_evidence_completeness", "file_evidence_completeness"),
)
def test_complete_profile_identity_cannot_mask_unknown_required_evidence(field):
    raw_b = _record(arm="B", observed_runtime="runtime-b")
    namespace = raw_b["metadata"]["aao_experiment_v1"]
    namespace["observed_profile"][field] = "unknown"

    assert namespace["profile_identity"]["completeness"] == "complete"

    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(raw_b),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert any(
        f"observed_profile.{field}" in reason and "unavailable" in reason
        for reason in result.reasons
    )


def test_unknown_required_evidence_beats_known_control_mismatch():
    raw_b = _record(arm="B", task_hash="task-b", observed_runtime="runtime-b")
    raw_b["metadata"]["aao_experiment_v1"]["observed_profile"][
        "tool_evidence_completeness"
    ] = "unknown"

    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(raw_b),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert "observed_profile.tool_evidence_completeness" in " ".join(result.reasons)
    assert "task_contract_sha256" in " ".join(result.reasons)


def test_invalid_duplicate_beats_unknown_required_evidence():
    raw = _record(arm="A")
    raw["metadata"]["aao_experiment_v1"]["observed_profile"][
        "tool_evidence_completeness"
    ] = "unknown"

    result = compare_experiment_pair(
        [parse_aao_experiment_record(raw), parse_aao_experiment_record(raw)]
    )

    assert result.status is ComparabilityStatus.INVALID
    assert any("duplicate" in reason for reason in result.reasons)
    assert "observed_profile.tool_evidence_completeness" in " ".join(result.reasons)


def test_duplicate_same_v1_arm_is_invalid():
    arm_a = parse_aao_experiment_record(_record(arm="A"))
    duplicate = parse_aao_experiment_record(_record(arm="A"))

    result = compare_experiment_pair([arm_a, duplicate])

    assert result.status is ComparabilityStatus.INVALID
    assert any("duplicate" in reason for reason in result.reasons)


def test_pair_identity_mismatch_is_invalid():
    raw_b = _record(arm="B", observed_runtime="runtime-b")
    raw_b["metadata"]["aao_experiment_v1"]["experiment"]["pair_id"] = "pair-2"

    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(raw_b),
        ]
    )

    assert result.status is ComparabilityStatus.INVALID
    assert "identity mismatch" in " ".join(result.reasons)


def test_invalid_duplicate_beats_incomplete_profile():
    arm_a = parse_aao_experiment_record(
        _record(
            arm="A",
            completeness="incomplete",
            effective_profile_id=None,
            observed_runtime=None,
            runtime_run_id=None,
            submission_attempted=False,
        )
    )
    duplicate = parse_aao_experiment_record(
        _record(
            arm="A",
            completeness="incomplete",
            effective_profile_id=None,
            observed_runtime=None,
            runtime_run_id=None,
            submission_attempted=False,
        )
    )

    result = compare_experiment_pair([arm_a, duplicate])

    assert result.status is ComparabilityStatus.INVALID
    assert any("duplicate" in reason for reason in result.reasons)
    assert "effective profile is incomplete" in result.reasons


def test_task_hash_mismatch_is_incomparable():
    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(_record(arm="B", task_hash="task-b")),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPARABLE
    assert "task_contract_sha256" in " ".join(result.reasons)


def test_incomplete_profile_beats_complete_control_mismatch():
    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(
                _record(
                    arm="B",
                    task_hash="task-b",
                    completeness="incomplete",
                    effective_profile_id=None,
                    observed_runtime=None,
                    runtime_run_id=None,
                    submission_attempted=False,
                )
            ),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPLETE
    assert "effective profile is incomplete" in result.reasons
    assert "task_contract_sha256" in " ".join(result.reasons)


def test_workspace_revision_mismatch_is_incomparable():
    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(
                _record(arm="B", starting_revision="b" * 40, observed_runtime="runtime-b")
            ),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPARABLE
    assert "workspace" in " ".join(result.reasons)


def test_required_budget_mismatch_is_incomparable():
    result = compare_experiment_pair(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(_record(arm="B", budget_id="budget-b")),
        ]
    )

    assert result.status is ComparabilityStatus.INCOMPARABLE
    assert "budget" in " ".join(result.reasons)


def test_malformed_present_namespace_is_invalid_not_legacy():
    raw = _record(arm="A")
    del raw["metadata"]["aao_experiment_v1"]["task"]

    with pytest.raises(ExperimentContractError, match="task"):
        parse_aao_experiment_record(raw)


def test_legacy_record_without_namespace_returns_no_v1_record():
    raw = _record(arm="A")
    del raw["metadata"]["aao_experiment_v1"]

    assert parse_aao_experiment_record(raw) is None


def test_input_is_not_mutated_during_parsing():
    raw = _record(arm="A")
    original = copy.deepcopy(raw)

    parse_aao_experiment_record(raw)

    assert raw == original


def test_pair_report_is_deterministic_and_does_not_judge(tmp_path):
    report = build_experiment_pair_report(
        [
            parse_aao_experiment_record(_record(arm="B", observed_runtime="runtime-b")),
            parse_aao_experiment_record(_record(arm="A")),
        ]
    )

    assert report["contract"] == "aao_experiment_v1"
    assert report["schema_version"] == "0.1"
    assert report["comparability"] == "comparable"
    assert report["logical_keys"] == [
        ["experiment-1", "pair-1", 1, "A"],
        ["experiment-1", "pair-1", 1, "B"],
    ]
    assert report["aao_verification"] == {"A": "pass", "B": "pass"}
    assert report["ae_judgment"] is None
    assert "estimated_cost" in report["evidence_layers"]
    assert "billed_cost" in report["evidence_layers"]
    assert report == build_experiment_pair_report(
        [
            parse_aao_experiment_record(_record(arm="A")),
            parse_aao_experiment_record(_record(arm="B", observed_runtime="runtime-b")),
        ]
    )
