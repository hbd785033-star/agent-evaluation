"""Strict AE consumer boundary for AAO ``aao_experiment_v1`` metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ExperimentContractError(ValueError):
    """The present V1 namespace is malformed or violates a truth invariant."""


class ComparabilityStatus(StrEnum):
    COMPARABLE = "comparable"
    INCOMPARABLE = "incomparable"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


_REQUIRED_SECTIONS = (
    "experiment",
    "task",
    "configured_profile",
    "observed_profile",
    "profile_identity",
    "budget",
    "execution_lifecycle",
)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{name} must be an object")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentContractError(f"{name} must be a non-empty string")
    return value


def _required_keys(value: dict[str, Any], name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ExperimentContractError(f"{name} missing required fields: {missing}")


def _evidence_unavailable(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() == "unknown"
    if isinstance(value, dict):
        return any(_evidence_unavailable(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_evidence_unavailable(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class ExperimentRecordV1:
    """Validated AAO V1 metadata plus the enclosing ExecutionRecord evidence."""

    raw: dict[str, Any]
    namespace: dict[str, Any]
    aao_verification: str | None

    @property
    def experiment(self) -> dict[str, Any]:
        return self.namespace["experiment"]

    @property
    def task(self) -> dict[str, Any]:
        return self.namespace["task"]

    @property
    def configured_profile(self) -> dict[str, Any]:
        return self.namespace["configured_profile"]

    @property
    def observed_profile(self) -> dict[str, Any]:
        return self.namespace["observed_profile"]

    @property
    def profile_identity(self) -> dict[str, Any]:
        return self.namespace["profile_identity"]

    @property
    def budget(self) -> dict[str, Any]:
        return self.namespace["budget"]

    @property
    def lifecycle(self) -> dict[str, Any]:
        return self.namespace["execution_lifecycle"]

    @property
    def logical_key(self) -> tuple[str, str, int, str]:
        experiment = self.experiment
        return (
            experiment["experiment_id"],
            experiment["pair_id"],
            experiment["trial_id"],
            experiment["arm_id"],
        )

    @property
    def ae_judge(self) -> None:
        """AAO verification is provenance; AE judgment is produced separately."""
        return None

    @classmethod
    def from_mapping(cls, raw: Any) -> ExperimentRecordV1:
        record = _mapping(raw, "ExecutionRecord")
        if record.get("schema_version") != "0.1":
            raise ExperimentContractError("ExecutionRecord schema_version must be 0.1")
        metadata = _mapping(record.get("metadata"), "ExecutionRecord.metadata")
        if "aao_experiment_v1" not in metadata:
            raise ExperimentContractError("aao_experiment_v1 namespace is absent")
        namespace = _mapping(metadata["aao_experiment_v1"], "aao_experiment_v1")
        if namespace.get("contract_version") != "1.0":
            raise ExperimentContractError("aao_experiment_v1 contract_version must be 1.0")
        _required_keys(namespace, "aao_experiment_v1", _REQUIRED_SECTIONS)

        experiment = _mapping(namespace["experiment"], "experiment")
        _required_keys(
            experiment,
            "experiment",
            (
                "experiment_id",
                "experiment_definition_revision",
                "experiment_definition_sha256",
                "comparison_kind",
                "pair_id",
                "trial_id",
                "arm_id",
            ),
        )
        for name in (
            "experiment_id",
            "experiment_definition_revision",
            "experiment_definition_sha256",
            "comparison_kind",
            "pair_id",
            "arm_id",
        ):
            _nonempty(experiment[name], f"experiment.{name}")
        trial_id = experiment["trial_id"]
        if not isinstance(trial_id, int) or isinstance(trial_id, bool) or trial_id < 1:
            raise ExperimentContractError("experiment.trial_id must be a positive integer")

        task = _mapping(namespace["task"], "task")
        _required_keys(
            task,
            "task",
            (
                "task_definition_id",
                "task_definition_revision",
                "task_contract_sha256",
                "prompt_sha256",
                "success_criteria_sha256",
                "dataset_or_fixture_revision",
            ),
        )
        for name in (
            "task_definition_id",
            "task_definition_revision",
            "task_contract_sha256",
            "prompt_sha256",
            "success_criteria_sha256",
            "dataset_or_fixture_revision",
        ):
            _nonempty(task[name], f"task.{name}")

        configured = _mapping(namespace["configured_profile"], "configured_profile")
        observed = _mapping(namespace["observed_profile"], "observed_profile")
        for section_name, section in (
            ("configured_profile", configured),
            ("observed_profile", observed),
        ):
            for name in ("runtime", "runtime_version", "model", "provider"):
                if section.get(name) is not None and not isinstance(section[name], str):
                    raise ExperimentContractError(f"{section_name}.{name} must be a string or null")
        workspace = _mapping(configured.get("workspace_contract"), "workspace_contract")
        _required_keys(
            workspace,
            "workspace_contract",
            ("starting_revision", "isolation_mode", "fixture_revision"),
        )

        identity = _mapping(namespace["profile_identity"], "profile_identity")
        _required_keys(
            identity,
            "profile_identity",
            (
                "configured_profile_id",
                "effective_profile_id",
                "completeness",
                "incompleteness_reasons",
            ),
        )
        if identity["completeness"] not in {"complete", "incomplete"}:
            raise ExperimentContractError("profile_identity.completeness is invalid")
        reasons = identity["incompleteness_reasons"]
        if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
            raise ExperimentContractError(
                "profile_identity.incompleteness_reasons must be a list of strings"
            )
        if identity["completeness"] == "incomplete":
            if identity["effective_profile_id"] is not None or not reasons:
                raise ExperimentContractError(
                    "incomplete profile must have null effective_profile_id and reasons"
                )
        elif (
            not isinstance(identity["effective_profile_id"], str)
            or not identity["effective_profile_id"].strip()
        ):
            raise ExperimentContractError("complete profile must have effective_profile_id")

        budget = _mapping(namespace["budget"], "budget")
        for name in ("configured_budget", "enforced_budget"):
            _mapping(budget.get(name), f"budget.{name}")
        _required_keys(
            budget,
            "budget",
            (
                "configured_budget",
                "enforced_budget",
                "observed_usage",
                "estimated_cost",
                "billed_cost",
            ),
        )
        _mapping(budget["estimated_cost"], "budget.estimated_cost")
        _mapping(budget["billed_cost"], "budget.billed_cost")

        lifecycle = _mapping(namespace["execution_lifecycle"], "execution_lifecycle")
        _required_keys(
            lifecycle,
            "execution_lifecycle",
            ("submission_attempted", "runtime_adapter_invoked", "runtime_run_id"),
        )
        if not isinstance(lifecycle["submission_attempted"], bool) or not isinstance(
            lifecycle["runtime_adapter_invoked"], bool
        ):
            raise ExperimentContractError("execution lifecycle booleans are required")
        if lifecycle["runtime_run_id"] is not None:
            _nonempty(lifecycle["runtime_run_id"], "execution_lifecycle.runtime_run_id")
        if lifecycle["submission_attempted"] is False:
            if lifecycle["runtime_run_id"] is not None:
                raise ExperimentContractError("pre-submission failure cannot have runtime_run_id")
            if budget["observed_usage"] is not None:
                raise ExperimentContractError("pre-submission failure cannot have observed usage")
            for name in ("estimated_cost", "billed_cost"):
                if budget[name].get("amount_usd") is not None:
                    raise ExperimentContractError(f"pre-submission failure cannot have {name}")

        verification = metadata.get("verification_status")
        if verification is not None and not isinstance(verification, str):
            raise ExperimentContractError("metadata.verification_status must be a string or null")
        return cls(record, namespace, verification)


def parse_aao_experiment_record(raw: Any) -> ExperimentRecordV1 | None:
    """Parse the additive namespace; absent metadata remains legacy-compatible."""
    if not isinstance(raw, dict):
        raise ExperimentContractError("ExecutionRecord must be an object")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or "aao_experiment_v1" not in metadata:
        return None
    return ExperimentRecordV1.from_mapping(raw)


@dataclass(frozen=True, slots=True)
class PairComparisonV1:
    status: ComparabilityStatus
    records: tuple[ExperimentRecordV1, ...]
    reasons: tuple[str, ...]
    aao_verification: dict[str, str | None]
    ae_judge: None = None


def _same(records: tuple[ExperimentRecordV1, ...], label: str, getter) -> str | None:
    values = [getter(record) for record in records]
    encoded = [json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values]
    if len(set(encoded)) > 1:
        return f"{label} mismatch: {encoded!r}"
    return None


def compare_experiment_pair(
    records: list[ExperimentRecordV1] | tuple[ExperimentRecordV1, ...],
) -> PairComparisonV1:
    rows = tuple(records)
    if len(rows) < 2:
        return PairComparisonV1(
            ComparabilityStatus.INCOMPLETE,
            rows,
            ("exactly two arms are required",),
            {},
            None,
        )
    if len(rows) > 2:
        return PairComparisonV1(
            ComparabilityStatus.INVALID,
            rows,
            ("exactly two arms are required",),
            {},
            None,
        )

    keys = [row.logical_key for row in rows]
    invalid_reasons: list[str] = []
    incomplete_reasons: list[str] = []
    incomparable_reasons: list[str] = []
    if len(set(keys)) != len(keys):
        invalid_reasons.append(f"duplicate logical key: {keys[0]!r}")
    if (
        len({row.experiment["experiment_id"] for row in rows}) > 1
        or len({row.experiment["pair_id"] for row in rows}) > 1
        or len({row.experiment["trial_id"] for row in rows}) > 1
    ):
        invalid_reasons.append("experiment/pair/trial identity mismatch")
    if any(row.profile_identity["completeness"] != "complete" for row in rows):
        incomplete_reasons.append("effective profile is incomplete")

    for label, getter in (
        ("experiment definition", lambda row: row.experiment["experiment_definition_sha256"]),
        ("task_contract_sha256", lambda row: row.task["task_contract_sha256"]),
        ("prompt_sha256", lambda row: row.task["prompt_sha256"]),
        ("success_criteria_sha256", lambda row: row.task["success_criteria_sha256"]),
        (
            "workspace",
            lambda row: (
                row.configured_profile["workspace_contract"]["starting_revision"],
                row.configured_profile["workspace_contract"]["fixture_revision"],
            ),
        ),
        ("budget", lambda row: row.budget["configured_budget"]),
    ):
        values = tuple(getter(row) for row in rows)
        if any(_evidence_unavailable(value) for value in values):
            incomplete_reasons.append(f"{label} evidence unavailable")
            continue
        mismatch = _same(rows, label, getter)
        if mismatch:
            incomparable_reasons.append(mismatch)

    status = (
        ComparabilityStatus.INVALID
        if invalid_reasons
        else ComparabilityStatus.INCOMPLETE
        if incomplete_reasons
        else ComparabilityStatus.INCOMPARABLE
        if incomparable_reasons
        else ComparabilityStatus.COMPARABLE
    )
    reasons = (*invalid_reasons, *incomplete_reasons, *incomparable_reasons)
    verification = {row.experiment["arm_id"]: row.aao_verification for row in rows}
    return PairComparisonV1(status, rows, reasons, verification, None)


def build_experiment_pair_report(
    records: list[ExperimentRecordV1] | tuple[ExperimentRecordV1, ...],
) -> dict[str, Any]:
    """Build a deterministic mechanical report; AE judgment remains unpopulated."""
    ordered = tuple(sorted(records, key=lambda record: record.logical_key))
    comparison = compare_experiment_pair(ordered)
    return {
        "contract": "aao_experiment_v1",
        "schema_version": "0.1",
        "comparability": comparison.status.value,
        "logical_keys": [list(record.logical_key) for record in ordered],
        "reasons": list(comparison.reasons),
        "aao_verification": dict(sorted(comparison.aao_verification.items())),
        "ae_judgment": comparison.ae_judge,
        "evidence_layers": {
            "configured_profile": [
                record.profile_identity["configured_profile_id"] for record in ordered
            ],
            "effective_profile": [
                record.profile_identity["effective_profile_id"] for record in ordered
            ],
            "configured_budget": [record.budget["configured_budget"] for record in ordered],
            "enforced_budget": [record.budget["enforced_budget"] for record in ordered],
            "observed_usage": [record.budget["observed_usage"] for record in ordered],
            "estimated_cost": [record.budget["estimated_cost"] for record in ordered],
            "billed_cost": [record.budget["billed_cost"] for record in ordered],
        },
    }
