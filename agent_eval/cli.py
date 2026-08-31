"""Command-line entry point for real agent evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .adapters import CommandAgentAdapter, RecordedAdapter
from .checkers import CheckerProfile, TaskCheckRegistry
from .dataset import load_dataset
from .execution_record import ExecutionRecordAdapter, WorkspaceAuthority, load_execution_records
from .judges import controlled_profile_judge
from .models import RunRecord
from .runner import EvalRunner, write_report


def _load_records(path: str | Path) -> list[RunRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("runs", raw) if isinstance(raw, dict) else raw
    records: list[RunRecord] = []
    for row in rows:
        record_raw = row.get("record", row) if isinstance(row, dict) else row
        records.append(RunRecord.from_dict(record_raw))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-eval")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run", help="execute and evaluate a versioned dataset")
    run.add_argument("--dataset")
    run.add_argument("--output-dir", default="reports")
    run.add_argument("--trials", type=int)
    run.add_argument("--model", required=True)
    run.add_argument("--provider", required=True)
    run.add_argument("--harness", default="command")
    run.add_argument("--source-cwd", help="source tree copied into each isolated trial")
    run.add_argument("--workspace-root", default=".agent-eval-workspaces")
    run.add_argument("--preserve-workspaces", action="store_true")
    run.add_argument("--checker-profile")
    run.add_argument("--judge-profile")
    run.add_argument("--workspace-authority")
    run.add_argument("--config", help="evaluation YAML config; CLI values take precedence")
    run.add_argument(
        "--trusted-record-workspace-root",
        help="control-plane trust root for replayed record paths",
    )
    run.add_argument(
        "--record-isolation-level",
        choices=("none", "workspace", "os"),
        default="none",
        help="control-plane-verified isolation used to produce replayed records",
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", help="re-score exported RunRecord JSON")
    source.add_argument("--execution-record", help="evaluate strict ExecutionRecord 0.1 JSON")
    source.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="harness command; reads task JSON on stdin and writes RunRecord JSON",
    )
    evaluate = subparsers.add_parser("evaluate", help="evaluate ExecutionRecord 0.1 JSON")
    evaluate.add_argument("execution_record")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output-dir", default="reports")
    evaluate.add_argument("--trials", type=int)
    evaluate.add_argument("--checker-profile")
    evaluate.add_argument("--judge-profile")
    evaluate.add_argument("--workspace-authority")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "evaluate":
        version, tasks = load_dataset(args.dataset)
        authority = (
            WorkspaceAuthority.load(args.workspace_authority) if args.workspace_authority else None
        )
        adapter = ExecutionRecordAdapter(
            load_execution_records(args.execution_record), authority=authority
        )
        checker_profile = None
        checker_registry = TaskCheckRegistry()
        if args.checker_profile:
            checker_raw = yaml.safe_load(Path(args.checker_profile).read_text(encoding="utf-8"))
            checker_profile = CheckerProfile.from_mapping(checker_raw)
        judge = controlled_profile_judge(args.judge_profile) if args.judge_profile else None
        report = EvalRunner(
            adapter,
            judge=judge,
            checker_registry=checker_registry,
            checker_profile=checker_profile,
            dataset_version=version,
        ).run(tasks, trials_override=args.trials)
        json_path, md_path = write_report(report, args.output_dir)
        passed_runs = sum(1 for row in report["runs"] if row["passed"] is True)
        print(
            json.dumps(
                {
                    "runs": report["run_count"],
                    "passed_runs": passed_runs,
                    "all_passed": passed_runs == report["run_count"] and report["run_count"] > 0,
                    "report_json": str(json_path),
                    "report_md": str(md_path),
                }
            )
        )
        return 0 if report["run_count"] > 0 and passed_runs == report["run_count"] else 1
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise SystemExit("--config must contain a YAML object")
        for name in (
            "dataset",
            "checker_profile",
            "judge_profile",
            "trials",
            "workspace_authority",
        ):
            value = config.get(name)
            if value is not None and getattr(args, name, None) in (None, ""):
                setattr(args, name, value)
    if not args.dataset:
        raise SystemExit("run requires --dataset or config.dataset")
    version, tasks = load_dataset(args.dataset)
    if args.execution_record:
        if args.trusted_record_workspace_root or args.record_isolation_level != "none":
            raise SystemExit("ExecutionRecord workspace authority requires --workspace-authority")
        authority = (
            WorkspaceAuthority.load(args.workspace_authority) if args.workspace_authority else None
        )
        adapter = ExecutionRecordAdapter(
            load_execution_records(args.execution_record), authority=authority
        )
    elif args.records:
        adapter = RecordedAdapter(
            _load_records(args.records),
            model=args.model,
            provider=args.provider,
            harness=args.harness,
            workspace_root=args.trusted_record_workspace_root,
            isolation_level=args.record_isolation_level,
        )
    else:
        if not args.command:
            raise SystemExit("--command requires at least one argument")
        adapter = CommandAgentAdapter(
            args.command,
            model=args.model,
            provider=args.provider,
            harness=args.harness,
            cwd=args.source_cwd,
            workspace_root=args.workspace_root,
            preserve_workspaces=args.preserve_workspaces,
        )
    checker_profile = None
    checker_registry = TaskCheckRegistry()
    if args.checker_profile:
        checker_raw = yaml.safe_load(Path(args.checker_profile).read_text(encoding="utf-8"))
        checker_profile = CheckerProfile.from_mapping(checker_raw)
    judge = controlled_profile_judge(args.judge_profile) if args.judge_profile else None
    report = EvalRunner(
        adapter,
        judge=judge,
        checker_registry=checker_registry,
        checker_profile=checker_profile,
        dataset_version=version,
    ).run(tasks, trials_override=args.trials)
    json_path, md_path = write_report(report, args.output_dir)
    passed_runs = sum(1 for run in report["runs"] if run["passed"] is True)
    all_passed = report["run_count"] > 0 and passed_runs == report["run_count"]
    print(
        json.dumps(
            {
                "runs": report["run_count"],
                "passed_runs": passed_runs,
                "all_passed": all_passed,
                "report_json": str(json_path),
                "report_md": str(md_path),
            }
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
