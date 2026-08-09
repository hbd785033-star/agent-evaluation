"""Command-line entry point for real agent evaluation runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import CommandAgentAdapter, RecordedAdapter
from .dataset import load_dataset
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
    run.add_argument("--dataset", required=True)
    run.add_argument("--output-dir", default="reports")
    run.add_argument("--trials", type=int)
    run.add_argument("--model", required=True)
    run.add_argument("--provider", required=True)
    run.add_argument("--harness", default="command")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--records", help="re-score exported RunRecord JSON")
    source.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="harness command; reads task JSON on stdin and writes RunRecord JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    version, tasks = load_dataset(args.dataset)
    if args.records:
        adapter = RecordedAdapter(_load_records(args.records))
    else:
        if not args.command:
            raise SystemExit("--command requires at least one argument")
        adapter = CommandAgentAdapter(
            args.command,
            model=args.model,
            provider=args.provider,
            harness=args.harness,
        )
    report = EvalRunner(adapter).run(tasks, trials_override=args.trials)
    report["dataset_version"] = version
    json_path, md_path = write_report(report, args.output_dir)
    print(json.dumps({
        "runs": report["run_count"],
        "report_json": str(json_path),
        "report_md": str(md_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
