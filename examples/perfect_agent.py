"""Controlled harness used only by the reproducible smoke demo."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

payload = json.load(sys.stdin)
workspace = Path(payload["workspace"])
(workspace / "result.txt").write_text("PASS\n", encoding="utf-8")
print(
    json.dumps(
        {
            "output": "PERFECT_AGENT_EVIDENCE: result.txt contains PASS",
            "files_changed": ["result.txt"],
            "tool_calls": [],
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "exit_status": "completed",
            "run_id": f"controlled-{uuid.uuid4().hex}",
        }
    )
)
