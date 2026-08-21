# AE-1 PR #2 Continuation Checkpoint

## Status

Development status:

**AE-1B IMPLEMENTATION AND REAL VERTICAL SLICE COMPLETE — REVIEW / CI / MERGE PENDING**

AE-1 implementation:

**TECHNICALLY COMPLETE AT `a79865ab570fa7ce8df3c16dff710eebb982448f`; NOT YET MERGED**

Current next technical phase:

**independent review → exact-head CI → PR #2 merge gate**

This document is the canonical cross-computer continuation checkpoint. It is
tracked in GitHub so the next session needs no prior computer, local worktree,
Hermes state, terminal history, or chat transcript.

## Repository

Repository:

`hbd785033-star/agent-evaluation`

Base branch:

`main`

Base SHA:

`3d46a42484981494191699b4c2ecf6d3d0453109`

Working branch:

`feat/r2-closed-loop-hardening`

Pull request:

[#2](https://github.com/hbd785033-star/agent-evaluation/pull/2)

PR state at checkpoint start:

**OPEN / DRAFT**

Original PR #2 head before this checkpoint:

`0c03fb5a0cef04f429c420a7d923b721cc071286`

## Existing PR #2 capability

PR #2 already contains substantial Agent Evaluation R2 work, including:

- executable deterministic criteria;
- versioned checker registry/profile;
- calibrated Judge adapter seam;
- strict provider-neutral `ExecutionRecord` ingestion;
- `ExecutionRecordAdapter`;
- `WorkspaceAuthority`;
- AAO ExecutionRecord evaluation CLI;
- workspace-authority regression coverage;
- controlled deterministic Judge fixture;
- controlled subprocess smoke.

Future AE-1 work must not blindly create a second adapter or rebuild the
evaluation framework from scratch. First reconcile the existing PR #2 seam.

## External AAO producer dependency

AAO repository:

`hbd785033-star/adaptive-agent-orchestrator`

AAO baseline:

`794fc5a5317205405a917f501304f18d3e292b71`

AAO E2E-3A:

**CLOSED**

Authoritative producer contract:

`contracts/execution.py`

Current AAO ExecutionRecord schema:

`0.1`

The AAO E2E-3A merge commit is the durable cross-repository reference. The
old AAO local `.hermes` plan artifact is not required for continuation.

## Architecture

```text
HMC
    Think / Plan
        ↓
AAO
    Route / Govern
        ↓
Runtime
    Act / Execute
        ↓
AAO ExecutionRecord 0.1
        ↓
Agent Evaluation
    Normalize / Measure / Judge
        ↓
EvaluatedRun
```

## Core truth boundaries

```text
Claimed != Configured != Planned != Observed != Verified != Judged
UNKNOWN != ZERO
UNKNOWN != EMPTY
planned runtime != observed runtime
runtime identity != model identity
runtime identity != provider identity
AAO EvalResult PASS != AE EvaluatedRun PASS
workspace isolation != OS sandbox
allowed_paths != hard sandbox
no writes observed != writes impossible
```

## AE-1 audit conclusion

AE-1 integration audit status:

**AE-1 INTEGRATION PLAN READY**

AAO already owns:

`contracts/execution.py::ExecutionRecord`

The contract explicitly describes itself as provider-neutral AAO output
consumed by independent evaluators. The preferred direction is therefore:

```text
AAO
→ versioned ExecutionRecord evidence
→ existing AE-side ExecutionRecordAdapter
→ RunRecord
→ existing AE evaluator pipeline
```

Do not make AE permanently read AAO SQLite. Do not make AAO depend on AE
evaluator internals. Do not create a competing adapter before reconciling the
existing PR #2 adapter.

## Important producer / consumer mismatch

AAO `ExecutionRecord 0.1` permits unavailable observations to remain `null` /
`None`. Important nullable producer fields include:

- `run_id`;
- `model`;
- `provider`;
- `input_tokens`;
- `output_tokens`;
- `cached_tokens`;
- `cost_usd`;
- `tool_calls`;
- `files_changed`;
- `output`;
- `workspace_root`;
- `isolation_level`.

The current PR #2 consumer requires several observations to be concrete. The
producer and consumer are therefore not fully semantically compatible yet.
This is a future reconciliation item, not an implementation performed here.

## Unknown telemetry issue

Future reconciliation must preserve:

```text
not observed != observed zero
```

This applies especially to:

- `input_tokens`;
- `output_tokens`;
- `cached_tokens`;
- `cost_usd`;
- latency;
- retry count;
- sub-agent count.

Unknown cost must not become fake `$0`. Unknown token telemetry must not become
measured zero.

## Unknown collection evidence

Preserve these distinctions:

```text
tool_calls=None != tool_calls=[]
files_changed=None != files_changed=[]
trajectory unavailable != trajectory known empty
output=None != output=""
```

Empty evidence must not automatically imply clean execution.

## Experiment identity vs execution provenance

Future reconciliation must distinguish AE experiment/control-plane identity:

- `model`;
- `provider`;
- `harness`;
- `trial`;
- dataset version;

from AAO execution provenance:

- planned runtime;
- selected runtime;
- observed runtime;
- execution mode;
- runtime `run_id`;
- optional runtime `session_id`.

Do not guess:

```text
model = hermes
provider = hermes
```

Hermes is a runtime/harness identity, not necessarily the executed model or
provider. HMC recommendation is planning evidence, not observed execution
provenance.

## Planned vs observed

Preserve separately:

```text
planning.runtime_plan.executor
planning.runtime_selection.selected_runtime
observed.runtime_adapter
```

Never derive observed runtime from `RuntimePlan`. Missing observed runtime must
remain missing. A planned/observed mismatch must be preserved and fail closed
as a provenance problem rather than silently reconciled.

## AAO Verified vs AE Judged

AAO `EvalResult` is upstream verification evidence. AE owns independent
measurement and judgment.

AAO PASS must never auto-promote to:

- AE deterministic PASS;
- AE Judge PASS;
- AE overall PASS.

The AE adapter may retain AAO verification status as bounded upstream metadata,
but AE must still run its own applicable layers.

## Workspace / isolation

PR #2 `WorkspaceAuthority` is directionally useful and should be audited for
salvage rather than replaced blindly.

External ExecutionRecord claims are data, not authority. `workspace_root` and
`isolation_level` must remain control-plane-authorized.

AAO E2E-2 and E2E-3A workspace claims do not prove OS sandboxing. In
particular:

```text
allowed_paths != hard sandbox
filesystem_read capability != read-only sandbox
workspace targeted != path escape impossible
no writes observed != writes impossible
```

The first truthful AAO replay may therefore fail AE's OS-isolation criterion
when the source evidence proves only AAO workspace isolation.

## Trajectory evidence

Current AAO trajectory evidence is **PARTIAL**.

AAO telemetry can contain useful runtime events and some tool payloads, but:

- generic `AgentEvent` does not guarantee a complete tool trace;
- summary export may contain only event types;
- stable total ordering/completeness is not yet a portable contract;
- runtime-native subagent events may not represent every internal action;
- retries are represented only partially through task/child attempt evidence.

Future AE integration must not claim COMPLETE trajectory evidence merely because
no tool calls are present in an exported record. Preserve a bounded
`complete` / `partial` / `unavailable` provenance classification if the
existing PR #2 metadata conventions support it.

## Current preferred integration ownership

Preferred owner:

**AE downstream adapter**

PR #2 already contains `ExecutionRecordAdapter`. Future work must first
reconcile or salvage that adapter instead of creating a competing
`aao_adapter.py`, unless a later read-only audit proves replacement necessary.

The adapter must:

- parse strict versioned JSON;
- preserve nullable evidence;
- never select or execute a runtime;
- never call a model or Judge merely to ingest evidence;
- route records through AE canonicalization and authority checks;
- remain generic for Hermes and future Runtime-B/LangGraph producers.

## Current design decision

Previous AE-1 audit classified the primary gap as:

**AE RUNRECORD / CONSUMER SEMANTIC GAP**

Because PR #2 already implements a substantial `ExecutionRecordAdapter`, the
next action is not new implementation from `main`.

The next action is:

**PR #2 RECONCILIATION**

followed by a bounded decision:

- `SALVAGE`; or
- `SUPERSEDE`.

Current architectural expectation: **likely SALVAGE**, but this is not a
proven final decision. Do not record SALVAGE as completed until the
reconciliation audit establishes it.

## Known P0-class risks for future work

- AAO verification PASS treated as AE PASS;
- unknown telemetry becoming zero;
- unknown collections becoming empty;
- runtime identity guessed as model/provider;
- planned runtime treated as observed runtime;
- untrusted workspace claims gaining authority;
- workspace isolation promoted to OS sandbox;
- synthetic execution identity hiding missing real run IDs;
- unknown cost entering aggregates as zero.

## Known P1-class risks for future work

- partial trajectory appearing complete;
- timestamp-only event ordering treated as total ordering;
- exact model/provider execution provenance unavailable;
- estimated cost reported as billed cost;
- runtime-native subagents not fully represented;
- failure/cancel/timeout distinctions collapsed;
- `RecordedAdapter` duplicate `(task_id, trial)` keys overwritten before
  validation;
- existing PR #2 consumer and AAO nullable producer semantics disagree.

## Existing PR #2 review state

The historical PR body states that an earlier independent review found
bounded-but-attacker-selected sibling workspace substitution, and commit
`0c03fb5` replaced that design with exact `WorkspaceAuthority` bindings.

At this checkpoint there is no GitHub review/comment evidence proving that a
later fresh independent review is currently in progress. Do not claim a review
completed unless future GitHub evidence proves it.

No new reviewer is dispatched in this checkpoint phase.

## Existing PR #2 CI state before checkpoint

Historical exact PR head:

`0c03fb5a0cef04f429c420a7d923b721cc071286`

Historical workflow:

`Agent Evaluation Framework CI`

Historical result:

**SUCCESS**

This proves only the historical PR head. The documentation checkpoint creates
a new PR head, so historical CI is not exact-head CI for the checkpoint commit.

## Deferred work

- formal PR #2 reconciliation audit;
- SALVAGE vs SUPERSEDE decision;
- unknown != zero implementation;
- unknown collection semantics implementation;
- experiment identity / execution provenance design;
- exact model/provider provenance;
- trajectory completeness contract;
- bounded AE-1 implementation;
- AE-1 independent review;
- AE-1 merge;
- live production Judge calibration;
- real Hermes benchmark;
- Runtime-B;
- LangGraph;
- same-task A/B evaluation;
- real cross-runtime selection;
- AE → AAO feedback;
- learned routing;
- benchmark expansion;
- cross-runtime leaderboard.

## Exact next phase when development resumes

Start:

**AE-1 PR #2 RECONCILIATION / SALVAGE AUDIT**

Requirements:

- read-only first;
- compare AAO producer `ExecutionRecord 0.1` with the PR #2 consumer;
- resolve UNKNOWN != ZERO;
- resolve UNKNOWN != EMPTY;
- resolve experiment identity versus execution provenance;
- preserve `WorkspaceAuthority`;
- preserve Verified != Judged;
- decide SALVAGE versus SUPERSEDE;
- do not implement before the decision.

## Fresh-machine recovery

```bash
git clone https://github.com/hbd785033-star/agent-evaluation.git
cd agent-evaluation
git fetch origin
git switch --track origin/feat/r2-closed-loop-hardening
git rev-parse HEAD
git status --short
```

Then read:

`docs/handoffs/AE-1-PR2-CONTINUATION.md`

and GitHub PR #2.

Also inspect the referenced AAO producer contract at the immutable commit:

Repository:

`hbd785033-star/adaptive-agent-orchestrator`

Commit:

`794fc5a5317205405a917f501304f18d3e292b71`

Path:

`contracts/execution.py`

The old computer, old local worktree, local `.hermes` state and prior chat
conversation are not required.

## Checkpoint scope

This checkpoint is documentation-only. It changes no production code, tests,
datasets, evaluator logic, adapter logic, Judge logic, dependency, AAO
repository, HMC, Hermes, Kanban or LangGraph state.

The only intended repository change is:

`docs/handoffs/AE-1-PR2-CONTINUATION.md`

## Recovery invariant

```text
REMOTE_ONLY_RECOVERABLE = YES
FRESH_CLONE_RECOVERABLE = YES
CONTINUATION_CRITICAL_LOCAL_ONLY_FILES = 0
```

The handoff document itself is the durable continuation state.

## AE-1B Evidence Truth Repair Checkpoint

This section supersedes the earlier planning-only and paused-state instructions above while
preserving them as historical context.

### Current immutable producer

AAO producer truth was repaired and merged before this consumer phase:

- repository: `hbd785033-star/adaptive-agent-orchestrator`;
- branch: `master`;
- merge SHA: `5a1ab3c844b7c19e1c73e13bb07abd7fe375f1fb`;
- ExecutionRecord schema: `0.1` (unchanged).

### AE-1B implementation candidate

The bounded consumer repair is committed on the existing PR #2 branch:

- implementation commit: `a79865ab570fa7ce8df3c16dff710eebb982448f`;
- parent/checkpoint: `e2d46e8adb4b45ea842d00c9eed580291d4829e6`;
- branch: `feat/r2-closed-loop-hardening`.

The implementation preserves these boundaries:

```text
UNKNOWN != ZERO
UNKNOWN != EMPTY
experiment identity != observed execution provenance
planned runtime != selected runtime != observed runtime
AAO Verified != AE Judged
claimed workspace != trusted workspace authority
workspace isolation != OS sandbox
reported/producer-estimated cost != AE-derived estimate != billed cost
```

Specific behavior:

- producer-nullable scalars, collections, output, workspace claims and isolation claims remain
  nullable through ingestion, `RunRecord`, evaluation and JSON reporting;
- `run_id=None` remains incomplete execution identity and is never replaced by an AE UUID;
- duplicate checks apply to non-null real run IDs, while duplicate task/trial bindings still fail
  closed;
- AAO model/provider/harness observations are retained separately from explicit
  `execution-record` experiment labels;
- planned, selected and actually invoked runtime identities remain separate nullable fields;
- unavailable trajectory is reported as `unavailable`, `evaluated=false`, with no fake zero-step
  statistics;
- missing files/action evidence fails closed when policy requires it;
- source cost and its semantics remain unchanged; AE estimates are separate and missing-aware;
- aggregates expose total, cost-available and cost-missing sample counts and never insert fake
  zeroes into means;
- existing `WorkspaceAuthority` exact `(task_id, trial)` binding remains the sole workspace trust
  source;
- `RecordedAdapter` now rejects duplicate task/trial keys before dictionary construction.

### Deterministic verification

On the implementation snapshot:

- focused AE-1B Evidence Truth regressions after review repair: `30 passed`;
- focused/adjacent compatibility suite after review repair: `106 passed, 1 skipped`;
- full repository suite after review repair: `106 passed, 1 skipped`;
- repository-standard Ruff scope: PASS;
- `git diff --check`: PASS.

The full-repository Ruff invocation still reports pre-existing issues in `tests/test_agent.py`,
which is outside the repository-standard CI Ruff scope and unchanged by AE-1B.

### Real cross-repository vertical slice

The mandatory truth slice used the actual producer and consumer CLIs:

```text
AAO master@5a1ab3c
→ aao run "AE-1B deterministic evidence truth slice" --mock --record-out record.json
→ actual MockHermesAdapter / Orchestrator / _build_execution_record / export
→ agent-eval evaluate record.json --dataset dataset.yaml --output-dir reports
→ load_execution_records / ExecutionRecordAdapter / EvalRunner / write_report
```

Local artifacts:

- `C:/Users/EDY/AppData/Local/Temp/ae1b-vertical-slice-5a1ab3c/record.json`;
- `C:/Users/EDY/AppData/Local/Temp/ae1b-vertical-slice-5a1ab3c/reports-review-fix/report.json`;
- `C:/Users/EDY/AppData/Local/Temp/ae1b-vertical-slice-5a1ab3c/reports-review-fix/report.md`.

Mechanical truth assertions passed. The real record retained `cached_tokens=null`,
`tool_calls=null`, `output=null`, and observed `files_changed=[]`. AE retained those distinctions,
kept the real runtime run ID, separated mock/fixture observations from experiment labels, did not
promote workspace claims, reported trajectory unavailable, labelled cost
`producer_estimated`, and emitted both normal report artifacts. After the independent-review
repair, the evaluated run truthfully FAILS because output evidence is unavailable; no empty output
is invented for security evaluation. That EvaluatedRun FAIL is an expected Evidence Truth result,
not a vertical-slice failure.

### Review, CI and merge state at this checkpoint

- initial independent AE-1B review: COMPLETE / BLOCK (`P0=2`, `P1=2`);
- bounded review repair: IMPLEMENTED; same-reviewer re-verification PENDING;
- exact-head GitHub Actions for the final PR head: PENDING;
- PR #2 draft state: expected OPEN / DRAFT / UNMERGED;
- merge: NOT PERFORMED.

Do not reinterpret this candidate checkpoint as merged closure. A later external review/CI/merge
record must bind to the exact final PR head. The authoritative current state is recoverable from
PR #2 plus the immutable SHAs above.
