# Governed Agentic Reproduction Workflow

A bounded multi-agent workflow that **reproduces** a known quantitative research
result under **structural governance**. Six single-purpose agents drive a
deterministic research pipeline through whitelisted tools; a seventh reviews the
result. No agent makes portfolio decisions, and no agent *can* place a trade —
the capability does not exist in any agent's toolset.

This is the **Level 2** ("research replication") system in the three-level
agentic taxonomy:

| Level | Name | What agents do | Status |
| --- | --- | --- | --- |
| 1 | Reporting | Describe a result a pipeline already produced | superseded |
| **2** | **Reproduction** | **Orchestrate the pipeline to produce the result, verify fidelity** | **this module** |
| 3 | Discovery | Propose new features / models / strategies | out of scope |

The thesis it demonstrates: *a governed multi-agent system can reproduce a
human-built quantitative research pipeline to within a stated tolerance, with
every stage bounded, audited, and reproducible.* It is **not** "agents beat
humans" or "agents decide allocations" — neither is claimed or supported.

---

## What it reproduces

The Notebook 03 ML-ensemble strategy: `MSR(Ensemble_mu_hat)` — Lasso, Random
Forest, and XGBoost expected-return forecasters, equal-weight ensembled, wrapped
in a long-only maximum-Sharpe optimizer, backtested out-of-sample.

- **Golden OOS Sharpe:** 2.579 (2023-01-01 → 2026, 29-asset universe)
- **Reproduced:** 2.5795 (absolute deviation 0.0005, within tolerance)
- The decomposed agent pipeline reproduces the monolithic engine **bit-for-bit**.

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │  Orchestrator (deterministic sequencing)  │
                         └─────────────────────────────────────────┘
                                          │
   Data ──▶ Feature ──▶ Model ──▶ Portfolio ──▶ Backtest ──▶ Risk ──▶ Review
  (load)  (features)  (Lasso/RF  (long-only    (lagged      (metrics) (governed
          + target)   /XGB →      MSR weights)  OOS returns)           memo)
                      ensemble)
     │         │          │            │             │           │        │
     └─────────┴──────────┴── each stage persists an auditable artifact ──┘
                              │
                    Fidelity check (vs golden) ──▶ Governance gate ──▶ Human review
```

**Six pipeline agents + one review agent**, each owning exactly one whitelisted
tool that wraps one deterministic engine function:

| Agent | Tool | Wraps | Writes |
| --- | --- | --- | --- |
| Data | `tool_load_data` | cache validation | — |
| Feature | `tool_compute_features` | feature panel + 21d target | `features.parquet`, `target.parquet` |
| Model | `tool_fit_predict` | fit Lasso/RF/XGB → ensemble | `predictions.parquet` |
| Portfolio | `tool_build_weights` | long-only MSR | `weights.parquet` |
| Backtest | `tool_backtest` | lagged-weight OOS returns | `strategy_returns.parquet` |
| Risk | `tool_evaluate` | metrics + diagnostics | `metrics.json`, `run_manifest.json`, `report.md` |
| Review | `tool_load_metrics_summary` | bounded read-only summary | `review_<reviewer>.md` |

### Governance is structural, not prompt-based

An agent's power equals its tool whitelist, enforced in code: calling a tool
outside the whitelist raises `PermissionError`. There is no trade tool, no
network-fetch tool, and no hyperparameter-mutation tool anywhere in the system —
so no agent can do those things, regardless of what any prompt says. This is a
stronger guarantee than asking an LLM to behave.

The agents do **not** reason about the pipeline: each pipeline stage is a Python
function call, fully deterministic. The **only** generative step is the Review
agent, which receives a bounded numeric summary (never raw data) and writes prose
— and it cannot alter any number, because it has no tool that mutates anything.

---

## The reviewer seam (cross-provider comparison)

The only swappable component is the Review agent, selectable at run time:

| `--reviewer` | Cost | Output |
| --- | --- | --- |
| `none` | zero tokens | deterministic template memo (control) |
| `anthropic` | Opus | LLM-narrated governed memo |
| `openai` | GPT | LLM-narrated governed memo |

All three receive *identical* bounded evidence and produce a governed memo that
must pass the same governance gate. Comparing the `anthropic` and `openai` memos
shows how two model families narrate identical evidence under identical
constraints — mirroring the project's BL-views experiment.

---

## Usage

```bash
# zero-token deterministic control
python scripts/run_agentic_reproduction.py --reviewer none

# LLM reviewers (need ANTHROPIC_API_KEY / OPENAI_API_KEY in .env)
python scripts/run_agentic_reproduction.py --reviewer anthropic
python scripts/run_agentic_reproduction.py --reviewer openai

# offline LLM validation (canned memo, no network)
python scripts/run_agentic_reproduction.py --reviewer anthropic --mock

# plain log output (CI / piped / non-TTY)
python scripts/run_agentic_reproduction.py --reviewer none --plain
```

The default run renders a **live terminal display**: each agent panel shows its
tool, action, and a bounded stat as it works, followed by the fidelity verdict
and the governed memo rendered in the terminal — the deliverable on screen.

### Outputs (per reviewer, under `results/agentic_reproduction/<reviewer>/`)

- `review_<reviewer>.md` — the governed review memo (the deliverable)
- `fidelity_<reviewer>.json` — reproduced vs golden Sharpe, deviation, verdict
- `run_log_<reviewer>.json` — per-agent tool-call audit trail
- `metrics.json`, `run_manifest.json`, `report.md`, and final parquets
- `stages/` — per-stage intermediate artifacts (regenerable; git-ignored)

---

## Why frozen cached data (a design choice, not a limitation)

The Data agent validates and loads a **pinned local cache**; it does not fetch
live data. This is deliberate and required: bit-for-bit reproduction of a known
result is only possible against fixed inputs. Live data changes every call, which
would make the fidelity check meaningless and introduce a network capability that
weakens the governance guarantee.

**Live-data "fresh run" mode is a documented future direction**, not a gap. The
decomposition makes it a clean extension: add a `tool_fetch_eodhd(...)` that pulls
from the data provider, **freezes the result to a timestamped snapshot**, and
returns the same `DataHandle` the rest of the pipeline already consumes. The six
downstream stages need no changes. A `--mode {reproduce, fresh}` switch would keep
the golden-fidelity check for reproduction and skip it for fresh runs (reporting
the new metrics instead). This belongs in its own branch with its own governance
review of the network capability.

---

## Tests

```bash
PYTHONPATH=src pytest tests/research_agents/test_agentic_reproduction.py -v
```

15 tests cover: each agent's single-tool whitelist; that no agent has a
trade/fetch/mutate tool; that calling outside a whitelist raises; per-stage
artifact persistence; the artifact contract; fidelity within tolerance; the
governance-phrase gate; the LLM reviewer seam; the streaming entry point; and an
opt-in live reproduction against the real 29-asset caches.

---

## Design notes

- **Deterministic core, observable execution.** `run_workflow` returns a result;
  `stream_workflow` yields per-stage events for the live display. Both drain one
  shared `_iter_workflow` generator, so they cannot drift.
- **Handles carry paths, not frames.** Each stage persists its output and passes
  a typed path-handle forward; agents never hold raw data in memory between
  stages, and the review agent only ever sees bounded summaries.
- **Framework-free.** No agent framework dependency; the "agent" is a bounded
  role object with a tool whitelist. Orchestration is deterministic Python.
```
