# Controller-Owned Stage Gates

The controller owns this fixed sequence. Gates advance automatically after validating and saving actual output. Model text cannot skip a stage or declare success, and elapsed time never authorizes a transition.

| Stage | Work | Evidence required to advance |
| --- | --- | --- |
| ACQUIRE | Download the selected market dataset. | Named OHLCV fields, positive finite prices, nonnegative volume, consistent OHLC relationships, and unique chronological timestamps. The streaming backtest requires at least 30 observations. |
| PREPARE | Calculate technical indicators. | Source timestamps, prices, and volume are preserved, and required indicator columns exist. Warm-up indicator values may be empty. |
| ANALYZE | Apply the strategy and ask Qwen for event-triggered decisions. | Completed row processing, signals within the source period, supported actions, valid confidence, and complete explanations and risk assessments. Trend analysis reads the acquired snapshot. |
| SIMULATE | Replay signals and settle simulated positions. | Full-period ledger coverage, valid executed trades, nonnegative cash and holdings, and reconciled portfolio values. |
| REVIEW | Calculate and check existing performance statistics. | Finite metrics, final value and return matching the ledger, correct executed-trade count, and valid win-rate and drawdown ranges. |
| REPORT | Format reviewed results for the dashboard. | Published trades, signals, performance, and strategy statistics match the reviewed evidence. |

A failed gate records its error and blocks every later stage. REVIEW checks consistency; it does not certify profitability or validate the strategy's economic assumptions. Existing trading rules and accounting formulas remain in the strategy and replay engine.

## Resource and Operation Ownership

One heavy stage runs at a time within this application installation. A reentrant thread lock and a host file lock serialize competing jobs and on-demand AI tools, including backend processes sharing the same lock path. Numerical thread pools default to four threads; Ollama has its separately configured inference-thread budget.

Pipeline Qwen calls are allowed only in ANALYZE. Market-data downloads are allowed only in ACQUIRE. The trend analyzer uses a frozen copy of the acquired dataset instead of fetching additional prices during analysis. The separate feedback and retrospective tools can use Qwen outside a pipeline and share the heavy-work slot.

## Saved Evidence and Run Ownership

Stage status and full handoff artifacts are stored in `data/stage-gates.sqlite3` under the project root. Each completed artifact has a SHA-256 digest. The controller verifies predecessor integrity before starting the next stage. Status and artifact reads also check integrity.

One run identifier ties together gate records, streaming events, reports, and analysis logs. New runs preserve previous runs' logs. Exact run lookup takes precedence over compatibility lookup for older upstream logs.

The dashboard displays pending, waiting, running, completed, failed, blocked, and cancelled states. **Refresh saved status** reads persisted state and can recover a finished report after a browser disconnect. A retry starts a new run; saved artifacts are an audit record rather than automatic execution-resume checkpoints.

## Stopping

**Stop after stage** requests a controlled stop. The active stage finishes and saves validated evidence; later stages are blocked. A queued stage checks for a stop request again before beginning work. Browser disconnection leaves the backend run available for status inspection.

## Configuration

- `PIPELINE_DB_PATH`: optional absolute path to the evidence database.
- `PIPELINE_LOCK_PATH`: optional shared host lock path; default `data/stage-work.lock` under the project root. Backend processes sharing an installation must use the same path.
- `PIPELINE_NUM_THREADS`: numerical thread budget, default `4`.
- `OLLAMA_NUM_THREAD`: inference thread budget, default `4`.

## Inspection API

Routes are under `/api/v1/llm-stream`:

| Route | Purpose |
| --- | --- |
| `GET /runs/{run_id}` | Read verified status, gate summaries, and hashes. |
| `GET /runs/{run_id}/artifacts/{stage}` | Read a handoff after checking its hash. |
| `POST /runs/{run_id}/stop` | Request a stop at a stage boundary. |

`stage` is `ACQUIRE`, `PREPARE`, `ANALYZE`, `SIMULATE`, `REVIEW`, or `REPORT`. Altered evidence produces an error rather than a successful report.
