# Autonomous paper trader

This is an independent forward paper-trading controller using the same local Ollama Qwen3 model and shared semantic decision validator as the historical backtester. It does not connect to any broker or place orders. The default daily performance objective is **$1 to $100 within one US regular trading session**. Every new challenge session starts at $1; previous days remain in the ledger and inform future decisions. The goal is recorded only when the account is flat and the realized balance reaches $100. Marked equity and incomplete sessions are reported separately. Performance is measured, never assumed.

## Run from the repository root

```bash
uv sync --locked
ollama pull qwen3:4b
PYTHONPATH=backend uv run python -m app.paper_trader run
# One scan/inference pass:
PYTHONPATH=backend uv run python -m app.paper_trader once
# Read current portfolio, session results and event counts:
PYTHONPATH=backend uv run python -m app.paper_trader status
# Ordinary persistent paper account (use a separate database):
PYTHONPATH=backend uv run python -m app.paper_trader run --normal --capital 10000 --db data/paper-normal.sqlite3
```

The scanner refreshes the Nasdaq Trader public exchange listings daily, cycles through bounded batches, and retains open positions, pending orders and recent plans in its refresh set. It ranks fresh candidates by observed momentum, volatility, relative volume, and compares alternatives in the model prompt. `--symbols SPY,AAPL` restricts the universe for reproducible testing; `--batch`, `--candidates`, and `--poll` control local resource consumption. A scan is rotating, not an instantaneous full exchange quote feed. Data provider delays, throttling, outages and CPU inference latency affect coverage.

The local controller uses completed five-minute Yahoo OHLCV bars, not executable quotes. It persists observations, validated model decisions and controller facts, queued orders, simulated fills, session valuations, forecast outcomes, and analytical errors in SQLite. Conditional numeric watches initiate fresh analysis; prose plans are retained for reevaluation. No conditional plan alone executes a trade. An immediate validated decision queues an order only while its data is fresh and before the session closes. The next eligible bar opening is the simulated execution opportunity, with modeled spread, slippage, fee, fractional-share rounding and a volume cap. No same-bar, retrospective, or zero-volume fill. The model chooses allocations within available simulated cash or shares; multiple pending buys reserve cash. It receives prior forecast errors, direction accuracy grouped by market phase, and recent realized trade performance across sessions. This is persistent episodic learning, not model-weight retraining.

The default daily challenge uses immediate reuse of paper proceeds to support intraday compounding. This is a hypothetical buying-power assumption, not a claim about brokerage permissions; select `--settled-cash` to delay reuse until the next observed trading session. Ordinary portfolio mode defaults to settled cash. Configure `--spread-bps`, `--slippage-bps`, `--fee`, and `--min-notional`; defaults are 10 bps spread, 5 bps slippage each side, zero fixed fee, one-cent simulated minimum, and millionth-share granularity. The regular US exchange holiday/early-close calendar handles common scheduled dates; exceptional exchange closures and provider gaps require verified data and incomplete results are not credited. `--close HH:MM` overrides the planned session cutoff. Do not reuse a database with a different configuration.

## Repeatable replay

```bash
PYTHONPATH=backend uv run python -m app.paper_trader replay --csv unseen.csv --db data/paper-replay.sqlite3
```

CSV columns: symbol,end,open,high,low,close,volume. `end` is the ISO-8601 timestamp of a completed five-minute bar, with timezone. Supply preceding market history. By default the first session is warmup only; use `--warmup-sessions N` to specify how many preceding sessions are loaded without decisions or fills. Each inference receives only observations available at that replay time; outcomes are scored only after their horizon has elapsed. A separate replay database prevents contamination of the forward account. Use an untouched file of held-out sessions and new ledger per evaluation; do not tune on the evaluation file. Replay advances on recorded bar times; forward mode uses wall-clock time for submission and rejects late decisions. Neither a fixture nor historical replay establishes live performance.

## Continuous operation

Run the command under a local service supervisor. A user systemd service installer is provided:

```bash
bash scripts/install-paper-service.sh
systemctl --user status paper-trader.service
journalctl --user -u paper-trader.service -f
systemctl --user stop paper-trader.service
```

The writer is serialized, SQLite commits each state transition and event together, and it resumes from saved state after restart. The service retries data/analysis errors without inventing trades. The existing web dashboard also exposes read-only paper status, using the configured PAPER_DB_PATH. For detailed records, query the SQLite tables `bars`, `decisions`, `outcomes`, `events`, `sessions`, `state`, and `config`. The system uses paper-only accounting and does not include Robinhood integration.


## Continuous intelligence

The same service runs a separate intelligence archive/research controller outside market hours and after the urgent portfolio pass during regular sessions. It preserves this controller's session calendar, model validation, fills, cash rules, challenge objective and database schema. Configure INTELLIGENCE_DB_PATH independently; see [market-intelligence controls and provenance](market-intelligence.md). INTELLIGENCE_ENABLED=false explicitly disables acquisition/research integration without changing the trading account.


## Forward timing and quotes

The forward service polls the held symbols and pending orders before discovery. It can ingest completed bars and account for pending fills while a single Qwen worker is occupied. The worker selects from the latest eligible snapshots at dispatch. Its responses are logged with queue and inference timing and are checked against current positions, cash, quote age and the regular session before submission. The feed warms a stock with five days of history and refreshes familiar stocks from the latest persisted bar with a one-bar overlap. Database keys deduplicate overlaps. Background research shares the same inference slot and runs when there is no eligible paper decision.

The default feed provides completed five-minute bars. A local bid/ask producer can optionally supply a daily, append-only JSON Lines file via --quotes. Each record contains symbol, time, received (ISO-8601 timestamps with time zones), bid, ask, bid_size, ask_size and optional fractional_eligible (true/false). Record arrival at ingestion; do not backdate receipt. With an eligible quoted fill, displayed size and instrument eligibility constrain quantity. Unknown eligibility uses whole-share sizing for quotes. The bar fallback retains the disclosed simulated fractional-share assumption and marks the fill as completed_bar_open. The file bridge consumes observations only; it has no market-data subscription or order API. For forecasts, the reference is the first completed bar at or after decision availability, followed by the prediction horizon. A late response cannot earn credit on the move between the input bar and that reference.

## Compare memory and score a fixed trial

Run a held-out comparison from the repository root using a fresh output directory:

    PYTHONPATH=backend uv run --locked python -m app.paper_compare held-out.csv data/comparison --warmup 1

Use --archive to supply a point-in-time archive. Each policy gets its own copy with the original availability times. Results include net marked P&L, realized trade P&L, mean absolute forecast error, decision time, and a hindsight diagnostic counting cost-adjusted positive next-bar moves with no matched paper buy. The latter is a description of the feed, not a number of tradeable signals established in advance.

When the laptop has passed a forward shakeout, choose the first day in advance and freeze the five consecutive exchange sessions. Use the same optional --quotes path for the service and the trial.

    PYTHONPATH=backend uv run --locked python -m app.paper_trial freeze data/five-session-trial.json --first-day YYYY-MM-DD
    PYTHONPATH=backend uv run --locked python -m app.paper_trial report data/five-session-trial.json

The report matches the account execution profile and each session's recorded commit, Ollama model digest, scan settings and feed. Missing and incomplete sessions remain in the five-day sequence. A session passes only when complete, flat and at or above the goal in realized paper proceeds after costs. Model and runtime identity are recorded when the service starts trading each day. Use the same scanner options for the service and manifest when overriding defaults.

## Private remote view

Preview the same allowlisted data that would be sent:

    PYTHONPATH=backend uv run --locked python -m app.paper_status_publish --preview --trial data/five-session-trial.json

The trial argument is optional. For remote viewing, configure a separate private GitHub repository and expose a repository-scoped Contents write token to the publisher as PAPER_STATUS_GITHUB_TOKEN. Run the publisher with --repo OWNER/REPO (and --branch if the destination uses a branch other than main). The destination can be viewed as paper-trader-status.json on GitHub. Add --interval 300 to refresh it every five minutes while the publisher command is running. The service-check timestamp distinguishes an active trader from a separate publisher that is still running. The publisher verifies private visibility before writing and omits raw prompts, system paths and account credentials. Publishing is separate from the trading service.
