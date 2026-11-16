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

CSV columns: symbol,end,open,high,low,close,volume. `end` is the ISO-8601 timestamp of a completed five-minute bar, with timezone. Supply preceding market history for indicator warmup. Each inference receives only observations available at that replay time; outcomes are scored only after their horizon has elapsed. A separate replay database prevents contamination of the forward account. Use an untouched file of held-out sessions and new ledger per evaluation; do not tune on the evaluation file. Replay advances on recorded bar times; forward mode uses wall-clock time for submission and rejects late decisions. Neither a fixture nor historical replay establishes live performance.

## Continuous operation

Run the command under a local service supervisor. A user systemd service installer is provided:

```bash
bash scripts/install-paper-service.sh
systemctl --user status paper-trader.service
journalctl --user -u paper-trader.service -f
systemctl --user stop paper-trader.service
```

The writer is serialized, SQLite commits each state transition and event together, and it resumes from saved state after restart. The service retries data/analysis errors without inventing trades. For detailed records, query the SQLite tables `bars`, `decisions`, `outcomes`, `events`, `sessions`, `state`, and `config`. The system uses paper-only accounting and does not include Robinhood integration.
