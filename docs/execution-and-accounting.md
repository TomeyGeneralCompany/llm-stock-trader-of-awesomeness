# Historical execution and accounting

The historical backtester uses a finite controller-owned long-only account. Signals are recommendations; creating one never spends cash or changes holdings. The next decision receives a snapshot of the fills already executed.

The first 30 observations are calibration warmup. Calibration cannot inspect the later test period. At each later daily bar, queued decisions from the previous completed close execute at the current open, protective orders are evaluated, the account is marked at close, and only then does analysis see that completed prefix. Last-bar decisions are cancelled because no later execution observation exists.

ANALYZE retains its chronological execution evidence. SIMULATE replays sealed signals and settings with the same fill kernel and checks exact equality against the account that supplied decision context. These stages remain sequential and share the existing workload lock. The forward autonomous paper trader remains independent and retains its own documented execution assumptions.

## Controls

The dashboard and streaming API expose starting cash, maximum position fraction, stop-loss fraction, take-profit fraction, proportional commission, and minimum commission. Defaults are $10,000, 30%, 5%, 10%, 0.1% per fill, and $0 minimum. A protective fraction of zero disables that bracket. An optional engine share cap is supported; there is no default 1,000-share order or $100-million account.

Allocation and affordability are checked again at the actual next opening price, including commission. A gap cannot force a negative cash balance. Protective stops use the opening price when the bar gaps past the stop; otherwise the configured price. If a daily bar touches both stop and target, the adverse stop is evaluated first because OHLC bars do not reveal the intrabar path.

End-of-test liquidation is explicit and pays the configured commission. Python callers can set liquidate_at_end=False to retain marked holdings. Settlement is idempotent. Daily account marks, fills, queued/rejected/cancelled orders, and frozen execution settings are retained in stage artifacts.

## Metrics

Equity is cash plus marked stock value. Returns and drawdown include starting equity; volatility uses changes in equity, not percentage changes in cumulative returns. Annualized return uses the actual run capital and elapsed calendar time; volatility uses 252 daily observations per year. A one-observation period has no annualized-return estimate and reports zero.

Entry commissions enter weighted-average cost basis. Partial exits allocate that basis proportionally, and realized profit subtracts both entry and exit commissions. Winning exits are classified by net realized profit. num_trades counts executed fills; strategy_statistics.total_trades counts completed exit fills, including partial exits and final settlement. cumulative_trade_return_rate is net realized profit divided by starting capital.

Buy-and-hold remains a gross price-return reference and is not presented as a matched-cost executable portfolio.

## Validation

Regression tests use hand-calculated gap fills, allocation caps, minimum and proportional fees, partial sales, final settlement, zero holdings, late cancelled orders, chronological calibration, and known equity-return paths. Existing staged Qwen and forward-paper tests remain in the suite. No live brokerage connection is added.
