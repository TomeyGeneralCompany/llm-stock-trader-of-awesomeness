# Interface Guide

The dashboard, analysis prompts, logs, comments, and documentation use English. Dates and numbers in the interface use the `en-US` locale.

## Backtest Setup

Open `http://localhost:3000` after starting the application.

| Control | Meaning |
| --- | --- |
| Stock symbol | The ticker to analyze, such as `AAPL`. |
| Backtest period | Historical period: six months, one year, two years, or five years. |
| Capital mode | The existing application uses a large fixed starting balance, described as unlimited-capital mode in the interface. |
| Start streaming backtest | Starts the historical replay and displays incoming decisions. |
| Stop | Closes the browser's progress stream. |

## Live Progress and Performance

During a run, the dashboard displays processing progress, completed trades, strategy win rate, cumulative realized P&L, and cumulative trade return.

When position data is available, the trading-state panel also shows unrealized P&L, the current trade's return, shares held, and risk level.

## Live Decision Logs

The log panel displays model decisions, their confidence and reasoning, generated buy or sell signals, and progress messages. Its colors distinguish buy, sell, hold, performance, and completion messages.

The model is instructed to write its reasoning and expected outcomes in English. Structured action values remain `BUY`, `SELL`, and `HOLD`.

## Backtest Results and Charts

The results section displays total trades, win rate, total return, maximum drawdown, and available realized-P&L statistics. The data overview lists price observations, trading signals, and model decisions.

The price chart displays candlesticks, volume, trade markers, and available decision markers. Chart labels and legends are in English.

## Daily Analysis and Strategy Discussion

Select an available trading date to inspect technical events and the model's decision reasoning. Event names, severity levels, and date labels are in English.

In the strategy discussion room, enter feedback about the selected day's decision and choose **Start discussion**. The assistant provides an assessment and proposed strategy-file changes in English. **Discuss again** clears that discussion so you can submit new feedback.

## Original Upstream Materials

The original uploaded `llm-agent-trader-master.zip` is retained as an unchanged source archive. It includes the original upstream screenshots. The active documentation uses this English guide in place of those Chinese-language screenshots.
