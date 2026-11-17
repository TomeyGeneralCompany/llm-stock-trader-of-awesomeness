"""Verify the paper trader with downloaded market observations and local Qwen."""
import argparse
from datetime import datetime, time, timedelta
import hashlib
import html
import json
from pathlib import Path

from app.config import settings
from app.paper_market import BAR_MINUTES, EASTERN, UTC, market_close, regular, yahoo_bars
from app.paper_trader import PaperTrader


def select_session(bars, symbol, received):
    """Use the most recent fully observed regular session, without price selection."""
    by_day = {}
    for bar in bars:
        if bar.symbol == symbol and bar.end <= received and regular(bar):
            by_day.setdefault(bar.day, {})[bar.end] = bar
    for day, records in sorted(by_day.items(), reverse=True):
        date = datetime.fromisoformat(day).date()
        if (received.astimezone(EASTERN).date()-date).days > 7:
            break
        opens = datetime.combine(date, time(9,30), EASTERN).astimezone(UTC)
        closes = datetime.combine(date, time(*market_close(day)), EASTERN).astimezone(UTC)
        if closes > received:
            continue
        cadence = timedelta(minutes=BAR_MINUTES)
        ends = [opens+cadence*index for index in range(1,(closes-opens)//cadence+1)]
        if len(ends) >= 48 and all(end in records for end in ends):
            return [records[end] for end in ends]
    raise RuntimeError(f"Yahoo Finance did not return a complete recent session for {symbol}. No market test ran.")


def write_summary(output, result):
    lines = ["# Qwen paper-trader check with observed market prices", ""]
    if result["status"] != "complete":
        lines += [f"**{result['status']}**", "", html.escape(result["error"])]
    else:
        decision = result["decision"]
        lines += [
            f"**Source:** Yahoo Finance via yfinance. **Symbol:** {result['symbol']}. **Session:** {result['session']} (Eastern).",
            f"Downloaded at {result['received_utc']}. Replayed {result['bars']} completed five-minute bars.",
            f"Qwen received {result['input_bars']} bars ending {result['as_of_utc']} and the $1 paper-account state. "
            "The remaining observations were withheld until after the decision.",
            "",
            f"**Qwen:** {decision['action']} · **Simulated fills:** {result['fills']} · **Forecasts scored:** {result['scored']}.",
            "This is historical replay on observed data. See the market-data artifact for the exact inputs and provenance.",
            "",
            "## Validated decision",
            "",
            "<pre>"+html.escape(json.dumps(decision, indent=2))+"</pre>",
        ]
    (output/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def check_market(output, symbol="SPY", *, fetch=yahoo_bars, clock=None, decide=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"paper.sqlite3").exists():
        raise ValueError("Choose a fresh output directory for each market check.")
    clock = clock or (lambda: datetime.now(UTC))
    phase = "provider_unavailable"
    try:
        requested = clock()
        downloaded = fetch([symbol], requested, period="5d")
        received = clock()
        session = select_session(downloaded, symbol, received)
        encoded = (json.dumps([bar.record() for bar in session], indent=2)+"\n").encode()
        (output/"market-bars.json").write_bytes(encoded)
        provenance = {
            "provider": "Yahoo Finance via yfinance", "symbol": symbol, "session": session[0].day,
            "received_utc": received.isoformat(), "interval": "5m", "period_requested": "5d",
            "bars": len(session), "first_bar_end": session[0].end.isoformat(),
            "last_bar_end": session[-1].end.isoformat(), "sha256": hashlib.sha256(encoded).hexdigest(),
        }
        (output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n", encoding="utf-8")
        phase = "model_or_paper_validation_failed"
        # Fixed choice before evaluating any price change. Subsequent bars cannot
        # enter a prompt, and fills use the same ledger as the normal paper trader.
        cutoff = 35
        as_of = session[cutoff-1].end
        paper = PaperTrader(output/"paper.sqlite3", decide=decide, mode="replay", max_candidates=1)
        paper.tick(session[:cutoff], as_of)
        decisions = paper.report()["latest_decisions"]
        if len(decisions) != 1:
            raise RuntimeError("Expected one validated Qwen decision: "+json.dumps(paper.report()["recent_events"]))
        for bar in session[cutoff:]:
            paper.tick([bar], bar.end, analyze=False)
        report = paper.report()
        if report["scored"] != 1 or not report["sessions"] or not report["sessions"][0]["complete"]:
            raise RuntimeError("Market replay did not produce a scored forecast and a complete session.")
        result = {**provenance, "status":"complete", "model":settings.OLLAMA_MODEL,
                  "as_of_utc":as_of.isoformat(), "input_bars":cutoff, "decision":decisions[0]["decision"],
                  "fills":report["events"].get("fill",0), "scored":report["scored"], "paper":report}
    except Exception as exc:
        result = {"status":phase,"error":str(exc)}
        write_summary(output,result)
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
        raise
    write_summary(output,result)
    (output/"result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="market-check")
    parser.add_argument("--symbol", default="SPY")
    args = parser.parse_args()
    try:
        check_market(args.output,args.symbol.upper())
    finally:
        summary = Path(args.output)/"summary.md"
        if summary.exists():
            print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
