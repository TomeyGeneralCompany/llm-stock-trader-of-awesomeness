"""Compare three paper policies on identical held-out observations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from app.config import settings
from app.intelligence.controller import IntelligenceController
from app.paper_market import market_close, read_replay
from app.paper_trader import LocalDecision, PaperTrader, replay


def momentum_baseline(ctx, _):
    change=ctx["technical_facts"]["return_6_bars"]
    held=ctx["shares"]>0
    action="BUY" if change>.005 and not held else "SELL" if change<-.005 and held else "HOLD"
    timeframe=ctx["observation_timeframe"]
    evidence=["technical_facts.return_6_bars","current.close"]
    reason=f"Observed six-bar return is {change:.4%}; the baseline threshold is 0.5%."
    return {"action":action,"confidence":.5,"risk_level":"medium","reasoning":reason,
            "expected_outcome":"The next completed bar measures continuation or reversal.",
            "allocation_fraction":0. if action=="HOLD" else .25 if action=="BUY" else 1.,
            "expected_return":change/6,"horizon_bars":1,
            "watch_conditions":[{"key":"technical_facts.return_6_bars","op":"<=" if held else ">=",
                                 "threshold":-.005 if held else .005,"required_position":"long" if held else "flat"}],
            "explanation":{
                "observed_state":{"timeframe":timeframe,"position":"long" if held else "flat",
                    "market_phase":ctx["trend"]["market_phase"],"macd_cross":ctx["macd_cross"],
                    "macd_alignment":ctx["macd_alignment"],"evidence":[reason]},
                "interpretation":reason,
                "forecast":{"scenario":"Recent momentum may persist.","timeframe":"Next completed bar",
                            "assumptions":["Recent market conditions persist."],"invalidation":"Momentum reverses."},
                "conditional_actions":[{"action":"HOLD","trigger":"Next observation","timeframe":"Next completed bar",
                    "required_position":"any","prerequisite":"Receive an updated market observation.",
                    "reasoning":"Recheck momentum and current holdings."}],
                "risk_assessment":{"evidence":evidence,"rationale":"Recent momentum can fade.","downside":"Price can move adversely."},
                "execution":{"mode":"wait" if action=="HOLD" else "now","evidence":evidence,"rationale":reason}}}


def outcomes(engine, report, seconds, bars, warmup):
    with engine.connect() as db:
        scored=[json.loads(row[0]) for row in db.execute("SELECT body FROM outcomes")]
        fills=[json.loads(row[0]) for row in db.execute("SELECT body FROM events WHERE kind='fill'")]
        entries = {(symbol, stamp) for symbol,stamp in db.execute("""SELECT d.symbol,d.bar FROM decisions d
            JOIN events e ON d.id=json_extract(e.body,'$.decision_id') WHERE e.kind='fill'
            AND json_extract(e.body,'$.action')='BUY'""")}
    histories=defaultdict(list)
    days=sorted({bar.day for bar in bars})[warmup:]
    for b in bars:
        if b.day in days and market_close(b.day):
            histories[b.symbol].append(b)
    # Hindsight diagnostic: next-bar gross move that would have survived modelled
    # spread/slippage and fixed fees at the starting paper balance.
    missed=0
    for symbol, history in histories.items():
        for before, after in zip(history,history[1:]):
            if before.day!=after.day or after.start!=before.end or min(before.volume,after.volume)<=0:
                continue
            costed=(engine.initial-engine.fee)*after.close*(1-engine.cost)/(after.open*(1+engine.cost))-engine.fee
            if costed>engine.initial and (symbol,before.end.isoformat()) not in entries:
                missed+=1
    return {"net_marked_pnl":sum(row["equity"]-row["start"] for row in report["sessions"]),
            "realized_trade_pnl":sum(row["realized_pnl"] for row in fills if row["realized_pnl"] is not None),
            "mean_absolute_forecast_error":sum(abs(row["error"]) for row in scored)/len(scored) if scored else None,
            "forecasts_scored":len(scored),"mean_decision_seconds":sum(seconds)/len(seconds) if seconds else None,
            "unfilled_positive_next_bar_moves":missed,"session_reports":report["sessions"]}


def compare(bars, directory, *, warmup=1, archive=None, local=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    model=local or LocalDecision()
    results={}
    for label, memory, policy in (("qwen_memory",True,model),("qwen_no_memory",False,model),
                                 ("deterministic",False,momentum_baseline)):
        seconds=[]
        def decide(context, lessons):
            start=perf_counter()
            try:
                return policy(context,lessons)
            finally:
                seconds.append(perf_counter()-start)
        intel_path=directory/(label+".intel.sqlite3")
        if archive:
            uri=Path(archive).resolve().as_uri()+"?mode=ro"
            from contextlib import closing
            with closing(sqlite3.connect(uri,uri=True)) as source, closing(sqlite3.connect(intel_path)) as target:
                source.backup(target)
        intel=IntelligenceController(intel_path,settings)
        engine=PaperTrader(directory/(label+".sqlite3"),decide=decide,mode="replay",
                           intelligence=intel,use_memory=memory)
        results[label]=outcomes(engine,replay(engine,bars,warmup),seconds,bars,warmup)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("output")
    parser.add_argument("--archive",help="Copy a local archive for each comparison. Original availability times still apply.")
    parser.add_argument("--warmup",type=int,default=1)
    args=parser.parse_args()
    result=compare(read_replay(args.csv),args.output,warmup=args.warmup,archive=args.archive)
    encoded=json.dumps(result,indent=2)
    (Path(args.output)/"comparison.json").write_text(encoded+"\n")
    print(encoded)


if __name__=="__main__":
    main()
