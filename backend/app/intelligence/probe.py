"""Real-Qwen acceptance probe with synthetic evidence and separate source checks."""
import argparse
import html
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.config import settings
from app.paper_market import Bar, EASTERN, UTC
from app.paper_trader import PaperTrader
from .controller import IntelligenceController
from .sources import DEFAULT_SOURCES
from .store import dump, stamp


def run(directory, network=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = settings.model_copy(update={"INTELLIGENCE_BENCHMARKS": ""})
    intel = IntelligenceController(directory/"fixture.sqlite3", config)
    start = datetime(2025,3,3,9,30,tzinfo=EASTERN)
    bars = [Bar("DEMO", start+timedelta(minutes=5*(i+1)), 10+i*.01, 10+i*.01+.03,
                10+i*.01-.03, 10+i*.01+.002, 100000+i*1000) for i in range(35)]
    now = bars[-1].end
    intel.store.security("DEMO", {"company": "DEMO Widgets", "industry": "Manufacturing"}, now)
    intel.store.security("METAL", {"company": "METAL Works", "industry": "Materials"}, now)
    intel.observe(bars, now)
    text = ("SYNTHETIC ACCEPTANCE FIXTURE, not real company news. "
            "DEMO Widgets purchases steel from METAL Works. "
            "DEMO Widgets announced a 20 percent increase in signed widget orders. "
            "Steel input costs have also increased, so higher orders alone do not establish higher profits.")
    intel.store.archive(url="https://example.org/synthetic-demo-orders", publisher="Synthetic fixture",
        title="DEMO Widgets synthetic order and supplier scenario", text=text,
        published=now-timedelta(hours=1), acquired=now, language="en")
    result = intel.research.step(now)
    if not result or not result["findings"] or not result["hypotheses"]:
        raise ValueError("Real Qwen did not produce an evidence-grounded measurable hypothesis.")
    paper = PaperTrader(directory/"paper.sqlite3", intelligence=intel, challenge=False, initial=1000, mode="replay")
    paper.tick(bars, now)
    report = paper.report()
    if not report["latest_decisions"]:
        raise ValueError("Real Qwen paper decision failed validation; inspect events and prompts.")
    with intel.store.connect() as db:
        forecast = db.execute("SELECT * FROM market_forecasts LIMIT 1").fetchone()
    observed = datetime.fromisoformat(forecast["target"])+timedelta(minutes=5)
    price = forecast["base_price"]*1.01
    intel.store.observe("DEMO", "5m", observed, dict(open=price,high=price,low=price,close=price,volume=100000), observed)
    intel.evaluate(observed)
    output = {"status": "completed", "model": settings.OLLAMA_MODEL,
        "data": "Synthetic acceptance fixture; no live prices or real company news in model input.",
        "research": result, "paper": report, "intelligence": intel.overview(observed)}
    if network:
        live = IntelligenceController(directory/"live-connectivity.sqlite3", config)
        health = []
        current = datetime.now(UTC)
        for ident, url, interval, kind in DEFAULT_SOURCES:
            if kind == "listing":
                continue
            try:
                live.acquisition.acquire({"source": ident, "kind": kind,
                    "payload": dump({"url": url})}, stamp(current))
                health.append({"source": ident, "status": "reachable_and_parsed"})
            except Exception as exc:
                health.append({"source": ident, "status": "unavailable_or_configuration_pending", "error": str(exc)[:400]})
        output["source_connectivity"] = health
    (directory/"result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary = ("# Market intelligence acceptance probe\n\n"
               "Real local Qwen research and a validated paper decision using synthetic evidence.\n"
               "This does not establish trading profitability or the daily objective.\n\n"
               "## Research\n\n<pre>"+html.escape(json.dumps(result,indent=2))+"</pre>\n\n"
               "## Paper decision\n\n<pre>"+html.escape(json.dumps(report["latest_decisions"],indent=2))+"</pre>\n")
    (directory/"summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    if network:
        print(json.dumps(output["source_connectivity"], indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description="Real-Qwen intelligence acceptance probe, never real orders.")
    parser.add_argument("--output", default="intelligence-probe")
    parser.add_argument("--network", action="store_true")
    args = parser.parse_args()
    run(args.output, args.network)


if __name__ == "__main__":
    main()
