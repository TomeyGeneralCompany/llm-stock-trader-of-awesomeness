"""Full acquisition/research/forecast/paper/learning path and chronology contracts."""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.intelligence.controller import IntelligenceController
from app.intelligence.sources import parse_feed, public_url
from app.intelligence.statistics import event_study, aggregate, score_forecast
from app.intelligence.store import IntelligenceStore, stamp, sha
from app.paper_market import UTC
from app.paper_trader import PaperTrader
from test_paper_trader import day_bars, next_bar, response

TEXT = "AAA buys steel from BBB. AAA reports a rise in signed widget orders. Increased input costs may offset the benefit."
WHEN = datetime(2025, 3, 3, 17, 25, tzinfo=UTC)


def config():
    return Settings(_env_file=None, INTELLIGENCE_BENCHMARKS="", INTELLIGENCE_NETWORK_JOBS=2,
                    INTELLIGENCE_FEEDS="[]", MEMORY_CHUNK_CHARS=256)


class ResearchModel:
    config = SimpleNamespace(num_ctx=8192, max_tokens=1800)
    def __init__(self, status="sufficient", corrupt=False):
        self.status, self.corrupt, self.prompts = status, corrupt, []
    def invoke(self, prompt):
        self.prompts.append(prompt)
        chunks = [json.loads(line) for line in prompt.splitlines() if line.startswith("{") and '"chunk_id"' in line]
        first = chunks[0]
        quote = "AAA buys steel from BBB."
        citation = {"chunk_id": "fabricated" if self.corrupt else first["chunk_id"], "quote": quote}
        value = {
            "status": self.status,
            "findings": [{"claim": "AAA has a direct input relationship with BBB.", "stance": "supports", "evidence": [citation]}],
            "relationships": [{"subject": "AAA", "subject_kind": "company", "object": "BBB", "object_kind": "company", "predicate": "purchases", "confidence": .8, "interpretation": "Documented purchasing exposure.", "evidence": [citation]}],
            "hypotheses": [{"event": "Signed orders increase", "classification": "demand", "affected_symbols": ["AAA"], "direction": 1, "expected_return": .02, "horizon_minutes": 5, "causal_chain": ["Orders may support revenue, partly offset by steel costs."], "confidence": .7, "assumptions": ["Orders are fulfilled."], "contradictions": [], "invalidation": "Order cancellations or costs overwhelm revenue.", "evidence": [citation]}],
            "followups": [{"question": "How much steel exposure?", "terms": "AAA BBB steel", "sources": ["gdelt"]}] if self.status=="continue" else [],
        }
        return SimpleNamespace(content=json.dumps(value))


def fixture(controller, now=WHEN):
    controller.store.security("AAA", {"company": "AAA Widgets", "industry": "Manufacturing"}, now)
    controller.store.security("BBB", {"company": "BBB Steel", "industry": "Materials"}, now)
    xml = ('<rss><channel><item><title>AAA signed widget orders increase</title>'
           '<link>https://example.org/aaa-orders</link>'
           '<pubDate>Mon, 03 Mar 2025 15:00:00 GMT</pubDate>'
           '<description>'+TEXT+'</description></item></channel></rss>').encode()
    controller.acquisition.fetch = lambda *args: xml
    controller.acquisition.acquire({"source": "bea", "kind": "feed",
        "payload": json.dumps({"url": "https://apps.bea.gov/rss/rss.xml"})}, stamp(now))


def test_end_to_end_acquisition_to_paper_fill_to_learned_retrieval(tmp_path):
    model = ResearchModel()
    intel = IntelligenceController(tmp_path/"intelligence.db", config(), client=model)
    bars = day_bars("AAA", n=35, growth=.005)
    now = bars[-1].end
    fixture(intel, now)
    intel.observe(bars, now)
    engine = PaperTrader(tmp_path/"paper.db", intelligence=intel, challenge=False, initial=1000,
                         decide=lambda ctx, lessons: response(ctx, "BUY", .5))
    before = engine.report()
    result = intel.research.step(now)
    assert result["hypotheses"] and result["relationships"]
    assert engine.report() == before  # research cannot mutate the portfolio
    boosted, context = intel.rank("AAA", .1, now)
    assert boosted > .1 and context["catalysts"]
    state = engine.tick(bars, now)
    assert "AAA" in state["pending"]
    later = next_bar(bars[-1], price=bars[-1].close*1.03)
    state = engine.tick([later], later.end)
    assert "AAA" in state["holdings"]
    assert intel.lessons("AAA", later.end)
    assert not intel.lessons("AAA", now-timedelta(seconds=1))
    overview = intel.overview(later.end)
    assert overview["counts"]["entity_relationships"] == 1
    assert overview["counts"]["forecast_outcomes"] == 1
    assert overview["accuracy"][0]["n"] == 1
    restored = IntelligenceController(tmp_path/"intelligence.db", config(), client=model)
    assert restored.lessons("AAA", later.end) == intel.lessons("AAA", later.end)


def test_versions_originals_offsets_and_future_revision_exclusion(tmp_path):
    store = IntelligenceStore(tmp_path/"archive.db")
    first = store.archive(url="https://example.org/story?utm_source=a", publisher="example.org",
        title="AAA event", text=TEXT, published=WHEN-timedelta(hours=1), acquired=WHEN, chunk_chars=64, overlap=8)
    duplicate = store.archive(url="https://example.org/story?utm_source=b", publisher="example.org",
        title="AAA event", text=TEXT, published=WHEN-timedelta(hours=1), acquired=WHEN, chunk_chars=64, overlap=8)
    assert first == duplicate
    second = store.archive(url="https://example.org/story", publisher="example.org", title="AAA event",
        text=TEXT+" Revised figures.", published=WHEN-timedelta(hours=1),
        acquired=WHEN+timedelta(hours=1), chunk_chars=64, overlap=8)
    old = store.retrieve("AAA", WHEN, 20)
    assert old and all(row["version"] == first for row in old)
    with store.connect() as db:
        versions = db.execute("SELECT id,original_text,hash FROM document_versions").fetchall()
        assert len(versions) == 2 and all(sha(row["original_text"]) == row["hash"] for row in versions)
    future = store.retrieve("AAA", WHEN+timedelta(hours=2), 20)
    assert {row["version"] for row in future} == {first, second}
    for chunk in old:
        assert TEXT[chunk["start"]:chunk["end"]] == chunk["body"]


def test_fabricated_references_are_rejected_and_unfinished_work_survives(tmp_path):
    intel = IntelligenceController(tmp_path/"archive.db", config(), client=ResearchModel(corrupt=True))
    fixture(intel)
    with pytest.raises(ValueError, match="citation"):
        intel.research.step(WHEN)
    with intel.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM causal_hypotheses").fetchone()[0] == 0
        assert db.execute("SELECT status FROM research_tasks LIMIT 1").fetchone()[0] == "queued"
        assert db.execute("SELECT status FROM research_runs ORDER BY rowid DESC LIMIT 1").fetchone()[0] == "rejected"


def test_followup_is_persisted_without_terminal_or_order_access(tmp_path):
    intel = IntelligenceController(tmp_path/"archive.db", config(), client=ResearchModel(status="continue"))
    fixture(intel)
    intel.research.step(WHEN)
    with intel.store.connect() as db:
        task = db.execute("SELECT * FROM research_tasks LIMIT 1").fetchone()
        assert task["status"] == "waiting_evidence" and task["steps"] == 1
        assert json.loads(task["dependencies"])[0]["sources"] == ["gdelt"]
        assert db.execute("SELECT COUNT(*) FROM acquisition_jobs WHERE kind='gdelt'").fetchone()[0] == 1


def test_weekend_acquisition_and_failure_isolation(tmp_path):
    saturday = datetime(2025,3,8,15,tzinfo=UTC)
    model = ResearchModel()
    intel = IntelligenceController(tmp_path/"archive.db", config(), client=model)
    def unavailable(*args):
        raise RuntimeError("provider unavailable")
    intel.acquisition.fetch = unavailable
    result = intel.cycle(saturday, research=False)
    assert result["status"] == "processed"
    with intel.store.connect() as db:
        rows = db.execute("SELECT status,error FROM acquisition_jobs").fetchall()
        assert rows and all(row["status"]=="queued" for row in rows)
        assert any(row["error"] for row in rows)
    assert not model.prompts


def test_unmatured_forecasts_are_not_fabricated(tmp_path):
    intel = IntelligenceController(tmp_path/"archive.db", config(), client=ResearchModel())
    fixture(intel)
    intel.store.observe("AAA", "5m", WHEN, dict(open=100,high=100,low=100,close=100,volume=100), WHEN)
    intel.research.step(WHEN)
    intel.evaluate(WHEN+timedelta(days=1))
    assert not intel.lessons("AAA", WHEN+timedelta(days=1))
    with intel.store.connect() as db:
        assert db.execute("SELECT status FROM market_forecasts").fetchone()[0] == "pending"


def test_future_acquisition_is_not_available_to_historical_replay(tmp_path):
    store = IntelligenceStore(tmp_path/"archive.db")
    store.observe("AAA", "5m", WHEN, dict(open=100,high=100,low=100,close=100,volume=100), WHEN+timedelta(days=1))
    assert not store.observations("AAA", WHEN)
    assert len(store.observations("AAA", WHEN+timedelta(days=1))) == 1


def test_event_study_uses_pre_event_fit_and_reports_samples():
    import math
    asset, benchmark = [], []
    ap, mp = 100., 100.
    for i in range(45):
        market_return = .001 + math.sin(i)*.002
        ap *= 1 + .0005 + 1.5*market_return + (.01 if i>=30 else 0)
        mp *= 1 + market_return
        end = stamp(WHEN+timedelta(minutes=i*5))
        asset.append(dict(end=end,close=ap,high=ap*1.01,low=ap*.99,volume=200 if i>=30 else 100))
        benchmark.append(dict(end=end,close=mp,volume=100))
    study = event_study(asset, benchmark, asset[30]["end"], pre=20, window=12)
    assert study["status"] == "complete" and study["sample_size"] == 20
    assert study["beta"] == pytest.approx(1.5)
    assert study["alpha"] == pytest.approx(.0005)
    assert study["cumulative_abnormal_return"] == pytest.approx(.12)
    assert study["abnormal_volume_peak"] == 2
    assert aggregate([study])["sample_size"] == 1
    assert aggregate([])["mean"] is None
    score = score_forecast(dict(expected_return=.05,direction=1,confidence=.8), .02)
    assert score["absolute_error"] == pytest.approx(.03)
    assert score["brier"] == pytest.approx(.04)


def test_untrusted_xml_and_private_network_are_rejected():
    with pytest.raises(ValueError, match="entities"):
        parse_feed(b'<!DOCTYPE rss [<!ENTITY x SYSTEM "file:///etc/passwd">]><rss/>', "https://example.org", WHEN)
    with pytest.raises(ValueError):
        public_url("https://127.0.0.1/news", {"127.0.0.1"})



def test_metadata_versions_do_not_leak_new_securities_or_industry(tmp_path):
    store = IntelligenceStore(tmp_path/"archive.db")
    store.security("AAA", {"company": "AAA Widgets", "industry": "Old sector"}, WHEN)
    store.security("AAA", {"industry": "New sector"}, WHEN+timedelta(days=1))
    store.security("NEW", {"company": "New listing"}, WHEN+timedelta(days=1))
    assert store.catalog(WHEN)["AAA"]["industry"] == "Old sector"
    assert "NEW" not in store.catalog(WHEN)
    assert store.catalog(WHEN+timedelta(days=2))["AAA"]["industry"] == "New sector"


def test_original_payloads_are_deduplicated_without_losing_document_text(tmp_path):
    store = IntelligenceStore(tmp_path/"archive.db")
    for index in range(2):
        store.archive(url=f"https://example.org/{index}",publisher="example.org",title=f"AAA {index}",
            text=TEXT+str(index),published=WHEN,acquired=WHEN,payload=b"one shared feed response")
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM source_payloads").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM document_versions").fetchone()[0] == 2
        assert db.execute("SELECT body FROM source_payloads").fetchone()[0] == b"one shared feed response"


def test_source_response_bound_and_private_destinations(monkeypatch):
    import httpx
    from app.intelligence import sources
    original = httpx.Client
    monkeypatch.setattr(sources, "public_url", lambda url, domains: url)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x"*2048)), **kwargs))
    with pytest.raises(ValueError, match="bound"):
        sources.PublicFetcher("fixture", 1024)("https://example.org/feed", {"example.org"})


def test_intelligence_endpoints_are_read_only_and_asof_scoped(tmp_path, monkeypatch):
    from app.config import settings
    from app.api.v1.endpoints import intelligence as api
    path = tmp_path/"archive.db"
    monkeypatch.setattr(settings, "INTELLIGENCE_DB_PATH", str(path))
    assert api.status()["status"] == "waiting" and not path.exists()
    store = IntelligenceStore(path)
    version = store.archive(url="https://example.org/aaa",publisher="example.org",title="AAA evidence",
        text=TEXT,published=WHEN,acquired=WHEN)
    chunk = store.retrieve("AAA", WHEN)[0]
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        api.evidence(chunk["id"], stamp(WHEN-timedelta(seconds=1)))
    assert error.value.status_code == 404
    assert api.evidence(chunk["id"], stamp(WHEN))["body"] == chunk["body"]
    assert api.original(version, stamp(WHEN))["original_text"] == TEXT
