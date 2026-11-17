"""Exercise the market-check plumbing separately from the network acceptance run."""
from dataclasses import replace
import hashlib
import json
from datetime import timezone

import pytest

from app.paper_market_check import check_market
from test_paper_trader import day_bars, response


def test_market_check_withholds_later_prices_and_preserves_provider_inputs(tmp_path):
    session=day_bars("SPY",day="2025-03-03",n=78,growth=.01)
    session[35] = replace(session[35], volume=0)
    partial=day_bars("SPY",day="2025-03-04",n=50,growth=.2)
    acquired=partial[-1].end
    calls=[]
    def decide(ctx,_lessons):
        calls.append(ctx["asof"])
        assert ctx["current"]["close"]==session[34].close
        return response(ctx,"BUY")
    result=check_market(tmp_path,fetch=lambda *args,**kw: session+partial,
                        clock=lambda:acquired,decide=decide)
    assert calls==[session[34].end.astimezone(timezone.utc).isoformat()]
    assert result["fills"]==0 and result["scored"]==1
    assert result["paper"]["events"]["unfilled"] == 1
    assert result["paper"]["sessions"][0]["coverage"]["observed_bars"] == 78
    assert result["session"]=="2025-03-03"
    assert result["sha256"]==hashlib.sha256((tmp_path/"market-bars.json").read_bytes()).hexdigest()


def test_provider_failure_never_invents_prices_or_calls_qwen(tmp_path):
    def unexpected(*args):
        pytest.fail("Must not start Qwen without provider data")
    with pytest.raises(RuntimeError, match="No market test ran"):
        check_market(tmp_path, fetch=lambda *args,**kwargs:[], decide=unexpected)
    assert json.loads((tmp_path/"result.json").read_text())["status"]=="provider_unavailable"
    assert not (tmp_path/"market-bars.json").exists()
    assert not (tmp_path/"paper.sqlite3").exists()
