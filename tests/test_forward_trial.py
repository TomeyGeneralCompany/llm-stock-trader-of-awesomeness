import json
from dataclasses import replace
from datetime import timedelta

import httpx

from app.paper_compare import compare
from app.paper_market import EASTERN
from app.paper_status_publish import publish, snapshot
from app.paper_trial import evaluation, profile_digest
from test_paper_trader import day_bars, response


def test_private_snapshot_is_allowlisted_and_refuses_public_destinations():
    paper={"equity":1.,"events":{},"state":{"secret":"never expose"},
           "latest_decisions":[{"symbol":"AAA","bar":"bar","decision":{"action":"HOLD","reasoning":"private prompt"}}],
           "recent_events":[],"sessions":[]}
    payload=snapshot(paper,day_bars()[-1].end)
    assert payload["recent_decisions"][0]["action"]=="HOLD"
    assert "secret" not in json.dumps(payload) and "private prompt" not in json.dumps(payload)
    calls=[]
    def public(request):
        calls.append(request.method)
        return httpx.Response(200,json={"visibility":"public"})
    import pytest
    with httpx.Client(transport=httpx.MockTransport(public)) as client:
        with pytest.raises(ValueError,match="private"):
            publish(client,"owner/repo","main",payload)
    assert calls==["GET"]
    def private(request):
        if request.method=="PUT":
            assert json.loads(request.content)["sha"]=="previous-version"
            calls.append("PUT")
            return httpx.Response(200)
        body={"sha":"previous-version"} if "/contents/" in str(request.url) else {"visibility":"private"}
        return httpx.Response(200,json=body)
    with httpx.Client(transport=httpx.MockTransport(private)) as client:
        publish(client,"owner/repo","main",payload)
    assert calls==["GET","PUT"]


def test_trial_cannot_substitute_later_day_or_count_open_equity():
    dates=["2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07"]
    runtime={"commit":"frozen","model":"qwen3:4b","feed":"bars"}
    profile={"challenge":True}
    manifest={"days":dates,"runtime":runtime,"profile_hash":profile_digest(profile),"start":1.,"target":100.}
    rows=[{"day":day,"start":1.,"complete":True,"achieved":True,"holdings":{},"runtime":runtime,
           "cash":100.,"unsettled":0.,"goal_reached_at":"timestamp"} for day in dates]
    report={"profile":profile,"sessions":rows}
    assert evaluation(manifest,report)["passed"]
    rows[0]["holdings"]={"AAA":{"shares":1}}
    rows[1]["runtime_changed"]=True
    rows[-1]["day"]="2025-03-10"
    result=evaluation(manifest,report)
    assert not result["passed"]
    assert [r["result"] for r in result["sessions"]]==["miss","configuration_changed","pass","pass","missing"]


def test_comparison_passes_identical_observations_to_all_policies(tmp_path):
    bars=day_bars(n=38,growth=.01)
    seen={"with":[],"without":[]}
    def local(context,lessons):
        seen["with" if lessons else "without"].append(context["asof"])
        return response(context)
    results=compare(bars,tmp_path/"comparison",warmup=0,local=local)
    assert set(results)=={"qwen_memory","qwen_no_memory","deterministic"}
    assert results["qwen_memory"]["forecasts_scored"]==results["qwen_no_memory"]["forecasts_scored"]>0
    assert results["deterministic"]["session_reports"][0]["complete"] is False
    assert seen["with"] and seen["without"]
