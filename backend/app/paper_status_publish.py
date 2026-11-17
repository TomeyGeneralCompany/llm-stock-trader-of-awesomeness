"""Publish an allowlisted snapshot from the laptop to a designated private GitHub repo."""
import argparse
import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path

import httpx

from app.paper_market import UTC
from app.paper_trader import read_report


def snapshot(paper, now, activity=None, trial=None):
    return {
        "reported_at":now.isoformat(), "last_quote_received":paper.get("last_quote_received"),
        "research_stage":activity, "equity":paper["equity"], "events":paper["events"],
        "recent_decisions":[{"symbol":row["symbol"],"bar":row["bar"],"action":row["decision"]["action"]}
                            for row in paper["latest_decisions"][:8]],
        "recent_fills":[{key:row["detail"].get(key) for key in ("symbol","action","quantity","price","fill_method")}
                        for row in paper["recent_events"] if row["kind"]=="fill"],
        "recent_issues":[{"time":row["time"],"kind":row["kind"],"symbol":row["detail"].get("symbol")}
                         for row in paper["recent_events"] if row["kind"] in ("analysis_error","order_not_queued","unfilled")],
        "sessions":[{key:row.get(key) for key in ("day","complete","achieved","equity","goal_reached_at")}
                    for row in paper["sessions"][-5:]],
        "trial":trial["sessions"] if trial else None,
    }


def publish(client, repository, branch, payload):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repository):
        raise ValueError("Expected a GitHub owner/repository.")
    repo_url=f"https://api.github.com/repos/{repository}"
    metadata=client.get(repo_url)
    metadata.raise_for_status()
    if metadata.json().get("visibility")!="private":
        raise ValueError("Paper status requires a private GitHub repository.")
    url=repo_url+"/contents/paper-trader-status.json"
    existing=client.get(url,params={"ref":branch})
    update={"message":"Refresh local paper trader status","branch":branch,
            "content":base64.b64encode((json.dumps(payload,indent=2)+"\n").encode()).decode()}
    if existing.status_code!=404:
        existing.raise_for_status()
        update["sha"]=existing.json()["sha"]
    response=client.put(url,json=update)
    response.raise_for_status()


def main():
    from app.config import settings
    from app.intelligence.store import IntelligenceStore
    from app.paper_trial import evaluation
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default=settings.PAPER_DB_PATH)
    parser.add_argument("--intelligence-db",default=settings.INTELLIGENCE_DB_PATH)
    parser.add_argument("--trial",type=Path)
    parser.add_argument("--repo")
    parser.add_argument("--branch",default="main")
    parser.add_argument("--preview",action="store_true")
    args=parser.parse_args()
    paper=read_report(args.db)
    activity=None
    if Path(args.intelligence_db).is_file():
        store=IntelligenceStore.__new__(IntelligenceStore)
        store.path,store.read_only=args.intelligence_db,True
        with store.connect() as db:
            row=db.execute("SELECT stamp,stage FROM intelligence_events ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            activity={"time":row["stamp"],"stage":row["stage"]}
    trial=evaluation(json.loads(args.trial.read_text()),paper) if args.trial else None
    payload=snapshot(paper,datetime.now(UTC),activity,trial)
    if args.preview:
        print(json.dumps(payload,indent=2))
        return
    token=os.environ.get("PAPER_STATUS_GITHUB_TOKEN")
    if not args.repo or not token:
        parser.error("Publishing requires --repo and PAPER_STATUS_GITHUB_TOKEN.")
    with httpx.Client(timeout=10,follow_redirects=False,trust_env=False,
                      headers={"Authorization":f"Bearer {token}","Accept":"application/vnd.github+json"}) as client:
        publish(client,args.repo,args.branch,payload)


if __name__=="__main__":
    main()
