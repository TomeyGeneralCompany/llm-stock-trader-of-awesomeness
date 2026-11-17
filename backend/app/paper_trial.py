"""Freeze and evaluate a five-session paper challenge using the existing ledger."""
import argparse
from datetime import date, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from app.paper_market import EASTERN, market_close
from app.paper_trader import OBJECTIVE, START, read_report


def runtime_identity(model, quotes=None, options=None):
    import httpx
    from app.config import settings
    repo=Path(__file__).resolve().parents[2]
    try:
        commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        dirty=bool(subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        commit,dirty=None,True
    digest = None
    try:
        with httpx.Client(timeout=3,trust_env=False) as client:
            response=client.get(settings.OLLAMA_BASE_URL.rstrip("/")+"/api/tags")
            response.raise_for_status()
        digest=next((row["digest"] for row in response.json()["models"] if row.get("name")==model),None)
    except (httpx.HTTPError,ValueError,KeyError,TypeError):
        pass
    defaults={"symbols":None,"batch":64,"candidates":3,"poll":60,"without_memory":False}
    scanner={key:getattr(options,key,value) for key,value in defaults.items()}
    config={"threads":settings.OLLAMA_NUM_THREAD,"context":settings.OLLAMA_NUM_CTX,
            "think":settings.OLLAMA_THINK,"intelligence":settings.INTELLIGENCE_ENABLED,
            "feed_config_hash":profile_digest({"feeds":settings.INTELLIGENCE_FEEDS,
                "benchmarks":settings.INTELLIGENCE_BENCHMARKS,"fred_series":settings.INTELLIGENCE_FRED_SERIES})}
    return {"commit":commit,"model":model,"model_digest":digest,"scanner":scanner,"config":config,
            "feed":"yahoo_completed_bars+local_bid_ask" if quotes else "yahoo_completed_bars",
            "quotes_path":str(Path(quotes).resolve()) if quotes else None,"dirty":dirty}


def freeze(path, first_day, model, quotes, db, options=None):
    if first_day <= datetime.now(EASTERN).date():
        raise ValueError("Choose the first session before the trial begins.")
    if market_close(first_day) is None:
        raise ValueError("The first date must be a regular trading session.")
    days, day = [], first_day
    while len(days)<5:
        if market_close(day):
            days.append(day.isoformat())
        day += timedelta(days=1)
    runtime=runtime_identity(model, quotes, options)
    if runtime["dirty"]:
        raise ValueError("Freeze a clean tracked checkout for the trial.")
    if not runtime["model_digest"]:
        raise ValueError("Could not verify the selected model digest from local Ollama.")
    path_db=Path(db).resolve()
    report=read_report(path_db)
    if report["mode"]!="forward" or not report["challenge"]:
        raise ValueError("The trial requires a forward-mode challenge account.")
    manifest={"days":days,"runtime":runtime,"db":str(path_db),
              "profile_hash":profile_digest(report["profile"]), "start":START,"target":OBJECTIVE}
    with open(path,"x",encoding="utf-8") as out:
        json.dump(manifest,out,indent=2)
    return manifest


def profile_digest(profile):
    return sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()


def evaluation(manifest, report):
    if profile_digest(report["profile"])!=manifest["profile_hash"]:
        raise ValueError("Account configuration differs from the frozen profile.")
    by_date={day["day"]:day for day in report["sessions"]}
    results=[]
    for day in manifest["days"]:
        row=by_date.get(day)
        if row is None:
            results.append({"day":day,"result":"missing"})
            continue
        realized=row["cash"]+row["unsettled"]
        identity_matches=row.get("runtime")==manifest["runtime"] and not row.get("runtime_changed")
        passed=(identity_matches and row["start"]==manifest["start"] and row["complete"] and row["achieved"]
                and not row["holdings"] and realized>=manifest["target"])
        results.append({"day":day,"result":"pass" if passed else "configuration_changed" if not identity_matches else "miss" if row["complete"] else "incomplete",
                        "realized":realized,"goal_reached_at":row["goal_reached_at"]})
    return {"passed": len(results)==5 and all(row["result"]=="pass" for row in results),"sessions":results,
            "frozen":manifest["runtime"]}


def main():
    from app.config import settings
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prepare=commands.add_parser("freeze")
    prepare.add_argument("manifest",type=Path)
    prepare.add_argument("--first-day",required=True,type=date.fromisoformat)
    prepare.add_argument("--db",default=settings.PAPER_DB_PATH)
    prepare.add_argument("--quotes",help="Same optional bid/ask input path as the service")
    prepare.add_argument("--symbols")
    prepare.add_argument("--batch",type=int,default=64)
    prepare.add_argument("--candidates",type=int,default=3)
    prepare.add_argument("--poll",type=int,default=60)
    prepare.add_argument("--without-memory",action="store_true")
    inspect=commands.add_parser("report")
    inspect.add_argument("manifest",type=Path)
    args=parser.parse_args()
    if args.command=="freeze":
        value=freeze(args.manifest,args.first_day,settings.OLLAMA_MODEL,args.quotes,args.db,args)
    else:
        manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
        value=evaluation(manifest,read_report(manifest["db"]))
    print(json.dumps(value,indent=2))


if __name__=="__main__":
    main()
