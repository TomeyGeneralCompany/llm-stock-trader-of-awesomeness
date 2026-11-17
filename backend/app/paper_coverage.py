"""Verify that a paper session was observed from open to close."""
from datetime import timedelta

from app.paper_market import BAR_MINUTES, utc

BAR_INTERVAL = timedelta(minutes=BAR_MINUTES)
ACTIONABLE_AGE = timedelta(minutes=15)


def session_coverage(db, opened, closed, mode):
    """Check exchange slots and, for forward sessions, actual receipt and poll times."""
    expected = {
        opened + BAR_INTERVAL * index
        for index in range(1, (closed - opened) // BAR_INTERVAL + 1)
    }
    observed, timely = set(), set()
    rows = db.execute(
        """SELECT DISTINCT b.end, r.received
           FROM bars b LEFT JOIN bar_receipts r ON b.symbol=r.symbol AND b.end=r.end
           WHERE b.end>? AND b.end<=?
             AND (?='replay' OR json_extract(b.body,'$.volume')>0)""",
        (opened.isoformat(), closed.isoformat(), mode),
    )
    for bar_end, received in rows:
        end = utc(bar_end)
        if end in expected:
            observed.add(end)
            if received and end <= utc(received) <= end + ACTIONABLE_AGE:
                timely.add(end)

    gaps = []
    if mode == "forward":
        points = [opened]
        points.extend(utc(row[0]) for row in db.execute(
            "SELECT stamp FROM session_heartbeats WHERE stamp>=? AND stamp<=? ORDER BY stamp",
            (opened.isoformat(), closed.isoformat()),
        ))
        points.append(closed)
        for left, right in zip(points, points[1:]):
            if right - left > BAR_INTERVAL:
                gaps.append({"from": left.isoformat(), "to": right.isoformat(),
                             "seconds": (right - left).total_seconds()})
    missing = expected - observed
    delayed = observed - timely if mode == "forward" else set()
    return {
        "verified": bool(expected) and not (missing or delayed or gaps),
        "mode": mode,
        "expected_bars": len(expected),
        "observed_bars": len(observed),
        "missing_bar_ends": [end.isoformat() for end in sorted(missing)],
        "late_or_unverified_bar_ends": [end.isoformat() for end in sorted(delayed)],
        "heartbeat_gaps": gaps,
    }
