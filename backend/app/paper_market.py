"""Forward/replay market observations. All timestamps denote completed bar ends."""
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import csv
import io
import math
import re

import pandas as pd
import requests
import yfinance as yf

EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc
BAR_MINUTES = 5


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("Market timestamps must include a timezone.")
    return stamp.tz_convert("UTC").to_pydatetime()


@dataclass(frozen=True)
class Bar:
    symbol: str
    end: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self):
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,15}", self.symbol):
            raise ValueError("Invalid symbol")
        object.__setattr__(self, "end", utc(self.end))
        values = (self.open, self.high, self.low, self.close, self.volume)
        if not all(math.isfinite(x) for x in values) or min(values[:4]) <= 0 or self.volume < 0:
            raise ValueError("Invalid OHLCV")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("Invalid bar range")

    @property
    def start(self):
        return self.end - timedelta(minutes=BAR_MINUTES)

    @property
    def day(self):
        return self.end.astimezone(EASTERN).date().isoformat()

    def record(self):
        return {**asdict(self), "end": self.end.isoformat()}

    @classmethod
    def from_record(cls, row):
        return cls(row["symbol"], utc(row["end"]), *(float(row[k]) for k in ("open", "high", "low", "close", "volume")))


def regular(bar, close_hour=16, close_minute=0):
    start, end = bar.start.astimezone(EASTERN), bar.end.astimezone(EASTERN)
    return (start.weekday() < 5 and start.date() == end.date()
            and (start.hour, start.minute) >= (9, 30)
            and (end.hour, end.minute) <= (close_hour, close_minute)
            and start.minute % BAR_MINUTES == 0
            and start.second == 0 and start.microsecond == 0)


def listed_symbols():
    """Discover US exchange listings from Nasdaq Trader's public directories."""
    base = "https://www.nasdaqtrader.com/dynamic/SymDir/"
    symbols = set()
    for name, key in (("nasdaqlisted.txt", "Symbol"), ("otherlisted.txt", "ACT Symbol")):
        response = requests.get(base + name, timeout=30)
        response.raise_for_status()
        rows = csv.DictReader(io.StringIO(response.text), delimiter="|")
        if key not in (rows.fieldnames or []):
            raise ValueError("Unexpected exchange listing format")
        for row in rows:
            symbol = (row.get(key) or "").replace(".", "-")
            if row.get("Test Issue") != "N" or not re.fullmatch(r"[A-Z0-9][A-Z0-9-]{0,9}", symbol):
                continue
            label = (row.get("Security Name") or "").lower()
            if any(term in label for term in ("warrant", "right to", "preferred", "depositary share", "unit consisting", "notes due")):
                continue
            symbols.add(symbol)
    if not symbols:
        raise ValueError("Exchange universe is empty")
    return sorted(symbols)


def normalize_history(frame, symbol, now):
    """Yahoo timestamps mark bar starts. Reject incomplete and extended-hours bars."""
    if frame is None or frame.empty:
        return []
    bars = []
    for stamp, row in frame.iterrows():
        try:
            end = utc(stamp) + timedelta(minutes=BAR_MINUTES)
            bar = Bar(symbol, end, *(float(row[k]) for k in ("Open", "High", "Low", "Close", "Volume")))
            if end <= now and regular(bar):
                bars.append(bar)
        except (ValueError, TypeError, KeyError):
            continue
    return bars


def yahoo_bars(symbols, now):
    """Bounded batches; missing symbols remain missing, never synthesized."""
    result = []
    for offset in range(0, len(symbols), 32):
        batch = symbols[offset:offset + 32]
        if not batch:
            continue
        frame = yf.download(batch, period="5d", interval="5m", group_by="ticker",
                            auto_adjust=False, prepost=False, progress=False,
                            threads=False, timeout=20)
        for symbol in batch:
            if isinstance(frame.columns, pd.MultiIndex):
                if symbol not in frame.columns.get_level_values(0):
                    continue
                part = frame[symbol]
            elif len(batch) == 1:
                part = frame
            else:
                continue
            result.extend(normalize_history(part, symbol, now))
    return result


def read_replay(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return sorted((Bar.from_record(row) for row in csv.DictReader(handle)),
                      key=lambda bar: (bar.end, bar.symbol))
