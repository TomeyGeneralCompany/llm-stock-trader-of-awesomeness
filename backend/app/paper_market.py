"""Forward/replay market observations. All timestamps denote completed bar ends."""
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import csv
import io
import math
import re

import pandas as pd
from functools import lru_cache
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
    USLaborDay, USThanksgivingDay)
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


def _new_year_observed(day):
    # NYSE does not observe a Saturday January 1 on the preceding Friday.
    return day + pd.DateOffset(days=1) if day.weekday() == 6 else day


class USMarketHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=_new_year_observed),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def market_close(day, override=None):
    """Regular US exchange calendar; exceptional closures must be supplied by operator/feed."""
    return _market_close(str(day), tuple(override) if override else None)


@lru_cache(maxsize=512)
def _market_close(day, override):
    d = date.fromisoformat(day)
    if d.weekday() >= 5 or len(USMarketHolidays().holidays(start=day, end=day)):
        return None
    if override:
        return override
    if (d.month, d.day) in ((7, 3), (12, 24)):
        return (13, 0)
    if d.month == 11 and d.weekday() == 4 and 23 <= d.day <= 29:
        return (13, 0)
    return (16, 0)


def regular(bar, close_hour=None, close_minute=None):
    start, end = bar.start.astimezone(EASTERN), bar.end.astimezone(EASTERN)
    close = market_close(end.date(), (close_hour, close_minute) if close_hour is not None else None)
    if close is None:
        return False
    return (start.weekday() < 5 and start.date() == end.date()
            and (start.hour, start.minute) >= (9, 30)
            and (end.hour, end.minute) <= close
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


def yahoo_bars(symbols, now, *, period="5d"):
    """Bounded batches; missing symbols remain missing, never synthesized."""
    result = []
    for offset in range(0, len(symbols), 32):
        batch = symbols[offset:offset + 32]
        if not batch:
            continue
        frame = yf.download(batch, period=period, interval="5m", group_by="ticker",
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



@dataclass(frozen=True)
class Quote:
    symbol: str
    time: datetime
    received: datetime
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    fractional_eligible: bool | None = None

    def __post_init__(self):
        if not self.symbol or self.time.tzinfo is None or self.received.tzinfo is None or self.time > self.received:
            raise ValueError("Quote requires a symbol and ordered, timezone-aware timestamps")
        if not all(math.isfinite(x) for x in (self.bid,self.ask,self.bid_size,self.ask_size)):
            raise ValueError("Quote must contain finite prices and sizes")
        if not 0 < self.bid <= self.ask or min(self.bid_size,self.ask_size) < 0:
            raise ValueError("Invalid bid, ask or displayed size")
        if self.fractional_eligible is not None and type(self.fractional_eligible) is not bool:
            raise ValueError("Fractional eligibility must be true, false or unknown")
