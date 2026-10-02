"""Download public research inputs and preserve auditable raw responses."""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ASOF = "2026-09-28"
SYMBOLS = {"SP500": "^GSPC", "DJIA": "^DJI", "NASDAQ": "^IXIC", "RUSSELL": "^RUT", "VIX": "^VIX"}


def fetch(item):
    name, url = item
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as response:
        blob = response.read()
    path = RAW / name
    path.write_bytes(blob)
    return name, {
        "url": url, "path": str(path.relative_to(ROOT)),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
    }


def yahoo_frame(name):
    result = json.loads((RAW / f"yahoo_{name}.json").read_text())["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    indicators = result["indicators"]
    assert "adjclose" in indicators, f"Missing adjusted close: {name}"
    frame = pd.DataFrame({"adjclose": indicators["adjclose"][0]["adjclose"],
                          "close": indicators["quote"][0]["close"]}, index=dates)
    assert frame.index.is_unique, name
    return frame.loc[:ASOF].sort_index()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    stop = int(dt.datetime(2026, 9, 29, tzinfo=dt.timezone.utc).timestamp())
    requests = [(f"yahoo_{k}.json", "https://query1.finance.yahoo.com/v8/finance/chart/"
                 + urllib.parse.quote(v, safe="") + f"?period1=0&period2={stop}&interval=1d")
                for k, v in SYMBOLS.items()]
    requests += [("gmu_DJId.csv", "https://mason.gmu.edu/~jgentle/books/statfinbk/Data/DJId.csv")]
    for key in ["SP500", "DJIA", "NASDAQCOM", "VIXCLS"]:
        requests.append((f"fred_{key}.csv", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={key}&cosd=2026-01-01&coed={ASOF}"))
    manifest = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        for name, meta in pool.map(fetch, requests):
            manifest[name] = meta
            print("downloaded", name, meta["bytes"], flush=True)
    (ROOT / "data" / "sources.json").write_text(json.dumps(manifest, indent=2))
    frames = {name: yahoo_frame(name) for name in SYMBOLS}
    audit = {"asof": ASOF, "series": {k: {"first": str(v.index.min().date()), "last": str(v.index.max().date()),
                   "rows": len(v), "missing_adjclose": int(v.adjclose.isna().sum()),
                   "max_close_adjusted_difference": float((v.close-v.adjclose).abs().max())} for k,v in frames.items()}}
    old = pd.read_csv(RAW / "gmu_DJId.csv", index_col="Date", parse_dates=True)["Adj Close"].dropna()
    assert old.index.is_unique
    overlap = pd.concat([old.rename("archive"), frames["DJIA"].adjclose.rename("yahoo")], axis=1).dropna()
    diff = (overlap.archive-overlap.yahoo).abs()
    audit["djia_overlap"] = {"n": len(overlap), "first": str(overlap.index.min().date()),
                             "last": str(overlap.index.max().date()), "max_abs_diff": float(diff.max()),
                             "median_abs_diff": float(diff.median()), "max_rel_diff": float((diff/overlap.yahoo).max())}
    # Index levels may differ in minor rounding, but a material mismatch blocks the splice.
    assert len(overlap) > 1000 and (diff/overlap.yahoo).max() < 0.001, audit["djia_overlap"]
    overlap.assign(abs_diff=diff).to_csv(ROOT / "data" / "djia_overlap_check.csv")
    djia_new = frames["DJIA"].adjclose.dropna()
    djia = pd.concat([old.loc[old.index < djia_new.index.min()], djia_new]).sort_index()
    audit["djia_splice"] = {"archive_last": str(old.loc[old.index < djia_new.index.min()].index.max().date()),
                            "yahoo_first": str(djia_new.index.min().date())}
    prices = pd.concat([frames["SP500"].adjclose.rename("SP500"), djia.rename("DJIA"),
                        frames["NASDAQ"].adjclose.rename("NASDAQ"), frames["RUSSELL"].adjclose.rename("RUSSELL")], axis=1)
    # The S&P calendar is the common reference. Preserve holes until AFTER differencing.
    calendar = frames["SP500"].adjclose.dropna().loc["1986-01-01":].index
    prices = prices.reindex(calendar)
    assert (prices.dropna() > 0).all().all()
    ret = np.log(prices).diff() * 100
    complete = prices.dropna()
    ret = ret.dropna()
    ret.index.name = prices.index.name = "date"
    prices.to_csv(ROOT / "data" / "prices_calendar.csv")
    ret.to_csv(ROOT / "data" / "returns.csv")
    frames["VIX"].adjclose.rename("VIX").to_csv(ROOT / "data" / "vix.csv", index_label="date")
    start = complete.index.min()
    audit["common_prices"] = {"first": str(start.date()), "last": str(complete.index.max().date()), "n": len(complete)}
    audit["common_returns"] = {"first": str(ret.index.min().date()), "last": str(ret.index.max().date()), "n": len(ret)}
    audit["missing_after_common_start"] = {k: [str(d.date()) for d in prices.loc[start:].index[prices.loc[start:,k].isna()]] for k in prices}
    audit["max_abs_daily_log_return_pct"] = ret.abs().max().to_dict()
    audit["large_returns_abs_over_10pct"] = [{"date": str(d.date()), "index": k, "log_return_pct": float(ret.loc[d,k])}
                                             for k in ret for d in ret.index[ret[k].abs() > 10]]
    fred_checks = {}
    for yf, fred in [("SP500","SP500"),("DJIA","DJIA"),("NASDAQ","NASDAQCOM"),("VIX","VIXCLS")]:
        f = pd.read_csv(RAW / f"fred_{fred}.csv", index_col=0, parse_dates=True).apply(pd.to_numeric, errors="coerce").iloc[:,0]
        pair = pd.concat([frames[yf].close.rename("yahoo"),f.rename("fred")],axis=1,sort=True).dropna()
        dif = (pair.yahoo-pair.fred).abs()
        fred_checks[yf] = {"n":len(pair), "last":str(pair.index.max().date()), "max_abs_diff":float(dif.max()),
                           "max_rel_diff": float((dif/pair.fred).max()), "latest_pair":pair.tail(1).iloc[0].to_dict()}
        pair.assign(abs_diff=dif).to_csv(ROOT / "data" / f"crosscheck_{yf}.csv")
    audit["fred_crosschecks"] = fred_checks
    (ROOT / "data" / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
