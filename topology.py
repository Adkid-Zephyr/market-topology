"""Causal daily four-index persistence landscapes, with exact landscape norms."""
from __future__ import annotations

import bisect
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.signal import periodogram
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
WINDOWS = (50, 100)


def landscape_norms(diagram):
    diagram = np.asarray(diagram, dtype=float).reshape(-1,2)
    if not np.isfinite(diagram).all():
        raise ValueError("H1 diagram contains non-finite intervals; do not silently discard censored loops")
    persistence = diagram[:,1] - diagram[:,0]
    if (persistence < 0).any():
        raise ValueError("Negative persistence")
    return float(np.sum(persistence**2)/4), float(np.sqrt(np.sum(persistence**3)/12))


def diagram_for(points):
    return ripser(np.asarray(points), maxdim=1, coeff=2, metric="euclidean", thresh=np.inf)["dgms"][1]


def window_metrics(points):
    l1,l2 = landscape_norms(diagram_for(points))
    std = points.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("Constant return coordinate")
    mean_var = float(np.mean(std**2))
    corr = np.corrcoef(points, rowvar=False)
    z_l1,_ = landscape_norms(diagram_for((points-points.mean(axis=0))/std))
    return {"l1": l1, "l2": l2, "mean_daily_variance": mean_var,
            "annual_rms_vol_pct": np.sqrt(mean_var*252), "mean_correlation":float(corr[np.triu_indices(4,1)].mean()),
            "scale_free_l1":l1/mean_var, "zscore_l1":z_l1}


def prior_percentile(series, lookback=None, minimum=756):
    """Empirical midrank against earlier values only, preserving zero vs missing."""
    x = series.to_numpy(float)
    out = np.full(len(x), np.nan)
    sorted_history = []
    for i,value in enumerate(x):
        if lookback is not None and i > lookback and np.isfinite(x[i-lookback-1]):
            sorted_history.pop(bisect.bisect_left(sorted_history, x[i-lookback-1]))
        if np.isfinite(value) and len(sorted_history) >= minimum:
            out[i] = 100*(bisect.bisect_left(sorted_history,value)+bisect.bisect_right(sorted_history,value))/(2*len(sorted_history))
        if np.isfinite(value):
            bisect.insort(sorted_history,value)
    return pd.Series(out,index=series.index)


def add_trailing_features(frame):
    x = frame.l1.to_numpy()
    frame = frame.copy()
    n = 500
    out = {k:np.full(len(x),np.nan) for k in ["variance500","acf1_500","psd_low500","psd_linear500","psd_1_16_500","psd_1_4_500"]}
    for i in range(n-1,len(x)):
        segment = x[i-n+1:i+1]
        out["variance500"][i] = np.var(segment,ddof=1)
        centered = segment-segment.mean()
        denom = np.dot(centered,centered)
        out["acf1_500"][i] = np.dot(centered[:-1],centered[1:])/denom if denom else np.nan
        _, psd = periodogram(segment,fs=1,window="boxcar",detrend="constant",scaling="density")
        _, linear = periodogram(segment,fs=1,window="boxcar",detrend="linear",scaling="density")
        for denom_,key in [(8,"psd_low500"),(16,"psd_1_16_500"),(4,"psd_1_4_500")]:
            bins = (len(psd)-1)//denom_
            out[key][i] = np.mean(psd[1:bins+1])
        out["psd_linear500"][i] = np.mean(linear[1:(len(linear)-1)//8+1])
    for key,arr in out.items():
        frame[key] = arr
    for key in ["l1","l2","scale_free_l1","zscore_l1","annual_rms_vol_pct","variance500","psd_low500"]:
        frame[key+"_prior_pct"] = prior_percentile(frame[key])
        frame[key+"_prior5y_pct"] = prior_percentile(frame[key],lookback=1260)
    for key in ["variance500","psd_low500","psd_linear500","psd_1_16_500","psd_1_4_500"]:
        a = frame[key].to_numpy()
        tau = np.full(len(a),np.nan)
        for i in range(249,len(a)):
            last = a[i-249:i+1]
            if np.isfinite(last).all():
                tau[i] = kendalltau(np.arange(250),last).statistic
        frame[key+"_tau250"] = tau
    return frame


def summarize(frames, prices):
    dates = {"Dot-com reference":"2000-03-10", "Lehman reference":"2008-09-15", "LTCM/Russia reference":"1998-08-31",
             "US downgrade reference":"2011-08-08", "COVID reference":"2020-03-16", "2022 reference":"2022-10-12"}
    keys=["l1","l2","l1_prior_pct","l1_prior5y_pct","scale_free_l1_prior_pct","zscore_l1_prior_pct", "annual_rms_vol_pct", "mean_correlation",
          "variance500_prior_pct","psd_low500_prior_pct","variance500_tau250","psd_low500_tau250",
          "psd_linear500_tau250","psd_1_16_500_tau250","psd_1_4_500_tau250"]
    snapshots=[]
    for w,f in frames.items():
        for lag in [0,21,63,126,250]:
            row=f.iloc[-1-lag]
            snapshots.append({"label":"current" if lag==0 else f"current_minus_{lag}_sessions", "window":w,"date":str(row.name.date()),**{k:row[k] for k in keys}})
        for event,date in dates.items():
            for timing,part in [("pre_event", f.loc[f.index<pd.Timestamp(date)]),("event_day",f.loc[:date])]:
                row=part.iloc[-1]
                snapshots.append({"label":event+"_"+timing,"window":w,"date":str(row.name.date()),**{k:row[k] for k in keys}})
    table=pd.DataFrame(snapshots)
    table.to_csv(ROOT/"results"/"snapshots.csv",index=False)
    annual=[]
    for w,f in frames.items():
        for year,g in f.groupby(f.index.year):
            peak=g.l1.idxmax()
            annual.append({"year":year,"window":w,"n":len(g),"l1_median":g.l1.median(),"l1_max":g.l1.max(),
                           "l1_peak_date":str(peak.date()),"l1_prior_pct_median":g.l1_prior_pct.median(),
                           "days_above_prior95":int((g.l1_prior_pct>=95).sum()),
                           "eligible_days":int(g.l1_prior_pct.notna().sum()),"scale_free_l1_median":g.scale_free_l1.median()})
    pd.DataFrame(annual).to_csv(ROOT/"results"/"annual.csv",index=False)
    correlations={str(w):{period:f.loc[begin:end,["l1","annual_rms_vol_pct","mean_correlation","scale_free_l1"]].corr(method="spearman").to_dict()
                            for period,begin,end in [("all",f.index.min(),f.index.max()),("post_paper","2017-01-01",f.index.max())]} for w,f in frames.items()}
    (ROOT/"results"/"correlations.json").write_text(json.dumps(correlations,indent=2))
    return table


def main():
    (ROOT/"results").mkdir(exist_ok=True)
    returns=pd.read_csv(ROOT/"data"/"returns.csv",index_col="date",parse_dates=True)
    prices=pd.read_csv(ROOT/"data"/"prices_calendar.csv",index_col="date",parse_dates=True)
    frames={}
    start=time.time()
    for w in WINDOWS:
        cache=ROOT/"results"/f"base_w{w}.csv"
        meta=cache.with_suffix(".meta.json")
        identity={"returns_sha256":hashlib.sha256((ROOT/"data"/"returns.csv").read_bytes()).hexdigest(),
                  "window":w,"core_version":"1.0-exact-landscape-raw-percent-return"}
        if cache.exists() and meta.exists() and json.loads(meta.read_text())==identity:
            f=pd.read_csv(cache,index_col="date",parse_dates=True)
        else:
            x=returns.to_numpy()
            points=(x[i-w+1:i+1] for i in range(w-1,len(x)))
            rows=[]
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                for i,row in enumerate(pool.map(window_metrics,points,chunksize=40)):
                    rows.append(row)
                    if i%1000==0: print(f"w={w} {i+1}/{len(x)-w+1} elapsed={time.time()-start:.1f}s",flush=True)
            f=pd.DataFrame(rows,index=returns.index[w-1:])
            f.to_csv(cache,index_label="date")
            meta.write_text(json.dumps(identity,indent=2))
        f=add_trailing_features(f)
        f.to_csv(ROOT/"results"/f"daily_w{w}.csv",index_label="date")
        frames[w]=f
    snapshots=summarize(frames,prices)
    print(snapshots.loc[snapshots.label.isin(["current","Dot-com reference_pre_event","Lehman reference_pre_event"])].to_string(index=False))
    print(f"Completed in {time.time()-start:.1f}s",flush=True)


if __name__=="__main__":
    main()
