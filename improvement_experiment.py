"""Frozen, purged monthly tail-loss experiment. See IMPROVEMENT_PROTOCOL.md."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss

from topology import diagram_for, landscape_norms

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "improvement_results"
SEED = 20260928
N_NULL = 64
HORIZONS = {"12m_20pct": (252, .20), "6m_15pct": (126, .15)}
RAW = ["raw50", "raw100", "variance_tau50", "psd_tau50"]
SCALED = ["scaled50", "scaled100", "scaled_delta50", "scaled_delta100"]
NULL = ["null_z50", "null_z100", "null_delta50", "null_delta100"]
MARKET = ["vol50", "vol100", "volratio", "mom63", "mom252", "drawdown252", "corr50", "skew50"]
MODELS = {"tda_raw": RAW, "tda_scaled": SCALED, "tda_null": NULL, "market": MARKET,
          "market_raw": MARKET+RAW, "market_scaled": MARKET+SCALED, "market_null": MARKET+NULL}


def monthly_dates(index):
    """Only completed calendar months; keep current partial month separately."""
    last = index.max()
    s = pd.Series(index, index=index).groupby(index.to_period("M")).last()
    return pd.DatetimeIndex(s[s.index < last.to_period("M")].to_numpy())


def forward_label(prices, position, horizon, threshold):
    if position+horizon >= len(prices):
        return np.nan, pd.NaT, np.nan
    future = prices.iloc[position+1:position+horizon+1]
    loss = float(future.min()/prices.iloc[position]-1)
    return float(loss <= -threshold), future.index[-1], loss


def null_metric(job):
    datestr, w, x = job
    rng = np.random.default_rng(np.random.SeedSequence([SEED, int(datestr.replace("-","")), w]))
    observed,_ = landscape_norms(diagram_for(x))
    cov = np.cov(x, rowvar=False, ddof=1)
    eig, vec = np.linalg.eigh(cov)
    root = vec @ np.diag(np.sqrt(np.maximum(eig,0)))
    norms = np.empty(N_NULL)
    for i in range(N_NULL):
        surrogate = rng.normal(size=x.shape) @ root.T
        norms[i] = landscape_norms(diagram_for(surrogate))[0]
    logs = np.log(norms+1e-8)
    z = (np.log(observed+1e-8)-logs.mean())/logs.std(ddof=1)
    return {"date":datestr, "window":w, "observed_l1":observed, "null_mean_l1":norms.mean(),
            "null_log_mean":logs.mean(), "null_log_std":logs.std(ddof=1), "null_z":z,
            "null_rank":(1+(norms<=observed).sum())/(N_NULL+1)}


def make_features():
    returns = pd.read_csv(ROOT/"data/returns.csv",index_col=0,parse_dates=True)
    prices = pd.read_csv(ROOT/"data/prices_calendar.csv",index_col=0,parse_dates=True).SP500.dropna()
    daily = {w:pd.read_csv(ROOT/"results"/f"daily_w{w}.csv",index_col=0,parse_dates=True) for w in (50,100)}
    frame = pd.DataFrame(index=daily[100].index)
    for w in (50,100):
        f=daily[w]
        frame[f"raw{w}"]=np.log1p(f.l1)
        frame[f"scaled{w}"]=np.log1p(f.scale_free_l1)
        frame[f"scaled_delta{w}"]=np.log1p(f.scale_free_l1).diff(63)
        frame[f"vol{w}"]=f.annual_rms_vol_pct
    frame["variance_tau50"]=daily[50].variance500_tau250
    frame["psd_tau50"]=daily[50].psd_low500_tau250
    frame["volratio"]=np.sqrt(returns.rolling(21).var().mean(axis=1)*252)/frame.vol100
    frame["mom63"]=np.log(prices).diff(63)
    frame["mom252"]=np.log(prices).diff(252)
    frame["drawdown252"]=prices/prices.rolling(252).max()-1
    frame["corr50"]=daily[50].mean_correlation
    frame["skew50"]=returns.SP500.rolling(50).skew()
    frame=frame.dropna()
    dates=monthly_dates(prices.index).intersection(frame.index)
    dates=dates.union(pd.DatetimeIndex([frame.index[-1]]))
    features=frame.loc[dates].copy()
    identity={"returns_sha256":hashlib.sha256((ROOT/"data/returns.csv").read_bytes()).hexdigest(),
              "dates":[str(d.date()) for d in dates],"seed":SEED,"n_null":N_NULL,"version":1}
    cache=OUT/"null_features.csv";meta=OUT/"null_features.meta.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text())==identity:
        nf=pd.read_csv(cache)
    else:
        jobs=[(str(date.date()),w,returns.loc[:date].tail(w).to_numpy()) for date in dates for w in (50,100)]
        rows=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            for i,row in enumerate(pool.map(null_metric,jobs,chunksize=10)):
                rows.append(row)
                if i%100==0:print(f"Null point clouds: {i+1}/{len(jobs)} endpoints",flush=True)
        nf=pd.DataFrame(rows)
        nf.to_csv(cache,index=False);meta.write_text(json.dumps(identity,indent=2))
    for w in (50,100):
        n=nf[nf.window==w].copy();n.index=pd.to_datetime(n.date)
        features[f"null_z{w}"]=n.null_z
        features[f"null_delta{w}"]=features[f"null_z{w}"].diff()
    features=features.dropna()
    features["complete_month"]=features.index.to_period("M") < prices.index[-1].to_period("M")
    for name,(h,threshold) in HORIZONS.items():
        records=[forward_label(prices,prices.index.get_loc(d),h,threshold) for d in features.index]
        features[name+"_y"]=[v[0] for v in records]
        features[name+"_end"]=[v[1] for v in records]
        features[name+"_loss"]=[v[2] for v in records]
    features.to_csv(OUT/"monthly_features_labels.csv",index_label="date")
    return features


def fitted_prediction(x_train, y_train, x_test):
    scaler=RobustScaler().fit(x_train)
    a=np.clip(scaler.transform(x_train),-5,5)
    b=np.clip(scaler.transform(x_test),-5,5)
    model=LogisticRegression(C=.1,solver="lbfgs",max_iter=2000,random_state=SEED)
    model.fit(a,y_train)
    train_p=model.predict_proba(a)[:,1]
    probability=float(model.predict_proba(b)[0,1])
    threshold=float(np.quantile(train_p,.9))
    return probability,threshold


def rolling_predictions(features):
    records=[];skipped=[]
    for target,(h,_) in HORIZONS.items():
        for i,(date,row) in enumerate(features.iterrows()):
            if date < pd.Timestamp("1999-01-01"):continue
            train=features[(features.complete_month)&(features[target+"_end"]<date)&features[target+"_y"].notna()]
            n=len(train);positive=int(train[target+"_y"].sum())
            if n<84 or positive<5 or n-positive<5:
                skipped.append({"date":str(date.date()),"target":target,"n_train":n,"positive_train":positive,"reason":"insufficient mature training months/classes"})
                continue
            common={"date":str(date.date()),"target":target,"y":row[target+"_y"],
                    "label_end":str(row[target+"_end"]),"future_loss":row[target+"_loss"],
                    "complete_month":bool(row.complete_month),"drawdown252":row.drawdown252,
                    "n_train":n,"positive_train":positive,"max_train_label_end":str(train[target+"_end"].max().date())}
            assert train[target+"_end"].max()<date
            records.append({**common,"model":"base_rate","probability":positive/n,"threshold":np.nan,"alarm":False})
            for name,cols in MODELS.items():
                p,threshold=fitted_prediction(train[cols],train[target+"_y"].astype(int),features.loc[[date],cols])
                records.append({**common,"model":name,"probability":p,"threshold":threshold,"alarm":p>=threshold})
            if i%60==0:print(target,str(date.date()),"train",n,"positive",positive,flush=True)
    predictions=pd.DataFrame(records)
    predictions.to_csv(OUT/"predictions.csv",index=False)
    pd.DataFrame(skipped).to_csv(OUT/"skipped.csv",index=False)
    return predictions


def score(y,p):
    y=np.asarray(y).astype(int);p=np.asarray(p)
    return {"brier":brier_score_loss(y,p),"log_loss":log_loss(y,p,labels=[0,1]),
            "auc":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            "ap":average_precision_score(y,p) if y.sum() else np.nan}


def evaluate(predictions):
    d=predictions.copy();d["date"]=pd.to_datetime(d.date)
    d=d[d.y.notna()&d.complete_month].copy()
    tables=[];episodes=[];bootstrap=[]
    for target in HORIZONS:
        q=d[d.target==target]
        for period,subset in [("all",q),("pre2017",q[q.date<"2017-01-01"]),("post2017",q[q.date>="2017-01-01"]),
                              ("not_in_large_drawdown",q[q.drawdown252>-.1])]:
            if not len(subset):continue
            ybase=subset[subset.model=="base_rate"].set_index("date")
            base_brier=score(ybase.y,ybase.probability)["brier"]
            for model,g in subset.groupby("model",sort=False):
                metrics=score(g.y,g.probability)
                alarm=g.alarm.astype(bool).to_numpy();y=g.y.to_numpy().astype(int)
                tp=((y==1)&alarm).sum();fp=((y==0)&alarm).sum()
                metrics.update({"target":target,"period":period,"model":model,"n":len(g),"positive":int(y.sum()),
                                "prevalence":float(y.mean()),"brier_skill":1-metrics["brier"]/base_brier,
                                "alarm_rate":float(alarm.mean()) if model!="base_rate" else np.nan,
                                "precision":tp/alarm.sum() if alarm.sum() else np.nan,
                                "recall":tp/y.sum() if y.sum() and model!="base_rate" else np.nan,
                                "false_positive_rate":fp/(y==0).sum() if (y==0).sum() and model!="base_rate" else np.nan,
                                "first":str(g.date.min().date()),"last":str(g.date.max().date())})
                tables.append(metrics)
            if period in ("all","post2017"):
                for new,old in [("tda_scaled","tda_raw"),("tda_null","tda_raw"),("market","tda_raw"),
                                ("market_raw","market"),("market_scaled","market"),("market_null","market")]:
                    g=subset.pivot(index="date",columns="model",values="probability")
                    y=ybase.y.reindex(g.index).to_numpy().astype(int)
                    a=g[new].to_numpy();b=g[old].to_numpy();n=len(y)
                    rng=np.random.default_rng(SEED)
                    bs=[];aps=[]
                    for _ in range(500):
                        starts=rng.integers(0,max(1,n-24+1),size=int(np.ceil(n/24)))
                        indices=np.concatenate([np.arange(s,min(s+24,n)) for s in starts])[:n]
                        if len(indices)<n or len(np.unique(y[indices]))<2:continue
                        yy=y[indices];aa=a[indices];bb=b[indices]
                        bs.append(np.mean((aa-yy)**2-(bb-yy)**2))
                        aps.append(average_precision_score(yy,aa)-average_precision_score(yy,bb))
                    bootstrap.append({"target":target,"period":period,"new":new,"old":old,
                                      "delta_brier":float(np.mean((a-y)**2-(b-y)**2)),
                                      "brier_lo":np.quantile(bs,.025),"brier_hi":np.quantile(bs,.975),
                                      "delta_ap":average_precision_score(y,a)-average_precision_score(y,b),
                                      "ap_lo":np.quantile(aps,.025),"ap_hi":np.quantile(aps,.975),"valid_bootstrap":len(bs)})
        ys=q[q.model=="base_rate"].sort_values("date")
        active=[]
        for row in ys.itertuples():
            if row.y==1:active.append(row.date)
            elif active:
                episodes.append({"target":target,"first_forecast_month":str(active[0].date()),"last_forecast_month":str(active[-1].date()),"n_positive_months":len(active)})
                active=[]
        if active:episodes.append({"target":target,"first_forecast_month":str(active[0].date()),"last_forecast_month":str(active[-1].date()),"n_positive_months":len(active)})
    metrics=pd.DataFrame(tables);metrics.to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/"paired_block_bootstrap.csv",index=False)
    pd.DataFrame(episodes).to_csv(OUT/"positive_forecast_runs.csv",index=False)
    print(metrics.loc[(metrics.target=="12m_20pct")&metrics.period.isin(["all","post2017"]),["period","model","n","positive","brier","brier_skill","auc","ap","alarm_rate","precision","recall"]].round(4).to_string(index=False))


def simulation():
    rng=np.random.default_rng(SEED);rows=[]
    for rho in [0,.5,.9,.99]:
        for i in range(100):
            f=rng.normal(size=(50,1));e=rng.normal(size=(50,4))
            unit=np.sqrt(rho)*f+np.sqrt(1-rho)*e
            for sigma in [1.,3.]:
                x=sigma*unit
                l1=landscape_norms(diagram_for(x))[0]
                rows.append({"rho":rho,"sigma":sigma,"replicate":i,"l1":l1,"scale_free_l1":l1/np.var(x,axis=0,ddof=1).mean()})
    pd.DataFrame(rows).to_csv(OUT/"simulation.csv",index=False)


def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    f=make_features();simulation();p=rolling_predictions(f);evaluate(p)
    identity={"protocol_sha256":hashlib.sha256((ROOT/"IMPROVEMENT_PROTOCOL.md").read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed":SEED,"null_replicates":N_NULL,"feature_rows":len(f),"elapsed_seconds":time.time()-start,
              "evaluation":"historical walk-forward; not a blinded or prospective trial; stock tail loss only"}
    (OUT/"run_manifest.json").write_text(json.dumps(identity,indent=2))
    print("Completed in",round(time.time()-start,1),"seconds",flush=True)


if __name__=="__main__":main()
