"""Independent saved-result audits for the monthly forward experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from improvement_experiment import HORIZONS, MODELS, fitted_prediction, null_metric

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"improvement_results"


def main():
    f=pd.read_csv(OUT/"monthly_features_labels.csv",index_col=0,parse_dates=True)
    p=pd.read_csv(OUT/"predictions.csv",parse_dates=["date","max_train_label_end"])
    assert (p.max_train_label_end<p.date).all()
    assert p.probability.between(0,1).all()
    sizes=p.groupby(["target","date"]).model.nunique()
    assert (sizes==8).all()
    prices=pd.read_csv(ROOT/"data/prices_calendar.csv",index_col=0,parse_dates=True).SP500
    n_labels=0
    for target,(h,threshold) in HORIZONS.items():
        f[target+"_end"]=pd.to_datetime(f[target+"_end"])
        for date,row in f.iterrows():
            position=prices.index.get_loc(date)
            if position+h>=len(prices):
                assert pd.isna(row[target+"_y"])
                continue
            minimum=min(prices.iloc[position+1:position+h+1])
            assert row[target+"_y"]==int(minimum<=prices.loc[date]*(1-threshold))
            assert row[target+"_end"]==prices.index[position+h]
            n_labels+=1
    retrained=0
    for target in HORIZONS:
        for date in ["2008-08-29","2019-12-31","2022-01-31","2025-08-29"]:
            t=pd.Timestamp(date)
            past=f.loc[:t].copy()
            train=past[(past.complete_month)&(past[target+"_end"]<t)&past[target+"_y"].notna()]
            for name in ["tda_raw","tda_null","market_scaled"]:
                cols=MODELS[name]
                prob,threshold=fitted_prediction(train[cols],train[target+"_y"].astype(int),past.loc[[t],cols])
                saved=p[(p.target==target)&(p.date==t)&(p.model==name)].iloc[0]
                np.testing.assert_allclose([prob,threshold],[saved.probability,saved.threshold],rtol=1e-9,atol=1e-10)
                retrained+=1
    null=pd.read_csv(OUT/"null_features.csv")
    returns=pd.read_csv(ROOT/"data/returns.csv",index_col=0,parse_dates=True)
    for date in ["2008-08-29","2026-09-28"]:
        for w in (50,100):
            actual=null_metric((date,w,returns.loc[:date].tail(w).to_numpy()))
            saved=null[(null.date==date)&(null.window==w)].iloc[0]
            np.testing.assert_allclose(actual["null_z"],saved.null_z,rtol=1e-10,atol=1e-10)
    metric=pd.read_csv(OUT/"metrics.csv")
    for row in metric[(metric.period=="post2017")].itertuples():
        q=p[(p.target==row.target)&(p.model==row.model)&p.y.notna()&p.complete_month&(p.date>="2017-01-01")]
        np.testing.assert_allclose(np.mean((q.probability-q.y)**2),row.brier,atol=1e-12)
    result={"status":"passed","all_train_label_ends_before_forecasts":True,"all_models_use_identical_forecast_dates":True,
            "independently_checked_labels":n_labels,"prefix_only_model_refits":retrained,"null_recomputations":4,
            "post2017_brier_manual_checks":16,"unmatured_labels_retained_as_unknown":True}
    (OUT/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
