"""Audit saved outputs, raw provenance and causal alignment without network requests."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from topology import window_metrics, prior_percentile

ROOT=Path(__file__).resolve().parent


def main():
    sources=json.loads((ROOT/"data"/"sources.json").read_text())
    for item in sources.values():
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()==item["sha256"]
    ret=pd.read_csv(ROOT/"data"/"returns.csv",index_col=0,parse_dates=True)
    assert ret.index.is_unique and ret.index.is_monotonic_increasing
    assert np.isfinite(ret.to_numpy()).all()
    rows=[]
    for w in [50,100]:
        f=pd.read_csv(ROOT/"results"/f"daily_w{w}.csv",index_col=0,parse_dates=True)
        assert len(f)==len(ret)-w+1
        assert f.index.equals(ret.index[w-1:])
        for name in ["l1","l2","mean_daily_variance","scale_free_l1","zscore_l1"]:
            assert np.isfinite(f[name]).all() and (f[name]>=0).all()
        assert f.variance500.first_valid_index()==f.index[499]
        assert f.psd_low500_tau250.first_valid_index()==f.index[748]
        for date in ["1999-12-31","2000-03-09","2008-09-12","2020-02-18","2026-09-28"]:
            if pd.Timestamp(date) < f.index.min():
                continue
            endpoint=f.loc[:date].index[-1]
            available=ret.loc[:endpoint]
            actual=window_metrics(available.tail(w).to_numpy())
            saved=f.loc[endpoint]
            for k,v in actual.items():
                np.testing.assert_allclose(v,saved[k],rtol=1e-9,atol=1e-10)
            past=f.loc[f.index<endpoint,"l1"]
            pct=100*((past<saved.l1).sum()+.5*(past==saved.l1).sum())/len(past)
            np.testing.assert_allclose(pct,saved.l1_prior_pct,atol=1e-10)
            rows.append({"window":w,"date":str(endpoint.date()),"checked":"metrics, prefix recomputation, past-only percentile"})
    a=json.loads((ROOT/"data"/"audit.json").read_text())
    provider_checks = a.get('input_route') != 'user-provided CSV'
    if provider_checks:
        assert a["djia_overlap"]["max_rel_diff"]<1e-8
        assert all(a["fred_crosschecks"][k]["max_rel_diff"]<1e-5 for k in ["SP500","DJIA","NASDAQ"])
    notes=["VIX differs between Yahoo and FRED on 2026-02-06; VIX is excluded from core computation and final charts.",
           "Current 2026-09-28 stock closes are from Yahoo. FRED cross-check reaches 2026-09-25.",
           "The publisher figure's undocumented preprocessing prevents a point-for-point replication claim."]
    if not provider_checks:
        notes = ['User-provided CSV: provider-specific DJIA and FRED cross-checks were not performed.',
                 "The publisher figure's undocumented preprocessing prevents a point-for-point replication claim."]
    result={"status":"passed","raw_source_hashes_checked":len(sources),"sample_checks":rows,
            "provider_crosschecks_performed":provider_checks,"notes":notes}
    (ROOT/"results"/"validation.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":"passed","raw_source_hashes_checked":len(sources),"sample_checks":len(rows),"notes":notes},indent=2))


if __name__=="__main__":main()
