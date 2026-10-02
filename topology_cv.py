"""Topology-preserving representations and nested chronological validation."""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score, roc_auc_score
from ripser import ripser
from topology import landscape_norms
from topology_visuals import landscape
from improvement_experiment import HORIZONS, RAW, MARKET

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"topology_cv_results"
SEED=20260928
GRID=np.linspace(0,6,64)
MODELS=["base_rate","raw_tda","market","landscape","market_landscape","delay","market_delay","diagram_kernel","market_kernel"]
CS=[.01,.1,1.]
KERNEL_PARAMS=[(a,b) for a in [10.,1.,.1] for b in [.5,1.,2.]]


def delay_cloud(s, end, w, tau):
    t=np.arange(end-w+1,end+1)
    ix=t[:,None]-tau*np.arange(4)[None,:]
    if ix.min()<0:raise ValueError("Insufficient past for delay embedding")
    return np.asarray(s)[ix]


def finite_diagrams(x):
    ds=ripser(x,maxdim=1,coeff=2,thresh=np.inf)["dgms"]
    assert np.isinf(ds[0][:,1]).sum()==1
    assert np.isfinite(ds[1]).all()
    return [ds[0][np.isfinite(ds[0][:,1])],ds[1]]


def representation(job):
    date,returns=job
    vector=[];diagrams=[];delay=[];audit=[]
    s=returns[:,0];end=len(s)-1
    for w in (50,100):
        x=returns[-w:]
        scale=np.sqrt(np.var(x,axis=0,ddof=1).mean())
        ds=finite_diagrams(x)
        for hom,d in enumerate(ds):
            diagrams.append(d.tolist())
            normalized=d/scale
            vector.extend(landscape(normalized,GRID,layers=5).ravel())
            audit.append({"window":w,"homology":hom,"bars":len(d),"deaths_beyond_grid":int((normalized[:,1]>6).sum()),
                          "max_death":float(normalized[:,1].max()) if len(d) else 0})
        for tau in (1,5):
            values=[]
            for endpoint in (end,end-63):
                cloud=delay_cloud(s,endpoint,w,tau)
                l1,_=landscape_norms(finite_diagrams(cloud)[1])
                values.append([np.log1p(l1),np.log1p(l1/np.var(cloud,axis=0,ddof=1).mean())])
            delay.extend(values[0]);delay.extend(np.array(values[0])-values[1])
    return {"date":date,"landscape":np.array(vector),"diagrams":diagrams,"delay":np.array(delay),"audit":audit}


def sw_distance(a,b,angles=16):
    a=np.asarray(a,float).reshape(-1,2);b=np.asarray(b,float).reshape(-1,2)
    if len(a)+len(b)==0:return 0.
    theta=np.linspace(-np.pi/2,np.pi/2,angles,endpoint=False)
    directions=np.stack([np.cos(theta),np.sin(theta)])
    ad=np.repeat(a.mean(axis=1,keepdims=True),2,axis=1)
    bd=np.repeat(b.mean(axis=1,keepdims=True),2,axis=1)
    x=np.sort(np.concatenate([a@directions,bd@directions],axis=0),axis=0)
    y=np.sort(np.concatenate([b@directions,ad@directions],axis=0),axis=0)
    return float(np.abs(x-y).sum(axis=0).mean())


_DIAGRAMS=None
def init_diagrams(d):
    global _DIAGRAMS
    _DIAGRAMS=d


def distance_row(i):
    n=len(_DIAGRAMS);result=np.zeros((n-i-1,4))
    for k,j in enumerate(range(i+1,n)):
        for ch in range(4):result[k,ch]=sw_distance(_DIAGRAMS[i][ch],_DIAGRAMS[j][ch])
    return i,result


def load_data():
    f=pd.read_csv(ROOT/"improvement_results/monthly_features_labels.csv",index_col=0,parse_dates=True)
    for target in HORIZONS:f[target+"_end"]=pd.to_datetime(f[target+"_end"])
    returns=pd.read_csv(ROOT/"data/returns.csv",index_col=0,parse_dates=True)
    ident={"returns":hashlib.sha256((ROOT/"data/returns.csv").read_bytes()).hexdigest(),
           "dates":[str(t.date()) for t in f.index],"representation_version":1}
    cache=OUT/"representations.npz";meta=OUT/"representations.meta.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text())==ident:
        a=np.load(cache);v=a["landscape"];delay=a["delay"]
        diagrams=json.loads((OUT/"diagrams.json").read_text())
    else:
        jobs=[(str(t.date()),returns.loc[:t].to_numpy()) for t in f.index]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:rows=list(p.map(representation,jobs,chunksize=10))
        v=np.stack([r["landscape"] for r in rows]);delay=np.stack([r["delay"] for r in rows]);diagrams=[r["diagrams"] for r in rows]
        np.savez_compressed(cache,landscape=v,delay=delay)
        (OUT/"diagrams.json").write_text(json.dumps(diagrams))
        (OUT/"representation_audit.json").write_text(json.dumps([{k:r[k] for k in ["date","audit"]} for r in rows],indent=2))
        meta.write_text(json.dumps(ident,indent=2))
    distpath=OUT/"diagram_distances.npz"
    distmeta=OUT/"diagram_distances.meta.json"
    if distpath.exists() and distmeta.exists() and json.loads(distmeta.read_text())==ident:
        distances=np.load(distpath)["distances"]
    else:
        n=len(f);distances=np.zeros((4,n,n))
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=init_diagrams,initargs=(diagrams,)) as pool:
            for i,r in pool.map(distance_row,range(n-1),chunksize=5):
                distances[:,i,i+1:]=r.T;distances[:,i+1:,i]=r.T
                if i%60==0:print("Distance rows",i+1,"/",n,flush=True)
        np.savez_compressed(distpath,distances=distances);distmeta.write_text(json.dumps(ident,indent=2))
    f.to_csv(OUT/"sample_index_labels.csv",index_label="date")
    return f,{"landscape":v,"delay":delay,"distances":distances}


def split_indices(f,target,cutoff,test_start,test_end,minimum=84,positive=5,maturity_cap=None):
    ends=f[target+"_end"]
    train=np.flatnonzero((f.index<cutoff)&f.complete_month.to_numpy()&(ends<cutoff).to_numpy()&f[target+"_y"].notna().to_numpy())
    test=np.flatnonzero((f.index>=test_start)&(f.index<test_end)&f.complete_month.to_numpy())
    if maturity_cap is not None:test=test[(ends.iloc[test]<maturity_cap).to_numpy()]
    yy=f.iloc[train][target+"_y"]
    if len(train)<minimum or yy.sum()<positive or len(train)-yy.sum()<positive:return None
    if len(test)==0:return None
    return train,test


def transformed(f,data,model,tr,te):
    if model in ("raw_tda","market","market_kernel"):
        x=f[RAW if model=="raw_tda" else MARKET].to_numpy()
    elif model in ("delay","market_delay"):
        x=data["delay"]
        if model=="market_delay":x=np.column_stack([x,f[MARKET].to_numpy()])
    elif model in ("landscape","market_landscape"):
        # Common-scale-normalized landscape; PCA sees only training rows.
        pca=PCA(n_components=8,svd_solver="randomized",random_state=SEED,iterated_power=4).fit(data["landscape"][tr])
        all_pc=pca.transform(data["landscape"])
        x=np.column_stack([all_pc,f[["raw50","raw100"]].to_numpy()])
        if model=="market_landscape":x=np.column_stack([x,f[MARKET].to_numpy()])
    else:raise ValueError(model)
    scaler=RobustScaler().fit(x[tr])
    return np.clip(scaler.transform(x[tr]),-5,5),np.clip(scaler.transform(x[te]),-5,5)


def prepare(f,data,model,tr,te):
    if model=="base_rate":return None
    if model=="diagram_kernel":
        train=data["distances"][:,tr][:,:,tr];test=data["distances"][:,te][:,:,tr]
        upper=np.triu_indices(len(tr),1)
        med=np.array([np.median(a[upper][a[upper]>0]) for a in train])
        med=np.where(np.isfinite(med)&(med>0),med,1.)
        return (train/med[:,None,None]).mean(axis=0),(test/med[:,None,None]).mean(axis=0)
    x,z=transformed(f,data,model,tr,te)
    if model=="market_kernel":
        d=cdist(x,x,"sqeuclidean");e=cdist(z,x,"sqeuclidean")
        upper=d[np.triu_indices(len(tr),1)];med=np.median(upper[upper>0])
        return d/med,e/med
    return x,z


def predict_prepared(model,param,prepared,y):
    prior=float(np.mean(y))
    if model=="base_rate":raise ValueError("base_rate handled by caller")
    a,b=prepared
    if model.endswith("kernel"):
        alpha,bandwidth=param
        k=np.exp(-a/bandwidth);kt=np.exp(-b/bandwidth)
        weights=solve(k+alpha*np.eye(len(y)),np.asarray(y)-prior,assume_a="pos",check_finite=False)
        return np.clip(prior+kt@weights,0,1)
    lr=LogisticRegression(C=param,max_iter=2000,solver="lbfgs",random_state=SEED)
    lr.fit(a,y)
    return lr.predict_proba(b)[:,1]


def cv(f,data):
    predictions=[];selection=[];foldlog=[]
    for target in HORIZONS:
        for year in range(2003,2027):
            cutoff=pd.Timestamp(year,1,1);finish=pd.Timestamp(year+1,1,1)
            outer=split_indices(f,target,cutoff,cutoff,finish)
            if outer is None:continue
            train,test=outer;y=f.iloc[train][target+"_y"].to_numpy().astype(int)
            # Only data and outcomes known at the outer training cutoff may tune models.
            inner=[]
            for k in (6,4,2):
                start=pd.Timestamp(year-k,1,1);end=pd.Timestamp(year-k+2,1,1)
                split=split_indices(f,target,start,start,end,minimum=60,positive=3,maturity_cap=cutoff)
                if split is not None and len(split[1])>=6:
                    inner.append(split)
                    foldlog.append({"target":target,"outer_year":year,"kind":"inner","cutoff":str(start.date()),
                                    "n_train":len(split[0]),"n_validation":len(split[1]),
                                    "max_train_label_end":str(f.iloc[split[0]][target+"_end"].max().date()),
                                    "max_validation_label_end":str(f.iloc[split[1]][target+"_end"].max().date())})
            foldlog.append({"target":target,"outer_year":year,"kind":"outer","cutoff":str(cutoff.date()),
                            "n_train":len(train),"n_validation":len(test),
                            "max_train_label_end":str(f.iloc[train][target+"_end"].max().date())})
            score_by_model={};pred_by_model={}
            for model in MODELS:
                params=[None] if model=="base_rate" else (KERNEL_PARAMS if model.endswith("kernel") else CS)
                totals={str(param):[] for param in params}
                for ti,vi in inner:
                    yi=f.iloc[ti][target+"_y"].to_numpy().astype(int)
                    yv=f.iloc[vi][target+"_y"].to_numpy().astype(int)
                    prep=prepare(f,data,model,ti,vi)
                    for param in params:
                        pp=np.repeat(yi.mean(),len(vi)) if model=="base_rate" else predict_prepared(model,param,prep,yi)
                        totals[str(param)].append(brier_score_loss(yv,pp))
                if inner:
                    chosen=min(params,key=lambda z:np.mean(totals[str(z)]))
                    cvscore=float(np.mean(totals[str(chosen)]))
                else:
                    chosen=None if model=="base_rate" else ((1.,1.) if model.endswith("kernel") else .1)
                    cvscore=np.nan
                score_by_model[model]=cvscore
                prep=prepare(f,data,model,train,test)
                pp=np.repeat(y.mean(),len(test)) if model=="base_rate" else predict_prepared(model,chosen,prep,y)
                pred_by_model[model]=pp
                selection.append({"target":target,"outer_year":year,"model":model,"parameter":str(chosen),
                                  "inner_brier":cvscore,"inner_folds":len(inner),"fallback":not bool(inner),
                                  "candidate_scores":json.dumps({k:float(np.mean(v)) if v else None for k,v in totals.items()})})
                for idx,prob in zip(test,pp):
                    row=f.iloc[idx]
                    predictions.append({"target":target,"date":str(row.name.date()),"outer_year":year,"model":model,
                                        "probability":float(prob),"y":row[target+"_y"],"label_end":str(row[target+"_end"]),
                                        "train_end":str(f.iloc[train][target+"_end"].max().date()),"n_train":len(train)})
            winner=min(MODELS,key=lambda m:score_by_model[m]) if inner else "base_rate"
            for idx,prob in zip(test,pred_by_model[winner]):
                row=f.iloc[idx]
                predictions.append({"target":target,"date":str(row.name.date()),"outer_year":year,"model":"inner_selector",
                                    "probability":float(prob),"y":row[target+"_y"],"label_end":str(row[target+"_end"]),
                                    "train_end":str(f.iloc[train][target+"_end"].max().date()),"n_train":len(train),"chosen_model":winner})
            print(target,year,"train",len(train),"inner",len(inner),"winner",winner,flush=True)
    pd.DataFrame(predictions).to_csv(OUT/"predictions.csv",index=False)
    pd.DataFrame(selection).to_csv(OUT/"selection.csv",index=False)
    pd.DataFrame(foldlog).to_csv(OUT/"folds.csv",index=False)
    return pd.DataFrame(predictions)


def metrics(y,p):
    return {"brier":brier_score_loss(y,p),"log_loss":log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),
            "auc":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            "ap":average_precision_score(y,p) if sum(y)>0 else np.nan}


def evaluate(p):
    p=p[p.y.notna()].copy();p["date"]=pd.to_datetime(p.date)
    output=[];bs=[]
    for target in HORIZONS:
        q=p[p.target==target]
        parts={"all":q,"post2017":q[q.date>="2017-01-01"]}
        for a,b in [(2003,2006),(2007,2011),(2012,2016),(2017,2020),(2021,2026)]:parts[f"{a}-{b}"]=q[(q.outer_year>=a)&(q.outer_year<=b)]
        for year in sorted(q.outer_year.unique()):parts[str(year)]=q[q.outer_year==year]
        for period,s in parts.items():
            if not len(s):continue
            base=s[s.model=="base_rate"]
            base_score=brier_score_loss(base.y,base.probability)
            for model,g in s.groupby("model"):
                m=metrics(g.y.astype(int),g.probability)
                output.append({"target":target,"period":period,"model":model,"n":len(g),"positive":int(g.y.sum()),
                               "first":str(g.date.min().date()),"last":str(g.date.max().date()),
                               **m,"brier_skill":1-m["brier"]/base_score})
            if period not in ("all","post2017"):continue
            wide=s.pivot(index="date",columns="model",values="probability")
            y=base.set_index("date").y.reindex(wide.index).to_numpy().astype(int)
            pairs=[(m,"raw_tda") for m in ["landscape","market_landscape","delay","market_delay","diagram_kernel"]]
            pairs += [("market_landscape","market"),("market_delay","market"),("diagram_kernel","market_kernel"),("inner_selector","base_rate")]
            for new,old in pairs:
                a=wide[new].to_numpy();b=wide[old].to_numpy();n=len(y);rng=np.random.default_rng(SEED);vals=[]
                for _ in range(500):
                    starts=rng.integers(0,max(1,n-24+1),size=int(np.ceil(n/24)))
                    ix=np.concatenate([np.arange(v,min(v+24,n)) for v in starts])[:n]
                    if len(ix)!=n:continue
                    vals.append(np.mean((a[ix]-y[ix])**2-(b[ix]-y[ix])**2))
                bs.append({"target":target,"period":period,"new":new,"old":old,"delta_brier":np.mean((a-y)**2-(b-y)**2),
                           "lo":np.quantile(vals,.025),"hi":np.quantile(vals,.975),"n_bootstrap":len(vals)})
    pd.DataFrame(output).to_csv(OUT/"metrics.csv",index=False)
    pd.DataFrame(bs).to_csv(OUT/"bootstrap.csv",index=False)
    print(pd.DataFrame(output).query("target=='12m_20pct' and period in ['all','post2017']")[["period","model","n","positive","brier","brier_skill","auc","ap"]].round(4).to_string(index=False))


def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    f,data=load_data();print("Representations ready",len(f),flush=True)
    p=cv(f,data);evaluate(p)
    manifest={"protocol_sha256":hashlib.sha256((ROOT/"TOPOLOGY_CV_PROTOCOL.md").read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"elapsed_seconds":time.time()-start,
              "seed":SEED,"n_feature_dates":len(f),"scope":"exploratory nested chronological validation, no final blind holdout"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Elapsed",round(time.time()-start,1),flush=True)


if __name__=="__main__":main()
