import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from persim import sliced_wasserstein
from topology_cv import load_data,prepare,predict_prepared,split_indices,representation,sw_distance,HORIZONS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"topology_cv_results"


def main():
    f,data=load_data()
    p=pd.read_csv(OUT/"predictions.csv",parse_dates=["date","train_end"])
    s=pd.read_csv(OUT/"selection.csv")
    folds=pd.read_csv(OUT/"folds.csv",parse_dates=["cutoff","max_train_label_end","max_validation_label_end"])
    assert (folds.max_train_label_end<folds.cutoff).all()
    inner=folds[folds.kind=="inner"]
    assert all(row.max_validation_label_end<pd.Timestamp(row.outer_year,1,1) for row in inner.itertuples())
    assert (p.train_end<pd.to_datetime(p.outer_year.astype(str)+"-01-01")).all()
    assert (p.groupby(["target","date"]).model.nunique()==10).all()
    assert p.probability.between(0,1).all()
    d=json.loads((OUT/"diagrams.json").read_text())
    reference_checks=0
    for i,j in [(0,10),(90,195),(300,420)]:
        for c in range(4):
            a=np.asarray(d[i][c]).reshape(-1,2);b=np.asarray(d[j][c]).reshape(-1,2)
            np.testing.assert_allclose(sw_distance(a,b),sliced_wasserstein(a,b,M=16),rtol=1e-6,atol=1e-7)
            reference_checks+=1
    returns=pd.read_csv(ROOT/"data/returns.csv",index_col=0,parse_dates=True)
    for date in ['2000-01-31','2008-08-29','2026-09-28']:
        i=f.index.get_loc(date)
        row=representation((date,returns.loc[:date].to_numpy()))
        np.testing.assert_allclose(row['landscape'],data['landscape'][i],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(row['delay'],data['delay'][i],rtol=1e-10,atol=1e-10)
    refits=0
    for target in HORIZONS:
        for year in [2008,2017,2022]:
            cutoff=pd.Timestamp(year,1,1);end=pd.Timestamp(year+1,1,1)
            tr,te=split_indices(f,target,cutoff,cutoff,end)
            n=te[-1]+1
            ff=f.iloc[:n]
            dd={"landscape":data['landscape'][:n],"delay":data['delay'][:n],"distances":data['distances'][:,:n,:n]}
            y=ff.iloc[tr][target+'_y'].astype(int).to_numpy()
            for model in ['landscape','market_delay','diagram_kernel','market_kernel']:
                chosen=s[(s.target==target)&(s.outer_year==year)&(s.model==model)].iloc[0]
                param=ast.literal_eval(chosen.parameter)
                prepared=prepare(ff,dd,model,tr,te)
                pp=predict_prepared(model,param,prepared,y)
                q=p[(p.target==target)&(p.outer_year==year)&(p.model==model)].sort_values('date')
                np.testing.assert_allclose(pp,q.probability.to_numpy(),rtol=1e-6,atol=1e-8)
                refits+=1
    for (target,year),g in s.groupby(['target','outer_year']):
        expected='base_rate' if g.fallback.all() else g.loc[g.inner_brier.idxmin(),'model']
        q=p[(p.target==target)&(p.outer_year==year)&(p.model=='inner_selector')]
        assert (q.chosen_model==expected).all()
    audit=json.loads((OUT/'representation_audit.json').read_text())
    rows=[r for v in audit for r in v['audit']]
    result={'status':'passed','outer_blocks':int((folds.kind=='outer').sum()),'inner_folds':len(inner),
            'mature_scored_outer_blocks':p[p.y.notna()][['target','outer_year']].drop_duplicates().shape[0],
            'persim_reference_checks':reference_checks,'prefix_feature_recomputations':3,'prefix_model_refits':refits,
            'all_inner_and_outer_label_maturity_checks':True,'all_model_selection_uses_inner_only':True,
            'landscape_grid_outside_bars':sum(r['deaths_beyond_grid'] for r in rows),
            'landscape_max_normalized_death':max(r['max_death'] for r in rows),
            'caveat':'Five landscape layers / finite grid are approximations; complete finite diagrams used in diagram kernel.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
