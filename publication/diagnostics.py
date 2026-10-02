"""Post-review diagnostics of fixed forecasts; no model fitting or selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,roc_auc_score,average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication'
BIN_EDGES=np.array([0,.05,.10,.20,.40,.60,1.000001])


def main():
    p=pd.read_csv(ROOT/'topology_cv_results/predictions.csv',parse_dates=['date'])
    p=p[(p.target=='12m_20pct')&p.y.notna()].copy()
    summaries=[];bins=[];deletion=[]
    for period,sub in [('all',p),('post2017',p[p.date>='2017-01-01'])]:
        for model,g in sub.groupby('model'):
            y=g.y.to_numpy();prob=g.probability.to_numpy();n=len(g)
            summaries.append({'period':period,'model':model,'n':n,'events':int(y.sum()),
                              'mean_prediction':prob.mean(),'observed_rate':y.mean(),'prediction_sd':prob.std(ddof=1),
                              'brier':np.mean((prob-y)**2),'calibration_mean_error':prob.mean()-y.mean()})
            rng=np.random.default_rng(20260929);resamples=[]
            for _ in range(1000):
                starts=rng.integers(0,n-24+1,size=int(np.ceil(n/24)))
                resamples.append(np.concatenate([np.arange(s,s+24) for s in starts])[:n])
            ix=np.array(resamples)
            for lo,hi in zip(BIN_EDGES[:-1],BIN_EDGES[1:]):
                mask=(prob>=lo)&(prob<hi);count=int(mask.sum())
                if not count:continue
                in_bin=(prob[ix]>=lo)&(prob[ix]<hi);denom=in_bin.sum(axis=1)
                valid=denom>0
                vals=(in_bin*y[ix]).sum(axis=1)[valid]/denom[valid]
                ci=np.quantile(vals,[.025,.975])
                bins.append({'period':period,'model':model,'lo':lo,'hi':min(1,hi),'n':count,
                             'mean_prediction':prob[mask].mean(),'observed_rate':y[mask].mean(),
                             'ci_low':ci[0],'ci_high':ci[1],'valid_resamples':len(vals)})
    # Evaluation deletion only, not leave-event-out retraining or validation.
    phases={'GFC_2007_2009':('2007-01-01','2009-12-31'),'COVID_2019_2020':('2019-01-01','2020-12-31'),
            'repricing_2021_2022':('2021-01-01','2022-12-31')}
    for excluded,(start,end) in phases.items():
        sub=p[~p.date.between(start,end)]
        base=sub[sub.model=='base_rate'];base_brier=brier_score_loss(base.y,base.probability)
        for model,g in sub.groupby('model'):
            brier=brier_score_loss(g.y,g.probability)
            deletion.append({'excluded':excluded,'model':model,'n':len(g),'positive':int(g.y.sum()),'brier':brier,
                             'delta_vs_base':brier-base_brier,'auc':roc_auc_score(g.y,g.probability)})
    pd.DataFrame(summaries).to_csv(OUT/'calibration_summary.csv',index=False)
    b=pd.DataFrame(bins);b.to_csv(OUT/'calibration_bins.csv',index=False)
    pd.DataFrame(deletion).to_csv(OUT/'event_deletion.csv',index=False)
    (OUT/'diagnostics_scope.json').write_text(json.dumps({'type':'fixed-forecast post hoc diagnostics','model_refitting':False,
                                                        'bootstrap_block_months':24,'bootstrap_samples':1000,'bin_edges':BIN_EDGES.tolist(),
                                                        'event_deletion_is_not_leave_event_out_validation':True},indent=2))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':.7,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.25),sharex=True,sharey=True)
    names=[('raw_tda','Original TDA','#234e70'),('delay','Delay TDA','#b67838'),('diagram_kernel','Diagram kernel','#478f82')]
    for a,period,title in zip(axes,['all','post2017'],['(a) 2003–2025','(b) 2017–2025']):
        a.plot([0,1],[0,1],ls='--',lw=.7,color='#888888')
        for model,label,color in names:
            g=b[(b.period==period)&(b.model==model)&(b.n>=5)].sort_values('mean_prediction')
            a.plot(g.mean_prediction,g.observed_rate,'o-',markersize=3.5,lw=.8,color=color,label=label)
        a.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability');a.set_title(title,loc='left')
        a.grid(alpha=.15,lw=.5)
    axes[0].set_ylabel('Observed event frequency');axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    fig.subplots_adjust(left=.09,right=.99,bottom=.15,top=.89,wspace=.13)
    fig.savefig(OUT/'figures/fig4_calibration.png',dpi=300,bbox_inches='tight')
    fig.savefig(OUT/'figures/fig4_calibration.svg',bbox_inches='tight')
    print(pd.DataFrame(summaries).query("period=='post2017'").round(4).to_string(index=False))
    print(pd.DataFrame(deletion).query("model in ['base_rate','raw_tda','diagram_kernel','delay']").round(4).to_string(index=False))


if __name__=='__main__':main()
