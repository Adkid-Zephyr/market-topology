"""Journal-style figures: neutral panel labels, methods in external captions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from topology_visuals import landscape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1,
                     'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
                     'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
COLORS={'raw_tda':'#234e70','diagram_kernel':'#478f82','delay':'#b67838','base_rate':'#777777'}


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def structure():
    js=json.loads((ROOT/'topology_visuals/snapshot_data.json').read_text())
    dates=['2017-06-30','2000-03-09','2008-09-12','2026-09-28']
    diagrams=[np.array(js['50_'+d]['h1_diagram']) for d in dates]
    limit=max(d.max() for d in diagrams)*1.06;grid=np.linspace(0,limit,800)
    ls=[landscape(d,grid,5) for d in diagrams];height=max(v.max() for v in ls)*1.12
    fig,axes=plt.subplots(2,4,figsize=(7.3,3.8))
    for i,(date,d,l) in enumerate(zip(dates,diagrams,ls)):
        a=axes[0,i];a.scatter(d[:,0],d[:,1],s=17,color='#234e70',linewidths=.3,edgecolors='white',zorder=3)
        a.plot([0,limit],[0,limit],color='#888888',ls='--',lw=.7)
        a.set(xlim=(0,limit),ylim=(0,limit),xlabel='Birth distance')
        a.set_aspect('equal',adjustable='box')
        a.set_title(f'({chr(97+i)}) {date}',loc='left',pad=6)
        a.xaxis.set_major_locator(MaxNLocator(3));a.yaxis.set_major_locator(MaxNLocator(3))
        if i==0:a.set_ylabel('Death distance')
        a=axes[1,i]
        for k in range(5):a.plot(grid,l[k],color='#234e70',alpha=1-k*.15,lw=1 if k==0 else .7)
        a.set(xlim=(0,limit),ylim=(0,height),xlabel='Filtration distance')
        a.text(.96,.93,f"$L^1={js['50_'+date]['l1']:.4f}$",ha='right',va='top',transform=a.transAxes,fontsize=9)
        a.xaxis.set_major_locator(MaxNLocator(3));a.yaxis.set_major_locator(MaxNLocator(3))
        if i==0:a.set_ylabel('Landscape height')
    fig.subplots_adjust(left=.07,right=.995,bottom=.11,top=.93,hspace=.42,wspace=.38)
    save(fig,'fig1_persistence')


def uncertainty():
    p=pd.read_csv(ROOT/'topology_cv_results/predictions.csv',parse_dates=['date'])
    p=p[(p.target=='12m_20pct')&p.y.notna()]
    names=['raw_tda','landscape','delay','diagram_kernel','market','market_landscape','market_delay','market_kernel','inner_selector']
    labels=['Original TDA features','Landscape representation','Delay-embedded TDA','Persistence-diagram kernel','Market features','Market + landscapes','Market + delay TDA','Market RBF kernel','Inner-validation selector']
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.9),sharey=True)
    records=[];lim=[]
    for a,period,sub in [(axes[0],'all',p),(axes[1],'post2017',p[p.date>='2017-01-01'])]:
        wide=sub.pivot(index='date',columns='model',values='probability')
        y=sub[sub.model=='base_rate'].set_index('date').y.reindex(wide.index).to_numpy()
        base=wide.base_rate.to_numpy();n=len(y);rng=np.random.default_rng(20260929)
        ix=[]
        for _ in range(1000):
            starts=rng.integers(0,n-24+1,size=int(np.ceil(n/24)))
            ix.append(np.concatenate([np.arange(s,s+24) for s in starts])[:n])
        ix=np.array(ix)
        for j,name in enumerate(names):
            pred=wide[name].to_numpy();loss=(pred-y)**2-(base-y)**2
            ci=np.quantile(loss[ix].mean(axis=1),[.025,.975]);estimate=loss.mean()
            a.plot(ci,[j,j],color='#234e70',lw=1)
            a.scatter([estimate],[j],color='#234e70',s=19,zorder=3)
            records.append({'period':period,'model':name,'n':n,'delta_brier':estimate,'ci_low':ci[0],'ci_high':ci[1]})
            lim.extend(ci)
        a.axvline(0,color='#555555',ls='--',lw=.8)
        a.set_yticks(range(len(names)),labels);a.set_xlabel(r'$\Delta$ Brier score')
        a.set_title(('(a) 2003–2025' if period=='all' else '(b) 2017–2025')+f' ($n={n}$)',loc='left')
        a.xaxis.set_major_locator(MaxNLocator(4));a.grid(axis='x',alpha=.15,lw=.5)
    lo=min(lim);hi=max(lim);pad=(hi-lo)*.06
    for a in axes:a.set_xlim(lo-pad,hi+pad)
    axes[0].invert_yaxis();fig.subplots_adjust(left=.26,right=.995,bottom=.15,top=.91,wspace=.15)
    save(fig,'fig2_prediction_error')
    pd.DataFrame(records).to_csv(ROOT/'publication/error_intervals.csv',index=False)


def history():
    fs={w:pd.read_csv(ROOT/'results'/f'daily_w{w}.csv',index_col=0,parse_dates=True) for w in [50,100]}
    fig,ax=plt.subplots(2,1,figsize=(7.3,4),gridspec_kw={'height_ratios':[1,1]})
    for w,color in [(50,'#234e70'),(100,'#b67838')]:
        f=fs[w];ax[0].plot(f.index,f.l1,lw=.65,color=color,label=f'{w}-day window')
        q=f.loc['2024-01-01':];ax[1].plot(q.index,q.l1_prior_pct,lw=.9,color=color)
    ax[0].set_title('(a) Persistence-landscape norms',loc='left');ax[0].set_ylabel(r'$L^1$ norm')
    ax[0].legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    ax[1].set_title('(b) Historical percentile',loc='left');ax[1].set_ylabel('Percentile');ax[1].set_ylim(0,103)
    ax[1].set_xlabel('Date')
    import matplotlib.dates as md
    ax[0].xaxis.set_major_locator(md.YearLocator(5));ax[0].xaxis.set_major_formatter(md.DateFormatter('%Y'))
    ax[1].xaxis.set_major_locator(md.MonthLocator(interval=6));ax[1].xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
    for a in ax:a.grid(alpha=.15,lw=.5)
    fig.subplots_adjust(left=.10,right=.99,bottom=.13,top=.94,hspace=.42)
    save(fig,'fig3_history')


if __name__=='__main__':structure();uncertainty();history()
