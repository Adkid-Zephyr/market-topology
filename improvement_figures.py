from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"improvement_figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":.17,"figure.facecolor":"#fafbfc","axes.facecolor":"#fafbfc"})


def save(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=170,bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight")
    plt.close(fig)


def mechanism():
    d=pd.read_csv(ROOT/"improvement_results/simulation.csv")
    g=d.groupby(["rho","sigma"])[["l1","scale_free_l1"]].mean()
    fig,ax=plt.subplots(1,2,figsize=(12.5,5))
    x=np.arange(4)
    for sigma,c in [(1,"#086788"),(3,"#de7739")]:
        for a,key in zip(ax,["l1","scale_free_l1"]):
            vals=g.xs(sigma,level="sigma")[key]
            a.plot(x,vals,marker="o",color=c,lw=2,label=f"Volatility scale = {sigma}")
            a.set_yscale("log");a.set_xticks(x,["0","0.50","0.90","0.99"])
            a.set_xlabel("Cross-market correlation (rho)")
    ax[0].set_title("Raw topology mixes scale and geometry",pad=13);ax[0].set_ylabel("Mean landscape L1 (log axis)")
    ax[0].legend(loc="lower left",fontsize=9)
    ax[1].set_title("Scale correction does not remove correlation effects",pad=13);ax[1].set_ylabel("Mean L1 / average variance (log axis)")
    fig.suptitle("A structural weakness: synchronized markets can have smaller loops",fontsize=17,y=1.03)
    fig.text(.06,-.045,"Controlled Gaussian experiment: 50 points, 4 dimensions, 100 paired draws per setting. Tripling scale multiplies L1 by 9.\nIncreasing synchronization shrinks loop geometry. This simulation tests the metric, not crisis forecasting.",fontsize=10,color="#4a5d70")
    fig.subplots_adjust(wspace=.32,bottom=.17)
    save(fig,"01_mechanism")


def comparison():
    m=pd.read_csv(ROOT/"improvement_results/metrics.csv")
    names=["base_rate","tda_raw","tda_scaled","tda_null","market","market_raw","market_scaled","market_null"]
    labels=["Past event rate","Original TDA features","Scale-adjusted TDA","Null-adjusted TDA","Simple market features","Market + original TDA","Market + scale-adjusted","Market + null-adjusted"]
    fig,ax=plt.subplots(1,2,figsize=(13.5,6.1),sharey=True)
    colors=["#64748b","#c45850","#08849b","#5b72ad","#9a7447","#9a7447","#9a7447","#9a7447"]
    for a,period,title in zip(ax,["all","post2017"],["All scored months: 2001–2025","Beyond original paper data: 2017–2025"]):
        q=m[(m.target=="12m_20pct")&(m.period==period)].set_index("model").loc[names]
        a.barh(np.arange(8),q.brier,color=colors,height=.65)
        a.axvline(q.loc["base_rate","brier"],color="#26364a",ls="--",lw=1)
        for i,val in enumerate(q.brier):a.text(val+.002,i,f"{val:.4f}",va="center",fontsize=10)
        a.set_yticks(np.arange(8),labels)
        a.set_xlim(0,.208);a.set_xlabel("Brier score — lower is better")
        a.set_title(title,fontsize=12,pad=16)
    ax[0].invert_yaxis()
    fig.suptitle("The first improvements do not establish reliable predictive gains",fontsize=18,y=.995)
    fig.text(.03,.015,"Target: S&P 500 loses at least 20% from the forecast close within 252 trading sessions. Purged monthly expanding-window forecasts.\nDashed line: historical event-rate benchmark. 2017+ sample: 104 months, 16 positive months concentrated around 2020 and 2022.",fontsize=10,color="#4a5d70")
    fig.subplots_adjust(left=.21,right=.98,top=.87,bottom=.15,wspace=.19)
    save(fig,"02_comparison")


def timeline():
    p=pd.read_csv(ROOT/"improvement_results/predictions.csv",parse_dates=["date"])
    p=p[(p.target=="12m_20pct")&p.y.notna()&p.complete_month&(p.date>="2017-01-01")]
    fig,ax=plt.subplots(figsize=(13,4.8))
    for model,c,label in [("base_rate","#64748b","Past event rate"),("tda_raw","#c45850","Original TDA features"),("tda_scaled","#08849b","Scale-adjusted TDA")]:
        q=p[p.model==model]
        ax.plot(q.date,q.probability,lw=1.5,color=c,label=label)
    ys=p[p.model=="base_rate"]
    ax.fill_between(ys.date,0,1,where=ys.y==1,step="mid",alpha=.12,color="#803d53",transform=ax.get_xaxis_transform(),label="Forecast months with later 20% loss")
    ax.set_ylim(0,1);ax.set_ylabel("Experimental forecast probability");ax.legend(loc="upper left",ncol=2,fontsize=9)
    ax.set_title("The timing problem: high scores can arrive after the useful warning period",fontsize=16,pad=15)
    fig.text(.09,-.02,"Shading is the eventual outcome of each forecast month, not the dates on which a crisis started. These models failed the performance checks.",fontsize=9,color="#4a5d70")
    save(fig,"03_timing")


if __name__=="__main__":mechanism();comparison();timeline()
