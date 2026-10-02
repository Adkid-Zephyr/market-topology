"""Static research figures, generated directly from saved daily outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"figures"
OUT.mkdir(exist_ok=True)
F={w:pd.read_csv(ROOT/"results"/f"daily_w{w}.csv",index_col=0,parse_dates=True) for w in (50,100)}
P=pd.read_csv(ROOT/"data"/"prices_calendar.csv",index_col=0,parse_dates=True)
C={50:"#086788",100:"#E07A3F"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.grid":True,"grid.alpha":.18,"axes.titleweight":"bold","figure.facecolor":"#fafbfC","axes.facecolor":"#fafbfC"})


def save(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=175,bbox_inches="tight")
    fig.savefig(OUT/(name+".svg"),bbox_inches="tight")
    plt.close(fig)


def overview():
    fig,ax=plt.subplots(4,1,figsize=(14,11),sharex=True,gridspec_kw={"height_ratios":[1,1.1,1.1,1]})
    q=P.loc[F[50].index.min():].SP500
    ax[0].plot(q.index,q,color="#243B53",lw=1)
    ax[0].set_yscale("log");ax[0].set_ylabel("S&P 500\n(log scale)")
    for w,f in F.items():
        ax[1].plot(f.index,f.l1,color=C[w],lw=.85,alpha=.85,label=f"{w} sessions")
        ax[2].plot(f.index,f.l1_prior_pct,color=C[w],lw=.75,alpha=.85)
        ax[3].plot(f.index,f.psd_low500_tau250,color=C[w],lw=.85)
    ax[1].set_ylabel("Landscape L1\n(percent-return units squared)");ax[1].legend(loc="upper right",ncol=2)
    ax[2].set_ylabel("L1 historical percentile\n(earlier values only)");ax[2].set_ylim(0,103);ax[2].axhline(95,color="#a33939",ls="--",lw=.8)
    ax[3].set_ylabel("Low-frequency PSD trend\nKendall tau, 250 sessions");ax[3].set_ylim(-1.05,1.05);ax[3].axhline(0,color="#666",lw=.7)
    for a in ax:
        for start,end in [("2000-03-10","2002-10-09"),("2007-10-09","2009-03-09"),("2020-02-19","2020-03-23"),("2022-01-03","2022-10-12")]:
            a.axvspan(pd.Timestamp(start),pd.Timestamp(end),color="#939dac",alpha=.12,zorder=0)
    ax[-1].xaxis.set_major_locator(mdates.YearLocator(4));ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Four-index market topology | 1987–2026",fontsize=20,x=.09,ha="left",y=.985)
    fig.text(.09,.948,"S&P 500 · DJIA · NASDAQ Composite · Russell 2000   /   Latest completed close: 28 Sep 2026",fontsize=11,color="#4a5d70")
    fig.text(.09,.014,"Shading marks selected historical drawdowns. Percentiles and trend coefficients are descriptive, not crisis probabilities.\nData: Yahoo Finance; pre-1992 DJIA from James E. Gentle / George Mason University. Exact full H1 landscape norms.",fontsize=9,color="#4a5d70")
    fig.subplots_adjust(top=.915,bottom=.075,hspace=.15,left=.1,right=.97)
    save(fig,"01_full_history")


def current():
    fig,ax=plt.subplots(4,1,figsize=(13,10.5),sharex=True)
    start="2024-09-01"
    q=P.loc[start:].SP500
    ax[0].plot(q.index,q,color="#243B53",lw=1.5);ax[0].set_ylabel("S&P 500")
    specs=[("l1_prior_pct","Raw L1\nhistorical percentile"),("scale_free_l1_prior_pct","L1 / mean daily variance\nhistorical percentile"),("psd_low500_tau250","Low-frequency PSD trend\nKendall tau")]
    for a,(key,label) in zip(ax[1:],specs):
        for w,f in F.items():
            q=f.loc[start:]
            a.plot(q.index,q[key],lw=1.3,color=C[w],label=f"{w} sessions")
            a.scatter(q.index[-1],q[key].iloc[-1],s=30,color=C[w],zorder=5)
        a.set_ylabel(label)
    for a in ax[1:3]:a.set_ylim(0,105);a.axhline(95,ls="--",lw=.8,color="#a33939")
    ax[3].axhline(0,color="#777",lw=.7);ax[3].set_ylim(-1.05,1.05)
    ax[1].legend(loc="lower left",ncol=2)
    ax[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle("Current reading: elevated short-window structure, mixed across windows",fontsize=17,x=.11,ha="left",y=.988)
    fig.text(.11,.952,f"28 Sep 2026  |  Raw L1 percentile: 50-day {F[50].l1_prior_pct.iloc[-1]:.1f}   ·   100-day {F[100].l1_prior_pct.iloc[-1]:.1f}",fontsize=12,color="#4a5d70")
    fig.text(.11,.015,"Dashed 95th percentile is a descriptive reference, not a validated alarm threshold. Scale adjustment is an added diagnostic.\nNo future values enter a daily observation; current percentiles use all earlier observations.",fontsize=9,color="#4a5d70")
    fig.subplots_adjust(top=.916,bottom=.078,left=.12,right=.97,hspace=.19)
    save(fig,"02_current")


def event_comparison():
    endpoints=[("Dot-com: pre-event","2000-03-09"),("Lehman: pre-event","2008-09-12"),("Current","2026-09-28")]
    fig,axes=plt.subplots(3,3,figsize=(15,9.5),sharex=True,sharey="row")
    for j,(name,date) in enumerate(endpoints):
        axes[0,j].set_title(name+"\n"+date,fontsize=12,pad=12)
        for w,f in F.items():
            q=f.loc[:date].tail(750)
            t=np.arange(-len(q)+1,1)
            for i,key in enumerate(["l1_prior_pct","scale_free_l1_prior_pct","psd_low500_tau250"]):
                axes[i,j].plot(t,q[key],color=C[w],lw=1.05,label=f"{w} sessions")
                axes[i,j].scatter(0,q[key].iloc[-1],s=20,color=C[w])
        for a in axes[:2,j]:a.set_ylim(0,104);a.axhline(95,ls="--",lw=.7,color="#a33939")
        axes[2,j].set_ylim(-1.05,1.05);axes[2,j].axhline(0,color="#777",lw=.7)
        axes[2,j].set_xlabel("Trading sessions before endpoint")
    axes[0,0].set_ylabel("Raw L1\nhistorical percentile")
    axes[1,0].set_ylabel("Scale-adjusted L1\nhistorical percentile")
    axes[2,0].set_ylabel("Low-frequency PSD trend\nKendall tau")
    axes[0,2].legend(loc="lower left")
    fig.suptitle("Historical comparison with identical calculations",fontsize=20,x=.09,ha="left",y=.99)
    fig.text(.09,.021,"Each historical observation is computed from its own past. Event-day returns are excluded in the first two columns.\nThe PSD curve depends on preprocessing; raw demeaned periodogram is shown. See spectral_sensitivity.csv for alternatives.",fontsize=9,color="#4a5d70")
    fig.subplots_adjust(top=.865,bottom=.105,left=.09,right=.98,hspace=.17,wspace=.13)
    save(fig,"03_event_comparison")


def paper_span():
    fig,ax=plt.subplots(2,1,figsize=(13,7.5),sharex=True)
    # Keep raw units identical across both historical episodes.
    for w,f in F.items():
        part=f.loc["1998":"2010"]
        ax[0].plot(part.index,part.l1,color=C[w],lw=1,label=f"L1, {w} sessions")
        ax[1].plot(part.index,part.l2,color=C[w],lw=1,label=f"L2, {w} sessions")
    for a in ax:
        for date,text in [("2000-03-10","Dot-com reference"),("2008-09-15","Lehman")]:
            a.axvline(pd.Timestamp(date),ls="--",lw=.8,color="#8a3636")
            a.text(pd.Timestamp(date),a.get_ylim()[1]*.71,text,fontsize=9,ha="left",color="#8a3636")
        a.legend(loc="upper right",ncol=2)
    ax[0].set_ylabel("L1 = sum(persistence²) / 4")
    ax[1].set_ylabel("L2 = sqrt(sum(persistence³) / 12)")
    fig.suptitle("Paper-era reimplementation: topology grows around both episodes",fontsize=18,x=.1,ha="left",y=.99)
    fig.text(.1,.015,"All four indices retain their actual return scales. A visible peak near a crash does not by itself establish advance prediction.",fontsize=10,color="#4a5d70")
    fig.subplots_adjust(top=.92,bottom=.09,left=.1,right=.97,hspace=.16)
    save(fig,"04_paper_era")


if __name__=="__main__":
    overview();current();event_comparison();paper_span()
    print("Wrote four PNG/SVG figure pairs")
