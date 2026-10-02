"""Visualize actual 4-D point clouds, H1 diagrams, barcodes and landscapes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ripser import ripser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from topology import landscape_norms

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"topology_visuals"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"PingFang SC","font.size":11,"axes.unicode_minus":False,
                     "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.18})
CASES=[("相对平静期","2017-06-30"),("互联网泡沫参考日前","2000-03-09"),("雷曼倒闭前","2008-09-12"),("当前快照","2026-09-28")]


def landscape(diagram,grid,layers=5):
    if len(diagram)==0:return np.zeros((layers,len(grid)))
    tents=np.maximum(0,np.minimum(grid[None,:]-diagram[:,0,None],diagram[:,1,None]-grid[None,:]))
    values=np.sort(tents,axis=0)[::-1]
    result=np.zeros((layers,len(grid)))
    result[:min(layers,len(values))]=values[:layers]
    return result


def main():
    returns=pd.read_csv(ROOT/"data/returns.csv",index_col=0,parse_dates=True)
    all_cases={}
    for w in (50,100):
        samples=[]
        for name,date in CASES:
            x=returns.loc[:date].tail(w)
            d=ripser(x.to_numpy(),maxdim=1,thresh=np.inf)["dgms"][1]
            l1,l2=landscape_norms(d)
            samples.append((name,date,x,d,l1,l2))
            all_cases[f"{w}_{date}"]={"name":name,"date":date,"window":w,"points":x.to_numpy().tolist(),
                                       "point_dates":[str(t.date()) for t in x.index],"h1_diagram":d.tolist(),"l1":l1,"l2":l2}
        limit=max(d.max() if len(d) else 0 for _,_,_,d,_,_ in samples)*1.07
        lim_cloud=max(abs(x[["SP500","NASDAQ"]].to_numpy()).max() for _,_,x,_,_,_ in samples)*1.06
        grid=np.linspace(0,limit,600)
        maxheight=max(landscape(d,grid).max() for _,_,_,d,_,_ in samples)*1.15
        fig,axes=plt.subplots(3,4,figsize=(16,10))
        for j,(name,date,x,d,l1,l2) in enumerate(samples):
            a=axes[0,j]
            a.scatter(x.SP500,x.NASDAQ,s=22,c=np.arange(len(x)),cmap="viridis",alpha=.85,edgecolor="none")
            a.axhline(0,lw=.6,color="#999");a.axvline(0,lw=.6,color="#999")
            a.set_xlim(-lim_cloud,lim_cloud);a.set_ylim(-lim_cloud,lim_cloud);a.set_aspect("equal",adjustable="box")
            a.set_xlabel("标普对数收益率（%）")
            if j==0:a.set_ylabel("纳斯达克对数收益率（%）")
            a.set_title(f"{name}\n{date}",fontweight="bold",pad=12)
            a=axes[1,j]
            a.plot([0,limit],[0,limit],"--",lw=.8,color="#999")
            if len(d):
                pers=d[:,1]-d[:,0]
                a.scatter(d[:,0],d[:,1],s=30+150*pers/limit,c="#086788",alpha=.85)
                i=np.argmax(pers)
                a.plot([d[i,0],d[i,0]],[d[i,0],d[i,1]],color="#e17839",lw=2)
            a.set_xlim(0,limit);a.set_ylim(0,limit);a.set_aspect("equal",adjustable="box")
            a.set_xlabel("出生距离 b")
            if j==0:a.set_ylabel("死亡距离 d")
            a.set_title(f"真实四维 H1：{len(d)} 个有限环",fontsize=11)
            a=axes[2,j]
            ls=landscape(d,grid)
            for k in range(5):a.plot(grid,ls[k],lw=1.5 if k==0 else 1,alpha=1-.12*k)
            a.set_xlim(0,limit);a.set_ylim(0,maxheight)
            a.set_xlabel("滤过距离")
            if j==0:a.set_ylabel("持续景观高度")
            a.set_title(f"完整景观 L1 = {l1:.4f}",fontsize=11)
        fig.suptitle(f"拓扑结构可以直接画出来｜{w} 个交易日，四个股指",fontsize=21,y=.995)
        fig.text(.065,.02,"上排：二维坐标投影，颜色由早到晚；投影中看起来像圈，不代表四维中真的有环。\n中排：每一点代表一个四维环，离对角线越远，存活范围越长。下排展示前 5 层景观；标注 L1 使用所有环。各列同尺度。",fontsize=11,color="#425466")
        fig.subplots_adjust(top=.90,bottom=.10,left=.065,right=.99,hspace=.46,wspace=.26)
        fig.savefig(OUT/f"structure_w{w}.png",dpi=165,bbox_inches="tight")
        fig.savefig(OUT/f"structure_w{w}.svg",bbox_inches="tight")
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for a,w in zip(axes,(50,100)):
        case=all_cases[f"{w}_2026-09-28"]
        d=np.array(case["h1_diagram"])
        if len(d):
            order=np.argsort(d[:,1]-d[:,0])[::-1]
            for i,(b,death) in enumerate(d[order]):a.plot([b,death],[i,i],lw=5,solid_capstyle="round",color="#086788")
        a.set_xlabel("四维欧氏距离：出生 → 死亡")
        a.set_ylabel("环，按寿命从长到短")
        a.set_yticks(range(len(d)),[str(i+1) for i in range(len(d))]);a.invert_yaxis()
        a.set_title(f"当前 {w} 日窗口｜L1 = {case['l1']:.4f}")
    largest=max(np.array(all_cases[f"{w}_2026-09-28"]["h1_diagram"]).max() for w in (50,100))
    for a in axes:a.set_xlim(0,largest*1.08)
    fig.suptitle("当前的持续条形码｜2026-09-28",fontsize=20,y=1.01)
    fig.text(.07,-.02,"每条线是真实计算出的 H1 环。长度代表跨滤过尺度的持久性，不是持续了多少天，也不是危机概率。",fontsize=11,color="#425466")
    fig.tight_layout()
    fig.savefig(OUT/"current_barcodes.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    (OUT/"snapshot_data.json").write_text(json.dumps(all_cases,ensure_ascii=False,indent=2))
    print("Saved 8 actual topology snapshots and three figures")


if __name__=="__main__":main()
