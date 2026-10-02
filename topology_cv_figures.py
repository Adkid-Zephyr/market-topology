from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from topology_cv import split_indices
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'topology_cv_figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'PingFang SC','font.size':11,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
NAMES={'base_rate':'历史发生率','raw_tda':'原始拓扑指标','landscape':'景观曲线 + PCA','delay':'延迟嵌入拓扑',
       'diagram_kernel':'完整持续图核','market':'普通市场特征','market_landscape':'市场 + 景观曲线',
       'market_delay':'市场 + 延迟拓扑','market_kernel':'普通市场 RBF 核','inner_selector':'仅按过去验证选模型'}


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def main():
    m=pd.read_csv(ROOT/'topology_cv_results/metrics.csv')
    order=['base_rate','raw_tda','landscape','delay','diagram_kernel','market','market_landscape','market_delay','market_kernel','inner_selector']
    fig,ax=plt.subplots(1,2,figsize=(13.5,6.5),sharey=True)
    colors=['#64748b','#c45850','#398ab9','#347678','#7775b5','#a88957','#a88957','#a88957','#a88957','#444444']
    for a,period,title in zip(ax,['all','post2017'],['2003–2025：全部成熟测试月','2017–2025：原论文样本之后']):
        q=m[(m.target=='12m_20pct')&(m.period==period)].set_index('model').loc[order]
        a.barh(np.arange(10),q.brier,color=colors,height=.64)
        a.set_yticks(np.arange(10),[NAMES[x] for x in order]);a.set_xlim(0,.207)
        a.axvline(q.loc['base_rate','brier'],ls='--',color='#334155',lw=1)
        for i,v in enumerate(q.brier):a.text(v+.002,i,f'{v:.4f}',va='center',fontsize=10)
        a.set_title(title,fontsize=13,pad=12);a.set_xlabel('Brier 概率误差，越低越好')
    ax[0].invert_yaxis()
    fig.suptitle('更完整的拓扑表示：有局部改善，未稳定超越简单基准',fontsize=19,y=.985)
    fig.text(.04,.015,'未来 252 个交易日是否较预测月末下跌至少 20%。每年向前测试，参数仅由更早的内层验证选择。\n虚线为历史发生率基准。“仅按过去验证选模型”评估选模流程本身，未事后挑赢家。',fontsize=11,color='#425466')
    fig.subplots_adjust(left=.16,right=.99,top=.87,bottom=.15,wspace=.2)
    save(fig,'01_nested_results')
    periods=['2003-2006','2007-2011','2012-2016','2017-2020','2021-2026']
    q=m[(m.target=='12m_20pct')&m.period.isin(periods)].pivot(index='model',columns='period',values='brier_skill').loc[order[1:],periods]
    v=q.to_numpy();lim=max(abs(v.min()),abs(v.max()))
    fig,a=plt.subplots(figsize=(11,7))
    im=a.imshow(v,cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
    a.set_xticks(range(5),['2003–2006','2007–2011','2012–2016','2017–2020','2021–2025*'])
    a.set_yticks(range(len(q)),[NAMES[x] for x in q.index]);a.grid(False)
    for i in range(len(q)):
        for j in range(5):a.text(j,i,f'{100*v[i,j]:+.0f}%',ha='center',va='center',color='white' if abs(v[i,j])>lim*.58 else '#172B4D')
    a.set_title('不同时间段的结果差异很大\n相对历史发生率的 Brier 改善，正数为改善，负数为变差',fontsize=16,pad=18)
    fig.colorbar(im,ax=a,label='Brier skill：1 − 模型误差 / 基准误差',shrink=.8)
    fig.text(.04,.02,'* 主任务成熟标签只到 2025 年 8 月。平静期预测低风险也可降低误差，因此不能只看某个蓝色区间。',fontsize=10,color='#425466')
    fig.tight_layout(rect=[0,.06,1,1]);save(fig,'02_time_blocks')
    f=pd.read_csv(ROOT/'topology_cv_results/folds.csv')
    sample=pd.read_csv(ROOT/'topology_cv_results/sample_index_labels.csv',index_col=0,parse_dates=True)
    sample['12m_20pct_end']=pd.to_datetime(sample['12m_20pct_end'])
    def yearpos(t):
        return t.year+(t.dayofyear-1)/365.25
    fig,a=plt.subplots(figsize=(12,5))
    y=0
    for year in [2008,2017,2022]:
        g=f[(f.target=='12m_20pct')&(f.outer_year==year)]
        for row in g.itertuples():
            cutoff=pd.Timestamp(row.cutoff)
            is_outer=row.kind=='outer'
            split=split_indices(sample,'12m_20pct',cutoff,cutoff,cutoff+pd.DateOffset(years=1 if is_outer else 2),
                                minimum=84 if is_outer else 60,positive=5 if is_outer else 3,
                                maturity_cap=None if is_outer else pd.Timestamp(year,1,1))
            ti,vi=split
            t0=yearpos(sample.index[ti[0]]);t1=yearpos(sample.index[ti[-1]]+pd.Timedelta(days=1))
            v0=yearpos(sample.index[vi[0]]);v1=yearpos(sample.index[vi[-1]]+pd.Timedelta(days=1))
            a.barh(y,t1-t0,left=t0,height=.65,color='#3e8ca8')
            a.barh(y,v0-t1,left=t1,height=.65,color='#d4d8dd',hatch='//')
            a.barh(y,v1-v0,left=v0,height=.65,color='#e19a54')
            a.text(v1+.12,y,f'{year} 年：'+('外层测试' if is_outer else '内层验证'),va='center',fontsize=10)
            y+=1
        y+=.7
    a.set_xlim(1990,2027);a.invert_yaxis();a.set_yticks([]);a.set_xlabel('年份；色带对应实际样本的预测起点，灰段为尚未成熟的训练标签')
    a.set_title('时间验证：训练 → 等待标签成熟 → 验证或测试',fontsize=18,pad=48)
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color='#3e8ca8',label='训练样本起点'),Patch(facecolor='#d4d8dd',hatch='//',label='未成熟标签清除区'),Patch(color='#e19a54',label='验证 / 测试起点')],loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,ncol=3,fontsize=10)
    fig.tight_layout();save(fig,'03_validation_design')


if __name__=='__main__':main()
