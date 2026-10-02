"""Four exact 3:4 research cards, drawn from saved evidence (no new model run)."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
font_names={f.name for f in font_manager.fontManager.ttflist}
CJK_FONT=next((n for n in ["PingFang SC","Noto Sans CJK SC","Microsoft YaHei","Arial Unicode MS"] if n in font_names), "DejaVu Sans")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
W, H, DPI = 1200, 1600, 150
BG = '#F5F3ED'
INK = '#173D40'
TEAL = '#187F80'
ORANGE = '#C75D35'
MUTED = '#637778'
LINE = '#D7DFD9'
plt.rcParams.update({'font.family': CJK_FONT, 'axes.unicode_minus': False,
                     'svg.fonttype': 'path', 'font.size': 11})
SNAP = pd.read_csv(ROOT/'results/snapshots.csv')
DAILY = {w: pd.read_csv(ROOT/f'results/daily_w{w}.csv', index_col=0, parse_dates=True)
         for w in [50, 100]}
METRICS = pd.read_csv(ROOT/'topology_cv_results/metrics.csv')
CERT = json.loads((ROOT/'explainer/certificates.json').read_text())
TEXTS = []
if set(SNAP.loc[SNAP.label == "current", "date"]) != {"2026-09-28"}:
    raise ValueError("These cards describe the 2026-09-28 snapshot. Create a new dated report for newer inputs.")

def text(ax, x, y, s, size=30, color=INK, weight='normal', ha='left', **kw):
    t=ax.text(x,y,s,fontsize=size*72/DPI,color=color,fontweight=weight,
              ha=ha,va='top',linespacing=1.35,**kw)
    TEXTS.append(t)
    return t

def box(ax, x, y, w, h, color='white', radius=24):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                              facecolor=color,edgecolor='none',zorder=0))

def rule(ax, y, x=72, end=1128):
    ax.plot([x,end],[y,y],color=LINE,lw=1)

def page(n, kicker):
    fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI,facecolor=BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
    text(ax,72,56,'金融市场拓扑 · 复现报告',25,TEAL,weight='bold')
    text(ax,1128,56,f'0{n} / 04',25,MUTED,ha='right')
    rule(ax,105)
    text(ax,72,124,kicker,25,MUTED)
    rule(ax,1480)
    text(ax,72,1495,'数据截至 2026-09-28｜四指数历史样本 1987–2026',20,MUTED)
    text(ax,72,1520,'论文：Gidea & Katz · Physica A (2018)｜arXiv:1703.04385',20,MUTED)
    text(ax,72,1545,'数据：Yahoo Finance；早期道指 GMU；FRED 用于交叉核验',20,MUTED)
    text(ax,72,1570,'代码与复现：github.com/Adkid-Zephyr/market-topology',20,TEAL)
    return fig,ax

def inset(fig,x,y,w,h):
    return fig.add_axes([x/W,1-(y+h)/H,w/W,h/H],facecolor='none')

def clean(a):
    a.spines[['top','right']].set_visible(False)
    for s in ['bottom','left']:a.spines[s].set_color(LINE)
    a.tick_params(colors=MUTED,labelsize=12,length=0,pad=7)
    a.grid(alpha=.16,color=MUTED,lw=.6)
    a.set_axisbelow(True)

def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    # Check every editorial text block remains on the page.
    for t in TEXTS:
        if t.figure is fig:
            b=t.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > W+1 or b.y1 > H+1:
                raise ValueError(f'Text outside canvas: {t.get_text()} {b}')
    fig.savefig(OUT/f'{name}.png',dpi=DPI,facecolor=BG)
    fig.savefig(OUT/f'{name}.svg',facecolor=BG)
    plt.close(fig)

def current(w):
    return SNAP[(SNAP.label=='current')&(SNAP.window==w)].iloc[0]

def cover():
    fig,a=page(1,'美国四股指｜近 39 年历史样本')
    text(a,72,184,'美股拓扑指标\n短窗口偏高',76,weight='bold')
    text(a,72,408,'原论文：.com 泡沫与雷曼倒闭前出现结构信号',31)
    text(a,72,456,'本次检验：历史形态重现，预测优势未建立',31,MUTED)
    box(a,72,538,1056,375,INK)
    text(a,108,567,'市场状态 · 2026 年 9 月 28 日',29,'#BDE5DF')
    for x,w,c,label in [(108,50,'#8DE1CC','50 个交易日窗口'),(642,100,'#F1C496','100 个交易日窗口')]:
        text(a,x,633,label,30,'white')
        text(a,x,680,f'{current(w).l1_prior_pct:.1f}',114,c,weight='bold')
        text(a,x,810,'历史分位（0–100）',28,'#D1E5DF')
    text(a,72,954,'短窗口偏高，长窗口未同步',48,weight='bold')
    text(a,72,1021,'85.3 是历史分位，不是 85.3% 的危机概率。',31,ORANGE,weight='bold')
    ca=inset(fig,108,1120,985,225)
    for w,col in [(50,TEAL),(100,ORANGE)]:
        f=DAILY[w].loc['2024-01-01':]
        ca.plot(f.index,f.l1_prior_pct,color=col,lw=1.5)
        ca.scatter(f.index[-1],f.l1_prior_pct.iloc[-1],color=col,s=22,zorder=5)
    ca.set_ylim(0,105);ca.set_yticks([0,50,100]);ca.xaxis.set_major_locator(md.YearLocator())
    ca.xaxis.set_major_formatter(md.DateFormatter('%Y'));clean(ca)
    text(a,1128,1080,'近年历史位置：50 日 / 100 日',26,MUTED,ha='right')
    text(a,72,1402,'结构变化可以描述；危机概率尚不能可靠估计。',33,weight='bold')
    save(fig,'01_cover')

def percentiles():
    fig,a=page(2,'历史分位：只与每个时点之前的数据比较')
    text(a,72,184,'历史分位对照\n短长窗口分化',70,weight='bold')
    text(a,72,384,'拓扑强度（景观 L1）排名｜85.3 = 高于约 85.3% 的过去值',27,MUTED)
    box(a,72,455,1056,518)
    text(a,100,480,'观察时点',28,MUTED,weight='bold')
    text(a,834,480,'50 日',28,TEAL,weight='bold',ha='center')
    text(a,1045,480,'100 日',28,ORANGE,weight='bold',ha='center')
    cases=[('2000-03-09','.com 泡沫参考日前'),('2007-10-08','美股见顶前一交易日'),
           ('2008-09-12','雷曼倒闭前一交易日'),('2020-02-18','疫情暴跌前参照'),
           ('2026-09-28','本次最新观测')]
    vals=[]
    for i,(date,label) in enumerate(cases):
        y=539+i*84
        if i==4:box(a,87,y-8,1026,82,'#E7F0E9',10)
        text(a,102,y,date,31,weight='bold')
        text(a,321,y+2,label,27)
        v=[float(DAILY[w].loc[date,'l1_prior_pct']) for w in [50,100]]
        vals.append({'date':date,'label':label,'percentile_50':v[0],'percentile_100':v[1]})
        text(a,834,y-4,f'{v[0]:.1f}',43,TEAL,weight='bold',ha='center')
        text(a,1045,y-4,f'{v[1]:.1f}',43,ORANGE,weight='bold',ha='center')
        if i<4:rule(a,y+67,100,1100)
    text(a,72,1003,'最新观测：参照范围与尺度敏感性',36,weight='bold')
    for i,(label,key) in enumerate([('此前约 5 年','l1_prior5y_pct'),('去除整体波动放大后','scale_free_l1_prior_pct'),('逐指数标准化后','zscore_l1_prior_pct')]):
        y=1065+i*62
        text(a,90,y,label,29)
        text(a,834,y,f'{current(50)[key]:.1f}',33,TEAL,weight='bold',ha='center')
        text(a,1045,y,f'{current(100)[key]:.1f}',33,ORANGE,weight='bold',ha='center')
    box(a,72,1277,1056,170,'#E7F0E9')
    text(a,100,1300,'结果解读',31,weight='bold')
    text(a,100,1350,'50 日偏高，100 日没有同步异常。\n两种窗口分别排名；分位不代表危机概率。',29)
    pd.DataFrame(vals).to_csv(OUT/'historical_percentiles.csv',index=False)
    save(fig,'02_percentiles')

def topology():
    fig,a=page(3,'真实市场数据 · 50 个交易日窗口')
    text(a,72,184,'收益率点云\n与拓扑环',68,weight='bold')
    text(a,72,379,'标普 + 道指 + 纳斯达克 + 罗素 2000',31,weight='bold')
    text(a,72,431,'一个点代表一天：四个股指收益率组成四个坐标。',29,MUTED)
    for i,(date,label) in enumerate([('2000-03-09','.com 泡沫参考日前'),('2026-09-28','本次最新观测')]):
        x=72+i*540
        box(a,x,505,516,750)
        text(a,x+24,527,label,30,weight='bold')
        text(a,x+24,571,date,26,MUTED)
        c=CERT[date];pts=np.array(c['projection_3d'])
        p=inset(fig,x+53,651,410,206)
        p.scatter(pts[:,0],pts[:,1],s=22,color='#91B4AE',alpha=.8)
        for u,v in c['cycle_edges']:
            p.plot(pts[[u,v],0],pts[[u,v],1],color=ORANGE,lw=2)
        lim=max(np.abs(np.array(CERT[k]['projection_3d'])[:,:2]).max()
                for k in ['2000-03-09','2026-09-28'])*1.12
        p.set_xlim(-lim,lim);p.set_ylim(-lim,lim);p.set_aspect('equal',adjustable='box')
        p.set_xticks([-4,0,4]);p.set_yticks([-4,0,4]);clean(p)
        text(a,x+24,613,'市场状态投影 · 橙线为实际计算出的环',24,ORANGE)
        text(a,x+24,909,'环的尺度分布 · 每个点代表一个环',24,TEAL)
        d=np.array(c['diagram']);p=inset(fig,x+78,962,355,180)
        p.plot([0,3.3],[0,3.3],ls='--',lw=.8,color=MUTED)
        p.scatter(d[:,0],d[:,1],s=30,color=TEAL)
        p.scatter(c['birth'],c['death'],s=60,color=ORANGE,zorder=5)
        p.set_xlim(0,3.3);p.set_ylim(0,3.3);p.set_aspect('equal',adjustable='box')
        p.set_xticks([0,1,2,3]);p.set_yticks([0,1,2,3]);clean(p)
        p.set_xlabel('出生距离 b',fontsize=11,color=MUTED)
        p.set_ylabel('死亡距离 d',fontsize=11,color=MUTED)
        text(a,x+24,1220,f"最长环跨尺度寿命：{c['persistence']:.3f}",25,weight='bold')
    text(a,72,1298,'拓扑环：相似市场状态围成的闭合结构。',34,weight='bold')
    text(a,72,1358,'放宽相似程度的标准：先连边成环，再被三角面填满。\n“寿命”是存在的距离范围；四维计算记为 H1。\n投影仅供观察。平静期也有环，不能据此判断危机。',28,MUTED)
    save(fig,'03_topology')

def results():
    fig,a=page(4,'样本外检验｜按时间先后训练与评分')
    text(a,72,184,'预测检验结果\n未建立稳定优势',67,weight='bold')
    text(a,72,374,'预测误差与简单基准接近，近期样本中更高。',35,ORANGE,weight='bold')
    text(a,72,435,'目标：标普未来约一年内，较预测月末价格下跌至少 20%。\n过去数据选参数，未来数据评分；检验的是股价损失。',27,MUTED)
    text(a,72,539,'预测误差（Brier）：分数与实际结果越接近，误差越低',27,weight='bold')
    box(a,72,594,1056,581)
    text(a,100,616,'方法',26,MUTED,weight='bold')
    text(a,815,616,'2003–2025',24,MUTED,weight='bold',ha='center')
    text(a,1010,616,'2017–2025',24,MUTED,weight='bold',ha='center')
    names=[('base_rate','历史发生率基准'),('raw_tda','原始拓扑特征'),('landscape','景观曲线 + PCA'),
           ('diagram_kernel','完整持续图核'),('delay','延迟嵌入拓扑'),('market','普通市场特征'),
           ('market_kernel','普通市场 RBF 核'),('market_landscape','市场 + 景观'),
           ('market_delay','市场 + 延迟'),('inner_selector','只按过去验证选模')]
    table=[]
    for i,(name,label) in enumerate(names):
        y=672+i*48
        values=[]
        for period in ['all','post2017']:
            row=METRICS[(METRICS.target=='12m_20pct')&(METRICS.period==period)&(METRICS.model==name)].iloc[0]
            values.append(float(row.brier))
        table.append({'model':name,'label':label,'all':values[0],'post2017':values[1]})
        if i==0:box(a,86,y-7,1028,47,'#E7F0E9',8)
        text(a,100,y,label,28,weight='bold' if i==0 else 'normal')
        for x,value in zip([815,1010],values):
            text(a,x,y,f'{value:.4f}',29,TEAL if i==0 else INK,weight='bold' if i==0 else 'normal',ha='center')
    text(a,72,1200,'全期评分至 2025-08：272 个月，其中 32 个月出现目标损失。',26,MUTED)
    text(a,72,1242,'2017 年以后：104 个月 / 16 个正标签月；月份间存在重叠。',26,MUTED)
    box(a,72,1295,1056,170,INK)
    text(a,100,1315,'历史信号的适用边界',30,'#BDE5DF',weight='bold')
    text(a,100,1364,'① 雷曼倒闭前，标普已较 2007 年高点下跌约 20%。\n② 低频趋势随预处理变号，尚无稳健的统一预警判据。',27,'white')
    pd.DataFrame(table).to_csv(OUT/'prediction_scores.csv',index=False)
    save(fig,'04_results')

def main():
    cover();percentiles();topology();results()
    from PIL import Image
    ims=[Image.open(OUT/f'{name}.png') for name in ['01_cover','02_percentiles','03_topology','04_results']]
    preview=Image.new('RGB',(1200,1600),BG)
    for i,im in enumerate(ims):
        assert im.size==(W,H)
        preview.paste(im.resize((600,800)),((i%2)*600,(i//2)*800))
    preview.save(OUT/'preview_4cards.jpg',quality=95)
    files=['results/snapshots.csv','results/daily_w50.csv','results/daily_w100.csv',
           'topology_cv_results/metrics.csv','explainer/certificates.json','REPORT.md']
    manifest={'data_asof':'2026-09-28','production_date':'2026-10-03','size':[W,H],
              'new_predictions_run':False,'status':'local_only_not_posted',
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
              'reference':'https://arxiv.org/abs/1703.04385',
              'pngs':[f'{n}.png' for n in ['01_cover','02_percentiles','03_topology','04_results']]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('Built 4 cards at 1200 x 1600; editorial text bounds checked.')

if __name__=='__main__':main()
