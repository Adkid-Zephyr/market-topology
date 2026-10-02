"""Export an explanatory scientific animation; all market geometry is certified."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'explainer/output'
OUT.mkdir(parents=True,exist_ok=True)
C=json.loads((ROOT/'explainer/certificates.json').read_text())
NAVY='#204c68';ORANGE='#d48539';BLUE='#77accb';GREY='#7b8792'
plt.rcParams.update({'font.family':'PingFang SC','font.size':12,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,
                     'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none'})


def point_order(edges):
    graph={}
    for u,v in edges:graph.setdefault(u,[]).append(v);graph.setdefault(v,[]).append(u)
    assert all(len(v)==2 for v in graph.values()),'Only simple cycles used in this animation'
    start=min(graph);path=[start];previous=None;current=start
    while True:
        nxt=next(v for v in graph[current] if v!=previous)
        if nxt==start:break
        path.append(nxt);previous,current=current,nxt
    assert len(path)==len(graph)
    return path+[start]


def stage_scale(t,b,d):
    # Hold the existing loop for readability, then reveal its filling.
    if t<.2:return b*(.70+.31*t/.2)
    if t<.65:return b+.03*(d-b)+(.85*(d-b))*(t-.2)/.45
    if t<.8:return b+.88*(d-b)+.17*(d-b)*(t-.65)/.15
    return d+.05*(d-b)


def status(e,b,d):
    if e<b:return '未形成选定环'
    if e<d:return '选定环存在'
    return '选定环被三角面填满'


def setup_text(fig,title,subtitle):
    fig.text(.055,.943,title,fontsize=24,color='#172b3d',weight='medium')
    fig.text(.055,.900,subtitle,fontsize=12,color='#566777')


def intro(fig):
    setup_text(fig,'四维收益率点云','每个交易日对应一个四维点；一个坐标对应一个股指')
    ret=pd.read_csv(ROOT/'data/returns.csv',index_col=0).loc['2026-09-28']
    names=['S&P 500','DJIA','NASDAQ','Russell 2000']
    xs=[.15,.38,.61,.84]
    for x,name,value in zip(xs,names,ret.to_numpy()):
        fig.text(x,.69,name,ha='center',fontsize=19,color=NAVY)
        fig.text(x,.57,f'{value:+.2f}%',ha='center',fontsize=28,color='#172b3d')
    fig.text(.5,.39,'同一天的四个收益率',ha='center',fontsize=18,color='#566777')
    fig.text(.5,.29,r'一个点  $x_t \in \mathbb{R}^{4}$',ha='center',fontsize=29,color=NAVY)
    fig.text(.5,.17,'取最近 50 个交易日，得到 50 个点',ha='center',fontsize=19)
    fig.text(.055,.065,'数据示例：2026-09-28；原论文以四个股指组成四维坐标，不是四个未来时刻。',fontsize=12,color='#566777')
    return lambda _:None


def setup_bar(fig,b,d,maximum):
    a=fig.add_axes([.12,.17,.77,.07]);a.set_xlim(0,maximum);a.set_ylim(-.7,.7)
    a.plot([0,maximum],[0,0],color='#d0d6db',lw=2)
    a.plot([b,d],[0,0],color=ORANGE,lw=7,solid_capstyle='round')
    a.text(b,.29,f'b = {b:.3f}',ha='right',fontsize=12)
    a.text(d,.29,f'd = {d:.3f}',ha='left',fontsize=12)
    line=a.axvline(0,ymin=0,ymax=.58,color=NAVY,lw=1.4)
    a.set_yticks([]);a.set_xlabel('距离阈值 ε',fontsize=13)
    a.spines[['left','right','top']].set_visible(False)
    for tick in a.get_xticklabels():tick.set_fontsize(11)
    return line


def toy(fig):
    s=C['square'];x=np.array(s['points_2d']);dist=np.array(s['distances']);b=s['birth'];d=s['death']
    setup_text(fig,'Vietoris–Rips 滤过','二维数学示意：两点足够近则连边；三点两两相连则加入三角面')
    a=fig.add_axes([.055,.265,.56,.60]);a.set_aspect('equal');a.set_xlim(-1.65,1.65);a.set_ylim(-1.65,1.65);a.axis('off')
    edges=LineCollection([],colors=GREY,linewidths=1.8);a.add_collection(edges)
    faces=PolyCollection([],facecolors=BLUE,alpha=.42,edgecolors='none');a.add_collection(faces)
    line,=a.plot([],[],color=ORANGE,lw=4,zorder=4)
    a.scatter(x[:,0],x[:,1],s=135,c=NAVY,zorder=5,edgecolors='white',linewidths=1.1)
    for i,(px,py) in enumerate(x):a.text(px*1.20,py*1.20,chr(65+i),ha='center',va='center',fontsize=17,color=NAVY)
    stage=fig.text(.665,.69,'',fontsize=21,color='#172b3d')
    eps=fig.text(.665,.61,'',fontsize=18,color=NAVY)
    fig.text(.665,.49,'H₁：一维环',fontsize=21,color='#172b3d')
    fig.text(.665,.39,'橙线：闭合边链\n蓝面：填充三角面',fontsize=15,color='#566777',linespacing=1.8)
    bar=setup_bar(fig,b,d,3.2)
    fig.text(.055,.055,'环的寿命是 d − b，即跨越的距离尺度；不是持续了多少天。',fontsize=12,color='#566777')
    pairs=list(combinations(range(4),2));cycle=x[point_order(s['cycle_edges'])]
    def update(t):
        e=stage_scale(t,b,d)
        edges.set_segments([x[list(pair)] for pair in pairs if dist[pair]<=e])
        if e>=b:line.set_data(cycle[:,0],cycle[:,1]);line.set_color(ORANGE if e<d else NAVY)
        else:line.set_data([],[])
        faces.set_verts([x[tri] for tri in s['filling_triangles'] if max(dist[pair] for pair in combinations(tri,2))<=e])
        stage.set_text(status(e,b,d).replace('选定',''))
        eps.set_text(f'ε = {e:.3f}')
        bar.set_xdata([e,e])
    return update


def market(fig,date):
    s=C[date];x=np.array(s['projection_3d']);dist=np.array(s['distances']);dgm=np.array(s['diagram']);b=s['birth'];d=s['death']
    title='历史窗口' if date=='2000-03-09' else '当前数据快照'
    setup_text(fig,f'四维市场点云：{title}',f'{date} 截止的 50 个交易日；所有连边与环均按原始四维距离计算')
    a=fig.add_axes([.015,.345,.375,.49],projection='3d')
    limits=np.max(np.abs(x),axis=0)*1.18
    a.set_xlim(-limits[0],limits[0]);a.set_ylim(-limits[1],limits[1]);a.set_zlim(-limits[2],limits[2]);a.set_box_aspect(limits)
    a.set_xlabel('PC1',fontsize=10,labelpad=0);a.set_ylabel('PC2',fontsize=10,labelpad=0);a.set_zlabel('PC3',fontsize=10,labelpad=0)
    a.set_xticks([]);a.set_yticks([]);a.set_zticks([]);a.grid(False)
    a.set_title('三维 PCA 投影',fontsize=14,pad=8)
    for axis in [a.xaxis,a.yaxis,a.zaxis]:axis.pane.fill=False
    points=a.scatter(*x.T,s=18,c='#8c9bab',alpha=.55,depthshade=False)
    edgeplot=Line3DCollection([],colors='#8494a0',linewidths=.55,alpha=.09);a.add_collection3d(edgeplot,autolim=False)
    faces=Poly3DCollection([],facecolors=BLUE,alpha=.48,edgecolors='#538bac',linewidths=.55);a.add_collection3d(faces,autolim=False)
    selected=a.plot([],[],[],color=ORANGE,lw=3.4,zorder=8)[0]
    vertices=sorted(set(v for pair in s['cycle_edges'] for v in pair));cycle=x[point_order(s['cycle_edges'])]
    a.scatter(*x[vertices].T,s=33,c=ORANGE,edgecolors='white',linewidths=.5,depthshade=False)
    # Schematic cycle layout preserves certified connectivity, not distances.
    order=point_order(s['cycle_edges'])[:-1]
    theta=np.linspace(np.pi/2,np.pi/2+2*np.pi,len(order),endpoint=False)
    layout={v:np.array([np.cos(t),np.sin(t)]) for v,t in zip(order,theta)}
    graph=fig.add_axes([.405,.345,.28,.46]);graph.set_aspect('equal');graph.set_xlim(-1.45,1.45);graph.set_ylim(-1.45,1.45);graph.axis('off')
    graph.set_title('选定环的连接关系',fontsize=14,pad=8)
    graph_edges=LineCollection([],colors=GREY,linewidths=1.6);graph.add_collection(graph_edges)
    graph_faces=PolyCollection([],facecolors=BLUE,alpha=.5,edgecolors='#538bac',linewidths=.6);graph.add_collection(graph_faces)
    graph_cycle,=graph.plot([],[],color=ORANGE,lw=3.4,zorder=4)
    coordinates=np.array([layout[v] for v in order]);graph.scatter(*coordinates.T,s=55,c=NAVY,zorder=5)
    for v,pos in layout.items():graph.text(*(pos*1.24),s['point_dates'][v][5:],ha='center',va='center',fontsize=9,color='#405668')
    graph.text(0,-1.43,'示意布局：位置不代表收益率',ha='center',va='center',fontsize=10,color='#566777')
    diag=fig.add_axes([.75,.39,.21,.39]);diag.plot([0,3.2],[0,3.2],'--',color='#adb5bd',lw=.9)
    diag.scatter(dgm[:,0],dgm[:,1],s=20,c=NAVY);diag.scatter([b],[d],s=60,c=ORANGE,zorder=3)
    diag.set(xlim=(0,3.2),ylim=(0,3.2),xlabel='Birth distance',ylabel='Death distance');diag.set_aspect('equal')
    diag.tick_params(labelsize=10);diag.xaxis.label.set_size(11);diag.yaxis.label.set_size(11)
    diag.set_title('H₁ 持续图',fontsize=14,pad=9)
    stage=fig.text(.07,.285,'',fontsize=17,color='#172b3d')
    info=fig.text(.63,.285,'',fontsize=14,color=NAVY)
    bar=setup_bar(fig,b,d,3.2)
    fig.text(.055,.060,'橙线：经四维核验的环代表；蓝面：其填充链。其余面省略，布局与投影不参与计算。',fontsize=12,color='#566777')
    fig.text(.055,.026,'相连的是相似的交易日状态，不是相邻日期；图形不能直接换算成危机概率。',fontsize=12,color='#566777')
    pairs=list(combinations(range(len(x)),2))
    def update(t):
        e=stage_scale(t,b,d)
        edgeplot.set_segments([x[list(pair)] for pair in pairs if dist[pair]<=e])
        fs=[x[tri] for tri in s['filling_triangles'] if max(dist[pair] for pair in combinations(tri,2))<=e]
        faces.set_verts(fs)
        graph_edges.set_segments([[layout[u],layout[v]] for u,v in combinations(order,2) if dist[u,v]<=e])
        graph_faces.set_verts([[layout[v] for v in tri] for tri in s['filling_triangles'] if max(dist[pair] for pair in combinations(tri,2))<=e])
        if e>=b:
            selected.set_data_3d(cycle[:,0],cycle[:,1],cycle[:,2]);selected.set_color(ORANGE if e<d else NAVY)
            schematic=np.array([layout[v] for v in order+[order[0]]]);graph_cycle.set_data(*schematic.T);graph_cycle.set_color(ORANGE if e<d else NAVY)
        else:
            selected.set_data_3d([],[],[]);graph_cycle.set_data([],[])
        a.view_init(elev=22+7*np.sin(t*np.pi*2),azim=-60+100*t)
        stage.set_text(status(e,b,d))
        info.set_text(f'ε = {e:.3f}     d − b = {d-b:.3f}')
        bar.set_xdata([e,e])
    return update


def storyboard():
    fig,ax=plt.subplots(1,3,figsize=(10.5,3.5))
    s=C['square'];x=np.array(s['points_2d']);D=np.array(s['distances'])
    for i,(a,e,label) in enumerate(zip(ax,[1.7,2.2,3.0],['(a) 连边之前','(b) 环存在','(c) 三角面填满'])):
        a.scatter(*x.T,s=50,c=NAVY,zorder=5)
        for pair in combinations(range(4),2):
            if D[pair]<=e:a.plot(*x[list(pair)].T,color=ORANGE if e<s['death'] else NAVY,lw=2)
        for tri in s['filling_triangles']:
            if max(D[p] for p in combinations(tri,2))<=e:a.fill(*x[tri].T,color=BLUE,alpha=.45)
        a.set_xlim(-1.6,1.6);a.set_ylim(-1.6,1.6);a.set_aspect('equal');a.axis('off');a.set_title(label,fontsize=14)
        a.text(0,-1.48,f'ε = {e:.1f}',ha='center',fontsize=12)
    fig.subplots_adjust(bottom=.16,top=.82,wspace=.13)
    fig.text(.5,.03,'数学示意：四维空间中的 H₁ 环，遵循相同的连边与填面规则。',ha='center',fontsize=12,color='#566777')
    fig.savefig(OUT/'filtration_storyboard.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/'filtration_storyboard.svg',bbox_inches='tight');plt.close(fig)


def main():
    storyboard()
    fps=20;segments=[('intro',4),('toy',8),('2000-03-09',9),('2026-09-28',9)]
    fig=plt.figure(figsize=(12.8,7.2),dpi=150)
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=-1,extra_args=['-crf','20','-pix_fmt','yuv420p','-movflags','+faststart'])
    path=OUT/'topology_explainer_1080p.mp4'
    with writer.saving(fig,str(path),dpi=150):
        for scene,duration in segments:
            fig.clear()
            update=intro(fig) if scene=='intro' else (toy(fig) if scene=='toy' else market(fig,scene))
            for frame in range(duration*fps):
                t=frame/(duration*fps-1);update(t)
                if frame in [int(duration*fps*.5),int(duration*fps*.9)]:
                    fig.savefig(OUT/(f'{scene}_'+('ring' if frame<int(duration*fps*.7) else 'filled')+'.png'),dpi=150)
                writer.grab_frame()
            print('Rendered',scene,duration,'seconds',flush=True)
    plt.close(fig)
    print(path,flush=True)


if __name__=='__main__':main()
