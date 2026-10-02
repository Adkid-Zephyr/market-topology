"""Compute a certified representative of the longest H1 interval over F2.

Unlike a cocycle visualization, the returned edge chain is an actual cycle. It
comes from a reduced triangle-boundary column and has an explicit filling chain
at its death scale. The representative is not unique.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import pdist,squareform
from ripser import ripser


def bits(value):
    while value:
        low=value & -value
        yield low.bit_length()-1
        value ^= low


def cycle_certificate(points):
    points=np.asarray(points,float)
    distances=squareform(pdist(points)).astype(np.float32).astype(float)
    edges=sorted(combinations(range(len(points)),2),key=lambda e:(distances[e],e))
    edge_index={e:i for i,e in enumerate(edges)}
    edge_values=np.array([distances[e] for e in edges])
    triangles=sorted(combinations(range(len(points)),3),key=lambda t:(max(distances[e] for e in combinations(t,2)),t))
    pivots={};chains={};deaths={};best=None;pairs=[]
    for ti,tri in enumerate(triangles):
        column=0
        for e in combinations(tri,2):column^=1<<edge_index[e]
        chain=1<<ti
        while column:
            pivot=column.bit_length()-1
            if pivot not in pivots:break
            column ^= pivots[pivot];chain ^= chains[pivot]
        if not column:continue
        pivot=column.bit_length()-1
        birth=float(edge_values[pivot]);death=float(max(distances[e] for e in combinations(tri,2)))
        if death-birth>1e-7:
            pairs.append((birth,death))
            if best is None or death-birth>best['persistence']:
                best={'birth':birth,'death':death,'persistence':death-birth,'cycle_mask':column,'filling_mask':chain}
        pivots[pivot]=column;chains[pivot]=chain;deaths[pivot]=death
    if best is None:raise ValueError('No positive H1 interval')
    cycle=[edges[i] for i in bits(best['cycle_mask'])]
    filling=[triangles[i] for i in bits(best['filling_mask'])]
    vertex_boundary=0
    for u,v in cycle:vertex_boundary^=(1<<u)^(1<<v)
    assert vertex_boundary==0
    boundary=0
    for tri in filling:
        for e in combinations(tri,2):boundary^=1<<edge_index[e]
    assert boundary==best['cycle_mask']
    assert max(distances[e] for e in cycle)<=best['birth']+1e-7
    assert max(max(distances[e] for e in combinations(t,2)) for t in filling)<=best['death']+1e-7
    # Before death this cycle cannot be expressed as a sum of triangle boundaries.
    remainder=best['cycle_mask']
    while remainder:
        pivot=remainder.bit_length()-1
        if pivot not in pivots or deaths[pivot]>=best['death']:break
        remainder ^= pivots[pivot]
    assert remainder!=0
    diagram=ripser(distances,distance_matrix=True,maxdim=1,thresh=np.inf)['dgms'][1]
    target=diagram[np.argmax(diagram[:,1]-diagram[:,0])]
    np.testing.assert_allclose([best['birth'],best['death']],target,atol=2e-6,rtol=1e-6)
    result={k:best[k] for k in ['birth','death','persistence']}
    result.update({'cycle_edges':[list(e) for e in cycle],'filling_triangles':[list(t) for t in filling],
                   'distances':distances.tolist(),'diagram':diagram.tolist(),
                   'checks':{'cycle_boundary_zero':True,'filling_boundary_equals_cycle':True,'cycle_nonboundary_before_death':True,
                             'birth_and_death_match_ripser':True},
                   'coefficients':'F2','representative_unique':False})
    return result


def main():
    root=Path(__file__).resolve().parents[1]
    snapshots=json.loads((root/'topology_visuals/snapshot_data.json').read_text())
    cases={}
    for date in ['2000-03-09','2026-09-28']:
        source=snapshots['50_'+date]
        points=np.array(source['points'])
        certificate=cycle_certificate(points)
        # PCA projection is for drawing only; all distances above remain four-dimensional.
        center=points.mean(axis=0);_,_,vt=np.linalg.svd(points-center,full_matrices=False)
        projection=(points-center)@vt[:3].T
        variance=np.linalg.svd(points-center,compute_uv=False)**2
        certificate.update({'date':date,'points_4d':points.tolist(),'point_dates':source['point_dates'],
                            'projection_3d':projection.tolist(),'projection_variance_fraction':float(variance[:3].sum()/variance.sum()),
                            'l1_all':source['l1']})
        cases[date]=certificate
        print(date,'birth/death',certificate['birth'],certificate['death'],'cycle_edges',len(certificate['cycle_edges']),
              'filling_triangles',len(certificate['filling_triangles']),flush=True)
    # A square gives the simplest possible non-trivial Vietoris-Rips loop.
    square=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
    toy=cycle_certificate(square)
    toy.update({'points_2d':square.tolist(),'type':'mathematical illustration; not market data'})
    cases['square']=toy
    (root/'explainer/certificates.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
