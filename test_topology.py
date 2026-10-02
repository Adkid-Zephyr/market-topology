import numpy as np
import pandas as pd
import pytest
from topology import diagram_for, landscape_norms, prior_percentile, window_metrics, add_trailing_features


def test_square_and_exact_integrals():
    square=np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float)
    d=diagram_for(square)
    np.testing.assert_allclose(d,[[1,np.sqrt(2)]],rtol=1e-6)
    a,b=landscape_norms(d)
    p=np.sqrt(2)-1
    assert a==pytest.approx(p*p/4,rel=1e-6)
    assert b==pytest.approx(np.sqrt(p**3/12),rel=1e-6)
    grid=np.linspace(0,4,100001)
    diagram=np.array([[0,2],[.4,1.2],[1,3.4]])
    tents=np.maximum(0,np.minimum(grid-diagram[:,0,None],diagram[:,1,None]-grid))
    l1,l2=landscape_norms(diagram)
    assert l1==pytest.approx(np.trapezoid(tents.sum(axis=0),grid),rel=1e-7)
    assert l2==pytest.approx(np.sqrt(np.trapezoid((tents**2).sum(axis=0),grid)),rel=1e-7)


def test_invariances_and_scaling():
    rng=np.random.default_rng(23)
    x=rng.normal(size=(50,4))
    l1,l2=landscape_norms(diagram_for(x))
    a,b=landscape_norms(diagram_for(x[rng.permutation(len(x))]+np.array([3,-2,6,1])))
    np.testing.assert_allclose([a,b],[l1,l2],rtol=1e-5)
    a,b=landscape_norms(diagram_for(x*3))
    np.testing.assert_allclose([a,b],[l1*9,l2*3**1.5],rtol=1e-5)
    assert landscape_norms(diagram_for(np.arange(12)[:,None]))==(0,0)
    with pytest.raises(ValueError):landscape_norms([[1,np.inf]])


def test_past_only_percentiles():
    s=pd.Series([1.,2.,3.,4.,0.,5.])
    p=prior_percentile(s,lookback=3,minimum=2)
    assert p.iloc[2]==100 and p.iloc[4]==0 and p.iloc[5]==100
    pd.testing.assert_series_equal(prior_percentile(s.iloc[:4],minimum=2),prior_percentile(s,minimum=2).iloc[:4])


def test_causal_window_and_trailing_metrics():
    x=np.random.default_rng(5).normal(size=(150,4))
    original=window_metrics(x[40:90])
    x[90:]*=100
    assert original==window_metrics(x[40:90])
    frame=pd.DataFrame({k:np.linspace(1,2,810)+np.sin(np.arange(810)/10)*.1 for k in ["l1","l2","scale_free_l1","zscore_l1","annual_rms_vol_pct"]})
    full=add_trailing_features(frame)
    short=add_trailing_features(frame.iloc[:790])
    pd.testing.assert_frame_equal(short,full.iloc[:790])
