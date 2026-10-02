import numpy as np
import pandas as pd
from topology_cv import sw_distance,delay_cloud,split_indices


def test_sw_geometry():
    a=np.array([[0,2],[.4,1.3]])
    b=np.array([[.2,2.4]])
    assert sw_distance([],[])==0
    assert sw_distance(a,a)==0
    assert sw_distance(a,b)>0
    np.testing.assert_allclose(sw_distance(a,b),sw_distance(b,a))
    np.testing.assert_allclose(sw_distance(a[::-1],b),sw_distance(a,b))
    np.testing.assert_allclose(sw_distance(3*a,3*b),3*sw_distance(a,b))
    assert sw_distance([[1,1]],[])==0


def test_delay_has_no_future():
    a=np.arange(100.)
    x=delay_cloud(a,80,50,5)
    assert x.max()==80 and x.min()==16
    assert np.array_equal(x[-1],[80,75,70,65])
    a[81:]=10000
    np.testing.assert_array_equal(x,delay_cloud(a,80,50,5))


def test_maturity_purge():
    ix=pd.date_range('2000-01-31',periods=180,freq='ME')
    f=pd.DataFrame({'complete_month':True,'t_y':np.arange(180)%2,'t_end':ix+pd.DateOffset(years=1)},index=ix)
    cut=pd.Timestamp('2010-01-01')
    tr,te=split_indices(f,'t',cut,cut,pd.Timestamp('2011-01-01'))
    assert f.iloc[tr].t_end.max()<cut
    assert f.index[te].min()>=cut
    # Inner validation outcomes must all be known at the outer cutoff.
    tr,te=split_indices(f,'t',pd.Timestamp('2008-01-01'),pd.Timestamp('2008-01-01'),cut,minimum=60,positive=3,maturity_cap=cut)
    assert f.iloc[te].t_end.max()<cut
