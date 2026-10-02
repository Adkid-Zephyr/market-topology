import numpy as np
import pandas as pd
from improvement_experiment import forward_label, monthly_dates, null_metric
from topology import diagram_for, landscape_norms


def test_horizon_end_and_unknown():
    p=pd.Series([100.,60.,100.,100.,100.],index=pd.date_range("2000-01-01",periods=5))
    y,end,loss=forward_label(p,0,3,.2)
    assert y==1 and end==p.index[3] and loss==-.4
    y,end,loss=forward_label(p,2,3,.2)
    assert np.isnan(y) and pd.isna(end)
    p2=pd.Series([50.,100.,100.,100.],index=pd.date_range("2000-01-01",periods=4))
    assert forward_label(p2,0,3,.2)[0]==0 # exclude loss before the forecast origin


def test_calendar_month_completion():
    idx=pd.bdate_range("2025-01-01","2025-03-28")
    out=monthly_dates(idx)
    assert list(out)==[pd.Timestamp("2025-01-31"),pd.Timestamp("2025-02-28")]


def test_null_reproducibility_and_sign_invariance():
    x=np.random.default_rng(6).normal(size=(50,4))
    a=null_metric(("2000-01-31",50,x));b=null_metric(("2000-01-31",50,x))
    assert a==b
    np.testing.assert_allclose(landscape_norms(diagram_for(x)),landscape_norms(diagram_for(-x)),rtol=1e-6)
