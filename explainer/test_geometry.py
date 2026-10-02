import numpy as np
from geometry import cycle_certificate


def test_square_has_one_certified_interval():
    x=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    result=cycle_certificate(x)
    np.testing.assert_allclose([result['birth'],result['death']],[1,np.sqrt(2)],rtol=1e-6)
    assert len(result['cycle_edges'])==4
    assert len(result['filling_triangles'])==2
    assert all(result['checks'].values())


def test_four_dimensional_isometric_embedding():
    x=np.array([[0,0],[1,0],[1,1],[0,1]],float)
    q,_=np.linalg.qr(np.random.default_rng(10).normal(size=(4,4)))
    y=np.column_stack([x,np.zeros((4,2))])@q
    a=cycle_certificate(x);b=cycle_certificate(y)
    np.testing.assert_allclose([a['birth'],a['death']],[b['birth'],b['death']],rtol=1e-6)
