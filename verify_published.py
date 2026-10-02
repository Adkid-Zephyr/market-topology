"""Verify saved report summaries, without downloading prices or fitting models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    s = pd.read_csv(ROOT/'results/snapshots.csv')
    checks = []
    for label, expected in [
        ('current', [85.3, 45.9]),
        ('Dot-com reference_pre_event', [99.7, 99.7]),
        ('Lehman reference_pre_event', [90.1, 75.4]),
    ]:
        row = s[s.label == label].sort_values('window')
        assert list(row.window) == [50, 100]
        np.testing.assert_allclose(row.l1_prior_pct.round(1), expected, atol=1e-8)
        checks.append(label)
    m = pd.read_csv(ROOT/'topology_cv_results/metrics.csv')
    for period, count, positive in [('all',272,32),('post2017',104,16)]:
        q=m[(m.target=='12m_20pct')&(m.period==period)]
        assert q.model.nunique()==10
        assert (q.n==count).all() and (q.positive==positive).all()
        assert (q['last']=='2025-08-29').all()
        base=float(q[q.model=='base_rate'].brier.iloc[0])
        assert (q[q.model!='base_rate'].brier>base).all()
        checks.append(period)
    result = {
        'status':'passed', 'checks':checks,
        'scope':'Saved summaries only; not independent market-input recomputation.',
        'new_download':False, 'new_model_fit':False,
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
