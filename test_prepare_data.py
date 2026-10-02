import numpy as np
import pandas as pd
import pytest
from prepare_data import prepare, COLUMNS


def fixture_file(tmp_path, values, dates=None):
    f = pd.DataFrame(values, columns=COLUMNS)
    f.insert(0, 'date', dates or ['2026-09-25', '2026-09-28', '2026-09-29'])
    source = tmp_path / 'input.csv'
    f.to_csv(source, index=False)
    return source


def test_returns_use_only_cutoff_prices(tmp_path):
    source = fixture_file(tmp_path, [[100]*4, [110]*4, [999]*4])
    out = tmp_path / 'project'
    audit = prepare(source, out, '2026-09-28')
    result = pd.read_csv(out / 'data/returns.csv')
    assert audit['common_prices']['n'] == 2
    assert list(result.date) == ['2026-09-28']
    np.testing.assert_allclose(result[COLUMNS], 100*np.log(1.1))


def test_bad_input_cannot_be_filled_or_duplicated(tmp_path):
    source = fixture_file(tmp_path, [[100]*4, [np.nan,110,110,110], [120]*4])
    with pytest.raises(ValueError, match='nonmissing'):
        prepare(source, tmp_path/'project', '2026-09-28')
    source = fixture_file(tmp_path, [[100]*4, [110]*4, [120]*4], ['2026-09-25']*3)
    with pytest.raises(ValueError, match='Duplicate'):
        prepare(source, tmp_path/'project', '2026-09-28')
