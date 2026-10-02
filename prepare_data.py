"""Prepare local inputs from an independently licensed four-index price CSV."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ['SP500', 'DJIA', 'NASDAQ', 'RUSSELL']


def prepare(source, root, asof, source_url=None):
    source, root = Path(source).resolve(), Path(root).resolve()
    frame = pd.read_csv(source)
    missing = set(['date'] + COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    dates = pd.to_datetime(frame['date'], errors='raise')
    if dates.dt.tz is not None:
        raise ValueError('Use date-only local trading dates, without timezone offsets.')
    if not dates.equals(dates.dt.normalize()):
        raise ValueError('Use one date-only row per trading session.')
    prices = frame[COLUMNS].apply(pd.to_numeric, errors='raise')
    prices.index = pd.DatetimeIndex(dates, name='date')
    if not prices.index.is_unique:
        raise ValueError('Duplicate dates are not permitted.')
    prices = prices.sort_index().loc[:pd.Timestamp(asof)]
    if len(prices) < 2 or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Need at least two rows of finite positive, nonmissing index prices.')
    data = root / 'data'
    raw = data / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    copy = raw / 'user_prices.csv'
    if copy.resolve() != source:
        shutil.copyfile(source, copy)
    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    prices.to_csv(data / 'prices_calendar.csv')
    returns = (np.log(prices).diff() * 100).iloc[1:]
    returns.to_csv(data / 'returns.csv')
    sources = {'user_prices.csv': {
        'url': source_url or 'user-provided licensed CSV',
        'path': 'data/raw/user_prices.csv',
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'sha256': digest, 'bytes': copy.stat().st_size,
    }}
    audit = {
        'input_route': 'user-provided CSV', 'asof': asof,
        'common_prices': {'first': str(prices.index[0].date()), 'last': str(prices.index[-1].date()), 'n': len(prices)},
        'common_returns': {'first': str(returns.index[0].date()), 'last': str(returns.index[-1].date()), 'n': len(returns)},
        'missing_values': 0, 'forward_filling': False,
        'provider_crosschecks': 'not performed for the user-provided CSV route',
    }
    (data / 'sources.json').write_text(json.dumps(sources, indent=2))
    (data / 'audit.json').write_text(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prices_csv', type=Path)
    parser.add_argument('--asof', default='2026-09-28')
    parser.add_argument('--source-url')
    args = parser.parse_args()
    print(json.dumps(prepare(args.prices_csv, ROOT, args.asof, args.source_url), indent=2))


if __name__ == '__main__':
    main()
