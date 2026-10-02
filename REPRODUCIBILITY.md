# Reproduction

The published snapshot is fixed at **2026-09-28**. Raw prices and returns are not included. Distinguish checking saved results from recalculating them from independently obtained inputs.

## 1. Check the public package without market inputs

```bash
python -m pytest -q
python demo.py
python verify_published.py
```

Tests cover known square homology, exact landscape norms, translation/permutation/sign behavior, scale changes, past-only percentiles, label maturity, delay coordinates, diagram distance and certified cycles. The demo uses a square and synthetic Gaussian points. The public-result check recalculates display tables from saved analytical summaries; it is not an independent data replication.

## 2. Obtain inputs

See [DATA_SOURCES.md](DATA_SOURCES.md) before retrieving or reusing third-party inputs.

Two supported paths:

1. **Historical provider route:** `python fetch_data.py` retrieves the fixed-date sources listed in `data/sources.json`, checks the DJIA overlap and prepares local input files. This is an opt-in network operation. It also retrieves FRED series for cross-checking only; the training input is the Yahoo/GMU four-index series. Public endpoints may fail or revise history.
2. **Your own licensed CSV:** prepare a CSV with `date,SP500,DJIA,NASDAQ,RUSSELL`, one row per common trading session, positive comparable index levels and no missing values. Then run:

```bash
python prepare_data.py /path/to/licensed_prices.csv --asof 2026-09-28
```

Both routes write `data/prices_calendar.csv`, `data/returns.csv`, `data/audit.json` and `data/sources.json` locally. The CSV route preserves a local copy and its SHA256 for auditing. Its source URL can be recorded with `--source-url`. The original provider manifest is separately preserved in `reference_provenance/sources_20260928.json`; compare hashes rather than assuming identical inputs.

## 3. Core and prediction calculations

```bash
python topology.py
python spectral_sensitivity.py
python verify_outputs.py
python make_figures.py
python improvement_experiment.py
python verify_improvement.py
python improvement_figures.py
python topology_cv.py
python verify_topology_cv.py
python topology_cv_figures.py
python topology_visuals.py
python explainer/geometry.py
python publication/figures.py
python publication/diagnostics.py
```

The second experiment depends on the monthly feature/label output of the first. Geometry depends on `topology_visuals.py`. Each script writes its normal project-local outputs. There are four processes in the original feature calculations. Kernels, Gaussian surrogates and geometry calculations can take time; a cached run is not evidence of fresh computation. Cached representations are keyed by the input hash and algorithm identity.

`verify_outputs.py` checks raw-input hashes and sampled past-prefix recalculations. For the provider route it also verifies the recorded DJIA/FRED cross-checks; for the custom-CSV route those provider-specific checks are marked unavailable. Do not claim that an alternative provider has reproduced the original provenance audit.

## 4. Regenerate the four report cards

For the original 2026-09-28 snapshot, after the steps above:

```bash
python publication/xiaohongshu_20261003_v2/build_cards.py
```

The generator needs daily results, prediction metrics and certified ring data. Saved PNG/SVG files are provided for reading without those inputs. Cards are exactly 1200×1600. Different Chinese fonts can alter the layout. The generator checks editorial text bounds; inspect rendered cards before publication.

Optional explanatory animation:

```bash
PYTHONPATH=explainer python explainer/render.py
```

The animation additionally needs FFmpeg available on PATH. The core calculations and report cards do not need FFmpeg.

## Exact snapshot versus method replication

- `reference_provenance/` preserves original input hashes and validation records. `provenance.json` records original script/protocol hashes.
- Downloads at a later time can differ. A hash mismatch must be reported as a revised/different input, not hidden.
- Reports and the cards contain fixed dates. A new date requires a new report release and corresponding labels; changing only a downloader date is insufficient.
- Original validation JSON files document historical runs. `public_package_validation.json` documents checks executed for this open-source package. Neither implies a prospective market test.
- The auxiliary 126-trading-day / 15% loss task and full model scores are in the result CSVs. Future labels that have not matured remain unknown.
