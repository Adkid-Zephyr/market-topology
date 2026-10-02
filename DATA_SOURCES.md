# Data sources and usage boundaries

Snapshot: **1987-09-10 through 2026-09-28**, 9,836 common price sessions and 9,835 return observations in the historical run.

| Input | Original source | Use |
|---|---|---|
| S&P 500 (`^GSPC`) | [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/history/) | Four-dimensional return coordinate; forward-loss label |
| DJIA (`^DJI`) | [Yahoo Finance](https://finance.yahoo.com/quote/%5EDJI/history/) | Four-dimensional return coordinate |
| Early DJIA | [James E. Gentle / GMU textbook archive](https://mason.gmu.edu/~jgentle/books/statfinbk/statfindata.htm) | Pre-Yahoo history; overlap checked before splice |
| NASDAQ Composite (`^IXIC`) | [Yahoo Finance](https://finance.yahoo.com/quote/%5EIXIC/history/) | Four-dimensional return coordinate |
| Russell 2000 (`^RUT`) | [Yahoo Finance](https://finance.yahoo.com/quote/%5ERUT/history/) | Four-dimensional return coordinate |
| SP500, DJIA, NASDAQCOM | [FRED](https://fred.stlouisfed.org/) | Independent 2026 price cross-check; not model input |
| VIX / VIXCLS | Yahoo / FRED | Diagnostic check; excluded from core computation and predictive features |

## Original download provenance

[reference_provenance/sources_20260928.json](reference_provenance/sources_20260928.json) lists exact URLs, UTC retrieval times, byte counts and SHA256 hashes. [audit_20260928.json](reference_provenance/audit_20260928.json) records coverage, missingness and comparison results. These files are provenance metadata, not a distribution of the supplier's price archive.

The inputs use daily percentage log returns. Missing prices are not forward-filled. The old DJIA is spliced before the first Yahoo observation after checking 6,550 overlapping sessions. The 1987-09-10 common start cannot provide a 50/100-day pre-Black-Monday warning window.

## Data rights

Public access does not establish an open-data redistribution license. This repository does not distribute downloaded supplier responses, historical price or return arrays, full point-cloud caches or the geometry certificates containing market coordinates. Code and analytical output attribution does not transfer the data providers' rights.

- Yahoo access/use is governed by [Yahoo terms](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) and applicable data-provider conditions. No unrestricted redistribution license was established for this snapshot.
- The GMU archive is cited as the original early-DJIA source. No explicit unrestricted redistribution license was established on the source page.
- FRED has [service terms and series-specific restrictions](https://fred.stlouisfed.org/legal/). Its terms include restrictions concerning AI/ML use. FRED data are not used as training inputs here; their historical role is a price cross-check. Do not substitute FRED-sourced series into an AI training pipeline merely because they are downloadable.

Obtain input data under rights appropriate to your intended use. The optional historical downloader reflects the original acquisition route; availability and permitted use are determined by the providers. A supplied licensed CSV can be used instead.

## What the public analytical files contain

Published tables contain topology summary measures, time-validation scores, fold dates, parameter selection, bootstrap intervals and post-hoc diagnostics. Plots show derived structures and summary results. Raw market input caches remain local. MIT applies to original code; third-party data and references are excluded from that grant.
