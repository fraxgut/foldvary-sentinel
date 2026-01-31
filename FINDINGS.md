# Findings Report

## Scope Reviewed
- Core pipeline: `sentinel.py`
- Backtest: `backtest.py`
- Legacy analysis helpers: `legacy/analyze_signal_combinations.py`, `legacy/test_*`, `legacy/generate_backtest_report.py`
- Docs & artifacts: `docs/*.md`, `docs/backtest_run.log`, `output/*.csv`
- Workflow & deps: `.github/workflows/sentinel.yml`, `requirements.txt`

## Summary
The system is coherent but not “finished.” The largest risks are (1) logic drift between production and backtest, (2) state updates that can be skipped on multi‑signal days, and (3) inconsistent documentation/backtest artifacts that make it hard to trust historical claims. Also, the backtest is trading‑oriented (SPX forward returns), which is misaligned with a “depression predictor” goal unless explicitly reframed.

## Findings

### Critical
1. **State updates skipped when multiple signals fire.** If a BUY/SELL coincides with any other trigger, `event` becomes `MULTIPLE_CRISIS` and the state updates for entries/exits are not applied. This breaks position tracking. (`sentinel.py:872`, `sentinel.py:885`)
2. **Backtest logic diverges from production logic.** Example: production’s WPM signal requires a 1‑day >5% drop *plus* Bollinger/RSI/volume; the backtest only checks Bollinger/RSI/volume. Temporal combos in backtest can fire same‑day; production relies on persisted state so same‑day combos won’t fire. Results are not representative. (`sentinel.py:575`, `sentinel.py:730` vs `backtest.py:150`, `backtest.py:165`)

### High
3. **COMBO_CRISIS statistics are inconsistent and likely over‑stated.** In current outputs, COMBO_CRISIS shows **12** occurrences with **90‑day crash accuracy 91.7%** but **180‑day win rate 100%** (mean +5.74%). This suggests a short‑horizon drawdown signal followed by recovery, and also hints at clustered, non‑independent samples. (`output/maximalist_backtest.csv`, `output/scientific_backtest.csv`)
4. **Event naming mismatch across code, outputs, and docs.** Backtests emit `TEMPORAL (Sugar+EM)`/`TEMPORAL_CRISIS (Sugar+EM)` while production uses `TEMPORAL_CRISIS`. This makes report comparisons error‑prone. (`backtest.py:174`, `sentinel.py:713`)
5. **Prompt lacks a title for `MULTIPLE_CRISIS`.** When multiple triggers occur, the prompt’s “TÍTULOS POR EVENTO” table has no entry for `MULTIPLE_CRISIS`, so formatting is undefined. (`sentinel.py:880`, `sentinel.py:1069`)
6. **PDF generation dependency missing from requirements.** `fpdf` is used in `backtest.py` and `generate_manual.py` but not listed in `requirements.txt`, so clean installs can’t generate PDFs. (`backtest.py:11`, `generate_manual.py:1`, `requirements.txt`)

### Medium
7. **Depression‑prediction goal not reflected in backtest metric.** Backtest evaluates SPX forward returns, which is a trading proxy, not a depression indicator. There’s no evaluation against macro outcomes (NBER recessions, unemployment, credit contraction, etc.).
8. **Potential unit ambiguity for ^TNX.** If Yahoo’s `^TNX` is in “yield × 10”, thresholds like 5.5 can be off by 10×. No explicit normalization is applied. (`sentinel.py:483`, `sentinel.py:758`)
9. **BTC treated as critical data.** If BTC feed fails, the system returns `DATA_OUTAGE` and skips crisis evaluation entirely, despite BTC being non‑essential for depression detection. (`sentinel.py:488`)
10. **Backtest counts overlapping days as independent samples.** Many signals fire on consecutive days; this inflates hit rates. No de‑duplication window is applied.
11. **Legacy combination script doesn’t match current CSV schema.** It expects `backtest_results_*.csv` with `Asset_Tracked` and `Fwd_90d_%`, but current outputs are `output/*.csv` with `Asset` and `Fwd_90d`. (`legacy/analyze_signal_combinations.py`)

### Low
12. **Docs are internally inconsistent.** Multiple files disagree on thresholds (adaptive vs fixed), signal intent, and results. This undermines trust in any single report.
13. **LABOUR/LABOR naming inconsistency.** Code uses `LABOUR_SHOCK`, but some docs/legacy scripts refer to `LABOR_SHOCK`, which can break scripts or reports.

## Resolution Status (2026-01-31)
- Fixed multi‑signal state updates in `sentinel.py` (BUY/SELL now applied even under `MULTIPLE_CRISIS`).
- Backtest logic aligned with production (WPM daily drop, temporal combo timing, EM fallback, FLASH_MOVE).
- Event naming normalized (`TEMPORAL_CRISIS` + `EventDetail`) and `MULTIPLE_CRISIS` title added to prompt.
- Added depression proxy outcome columns + combo analysis output.
- Added `fpdf2` to `requirements.txt`.
- Added TNX unit normalization in prod/backtest.

## Notes on User Comments
- **COMBO_CRISIS “100% crash + 100% win at 180d”:** Current CSVs show 90‑day crash accuracy ~92% with 180‑day win rate 100%. That is not inherently contradictory; it implies a short‑term drawdown signal followed by recovery. However, the sample size is only 12 and appears clustered, so the statistic is fragile.
- **“Test all combinations”:** There is already a legacy combo‑analysis script, but it expects a different CSV format and uses a strict same‑day intersection. It should be updated and extended to test windowed combinations, then tag “trivial” combos explicitly instead of removing them.
