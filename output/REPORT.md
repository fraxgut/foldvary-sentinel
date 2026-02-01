# Backtest Comparison Report
**Date generated:** 2026-01-31

## Coverage
- **Long-history run:** 1962-05-28 → 2026-01-30 (11,871 signals)
- **Modern run:** 2015-01-05 → 2026-01-30 (3,200 signals)

## What NBER Means
The NBER (National Bureau of Economic Research) is the official US recession dating body. The `USREC` series is a binary recession indicator: **1 = recession, 0 = expansion**. In these reports it is **context only** and not part of the depression proxy score.

## Key Signal Comparison
| Signal | Long Count | Long Dep% | Long Crash90 | Long Win180 | Modern Count | Modern Dep% | Modern Crash90 | Modern Win180 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DEPRESSION_ALERT | 240 | 66.7% | 25.4% | 61.3% | 149 | 70.5% | 44.3% | 75.8% |
| DEPRESSION_WATCH | 161 | 22.4% | 22.4% | 77.6% | 74 | 29.7% | 29.7% | 85.1% |
| COMBO_CRISIS | 12 | 0.0% | 91.7% | 100.0% | 6 | 0.0% | 0.0% | 100.0% |
| HOUSING_BUST | 557 | 35.4% | 73.1% | 27.8% | 0 | 0.0% | 0.0% | 0.0% |
| SOLVENCY_DEATH | 3128 | 14.8% | 37.1% | 67.4% | 519 | 1.3% | 17.3% | 92.5% |
| INTERBANK_STRESS | 92 | 21.7% | 14.1% | 40.2% | 65 | 23.1% | 4.6% | 44.6% |
| SUGAR_CRASH | 554 | 11.9% | 25.8% | 81.0% | 173 | 20.2% | 35.8% | 88.4% |

## Housing Bust Focus
- **Long-history HOUSING_BUST:** 557 signals, depression 35.4%, NBER 100.0%, 90d crash 73.1%, 180d win 27.8%.
Housing data starts in 1959, so pre-1960 coverage is limited regardless of start date.

## COMBO_CRISIS Note
- **Long-history COMBO_CRISIS:** 12 signals, 90d crash 91.7%, 180d win 100.0%, depression 0.0%.
- **Modern COMBO_CRISIS:** 6 signals, 90d crash 0.0%, 180d win 100.0%, depression 0.0%.
COMBO_CRISIS remains a **short-horizon crash signal** with mean-reversion by 180d, and very small sample size (clustered).

## Crisis/Depression Predictor Ranking (Long-history)
| Rank | Signal | Count | Depression Rate | NBER Rate |
| --- | --- | --- | --- | --- |
| 1 | DEPRESSION_ALERT | 240 | 66.7% | 69.2% |
| 2 | HOUSING_BUST | 557 | 35.4% | 100.0% |
| 3 | DEPRESSION_WATCH | 161 | 22.4% | 22.4% |
| 4 | INTERBANK_STRESS | 92 | 21.7% | 21.7% |
| 5 | SOLVENCY_DEATH | 3128 | 14.8% | 28.5% |
| 6 | SUGAR_CRASH | 554 | 11.9% | 14.4% |
| 7 | FLASH_MOVE | 3739 | 9.8% | 20.1% |
| 8 | WAR_PROTOCOL | 26 | 7.7% | 26.9% |
| 9 | EM_CURRENCY_STRESS | 1397 | 7.4% | 16.5% |
| 10 | LABOUR_SHOCK | 519 | 4.2% | 70.1% |
| 11 | TEMPORAL_CRISIS | 495 | 3.6% | 10.3% |
| 12 | BOND_FREEZE | 859 | 0.9% | 25.8% |
| 13 | COMBO_CRISIS | 12 | 0.0% | 0.0% |

## Crisis/Depression Predictor Ranking (Modern)
| Rank | Signal | Count | Depression Rate | NBER Rate |
| --- | --- | --- | --- | --- |
| 1 | DEPRESSION_ALERT | 149 | 70.5% | 72.5% |
| 2 | DEPRESSION_WATCH | 74 | 29.7% | 29.7% |
| 3 | INTERBANK_STRESS | 65 | 23.1% | 23.1% |
| 4 | SUGAR_CRASH | 173 | 20.2% | 20.8% |
| 5 | FLASH_MOVE | 1491 | 9.9% | 12.5% |
| 6 | LABOUR_SHOCK | 58 | 6.9% | 44.8% |
| 7 | SOLVENCY_DEATH | 519 | 1.3% | 7.3% |
| 8 | EM_CURRENCY_STRESS | 327 | 0.0% | 0.0% |
| 9 | TEMPORAL_CRISIS | 125 | 0.0% | 0.0% |
| 10 | BOND_FREEZE | 123 | 0.0% | 0.0% |
| 11 | WAR_PROTOCOL | 7 | 0.0% | 0.0% |
| 12 | COMBO_CRISIS | 6 | 0.0% | 0.0% |

## Notes
- Rankings are based on depression-proxy rate, not short-horizon crash accuracy.
- Signals with low counts can look extreme; treat those rates cautiously.
