# Depression Backtest Summary
**Date generated:** 2026-01-31
**Data range:** 2015-01-05 to 2026-01-30
**Total signals:** 3,200

## Depression Proxy (Outcome Definition)
A forward-looking 12-month depression proxy is computed per signal date. A date is flagged when **>=3 of 5** statistically extreme conditions occur within the next 12 months, using **prior 5-year distributions** as baselines:
1. **Equity crash:** 12m max drawdown <= 5th percentile of prior 5y drawdowns.
2. **Credit freeze:** HY spread max z-score >= 2.0.
3. **Labour shock:** ICSA 4-week avg max z-score >= 2.0.
4. **Housing bust:** Housing starts min z-score <= -2.0 **and** mortgage rate max z-score >= +1.0.
5. **Interbank stress:** SOFR–DFF spread max z-score >= 2.0.

**Context only:** NBER recession flag (USREC) recorded separately, not part of the score.

## Top Signals by Depression-Proxy Rate
| Signal | Occurrences | Depression Rate | Avg Depression Score | NBER Rate |
| --- | --- | --- | --- | --- |
| DEPRESSION_ALERT | 149 | 70.5% | 3.17 | 72.5% |
| DEPRESSION_WATCH | 74 | 29.7% | 1.23 | 29.7% |
| INTERBANK_STRESS | 65 | 23.1% | 1.65 | 23.1% |
| SUGAR_CRASH | 173 | 20.2% | 1.24 | 20.8% |
| FLASH_MOVE | 1491 | 9.9% | 0.97 | 12.5% |
| LABOUR_SHOCK | 58 | 6.9% | 1.43 | 44.8% |

## Depression Watch Signals
- **DEPRESSION_ALERT:** 149 occurrences, depression rate 70.5%, avg score 3.17, NBER rate 72.5%.
- **DEPRESSION_WATCH:** 74 occurrences, depression rate 29.7%, avg score 1.23, NBER rate 29.7%.

## COMBO_CRISIS Note
COMBO_CRISIS remains a short-horizon crash signal (90d crash rate 0.0%), but it **does not score as a depression proxy**. 180d win rate is 100.0% (mean reversion after shocks).
