# Foldvary Sentinel

Automated macro-economic monitoring system based on the Foldvary Cycle (Geo-Austrian economics). Runs on GitHub Actions, pulls live data from Yahoo Finance and FRED, evaluates crisis scenarios, generates commentary via Gemini, and sends alerts to Telegram.

The repo also contains `foldvary_sentinel.pine`, the original TradingView Pine Script prototype. It works but requires a paid TV subscription for reliable multi-ticker alerts, so the Python version replaced it.

## Setup

Fork this repo, then add secrets under Settings > Secrets > Actions:

**Required:**
- `GEMINI_API_KEY` - Free key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- `TELEGRAM_TOKEN` - Create a bot via `@BotFather` on Telegram
- `TELEGRAM_CHAT_ID` - Get yours from `@userinfobot`

**Optional (for exit signals):**
- `GIST_TOKEN` - GitHub Personal Access Token with `gist` scope
- `STATE_GIST_ID` - ID of a secret gist for state storage

See [SETUP_EXIT_SIGNALS.md](SETUP_EXIT_SIGNALS.md) for detailed instructions on enabling exit signals.

Enable the workflow under the Actions tab. It runs every 4 hours on weekdays (`0 */4 * * 1-5`).

To run locally:

```
pip install -r requirements.txt
export GEMINI_API_KEY="..." TELEGRAM_TOKEN="..." TELEGRAM_CHAT_ID="..."
python sentinel.py
```

## How it works

The script executes a four-stage pipeline: **Fetch → Analyse → Generate → Notify**.

Data comes from Yahoo Finance (prices, yields, volumes) and FRED (credit spreads, Fed balance sheet, RRP, TGA). The analysis evaluates 7 scenario triggers in priority order. If any fires, Gemini generates a Spanish-language alert in an "Austrian Economics" persona and dispatches it to Telegram.

Normal status reports are suppressed during the day and only sent as an end-of-day heartbeat (20:00+ UTC) to avoid spam. Crisis alerts fire immediately regardless of time.

## Scenarios

**Crisis alerts:**

| # | Event | Condition | Meaning |
|---|-------|-----------|---------|
| 1 | `SOLVENCY_DEATH` | HY Spread > 5.0% for 3+ consecutive observations | Corporate bond market is shutting down |
| 2 | `HOUSING_BUST` | Housing starts down >15% from 12mo max + Mortgage rate >6.5% | Foldvary land cycle rolling over |
| 3 | `BOND_FREEZE` | 10Y Yield > 5.5% + 10Y RSI > 70 | Panic in the Treasury market |
| 4 | `CLP_DEVALUATION` | DXY > 107 + 10Y > 4.2% | Emerging market currency destruction |
| 5 | `WAR_PROTOCOL` | Oil > $95 + Gold at 20-day high + SPX at 20-day low | Geopolitical correlation breakdown |
| 6 | `SUGAR_CRASH` | SPX at 50-day high + RSI divergence + VIX < 13 | Fake rally running on fumes |

**Entry signals:**

| Event | Condition | Action |
|-------|-----------|--------|
| `BUY_WPM_NOW` | WPM drops >5% + RSI < 30 + Volume > 2x avg | Capitulation entry in Wheaton Precious Metals |
| `BUY_BTC_NOW` | Net Liquidity crosses above 20-day SMA + BTC RSI < 60 | Fed liquidity pivot, buy Bitcoin |

**Exit signals** (requires state persistence setup):

| Event | Condition | Action |
|-------|-----------|--------|
| `SELL_WPM_NOW` | RSI > 70 OR price up >30% from entry | Take profit on WPM |
| `SELL_BTC_NOW` | Net Liquidity crosses below SMA OR BTC RSI > 80 | Exit BTC position |
| `SELL_ALL_NOW` | Solvency crisis while holding positions | Emergency liquidation |

## Data sources

**Yahoo Finance:** `^TNX` (10Y yield), `^SPX`, `^VIX`, `CL=F` (oil), `GC=F` (gold), `HG=F` (copper), `DX-Y.NYB` (DXY), `WPM`, `BTC-USD`

**FRED:** `BAMLH0A0HYM2` (HY spread), `RRPONTSYD` (reverse repo), `WALCL` (Fed assets), `WTREGEN` (TGA)

Net Liquidity formula: `Fed Assets - TGA - RRP`

Note that FRED data (especially `WALCL`) updates weekly, not daily. The 20-day SMA crossover for the BTC signal is effectively a multi-week lookback.

## Dependencies

- `yfinance`
- `pandas`
- `pandas_datareader`
- `requests`
- `google-generativeai`

## Limitations

- The MOVE Index (bond market volatility) is not available through Yahoo Finance or FRED. The `BOND_FREEZE` scenario uses yield + RSI confirmation instead of the full three-condition check in the Pine Script version.
- FRED macro indicators are not updated daily. Forward-fill handles gaps but introduces lag.
- The free Gemini tier has rate limits. Under normal operation (max 1 alert per 4-hour window) this is not a concern.

## Licence

AGPL-3.0
