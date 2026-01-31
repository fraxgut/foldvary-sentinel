# F2628 (Centinela)

## Project Overview
**F2628** is a serverless market monitoring bot written in Python. It is designed to run automatically via GitHub Actions to analyse macro-economic anomalies based on the "18-year Cycle" framework.

It acts as a "Depression Detector," monitoring variables like Credit Spreads, Reverse Repo, and Bond Yields. When critical thresholds are breached, it uses the Google Gemini API to generate a stylized alert (in Spanish, utilizing a "Centinela" persona) and delivers it via Telegram.
Trading signals exist but are auxiliary; the primary mission is depression/crisis detection.

> **Note:** Fred Foldvary was the main inspiration for this project.

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

Data comes from Yahoo Finance (prices, yields, volumes) and FRED (credit spreads, Fed balance sheet, RRP, TGA, SOFR, jobless claims, housing). The analysis evaluates multiple crisis and stress triggers in priority order. If any fires, Gemini generates a Spanish-language alert in a technical Spanish persona and dispatches it to Telegram.

Normal status reports are suppressed during the day and only sent as an end-of-day heartbeat (20:00+ UTC) to avoid spam. Crisis alerts fire immediately regardless of time.

## Scenarios

**Crisis / stress alerts:**

| Event | Condition (summary) | Meaning |
|---|---|---|
| `COMBO_CRISIS` | `SOLVENCY_DEATH` + `SUGAR_CRASH` same day | Credit freeze + euphoria = maximum systemic risk |
| `DEPRESSION_ALERT` | Euforia + Estrés interbancario + shock de volatilidad (30d) | Riesgo de contracción prolongada |
| `DEPRESSION_WATCH` | Euforia + Estrés interbancario o laboral (30d) | Vigilancia de depresión |
| `TEMPORAL_CRISIS` | SUGAR+EM or SOLVENCY+WAR within 30d | Converging structural fractures |
| `SOLVENCY_DEATH` | HY spread > 5.0% for 3+ days | Credit market freeze |
| `HOUSING_BUST` | Housing starts < -1.5σ and rates > 52w avg | Land/housing cycle rollover |
| `BOND_FREEZE` | US10Y > 2σ + RSI > 70 | Bond market panic |
| `EM_CURRENCY_STRESS` | DXY 95th pct + US10Y > SMA250 (fallback DXY>107) | EM stress / USD squeeze |
| `WAR_PROTOCOL` | Oil > 2σ + Gold high + SPX low | Geopolitical shock |
| `SUGAR_CRASH` | SPX 50d high + RSI divergence + VIX < 13 | Euphoria / late-cycle warning |
| `LABOUR_SHOCK` | ICSA 4w avg > 2σ | Labor deterioration |
| `INTERBANK_STRESS` | SOFR > DFF + 10bps | Funding stress |
| `FLASH_MOVE` | >5% single-session shock | Sudden volatility event |

**Entry signals:**

| Event | Condition | Action |
|-------|-----------|--------|
| `BUY_WPM_NOW` | WPM drops >5% + RSI < 30 + Volume > 2x avg + Bollinger lower break | Capitulation entry in Wheaton Precious Metals |
| `BUY_BTC_NOW` | Net Liquidity crosses above 20-day SMA + BTC RSI < 60 | Fed liquidity pivot, buy Bitcoin |

**Exit signals** (requires state persistence setup):

| Event | Condition | Action |
|-------|-----------|--------|
| `SELL_WPM_NOW` | RSI > 70 OR price up >30% from entry | Take profit on WPM |
| `SELL_BTC_NOW` | Net Liquidity crosses below SMA OR BTC RSI > 80 | Exit BTC position |
| `SELL_ALL_NOW` | Solvency crisis while holding positions | Emergency liquidation |

## Data sources

**Yahoo Finance:** `^TNX` (10Y yield), `^SPX`, `^VIX`, `CL=F` (oil), `GC=F` (gold), `SI=F` (silver), `HG=F` (copper), `HRC=F` (steel), `DX-Y.NYB` (DXY), `ITA`, `WPM`, `BTC-USD`

**FRED:** `BAMLH0A0HYM2` (HY spread), `RRPONTSYD` (reverse repo), `WALCL` (Fed assets), `WTREGEN` (TGA), `ICSA` (claims), `SOFR` (interbank), `DFF` (Fed funds), `USREC` (NBER recession flag), `HOUST`, `MORTGAGE30US`, `CSUSHPINSA`, `DRSFRMACBS`

Net Liquidity formula: `Fed Assets - TGA - RRP`

Note that FRED data (especially `WALCL`) updates weekly, not daily. The 10-day SMA crossover for the BTC signal is effectively a multi-week lookback.

## Backtests (Perspective Only)
- Latest results: `output/maximalist_backtest.csv`
- Depression proxy summary: `docs/DEPRESSION_BACKTEST_SUMMARY.md`
- Combo analysis output: `output/combo_analysis.csv`

## Dependencies

- `yfinance`
- `pandas`
- `pandas_datareader`
- `requests`
- `google-genai`
- `fpdf2` (PDF reports)

## Limitations

- The MOVE Index (bond market volatility) is not available through Yahoo Finance or FRED. The `BOND_FREEZE` scenario uses yield + RSI confirmation instead of the full three-condition check in the Pine Script version.
- FRED macro indicators are not updated daily. Forward-fill handles gaps but introduces lag.
- The free Gemini tier has rate limits. Under normal operation (max 1 alert per 4-hour window) this is not a concern.

## Licence

AGPL-3.0
