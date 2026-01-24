"""
Backtesting Script for Foldvary Sentinel

Replays the trigger logic day-by-day over 10+ years of historical data.
Outputs signal dates and forward returns to evaluate predictive accuracy.
"""

import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import time

# --- CONFIGURATION ---
LOOKBACK_YEARS = 12
FORWARD_WINDOWS = [30, 60, 90]  # Days to measure forward performance


# --- TECHNICAL INDICATORS (Same as sentinel.py) ---
def rsi(series, period=14):
    """Standard RSI: 100 - (100 / (1 + RS))."""
    if series.empty or len(series) < period:
        return pd.Series([50.0] * len(series), index=series.index)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- ADAPTIVE THRESHOLDS (from sentinel.py) ---
def calculate_z_score_threshold(series, window=250, num_std=2.0):
    """
    Calculates an adaptive threshold based on z-score (mean + N*std).
    Returns (threshold, z_score) tuple.
    """
    if len(series) < window:
        return None, None

    rolling_mean = series.rolling(window=window).mean().iloc[-1]
    rolling_std = series.rolling(window=window).std().iloc[-1]

    if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0:
        return None, None

    threshold = rolling_mean + (num_std * rolling_std)
    current_value = series.iloc[-1]
    z_score = (current_value - rolling_mean) / rolling_std if rolling_std > 0 else 0

    return threshold, z_score


def calculate_percentile_threshold(series, window=250, percentile=95):
    """
    Calculates an adaptive threshold based on rolling percentile.
    Returns (threshold, current_percentile) tuple.
    """
    if len(series) < window:
        return None, None

    rolling_window = series.rolling(window=window)
    threshold = rolling_window.quantile(percentile / 100.0).iloc[-1]

    if pd.isna(threshold):
        return None, None

    recent_data = series.iloc[-window:]
    current_value = series.iloc[-1]
    current_pctl = (recent_data < current_value).sum() / len(recent_data) * 100

    return threshold, current_pctl


# --- DATA FETCHING ---
def fetch_historical_data():
    """
    Fetches 12 years of historical market and macro data.
    Returns aligned dataframes for market, FRED, and housing data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")

    # Yahoo Finance tickers
    tickers = [
        "^TNX", "^SPX", "^VIX", "CL=F", "GC=F", "SI=F", "HG=F", "HRC=F",
        "ITA", "DX-Y.NYB", "WPM", "BTC-USD"
    ]

    print("Downloading Yahoo Finance data...")
    try:
        # Download in chunks to avoid rate limits
        market_data_list = []
        for ticker in tickers:
            print(f"  Fetching {ticker}...")
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not ticker_data.empty:
                # Extract close prices and volume
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    close_series = ticker_data['Close'].iloc[:, 0]
                    vol_series = ticker_data['Volume'].iloc[:, 0] if 'Volume' in ticker_data.columns.get_level_values(0) else None
                else:
                    close_series = ticker_data['Close']
                    vol_series = ticker_data['Volume'] if 'Volume' in ticker_data.columns else None

                close_series.name = ticker
                market_data_list.append(close_series)

                # Store WPM volume separately
                if ticker == "WPM" and vol_series is not None:
                    vol_series.name = "WPM_Volume"
                    market_data_list.append(vol_series)

            time.sleep(0.5)  # Rate limit protection
    except Exception as e:
        print(f"Yahoo Finance Error: {e}")
        return None, None, None

    market_df = pd.concat(market_data_list, axis=1).ffill()

    # FRED Data
    print("Downloading FRED macro data...")
    try:
        fred_df = web.DataReader(
            ["BAMLH0A0HYM2", "RRPONTSYD", "WALCL", "WTREGEN", "ICSA", "SOFR", "DFF"],
            "fred",
            start_date,
            end_date
        ).ffill()
    except Exception as e:
        print(f"FRED Error: {e}")
        fred_df = pd.DataFrame()

    # FRED Housing Data
    print("Downloading FRED housing data...")
    try:
        housing_df = web.DataReader(
            ["HOUST", "MORTGAGE30US", "CSUSHPINSA", "DRSFRMACBS"],
            "fred",
            start_date,
            end_date
        ).ffill()
    except Exception as e:
        print(f"FRED Housing Error: {e}")
        housing_df = pd.DataFrame()

    print(f"Data fetched: {len(market_df)} trading days")
    return market_df, fred_df, housing_df


# --- TRIGGER LOGIC (Replicated from sentinel.py) ---
def check_triggers_for_date(market_df, fred_df, housing_df, date_idx):
    """
    Evaluates all trigger conditions for a specific date index.
    Returns dict of triggered events and their reasons.
    """
    if date_idx < 100:  # Need minimum history for indicators
        return {}

    # Slice data up to current date
    market_window = market_df.iloc[:date_idx+1]
    fred_window = fred_df.loc[:market_df.index[date_idx]] if not fred_df.empty else pd.DataFrame()
    housing_window = housing_df.loc[:market_df.index[date_idx]] if not housing_df.empty else pd.DataFrame()

    # Extract latest values
    def get_latest(df, col):
        if col in df.columns and not df[col].empty:
            val = df[col].iloc[-1]
            return val if pd.notna(val) else None
        return None

    us10y = get_latest(market_window, "^TNX")
    spx = get_latest(market_window, "^SPX")
    btc = get_latest(market_window, "BTC-USD")
    wpm = get_latest(market_window, "WPM")
    dxy = get_latest(market_window, "DX-Y.NYB") or 100.0
    oil = get_latest(market_window, "CL=F") or 70.0
    gold = get_latest(market_window, "GC=F") or 2000.0
    vix = get_latest(market_window, "^VIX") or 15.0

    # FRED metrics
    spread = get_latest(fred_window, "BAMLH0A0HYM2") or 3.5
    rrp = get_latest(fred_window, "RRPONTSYD") or 500
    fed_assets = get_latest(fred_window, "WALCL") or 7000000
    tga = get_latest(fred_window, "WTREGEN") or 700000

    if us10y is None or spx is None or btc is None:
        return {}  # Critical data missing

    # Net Liquidity
    net_liquidity = fed_assets - (tga + rrp)

    # Net Liquidity SMA crossover (changed from 20-day to 10-day for faster signals)
    if not fred_window.empty and len(fred_window) >= 10:
        net_liq_series = fred_window["WALCL"] - (fred_window["WTREGEN"] + fred_window["RRPONTSYD"])
        net_liq_sma = net_liq_series.rolling(window=10).mean().iloc[-1]
        net_liq_prev = net_liq_series.iloc[-2] if len(net_liq_series) > 1 else net_liq_series.iloc[-1]
        net_liq_sma_prev = net_liq_series.rolling(window=10).mean().iloc[-2] if len(net_liq_series) > 10 else net_liq_sma

        # Spread 3-day confirmation
        spread_series = fred_window["BAMLH0A0HYM2"].dropna()
        spread_3d_confirm = (
            len(spread_series) >= 3
            and all(spread_series.iloc[-i] > 5.0 for i in range(1, 4))
        )
    else:
        net_liq_sma = net_liquidity
        net_liq_prev = net_liquidity
        net_liq_sma_prev = net_liquidity
        spread_3d_confirm = False

    # RSI calculations
    wpm_rsi = rsi(market_window["WPM"]).iloc[-1] if "WPM" in market_window.columns else 50.0
    btc_rsi = rsi(market_window["BTC-USD"]).iloc[-1] if "BTC-USD" in market_window.columns else 50.0
    us10y_rsi = rsi(market_window["^TNX"]).iloc[-1] if "^TNX" in market_window.columns else 50.0
    spx_rsi = rsi(market_window["^SPX"]).iloc[-1] if "^SPX" in market_window.columns else 50.0

    # WPM crash check
    wpm_crash = False
    wpm_vol_spike = False
    if len(market_window) > 1 and "WPM" in market_window.columns:
        wpm_price_yest = market_window["WPM"].iloc[-2]
        wpm_crash = wpm < (wpm_price_yest * 0.95)

        if "WPM_Volume" in market_window.columns:
            wpm_vol = market_window["WPM_Volume"].iloc[-1]
            wpm_vol_avg = market_window["WPM_Volume"].rolling(window=20).mean().iloc[-1]
            wpm_vol_spike = wpm_vol > (wpm_vol_avg * 2) if wpm_vol_avg > 0 else False

    # SPX rolling highs/lows
    spx_high_50 = market_window["^SPX"].rolling(window=50).max().iloc[-1] if len(market_window) >= 50 else spx
    spx_rsi_series = rsi(market_window["^SPX"])
    spx_rsi_high_50 = spx_rsi_series.rolling(window=50).max().iloc[-1] if len(spx_rsi_series) >= 50 else 50.0

    gold_high_20 = market_window["GC=F"].rolling(window=20).max().iloc[-2] if len(market_window) >= 21 and "GC=F" in market_window.columns else gold
    spx_low_20 = market_window["^SPX"].rolling(window=20).min().iloc[-2] if len(market_window) >= 21 else spx

    # Housing metrics
    housing_bust = False
    if not housing_window.empty:
        mortgage_rate = get_latest(housing_window, "MORTGAGE30US") or 6.0
        houst = housing_window["HOUST"].dropna()

        if len(houst) >= 12:
            houst_current = houst.iloc[-1]
            houst_12m_max = houst.rolling(window=12).max().iloc[-1]
            houst_decline_pct = (houst_12m_max - houst_current) / houst_12m_max * 100 if houst_12m_max > 0 else 0
            housing_bust = houst_decline_pct > 15 and mortgage_rate > 6.5
        else:
            houst_decline_pct = 0
    else:
        mortgage_rate = 0
        houst_decline_pct = 0

    # --- ADAPTIVE THRESHOLDS ---
    # Calculate dynamic thresholds if sufficient data available (250+ days)

    # DXY: 95th percentile for EM currency stress
    dxy_threshold, dxy_percentile = None, None
    if "DX-Y.NYB" in market_window.columns and len(market_window) >= 250:
        dxy_threshold, dxy_percentile = calculate_percentile_threshold(market_window["DX-Y.NYB"], window=250, percentile=95)

    # US10Y: z-score for bond panic
    us10y_threshold, us10y_zscore = None, None
    if "^TNX" in market_window.columns and len(market_window) >= 250:
        us10y_threshold, us10y_zscore = calculate_z_score_threshold(market_window["^TNX"], window=250, num_std=2.0)

    # Oil: z-score for supply shocks
    oil_threshold, oil_zscore = None, None
    if "CL=F" in market_window.columns and len(market_window) >= 250:
        oil_threshold, oil_zscore = calculate_z_score_threshold(market_window["CL=F"], window=250, num_std=2.0)

    # --- TRIGGER EVALUATION ---
    triggers = {}

    # 1. SOLVENCY_DEATH
    if spread > 5.0 and spread_3d_confirm:
        triggers["SOLVENCY_DEATH"] = {"spread": spread}

    # 2. BUY_WPM_NOW
    if wpm_crash and wpm_rsi < 30 and wpm_vol_spike:
        triggers["BUY_WPM_NOW"] = {"wpm": wpm, "wpm_rsi": wpm_rsi}

    # 3. BUY_BTC_NOW
    if net_liquidity > net_liq_sma and net_liq_prev < net_liq_sma_prev and btc_rsi < 60:
        triggers["BUY_BTC_NOW"] = {"btc": btc, "btc_rsi": btc_rsi, "net_liq": net_liquidity}

    # 4. HOUSING_BUST
    if housing_bust:
        triggers["HOUSING_BUST"] = {"houst_decline": houst_decline_pct, "mortgage_rate": mortgage_rate}

    # 5. BOND_FREEZE - Adaptive threshold (z-score) with fixed fallback
    if us10y_threshold and us10y and us10y > us10y_threshold and us10y_rsi > 70:
        triggers["BOND_FREEZE"] = {"us10y": us10y, "us10y_rsi": us10y_rsi, "threshold_type": "adaptive", "threshold": us10y_threshold}
    elif not us10y_threshold and us10y and us10y > 5.5 and us10y_rsi > 70:
        triggers["BOND_FREEZE"] = {"us10y": us10y, "us10y_rsi": us10y_rsi, "threshold_type": "fixed", "threshold": 5.5}

    # 6. EM_CURRENCY_STRESS (formerly CLP_DEVALUATION) - Adaptive threshold (percentile) with fixed fallback
    if dxy_threshold and dxy > dxy_threshold and us10y and us10y > 4.2:
        triggers["EM_CURRENCY_STRESS"] = {"dxy": dxy, "us10y": us10y, "threshold_type": "adaptive", "threshold": dxy_threshold, "percentile": dxy_percentile}
    elif not dxy_threshold and dxy > 107 and us10y and us10y > 4.2:
        triggers["EM_CURRENCY_STRESS"] = {"dxy": dxy, "us10y": us10y, "threshold_type": "fixed", "threshold": 107}

    # 7. WAR_PROTOCOL - Adaptive threshold (z-score) with fixed fallback
    if oil_threshold and oil > oil_threshold and gold > gold_high_20 and spx < spx_low_20:
        triggers["WAR_PROTOCOL"] = {"oil": oil, "gold": gold, "spx": spx, "threshold_type": "adaptive", "threshold": oil_threshold}
    elif not oil_threshold and oil > 95 and gold > gold_high_20 and spx < spx_low_20:
        triggers["WAR_PROTOCOL"] = {"oil": oil, "gold": gold, "spx": spx, "threshold_type": "fixed", "threshold": 95}

    # 8. SUGAR_CRASH
    if spx >= spx_high_50 and spx_rsi < spx_rsi_high_50 and vix < 13:
        triggers["SUGAR_CRASH"] = {"spx": spx, "spx_rsi": spx_rsi, "vix": vix}

    # 9. LABOR_SHOCK (ICSA-based)
    labor_shock = False
    if not fred_window.empty and "ICSA" in fred_window.columns:
        icsa_series = fred_window["ICSA"].dropna()
        if len(icsa_series) >= 130:  # 26 weeks * 5 trading days ~= 6 months
            icsa_4w_avg = icsa_series.rolling(window=20).mean().iloc[-1]  # 4 weeks * 5 days
            icsa_6m_low = icsa_series.rolling(window=130).min().iloc[-1]  # 26 weeks * 5 days
            if icsa_4w_avg > icsa_6m_low * 1.2:
                labor_shock = True
                triggers["LABOR_SHOCK"] = {"icsa_4w_avg": icsa_4w_avg, "icsa_6m_low": icsa_6m_low}

    # 10. INTERBANK_STRESS (SOFR-based)
    interbank_stress = False
    if not fred_window.empty and "SOFR" in fred_window.columns and "DFF" in fred_window.columns:
        sofr_series = fred_window["SOFR"].dropna()
        dff_series = fred_window["DFF"].dropna()
        if len(sofr_series) > 0 and len(dff_series) > 0:
            sofr = sofr_series.iloc[-1]
            fed_funds = dff_series.iloc[-1]
            # SOFR spiking above Fed Funds (>10bps) = interbank stress
            if sofr > fed_funds + 0.10:
                interbank_stress = True
                triggers["INTERBANK_STRESS"] = {"sofr": sofr, "fed_funds": fed_funds}

    return triggers


# --- FORWARD RETURN CALCULATION ---
def calculate_forward_returns(market_df, signal_date_idx, asset_col, windows):
    """
    Calculates forward returns for specified windows (30, 60, 90 days).
    Returns dict of {window: return_pct}.
    """
    if asset_col not in market_df.columns:
        return {w: None for w in windows}

    entry_price = market_df[asset_col].iloc[signal_date_idx]
    if pd.isna(entry_price) or entry_price == 0:
        return {w: None for w in windows}

    returns = {}
    for window in windows:
        future_idx = signal_date_idx + window
        if future_idx < len(market_df):
            exit_price = market_df[asset_col].iloc[future_idx]
            if pd.notna(exit_price) and exit_price > 0:
                returns[window] = ((exit_price - entry_price) / entry_price) * 100
            else:
                returns[window] = None
        else:
            returns[window] = None  # Not enough future data

    return returns


def calculate_max_drawdown(market_df, signal_date_idx, asset_col, window):
    """
    Calculates maximum drawdown from peak during the forward window.
    Returns drawdown percentage (negative value).
    """
    if asset_col not in market_df.columns:
        return None

    future_idx = signal_date_idx + window
    if future_idx >= len(market_df):
        future_idx = len(market_df) - 1

    # Get price series during the forward window
    price_series = market_df[asset_col].iloc[signal_date_idx:future_idx+1]
    if price_series.empty or len(price_series) < 2:
        return None

    # Calculate running maximum and drawdown
    running_max = price_series.expanding().max()
    drawdown = ((price_series - running_max) / running_max) * 100
    max_dd = drawdown.min()

    return max_dd if pd.notna(max_dd) else None


def calculate_advanced_metrics(results_df):
    """
    Calculates advanced performance metrics for each event type.
    Returns dict with Sharpe ratio, win rate, max drawdown, etc.
    """
    if results_df.empty:
        return {}

    metrics_by_event = {}

    for event in results_df['Event'].unique():
        event_df = results_df[results_df['Event'] == event].copy()

        metrics = {}

        for window in [30, 60, 90]:
            col = f"Fwd_{window}d_%"
            returns = event_df[col].dropna()

            if len(returns) == 0:
                continue

            # Win rate
            win_rate = (returns > 0).sum() / len(returns) * 100

            # Average return
            avg_return = returns.mean()

            # Sharpe ratio (assuming 252 trading days per year, risk-free rate ~0 for simplicity)
            # Annualized: return/std * sqrt(252/window)
            if returns.std() > 0:
                sharpe = (avg_return / returns.std()) * (252 / window) ** 0.5
            else:
                sharpe = 0

            # Max drawdown
            dd_col = f"MaxDD_{window}d_%"
            if dd_col in event_df.columns:
                max_drawdown = event_df[dd_col].min()
            else:
                max_drawdown = None

            # Best/worst outcomes
            best = returns.max()
            worst = returns.min()

            metrics[f"{window}d"] = {
                "count": len(returns),
                "win_rate_%": round(win_rate, 1),
                "avg_return_%": round(avg_return, 2),
                "median_return_%": round(returns.median(), 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_%": round(max_drawdown, 2) if max_drawdown else None,
                "best_%": round(best, 2),
                "worst_%": round(worst, 2),
                "std_dev_%": round(returns.std(), 2)
            }

        metrics_by_event[event] = metrics

    return metrics_by_event


# --- BACKTEST EXECUTION ---
def run_backtest():
    """
    Main backtesting loop.
    Iterates through historical data day-by-day and records signal firings.
    """
    market_df, fred_df, housing_df = fetch_historical_data()

    if market_df is None or market_df.empty:
        print("Failed to fetch data. Aborting backtest.")
        return

    print("\n--- STARTING BACKTEST ---")
    print(f"Date range: {market_df.index[0].date()} to {market_df.index[-1].date()}")
    print(f"Total trading days: {len(market_df)}\n")

    results = []

    # Iterate through each date (skip first 100 days for indicator warmup)
    for idx in range(100, len(market_df)):
        current_date = market_df.index[idx]

        # Check triggers
        triggers = check_triggers_for_date(market_df, fred_df, housing_df, idx)

        if not triggers:
            continue

        # A signal fired - record it
        for event, metrics in triggers.items():
            # Determine asset to track for forward returns
            # MAKE MONEY MODE: Entry signals track their respective assets
            if event in ["BUY_WPM_NOW", "SELL_WPM_NOW"]:
                asset_col = "WPM"
            elif event in ["BUY_BTC_NOW", "SELL_BTC_NOW"]:
                asset_col = "BTC-USD"
            # CRISIS PREDICTOR MODE: All crisis signals track SPX crashes
            else:
                asset_col = "^SPX"  # Track equity market crashes

            # Calculate forward returns
            fwd_returns = calculate_forward_returns(market_df, idx, asset_col, FORWARD_WINDOWS)

            # Calculate max drawdowns for each window
            max_dd_30 = calculate_max_drawdown(market_df, idx, asset_col, 30)
            max_dd_60 = calculate_max_drawdown(market_df, idx, asset_col, 60)
            max_dd_90 = calculate_max_drawdown(market_df, idx, asset_col, 90)

            # Record result
            result = {
                "Date": current_date.strftime("%Y-%m-%d"),
                "Event": event,
                "Asset_Tracked": asset_col,
                **metrics,
                "Fwd_30d_%": fwd_returns.get(30),
                "Fwd_60d_%": fwd_returns.get(60),
                "Fwd_90d_%": fwd_returns.get(90),
                "MaxDD_30d_%": max_dd_30,
                "MaxDD_60d_%": max_dd_60,
                "MaxDD_90d_%": max_dd_90,
            }
            results.append(result)

            fwd_30 = fwd_returns.get(30)
            fwd_60 = fwd_returns.get(60)
            fwd_90 = fwd_returns.get(90)

            # Format forward returns, handling None values
            fwd_30_str = f"{fwd_30:+.1f}%" if fwd_30 is not None else "N/A"
            fwd_60_str = f"{fwd_60:+.1f}%" if fwd_60 is not None else "N/A"
            fwd_90_str = f"{fwd_90:+.1f}%" if fwd_90 is not None else "N/A"

            print(f"[{current_date.date()}] {event} triggered | {asset_col} forward: "
                  f"30d={fwd_30_str} 60d={fwd_60_str} 90d={fwd_90_str}")

    # Convert to DataFrame and save
    if results:
        results_df = pd.DataFrame(results)
        output_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)

        print(f"\n{'='*80}")
        print(f"BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"Total signals fired: {len(results_df)}")
        print(f"Date range: {results_df['Date'].min()} to {results_df['Date'].max()}")

        # Basic summary
        print(f"\n{'='*80}")
        print("SIGNAL FREQUENCY BY EVENT")
        print(f"{'='*80}")
        signal_counts = results_df['Event'].value_counts().sort_values(ascending=False)
        for event, count in signal_counts.items():
            print(f"  {event:25s} : {count:3d} signals")

        # Advanced metrics
        print(f"\n{'='*80}")
        print("PERFORMANCE METRICS BY EVENT")
        print(f"{'='*80}")
        advanced_metrics = calculate_advanced_metrics(results_df)

        for event, windows_data in advanced_metrics.items():
            print(f"\n{event}:")
            for window, metrics in windows_data.items():
                print(f"  {window} window:")
                print(f"    Count: {metrics['count']}")
                print(f"    Win Rate: {metrics['win_rate_%']}%")
                print(f"    Avg Return: {metrics['avg_return_%']:+.2f}% | Median: {metrics['median_return_%']:+.2f}%")
                print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                if metrics['max_drawdown_%']:
                    print(f"    Max Drawdown: {metrics['max_drawdown_%']:.2f}%")
                print(f"    Best: {metrics['best_%']:+.2f}% | Worst: {metrics['worst_%']:+.2f}%")
                print(f"    Std Dev: {metrics['std_dev_%']:.2f}%")

        # Crisis validation (did crashes happen after crisis signals?)
        print(f"\n{'='*80}")
        print("CRISIS SIGNAL VALIDATION")
        print(f"{'='*80}")
        crisis_events = ["SOLVENCY_DEATH", "HOUSING_BUST", "BOND_FREEZE", "SUGAR_CRASH", "WAR_PROTOCOL", "EM_CURRENCY_STRESS", "LABOR_SHOCK", "INTERBANK_STRESS"]
        for event in crisis_events:
            event_data = results_df[results_df['Event'] == event]
            if not event_data.empty:
                # Crisis signals should predict negative SPX returns
                fwd_90 = event_data['Fwd_90d_%'].dropna()
                if len(fwd_90) > 0:
                    crash_accuracy = (fwd_90 < 0).sum() / len(fwd_90) * 100
                    avg_decline = fwd_90[fwd_90 < 0].mean() if (fwd_90 < 0).any() else 0
                    print(f"  {event:25s} : {crash_accuracy:.0f}% predicted crash correctly")
                    if avg_decline < 0:
                        print(f"    {'':27s}   Avg decline when correct: {avg_decline:.1f}%")

        # Entry signal validation (did buys profit?)
        print(f"\n{'='*80}")
        print("ENTRY SIGNAL VALIDATION")
        print(f"{'='*80}")
        entry_events = ["BUY_WPM_NOW", "BUY_BTC_NOW"]
        for event in entry_events:
            event_data = results_df[results_df['Event'] == event]
            if not event_data.empty:
                fwd_90 = event_data['Fwd_90d_%'].dropna()
                if len(fwd_90) > 0:
                    profit_rate = (fwd_90 > 0).sum() / len(fwd_90) * 100
                    avg_gain = fwd_90[fwd_90 > 0].mean() if (fwd_90 > 0).any() else 0
                    print(f"  {event:25s} : {profit_rate:.0f}% profitable at 90d")
                    if avg_gain > 0:
                        print(f"    {'':27s}   Avg gain when profitable: +{avg_gain:.1f}%")

        print(f"\n{'='*80}\n")

    else:
        print("\n--- BACKTEST COMPLETE ---")
        print("No signals fired during the historical period.")


if __name__ == "__main__":
    run_backtest()
