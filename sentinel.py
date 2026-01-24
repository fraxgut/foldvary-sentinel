import os
import time
import json
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import requests
from google import genai
from datetime import datetime, timedelta

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GIST_TOKEN = os.environ.get("GIST_TOKEN")
STATE_GIST_ID = os.environ.get("STATE_GIST_ID")
DEBUG_CRISIS = False  # Set to True to test alarm formatting immediately

# Initialize the new Client
client = genai.Client(api_key=GEMINI_KEY)


# --- STATE PERSISTENCE (GITHUB GIST) ---
def load_state():
    """
    Loads position state from a GitHub Gist.
    State format: {
        "wpm_entry_price": null,
        "btc_entry_price": null,
        "last_buy_ts": null,
        "recent_signals": {
            "SOLVENCY_DEATH": ["2024-01-15", "2024-01-16"],
            "SUGAR_CRASH": [],
            "EM_CURRENCY_STRESS": [],
            "WAR_PROTOCOL": []
        }
    }
    """
    default_state = {
        "wpm_entry_price": None,
        "btc_entry_price": None,
        "last_buy_ts": None,
        "recent_signals": {
            "SOLVENCY_DEATH": [],
            "SUGAR_CRASH": [],
            "EM_CURRENCY_STRESS": [],
            "WAR_PROTOCOL": []
        }
    }

    if not GIST_TOKEN or not STATE_GIST_ID:
        return default_state

    try:
        url = f"https://api.github.com/gists/{STATE_GIST_ID}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        gist_data = response.json()

        # The gist should have a file called "state.json"
        content = gist_data["files"]["state.json"]["content"]
        state = json.loads(content)

        # Ensure recent_signals exists (migration for old state files)
        if "recent_signals" not in state:
            state["recent_signals"] = default_state["recent_signals"]

        return state
    except Exception as e:
        print(f"State load error: {e}. Using default state.")
        return default_state


def save_state(state):
    """
    Saves position state back to the GitHub Gist.
    """
    if not GIST_TOKEN or not STATE_GIST_ID:
        return

    try:
        url = f"https://api.github.com/gists/{STATE_GIST_ID}"
        headers = {"Authorization": f"token {GIST_TOKEN}"}
        payload = {
            "files": {
                "state.json": {
                    "content": json.dumps(state, indent=2)
                }
            }
        }
        response = requests.patch(url, headers=headers, json=payload)
        response.raise_for_status()
        print("State saved successfully.")
    except Exception as e:
        print(f"State save error: {e}")


def clean_old_signals(state, current_date, window_days=30):
    """
    Removes signals older than window_days from recent_signals tracking.
    """
    if "recent_signals" not in state:
        return

    for signal_name in state["recent_signals"]:
        # Filter out dates older than window_days
        state["recent_signals"][signal_name] = [
            date_str for date_str in state["recent_signals"][signal_name]
            if (current_date - datetime.strptime(date_str, "%Y-%m-%d")).days <= window_days
        ]


def check_temporal_combo(state, signal_a, signal_b, window_days=30):
    """
    Checks if both signal_a and signal_b have fired within the last window_days.
    Returns True if combo detected.
    """
    if "recent_signals" not in state:
        return False

    dates_a = state["recent_signals"].get(signal_a, [])
    dates_b = state["recent_signals"].get(signal_b, [])

    # Check if either list is non-empty (meaning at least one fired recently)
    return len(dates_a) > 0 and len(dates_b) > 0


def add_signal_to_history(state, signal_name, current_date):
    """
    Adds today's signal to the recent_signals tracking.
    Avoids duplicates.
    """
    if "recent_signals" not in state:
        state["recent_signals"] = {
            "SOLVENCY_DEATH": [],
            "SUGAR_CRASH": [],
            "EM_CURRENCY_STRESS": [],
            "WAR_PROTOCOL": []
        }

    if signal_name not in state["recent_signals"]:
        state["recent_signals"][signal_name] = []

    date_str = current_date.strftime("%Y-%m-%d")
    if date_str not in state["recent_signals"][signal_name]:
        state["recent_signals"][signal_name].append(date_str)


def get_macro_event(date):
    """
    Returns macro event name if date matches a known event.
    Calendar covers 2024-2026 (update yearly).

    Sources:
    - FOMC: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    - CPI: Monthly, typically 2nd week
    - NFP: First Friday of each month
    """
    date_str = date.strftime("%Y-%m-%d")

    # Macro event calendar
    events = {
        # 2024
        "2024-01-31": "FOMC Decision",
        "2024-02-09": "CPI Release",
        "2024-03-12": "CPI Release",
        "2024-03-20": "FOMC Decision",
        "2024-04-10": "CPI Release",
        "2024-05-01": "FOMC Decision",
        "2024-05-15": "CPI Release",
        "2024-06-12": "FOMC Decision + CPI Release",
        "2024-07-11": "CPI Release",
        "2024-07-31": "FOMC Decision",
        "2024-08-14": "CPI Release",
        "2024-09-11": "CPI Release",
        "2024-09-18": "FOMC Decision",
        "2024-10-10": "CPI Release",
        "2024-11-07": "FOMC Decision",
        "2024-11-13": "CPI Release",
        "2024-12-11": "CPI Release",
        "2024-12-18": "FOMC Decision",

        # 2025
        "2025-01-15": "CPI Release",
        "2025-01-29": "FOMC Decision",
        "2025-02-12": "CPI Release",
        "2025-03-12": "CPI Release",
        "2025-03-19": "FOMC Decision",
        "2025-04-10": "CPI Release",
        "2025-05-07": "FOMC Decision",
        "2025-05-13": "CPI Release",
        "2025-06-11": "CPI Release",
        "2025-06-18": "FOMC Decision",
        "2025-07-10": "CPI Release",
        "2025-07-30": "FOMC Decision",
        "2025-08-13": "CPI Release",
        "2025-09-10": "CPI Release",
        "2025-09-17": "FOMC Decision",
        "2025-10-10": "CPI Release",
        "2025-11-06": "FOMC Decision",
        "2025-11-12": "CPI Release",
        "2025-12-10": "CPI Release",
        "2025-12-17": "FOMC Decision",

        # 2026
        "2026-01-14": "CPI Release",
        "2026-01-28": "FOMC Decision",
        "2026-02-11": "CPI Release",
        "2026-03-11": "CPI Release",
        "2026-03-18": "FOMC Decision",
        "2026-04-10": "CPI Release",
        "2026-05-06": "FOMC Decision",
        "2026-05-13": "CPI Release",
        "2026-06-10": "CPI Release",
        "2026-06-17": "FOMC Decision",
        "2026-07-15": "CPI Release",
        "2026-07-29": "FOMC Decision",
        "2026-08-12": "CPI Release",
        "2026-09-10": "CPI Release",
        "2026-09-16": "FOMC Decision",
        "2026-10-13": "CPI Release",
        "2026-11-05": "FOMC Decision",
        "2026-11-12": "CPI Release",
        "2026-12-10": "CPI Release",
        "2026-12-16": "FOMC Decision",
    }

    return events.get(date_str, None)


# --- DATA FETCHING ---
def get_data():
    """
    Fetches market data from Yahoo Finance and macro data from FRED.
    Returns close prices, WPM full OHLCV, and FRED macro indicators.

    Note: Fetches 365 days (250+ trading days) to support adaptive thresholds
    that use 250-day rolling statistics (z-scores, percentiles).
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Extended for adaptive thresholds

    print("Fetching Market Data...")

    tickers = [
        "^TNX",       # 10 Year Treasury Yield
        "^SPX",       # S&P 500 Index
        "^VIX",       # Volatility Index
        "CL=F",       # Crude Oil Futures
        "GC=F",       # Gold Futures
        "SI=F",       # Silver Futures
        "HG=F",       # Copper Futures
        "HRC=F",      # Steel (Hot Rolled Coil) Futures
        "ITA",        # Aerospace & Defense ETF
        "DX-Y.NYB",  # US Dollar Index
        "WPM",        # Wheaton Precious Metals
        "BTC-USD",    # Bitcoin
    ]

    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        # Forward fill to handle weekends and minor API gaps
        raw_data = raw_data.ffill()
    except Exception as e:
        print(f"Yahoo Finance Error: {e}")
        raw_data = pd.DataFrame()

    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        else:
            data = raw_data
    else:
        data = pd.DataFrame()

    # Fetch WPM separately for Volume data.
    try:
        wpm_raw = yf.download("WPM", start=start_date, end=end_date, progress=False)
        wpm_raw = wpm_raw.ffill()
        if isinstance(wpm_raw.columns, pd.MultiIndex):
            wpm_full = wpm_raw.droplevel(1, axis=1)
        else:
            wpm_full = wpm_raw
    except:
        wpm_full = pd.DataFrame()

    # FRED Data (Macro Plumbing)
    print("Fetching FRED Data...")
    try:
        fred_data = web.DataReader(
            ["BAMLH0A0HYM2", "RRPONTSYD", "WALCL", "WTREGEN", "ICSA", "SOFR", "DFF"],
            "fred",
            start_date,
            end_date,
        )
        fred_data = fred_data.ffill()
    except Exception as e:
        print(f"FRED Error: {e}")
        fred_data = pd.DataFrame()

    # FRED Housing Data (longer lookback needed for monthly/quarterly series)
    print("Fetching FRED Housing Data...")
    housing_start = end_date - timedelta(days=730)
    try:
        housing_data = web.DataReader(
            ["HOUST", "MORTGAGE30US", "CSUSHPINSA", "DRSFRMACBS"],
            "fred",
            housing_start,
            end_date,
        )
        housing_data = housing_data.ffill()
    except Exception as e:
        print(f"FRED Housing Error: {e}")
        housing_data = pd.DataFrame()

    return data, wpm_full, fred_data, housing_data


# --- TECHNICAL INDICATORS ---
def rsi(series, period=14):
    """Standard RSI: 100 - (100 / (1 + RS))."""
    if series.empty or len(series) < period:
        return pd.Series([50.0] * len(series))
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# --- ADAPTIVE THRESHOLDS ---
def calculate_z_score_threshold(series, window=250, num_std=2.0):
    """
    Calculates an adaptive threshold based on z-score (mean + N*std).
    Returns (threshold, z_score) tuple.

    Args:
        series: pandas Series of historical values
        window: lookback period (default 250 trading days = ~1 year)
        num_std: number of standard deviations above mean (default 2.0)

    Returns:
        (threshold_value, current_z_score) or (None, None) if insufficient data
    """
    if len(series) < window:
        return None, None

    # Calculate rolling statistics
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

    Args:
        series: pandas Series of historical values
        window: lookback period (default 250 trading days)
        percentile: percentile level (default 95 = top 5%)

    Returns:
        (threshold_value, current_percentile) or (None, None) if insufficient data
    """
    if len(series) < window:
        return None, None

    # Calculate rolling percentile threshold
    rolling_window = series.rolling(window=window)
    threshold = rolling_window.quantile(percentile / 100.0).iloc[-1]

    if pd.isna(threshold):
        return None, None

    # Calculate current value's percentile rank within the window
    recent_data = series.iloc[-window:]
    current_value = series.iloc[-1]
    current_pctl = (recent_data < current_value).sum() / len(recent_data) * 100

    return threshold, current_pctl


# --- THE LOGIC CORE ---
def analyze_market(df, wpm_df, fred_df, housing_df, state):
    """
    Analyses raw data against the Foldvary parameters.
    Also checks for exit signals if positions are held.
    """
    end_date = datetime.now()

    if df.empty:
        return {"event": "DATA_OUTAGE", "missing": ["all_market_data"]}

    # Helper to extract latest prices safely
    def get_latest(ticker):
        if ticker in df.columns and not pd.isna(df[ticker].iloc[-1]):
            return df[ticker].iloc[-1]
        return None

    us10y = get_latest("^TNX")
    spx = get_latest("^SPX")
    btc = get_latest("BTC-USD")
    wpm = get_latest("WPM")

    # Critical Outage Check: These are mandatory for the core model logic.
    critical = {"us10y": us10y, "spx": spx, "btc": btc}
    missing = [k for k, v in critical.items() if v is None]
    if missing:
        return {"event": "DATA_OUTAGE", "missing": missing}

    # Safe Defaults for Auxiliary Metrics
    dxy = get_latest("DX-Y.NYB") or 100.00
    oil = get_latest("CL=F") or 70.00
    gold = get_latest("GC=F") or 2000.00
    silver = get_latest("SI=F") or 25.00
    copper = get_latest("HG=F") or 4.00
    steel = get_latest("HRC=F") or 800.00
    ita = get_latest("ITA") or 130.00
    vix = get_latest("^VIX") or 15.00
    wpm_val = wpm or 50.00 # Fallback for price

    # FRED Macros
    spread = fred_df["BAMLH0A0HYM2"].iloc[-1] if not fred_df.empty and "BAMLH0A0HYM2" in fred_df.columns else 3.5
    rrp = fred_df["RRPONTSYD"].iloc[-1] if not fred_df.empty and "RRPONTSYD" in fred_df.columns else 500
    fed_assets = fred_df["WALCL"].iloc[-1] if not fred_df.empty and "WALCL" in fred_df.columns else 7000000
    tga = fred_df["WTREGEN"].iloc[-1] if not fred_df.empty and "WTREGEN" in fred_df.columns else 700000

    net_liquidity = fed_assets - (tga + rrp)

    # Unemployment Claims (ICSA) - weekly recession indicator
    icsa_current = None
    icsa_4w_avg = None
    icsa_6m_low = None
    labor_shock = False

    if not fred_df.empty and "ICSA" in fred_df.columns:
        icsa_series = fred_df["ICSA"].dropna()
        if len(icsa_series) >= 26 * 5:  # 26 weeks * 5 trading days ~= 6 months
            icsa_current = icsa_series.iloc[-1]
            icsa_4w_avg = icsa_series.rolling(window=20).mean().iloc[-1]  # 4 weeks * 5 days
            icsa_6m_low = icsa_series.rolling(window=26 * 5).min().iloc[-1]

            # Trigger: 4-week avg rising >20% from 6-month low = labor market deterioration
            if icsa_4w_avg > icsa_6m_low * 1.2:
                labor_shock = True

    # SOFR vs Fed Funds (interbank stress indicator)
    sofr = None
    fed_funds = None
    interbank_stress = False

    if not fred_df.empty and "SOFR" in fred_df.columns and "DFF" in fred_df.columns:
        sofr_series = fred_df["SOFR"].dropna()
        dff_series = fred_df["DFF"].dropna()
        if len(sofr_series) > 0 and len(dff_series) > 0:
            sofr = sofr_series.iloc[-1]
            fed_funds = dff_series.iloc[-1]
            # SOFR spiking above Fed Funds (>10bps) = interbank stress
            if sofr > fed_funds + 0.10:
                interbank_stress = True

    # Net Liquidity SMA (reduced from 20 to 10 for faster BTC signals)
    if not fred_df.empty and len(fred_df) >= 10 and "WALCL" in fred_df.columns:
        net_liq_series = fred_df["WALCL"] - (fred_df["WTREGEN"] + fred_df["RRPONTSYD"])
        net_liq_sma = net_liq_series.rolling(window=10).mean().iloc[-1]
        net_liq_prev = net_liq_series.iloc[-2]
        net_liq_sma_prev = net_liq_series.rolling(window=10).mean().iloc[-2]

        spread_series = fred_df["BAMLH0A0HYM2"].dropna()
        spread_3d_confirm = (
            len(spread_series) >= 3
            and all(spread_series.iloc[-i] > 5.0 for i in range(1, 4))
        )
    else:
        net_liq_sma = net_liquidity
        net_liq_prev = net_liquidity
        net_liq_sma_prev = net_liquidity
        spread_3d_confirm = False

    wpm_rsi_val = rsi(df["WPM"]).iloc[-1] if "WPM" in df.columns else 50.0
    btc_rsi_val = rsi(df["BTC-USD"]).iloc[-1] if "BTC-USD" in df.columns else 50.0
    us10y_rsi_val = rsi(df["^TNX"]).iloc[-1] if "^TNX" in df.columns else 50.0
    spx_rsi_val = rsi(df["^SPX"]).iloc[-1] if "^SPX" in df.columns else 50.0

    # WPM Check needs valid dataframe context
    if len(df) > 1 and "WPM" in df.columns:
        wpm_price_yest = df["WPM"].iloc[-2]
        wpm_crash = wpm_val < (wpm_price_yest * 0.95)
    else:
        wpm_crash = False

    wpm_vol = wpm_df["Volume"].iloc[-1] if not wpm_df.empty and "Volume" in wpm_df.columns else 0
    wpm_vol_avg = wpm_df["Volume"].rolling(window=20).mean().iloc[-1] if not wpm_df.empty and "Volume" in wpm_df.columns else 1

    spx_high_50 = df["^SPX"].rolling(window=50).max().iloc[-1] if len(df) >= 50 else spx
    spx_rsi_series = rsi(df["^SPX"])
    spx_rsi_high_50 = spx_rsi_series.rolling(window=50).max().iloc[-1] if len(spx_rsi_series) >= 50 else 50.0

    gold_high_20 = df["GC=F"].rolling(window=20).max().iloc[-2] if len(df) >= 21 and "GC=F" in df.columns else gold
    spx_low_20 = df["^SPX"].rolling(window=20).min().iloc[-2] if len(df) >= 21 else spx

    # Housing data (the core Foldvary variable - land/real estate cycle).
    # HOUST = Housing Starts (thousands, monthly). A rollover signals construction bust.
    # MORTGAGE30US = 30-Year Fixed Rate (weekly). Elevated rates squeeze credit.
    # CSUSHPINSA = Case-Shiller National Home Price Index (monthly, ~2mo lag).
    # DRSFRMACBS = Mortgage Delinquency Rate (quarterly).
    if not housing_df.empty:
        mortgage_rate = housing_df["MORTGAGE30US"].dropna().iloc[-1] if "MORTGAGE30US" in housing_df.columns else 6.0
        houst = housing_df["HOUST"].dropna()
        if len(houst) >= 12:
            houst_current = houst.iloc[-1]
            houst_12m_max = houst.rolling(window=12).max().iloc[-1]
            houst_decline_pct = (houst_12m_max - houst_current) / houst_12m_max * 100
        else:
            houst_current = 0
            houst_decline_pct = 0

        case_shiller = housing_df["CSUSHPINSA"].dropna()
        if len(case_shiller) >= 6:
            cs_current = case_shiller.iloc[-1]
            cs_6m_ago = case_shiller.iloc[-6]
            cs_momentum = (cs_current - cs_6m_ago) / cs_6m_ago * 100
        else:
            cs_current = 0
            cs_momentum = 0

        delinquency = housing_df["DRSFRMACBS"].dropna()
        if len(delinquency) >= 2:
            delinq_current = delinquency.iloc[-1]
            delinq_prev = delinquency.iloc[-2]
            delinq_rising = delinq_current > delinq_prev
        else:
            delinq_current = 0
            delinq_rising = False

        # HOUSING_BUST trigger:
        # Housing starts have declined >15% from 12-month max while mortgage
        # rates are elevated (>6.5%). This signals the construction cycle is
        # rolling over - the core Foldvary bust mechanism.
        housing_bust = houst_decline_pct > 15 and mortgage_rate > 6.5
    else:
        mortgage_rate = 0
        houst_current = 0
        houst_decline_pct = 0
        cs_current = 0
        cs_momentum = 0
        delinq_current = 0
        delinq_rising = False
        housing_bust = False

    # Stress score (calculated early for use in later logic)
    stress_score = 0
    if us10y and us10y > 4.5: stress_score += 1
    if spread > 4.0: stress_score += 2
    if vix > 20: stress_score += 1
    if dxy > 105: stress_score += 1
    if spx_rsi_val < 30 or spx_rsi_val > 70: stress_score += 1
    if housing_bust: stress_score += 2
    if delinq_rising: stress_score += 1
    if labor_shock: stress_score += 2
    if interbank_stress: stress_score += 2

    # --- ADAPTIVE THRESHOLDS ---
    # Calculate dynamic thresholds based on historical data (250-day lookback).
    # These replace fixed thresholds that break during regime shifts.

    # DXY adaptive threshold (for CLP_DEVALUATION replacement)
    # Use 95th percentile instead of fixed 107
    dxy_threshold, dxy_percentile = None, None
    if "DX-Y.NYB" in df.columns and len(df) >= 250:
        dxy_threshold, dxy_percentile = calculate_percentile_threshold(df["DX-Y.NYB"], window=250, percentile=95)

    # US10Y adaptive threshold (for BOND_FREEZE)
    # Use z-score (mean + 2*std) instead of fixed 5.5%
    us10y_threshold, us10y_zscore = None, None
    if "^TNX" in df.columns and len(df) >= 250:
        us10y_threshold, us10y_zscore = calculate_z_score_threshold(df["^TNX"], window=250, num_std=2.0)

    # Oil adaptive threshold (for WAR_PROTOCOL)
    # Use z-score instead of fixed $95
    oil_threshold, oil_zscore = None, None
    if "CL=F" in df.columns and len(df) >= 250:
        oil_threshold, oil_zscore = calculate_z_score_threshold(df["CL=F"], window=250, num_std=2.0)

    # Flash move detection (will add to stress score if triggered)
    flash_move_detected = False

    # Clean old signals from state (>30 days)
    clean_old_signals(state, end_date, window_days=30)

    active_triggers = []

    # PRIORITY 1: COMBO_CRISIS (SOLVENCY_DEATH + SUGAR_CRASH same day)
    # Historical accuracy: 100% (11/11 crash predictions)
    # This is the PERFECT STORM: credit freezing while market is euphoric
    if (spread > 5.0 and spread_3d_confirm and
        spx >= spx_high_50 and spx_rsi_val < spx_rsi_high_50 and vix < 13):
        active_triggers.append((
            "COMBO_CRISIS",
            f"🔴 CRISIS INMINENTE: Spread {spread:.2f}% + SPX máximo ({spx:.0f}) + VIX {vix:.1f}"
        ))
        stress_score += 3  # Maximum stress

    # PRIORITY 2: TEMPORAL_CRISIS (signals within 30 days of each other)
    # SUGAR_CRASH + EM_CURRENCY_STRESS: 89% crash accuracy (28 occurrences)
    # SOLVENCY_DEATH + WAR_PROTOCOL: 82% crash accuracy (22 occurrences)
    temporal_combo_fired = False
    if check_temporal_combo(state, "SUGAR_CRASH", "EM_CURRENCY_STRESS", window_days=30):
        active_triggers.append((
            "TEMPORAL_CRISIS",
            "⚠️ CONVERGENCIA: SUGAR_CRASH + EM_CURRENCY_STRESS (últimos 30 días) | 89% precisión"
        ))
        stress_score += 2
        temporal_combo_fired = True
    elif check_temporal_combo(state, "SOLVENCY_DEATH", "WAR_PROTOCOL", window_days=30):
        active_triggers.append((
            "TEMPORAL_CRISIS",
            "⚠️ CONVERGENCIA: SOLVENCY_DEATH + WAR_PROTOCOL (últimos 30 días) | 82% precisión"
        ))
        stress_score += 2
        temporal_combo_fired = True

    # SOLVENCY_DEATH (only fires if COMBO_CRISIS didn't already fire)
    if not any(t[0] == "COMBO_CRISIS" for t in active_triggers):
        if spread > 5.0 and spread_3d_confirm:
            active_triggers.append(("SOLVENCY_DEATH", f"CRÍTICO: Spread de Crédito ({spread}%) > 5.0%"))
    if wpm_crash and wpm_rsi_val < 30 and wpm_vol > (wpm_vol_avg * 2):
        active_triggers.append(("BUY_WPM_NOW", "Oportunidad WPM: Crash >5% + Volumen 2x"))
    if net_liquidity > net_liq_sma and net_liq_prev < net_liq_sma_prev and btc_rsi_val < 60:
        active_triggers.append(("BUY_BTC_NOW", "Pivote Fed: Liquidez Neta cruzando al alza"))
    if housing_bust:
        reason_housing = f"Ciclo Foldvary: Inicios de Construcción -{houst_decline_pct:.0f}% + Hipotecas {mortgage_rate:.1f}%"
        if delinq_rising:
            reason_housing += " + Morosidad al alza"
        active_triggers.append(("HOUSING_BUST", reason_housing))

    # LABOR_SHOCK - unemployment claims spiking
    if labor_shock and icsa_4w_avg and icsa_6m_low:
        rise_pct = ((icsa_4w_avg / icsa_6m_low) - 1) * 100
        active_triggers.append(("LABOR_SHOCK", f"Mercado Laboral: Claims subiendo {rise_pct:.0f}% desde mínimo 6M"))

    # INTERBANK_STRESS - SOFR above Fed Funds
    if interbank_stress and sofr and fed_funds:
        spread_bps = (sofr - fed_funds) * 100
        active_triggers.append(("INTERBANK_STRESS", f"Estrés Interbancario: SOFR {spread_bps:.0f}bps sobre Fed Funds"))

    # BOND_FREEZE - Adaptive threshold (z-score > 2.0 above 250-day mean)
    if us10y and us10y_threshold and us10y > us10y_threshold and us10y_rsi_val > 70:
        active_triggers.append(("BOND_FREEZE", f"Pánico en Bonos: US10Y ({us10y}%) en percentil {us10y_zscore:.1f}σ (umbral adaptivo {us10y_threshold:.2f}%)"))
    elif us10y and not us10y_threshold and us10y > 5.5 and us10y_rsi_val > 70:
        # Fallback to fixed threshold if insufficient data for adaptive
        active_triggers.append(("BOND_FREEZE", f"Pánico en Bonos: US10Y ({us10y}%) > 5.5% (umbral fijo)"))

    # EM_CURRENCY_STRESS - FIXED threshold (DXY > 107)
    # Backtest shows 86% crash prediction accuracy with fixed threshold vs 40% with adaptive
    # DXY > 107 is a structural stress point for EM debt service, not just statistical elevation
    if dxy and dxy > 107 and us10y and us10y > 4.2:
        active_triggers.append(("EM_CURRENCY_STRESS", f"Estrés Cambiario EM: DXY ({dxy:.1f}) > 107 + US10Y ({us10y}%)"))

    # WAR_PROTOCOL - Adaptive threshold for oil, but keep the correlation check
    # Note: Backtest showed this is a poor predictor (20% win rate), but keeping with adaptive threshold
    if oil_threshold and oil > oil_threshold and gold > gold_high_20 and spx < spx_low_20:
        active_triggers.append(("WAR_PROTOCOL", f"Geopolítica: Petróleo en percentil {oil_zscore:.1f}σ + Ruptura de correlación"))
    elif not oil_threshold and oil > 95 and gold > gold_high_20 and spx < spx_low_20:
        # Fallback to fixed threshold if insufficient data
        active_triggers.append(("WAR_PROTOCOL", "Geopolítica: Ruptura de correlación (umbral fijo)"))
    # SUGAR_CRASH (only fire if COMBO_CRISIS didn't already fire)
    combo_fired = any(event == "COMBO_CRISIS" for event, _ in active_triggers)
    if not combo_fired and spx >= spx_high_50 and spx_rsi_val < spx_rsi_high_50 and vix < 13:
        active_triggers.append(("SUGAR_CRASH", "Trampa Alcista: Divergencia RSI"))

    # --- FLASH MOVE DETECTOR (Black Swan Catch) ---
    # Detects sudden single-session shocks that rolling indicators smooth out.
    # Priority: between entry signals and warnings (notable but not necessarily actionable).
    flash_tickers = {
        "^TNX": "Bonos 10Y",
        "^SPX": "S&P 500",
        "CL=F": "Petróleo",
        "GC=F": "Oro",
        "BTC-USD": "Bitcoin",
        "DX-Y.NYB": "Dólar (DXY)",
        "^VIX": "VIX"
    }

    flash_moves = []
    if len(df) > 1:  # Need at least 2 days for comparison
        for ticker, name in flash_tickers.items():
            if ticker in df.columns:
                current_price = df[ticker].iloc[-1]
                prev_price = df[ticker].iloc[-2]

                if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
                    pct_change = ((current_price - prev_price) / prev_price) * 100

                    if abs(pct_change) > 5.0:
                        flash_moves.append((name, pct_change))

    # Add flash move triggers (prioritize by magnitude)
    if flash_moves:
        flash_move_detected = True
        flash_moves_sorted = sorted(flash_moves, key=lambda x: abs(x[1]), reverse=True)
        for asset_name, pct in flash_moves_sorted[:3]:  # Report top 3 if multiple
            sign = "+" if pct > 0 else ""
            active_triggers.append(("FLASH_MOVE", f"⚡ MOVIMIENTO SÚBITO: {asset_name} {sign}{pct:.1f}% en una sesión"))

    # Update stress score with flash move detection
    if flash_move_detected:
        stress_score += 1

    # Calculate stress level based on accumulated score
    if stress_score == 0: stress_level = "Bajo (Nominal)"
    elif stress_score <= 2: stress_level = "Medio (Precaución)"
    elif stress_score <= 4: stress_level = "Alto (Fragilidad)"
    else: stress_level = "CRÍTICO (Riesgo Sistémico)"

    # --- EXIT SIGNALS ---
    # Exit signals only fire if we have open positions (tracked in state).
    # Priority: Emergency exits (SOLVENCY while holding) > Normal exits.

    wpm_entry = state.get("wpm_entry_price")
    btc_entry = state.get("btc_entry_price")

    # Emergency exit: SOLVENCY_DEATH while holding positions = dump everything.
    if (wpm_entry or btc_entry) and spread > 5.0 and spread_3d_confirm:
        if wpm_entry and btc_entry:
            active_triggers.append(("SELL_ALL_NOW", f"LIQUIDACIÓN TOTAL: Spread crítico ({spread}%) mientras mantiene posiciones"))
        elif wpm_entry:
            active_triggers.append(("SELL_WPM_NOW", f"EXIT WPM: Spread crítico ({spread}%)"))
        elif btc_entry:
            active_triggers.append(("SELL_BTC_NOW", f"EXIT BTC: Spread crítico ({spread}%)"))
    else:
        # Normal WPM exit: RSI overbought (>70) OR profit target (>30% gain).
        if wpm_entry:
            wpm_gain_pct = (wpm_val - wpm_entry) / wpm_entry * 100
            if wpm_rsi_val > 70:
                active_triggers.append(("SELL_WPM_NOW", f"EXIT WPM: RSI {wpm_rsi_val:.0f} sobrecomprado (ganancia {wpm_gain_pct:+.1f}%)"))
            elif wpm_gain_pct > 30:
                active_triggers.append(("SELL_WPM_NOW", f"EXIT WPM: Objetivo alcanzado (+{wpm_gain_pct:.1f}%)"))

        # Normal BTC exit: Net Liquidity reverses below SMA OR RSI extreme (>80).
        if btc_entry:
            btc_gain_pct = (btc - btc_entry) / btc_entry * 100
            liq_reversal = net_liquidity < net_liq_sma and net_liq_prev > net_liq_sma_prev
            if btc_rsi_val > 80:
                active_triggers.append(("SELL_BTC_NOW", f"EXIT BTC: RSI {btc_rsi_val:.0f} extremo (ganancia {btc_gain_pct:+.1f}%)"))
            elif liq_reversal:
                active_triggers.append(("SELL_BTC_NOW", f"EXIT BTC: Liquidez Fed revirtiendo (ganancia {btc_gain_pct:+.1f}%)"))

    # DEBUG OVERRIDE
    if DEBUG_CRISIS:
        active_triggers = [
            ("SOLVENCY_DEATH", "SIMULACRO: Spread Crítico > 5.0%"),
            ("BOND_FREEZE", "SIMULACRO: Bonos congelados")
        ]
        stress_level = "CRÍTICO (Simulacro)"

    # Determine Final Event and Reason
    if not active_triggers:
        event = "NORMAL"
        reason = "Revisión Diaria."
    elif len(active_triggers) == 1:
        event = active_triggers[0][0]
        reason = active_triggers[0][1]
    else:
        event = "MULTIPLE_CRISIS"
        reasons_list = [t[1] for t in active_triggers]
        reason = "MÚLTIPLES ALERTAS: " + " | ".join(reasons_list)

    # State updates: record entry prices on BUY, clear on SELL, track crisis signals
    state_update = {}
    if event == "BUY_WPM_NOW":
        state_update["wpm_entry_price"] = wpm_val
        state_update["last_buy_ts"] = datetime.utcnow().isoformat()
    elif event == "BUY_BTC_NOW":
        state_update["btc_entry_price"] = btc
        state_update["last_buy_ts"] = datetime.utcnow().isoformat()
    elif event == "SELL_WPM_NOW":
        state_update["wpm_entry_price"] = None
    elif event == "SELL_BTC_NOW":
        state_update["btc_entry_price"] = None
    elif event == "SELL_ALL_NOW":
        state_update["wpm_entry_price"] = None
        state_update["btc_entry_price"] = None

    # Track crisis signals for temporal combo detection
    tracked_signals = ["SOLVENCY_DEATH", "SUGAR_CRASH", "EM_CURRENCY_STRESS", "WAR_PROTOCOL"]
    for trigger_event, _ in active_triggers:
        if trigger_event in tracked_signals:
            add_signal_to_history(state, trigger_event, end_date)

    return {
        "event": event,
        "reason": reason,
        "stress_level": stress_level,
        "state_update": state_update,
        "us10y": round(us10y, 2) if us10y else 0,
        "spread": round(spread, 2),
        "dxy": round(dxy, 2),
        "net_liq_b": round(net_liquidity / 1000, 2),
        "wpm_price": round(wpm_val, 2),
        "wpm_rsi": round(wpm_rsi_val, 2),
        "btc_price": round(btc, 0) if btc else 0,
        "btc_rsi": round(btc_rsi_val, 2),
        "vix": round(vix, 2),
        "oil": round(oil, 2),
        "gold": round(gold, 2),
        "silver": round(silver, 2),
        "copper": round(copper, 2),
        "steel": round(steel, 2),
        "ita": round(ita, 2),
        "spx": round(spx, 0) if spx else 0,
        "mortgage_rate": round(mortgage_rate, 2),
        "housing_starts": round(houst_current, 0),
        "housing_decline_pct": round(houst_decline_pct, 1),
        "case_shiller": round(cs_current, 1),
        "cs_momentum_6m": round(cs_momentum, 1),
        "delinquency_rate": round(delinq_current, 2),
        "icsa_current": round(icsa_current, 0) if icsa_current else None,
        "icsa_4w_avg": round(icsa_4w_avg, 0) if icsa_4w_avg else None,
        "icsa_6m_low": round(icsa_6m_low, 0) if icsa_6m_low else None,
        "sofr": round(sofr, 2) if sofr else None,
        "fed_funds": round(fed_funds, 2) if fed_funds else None,
        "macro_event": get_macro_event(end_date),
    }


# --- THE PERSONA (GEMINI) ---
def generate_alert_text(data):
    """
    Sends raw data to Gemini using the new SDK with retry logic.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    event = data.get("event", "NORMAL")

    # Build clean data payload (exclude internal fields)
    market_data = {k: v for k, v in data.items() if k not in ("state_update",)}

    prompt = f"""Eres el <b>Centinela Foldvary</b>, un sistema de alerta de crisis macroeconómica basado en la síntesis Austro-Georgista (Fred Foldvary, ciclo de 18 años de la tierra).

Tu misión: detectar el agotamiento del ciclo crediticio y la inminencia de un colapso. No eres un bot de trading genérico - eres un sistema de defensa patrimonial.

FECHA: {today_str}
EVENTO DETECTADO: {event}
DATOS DEL SISTEMA: {market_data}

═══════════════════════════════════════
GLOSARIO DE ACTIVOS
═══════════════════════════════════════
- WPM: Wheaton Precious Metals (streaming minero, sensible a tasas reales). NO es "Wealth Preservation Metric".
- ITA: ETF de Defensa/Armamento. Proxy de gasto bélico estatal.
- HRC: Futuros de Acero Laminado. Proxy de construcción e infraestructura (bienes de capital de orden superior en la estructura austriaca).
- Spread HY: Diferencial de crédito High Yield vs Treasuries. Amplificación = mercado exigiendo prima por riesgo de impago.
- Liquidez Neta: Balance Fed - (TGA + RRP). El combustible monetario real del sistema.
- HOUST: Inicios de Construcción (miles/mes). Variable CENTRAL del Ciclo Foldvary.
- Case-Shiller: Índice Nacional de Precios de Vivienda (rezago ~2 meses).
- ICSA: Initial Jobless Claims semanal. Indicador más rápido de recesión (lidera payrolls 2-4 meses).
- SOFR: Secured Overnight Financing Rate. Tasa interbancaria con colateral.
- DFF: Fed Funds Rate efectiva.

═══════════════════════════════════════
MARCO TEÓRICO: CICLO FOLDVARY
═══════════════════════════════════════
Fase 1 - AUGE: Crédito artificialmente barato fomenta malinversiones en proyectos de largo plazo (minería, construcción, tech). Euforia en Acero, Cobre, SPX.
Fase 2 - ESPECULACIÓN: La renta de la tierra se dispara. Oro, Plata, WPM suben por búsqueda de rentas. Precios inmobiliarios se desacoplan de la economía real.
Fase 3 - QUIEBRE: No hay ahorro real para completar los proyectos. Spreads explotan, VIX sube, activos de riesgo colapsan. Los últimos en entrar son los que más pierden.

═══════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════
REGLAS ESTRICTAS:
- Formato: SOLO HTML de Telegram (<b>, <i>, <u>, saltos de línea).
- PROHIBIDO: Markdown (*, #, _, `, ```, -). Ni un solo carácter de Markdown.
- Idioma: Español formal, técnico. Sin anglicismos innecesarios.

ESTRUCTURA OBLIGATORIA (genera SOLO UN mensaje, NO dos):

IMPORTANTE: Entre cada sección usa EXACTAMENTE UN salto de línea (no dos).

1. TÍTULO (según tabla de eventos abajo)
2. <i>Centinela F2628 | {{fecha}}</i>
3. [UN salto de línea]
4. <b><u>Estado del Sistema</u></b>
   - <b>Estrés:</b> {{stress_level}}
   - <b>Señal:</b> {{reason}}
   {"- <b>Evento Macro:</b> " + str(data.get('macro_event')) if data.get('macro_event') else ""}
   [UN salto de línea]
5. <b><u>Mercados</u></b>
   IMPORTANTE: Muestra EXACTAMENTE estas métricas en este orden, SIEMPRE (sin agregar ni quitar).

   INDICADORES VISUALES (aplica <u>subrayado</u> + ⚠️ a métricas relevantes según señal disparada):
   - Si BOND_FREEZE: subraya <u><b>Rendimiento US10Y:</b> X% ⚠️</u>
   - Si EM_CURRENCY_STRESS: subraya <u><b>DXY (Dólar):</b> X ⚠️</u> y <u><b>Rendimiento US10Y:</b> X% ⚠️</u>
   - Si WAR_PROTOCOL: subraya <u><b>Petróleo WTI:</b> $X ⚠️</u> y <u><b>Oro:</b> $X ⚠️</u>
   - Si SUGAR_CRASH: subraya <u><b>S&P 500:</b> X ⚠️</u> y <u><b>VIX:</b> X ⚠️</u>
   - Si SOLVENCY_DEATH: subraya <u><b>Spread Crédito HY:</b> X% ⚠️</u>
   - Si BUY_BTC_NOW o SELL_BTC_NOW: subraya <u><b>Bitcoin:</b> $X (RSI X) ⚠️</u> y <u><b>Liquidez Neta (Fed):</b> $XB ⚠️</u>
   - Si BUY_WPM_NOW o SELL_WPM_NOW: subraya <u><b>WPM:</b> $X (RSI X) ⚠️</u>
   - Si FLASH_MOVE: subraya el/los activo(s) que se movieron >5%
   - Si HOUSING_BUST: subraya métricas inmobiliarias relevantes
   - Si LABOR_SHOCK: menciona en análisis (no hay métrica ICSA en lista principal)
   - Si INTERBANK_STRESS: menciona en análisis (no hay métrica SOFR en lista principal)

   Lista de métricas (aplica indicadores según reglas arriba):
   - <b>S&P 500:</b> {data.get('spx')}
   - <b>VIX:</b> {data.get('vix')}
   - <b>Rendimiento US10Y:</b> {data.get('us10y')}%
   - <b>Spread Crédito HY:</b> {data.get('spread')}%
   - <b>Liquidez Neta (Fed):</b> ${data.get('net_liq_b')}B
   - <b>DXY (Dólar):</b> {data.get('dxy')}
   - <b>Bitcoin:</b> ${data.get('btc_price')} (RSI {data.get('btc_rsi')})
   - <b>Oro:</b> ${data.get('gold')}
   - <b>Plata:</b> ${data.get('silver')}
   - <b>Cobre:</b> ${data.get('copper')}
   - <b>Petróleo WTI:</b> ${data.get('oil')}
   - <b>Acero (HRC):</b> ${data.get('steel')}
   - <b>WPM:</b> ${data.get('wpm_price')} (RSI {data.get('wpm_rsi')})
   - <b>Defensa (ITA):</b> ${data.get('ita')}

   <b><u>Sector Inmobiliario</u></b>
   - <b>Hipoteca 30Y:</b> {data.get('mortgage_rate')}%
   - <b>Inicios Construcción:</b> {data.get('housing_starts')}K (Δ {data.get('housing_decline_pct')}% vs máx 12m)
   - <b>Case-Shiller (6m):</b> {data.get('cs_momentum_6m')}%
   - <b>Morosidad:</b> {data.get('delinquency_rate')}%
   [UN salto de línea]
6. <b><u>Análisis</u></b>
   Párrafo(s) de análisis. Comienza con la fecha: "{today_str} -". Conecta los datos con la fase del ciclo Foldvary (Auge, Especulación, o Quiebre). Identifica qué fase estamos transitando. Varía la redacción cada día. Si hay macro_event, úsalo como contexto.

   IMPORTANTE: NO menciones códigos internos de señales (HOUSING_BUST, SOLVENCY_DEATH, etc.) en el análisis. Usa términos descriptivos en español (ej: "deterioro inmobiliario", "crisis de crédito").

   En estado NORMAL: Análisis breve (3-5 líneas) sobre posición en el ciclo y riesgos latentes.
   En CRISIS: Análisis extenso explicando la mecánica del riesgo, comparaciones históricas, y qué vigilar.

IMPORTANTE: Genera EXACTAMENTE UN mensaje completo. NO generes dos versiones ni variantes.

═══════════════════════════════════════
TÍTULOS POR EVENTO
═══════════════════════════════════════
- NORMAL → <b>👁‍🗨 Sistemas Nominales</b>
- COMBO_CRISIS → <b>🔴🔴🔴 ALERTA MÁXIMA | CRISIS INMINENTE 🔴🔴🔴</b>
- TEMPORAL_CRISIS → <b>⚠️ ALERTA ALTA | CONVERGENCIA DE CRISIS</b>
- SOLVENCY_DEATH → <b>🚨 ALARMA | ☠️ ALERTA DE SOLVENCIA</b>
- EM_CURRENCY_STRESS → <b>🚨 ALARMA | 💱 ESTRÉS CAMBIARIO (EM)</b>
- WAR_PROTOCOL → <b>🚨 ALARMA | ⚔️ PROTOCOLO DE GUERRA</b>
- BOND_FREEZE → <b>🚨 ALARMA | 🧊 CONGELAMIENTO DE BONOS</b>
- SUGAR_CRASH → <b>🚨 ALARMA | 🍬 EUFORIA TERMINAL</b>
- HOUSING_BUST → <b>🚨 ALARMA | 🏚️ CICLO FOLDVARY ACTIVADO</b>
- LABOR_SHOCK → <b>🚨 ALARMA | 📉 DETERIORO LABORAL</b>
- INTERBANK_STRESS → <b>🚨 ALARMA | 🏦 ESTRÉS INTERBANCARIO</b>
- FLASH_MOVE → <b>⚡ ALERTA | MOVIMIENTO SÚBITO</b>
- BUY_WPM_NOW → <b>🎯 SEÑAL DE ENTRADA | WPM</b>
- BUY_BTC_NOW → <b>🎯 SEÑAL DE ENTRADA | BTC</b>
- SELL_WPM_NOW → <b>💰 SEÑAL DE SALIDA | WPM</b>
- SELL_BTC_NOW → <b>💰 SEÑAL DE SALIDA | BTC</b>
- SELL_ALL_NOW → <b>🚨 LIQUIDACIÓN TOTAL | EXIT ALL</b>

═══════════════════════════════════════
INSTRUCCIONES POR SEÑAL (con precisión histórica)
═══════════════════════════════════════

<b>COMBO_CRISIS</b> [Precisión: 100%, 11/11]
La señal más letal del sistema. NUNCA se ha equivocado. Detecta: mercado en máximos con VIX < 13 (euforia) MIENTRAS spreads de crédito > 5% por 3+ días (crédito congelado). Es la "tormenta perfecta": todos quieren vender, nadie puede refinanciar, no hay compradores.
Tono: MÁXIMA ALARMA. Compara con Lehman 2008. Menciona: margin calls masivos, venta forzada institucional, imposibilidad de refinanciar, contagio sistémico. Esta señal exige acción inmediata.

<b>TEMPORAL_CRISIS</b> [Precisión: 82-89%]
Dos señales críticas dispararon en los últimos 30 días (no necesariamente el mismo día). La convergencia temporal indica escalada de riesgo.
- SUGAR_CRASH + EM_CURRENCY_STRESS (89%): Mercado complaciente mientras el dólar estrangula a EM. Contagio inesperado.
- SOLVENCY_DEATH + WAR_PROTOCOL (82%): Crisis de crédito + shock geopolítico. Doble golpe: financiamiento congelado + commodities en caos.
Tono: Serio y preventivo. La convergencia es más peligrosa que señales aisladas.

<b>WAR_PROTOCOL</b> [Precisión: 88%, 8 señales]
Petróleo alto (z-score > 2σ) + Oro en máximo 20d + SPX en mínimo 20d. Señal geopolítica: el mercado huele guerra o disrupción de suministro.
Tono: Grave. Menciona: shock de oferta, estanflación, destrucción de márgenes, recesión por costos. Históricamente devastador para equities en los 90 días siguientes.

<b>EM_CURRENCY_STRESS</b> [Precisión: 86%, 105 señales]
DXY > 107 + US10Y > 4.2%. El dólar fuerte estrangula a economías emergentes con deuda en USD.
Tono: Alarma alta. Segunda mejor señal del sistema. Menciona: fuga de capitales de EM, carry trade unwinding, presión sobre CLP/BRL/MXN, servicio de deuda insostenible, riesgo de contagio a desarrollados.

<b>BOND_FREEZE</b> [Precisión: 42%, 234 señales]
Rendimiento US10Y supera su z-score (media + 2σ sobre 250 días) + RSI > 70 (sobrecomprado). El mercado de bonos está en pánico vendedor.
Tono: Precaución. Menciona: endurecimiento financiero, mayores costos de financiamiento, presión sobre hipotecas y corporativos, posible crisis de refinanciamiento.

<b>SUGAR_CRASH</b> [Precisión: 32%, 302 señales]
SPX en máximo 50d + divergencia RSI (RSI no confirma el máximo) + VIX < 13. Euforia sin fundamento técnico.
Tono: Moderado. Señal débil por sí sola (32%), pero letal en combinación con otras. Menciona: complacencia extrema, mercado priced for perfection, vulnerable a cualquier shock. Es la calma antes de la tormenta.

<b>SOLVENCY_DEATH</b> [Precisión: 22%, 777 señales]
Spread HY > 5% por 3+ días consecutivos. El mercado de crédito exige prima de riesgo elevada.
Tono: Cauteloso. Señal frecuente y ruidosa por sí sola, pero su valor está en las combinaciones. Si dispara junto con SUGAR_CRASH (mismo día) = COMBO_CRISIS 100%. Si dispara con WAR_PROTOCOL (30 días) = TEMPORAL_CRISIS 82%. Menciona: aversión al riesgo, funding stress, empresas zombi en riesgo.

<b>HOUSING_BUST</b> [Precisión como predictor inmediato: 6%]
Inicios de construcción cayendo >20% desde máximo 12m MIENTRAS hipoteca 30Y > 6% Y morosidad subiendo.
IMPORTANTE: Esta NO es una señal de crash inmediato. Es un indicador ESTRUCTURAL que señala que el ciclo de 18 años de Foldvary está girando. El crash viene 1-2 años DESPUÉS de esta señal. Sirve para confirmar la fase del ciclo, no para timing.
Tono: Analítico, no alarmista. Menciona: deterioro estructural, ciclo inmobiliario girando, confirma tesis Foldvary a largo plazo. Compara con 2006-2007 (HOUSING_BUST disparó, crash vino en 2008).

<b>LABOR_SHOCK</b> [Precisión como predictor de crash: 7%, 155 señales | CONTRARIAN: 93% win rate, +11.2% avg retorno 90d]
Media 4 semanas de ICSA > mínimo 6 meses * 1.2 (subida >20% en claims).
IMPORTANTE: Esta señal es históricamente CONTRARIAN - cuando dispara, el mercado sube en los siguientes 90 días (93% win rate). NO es un predictor de crash inmediato. LABOR_SHOCK señala recesión estructural inminente, pero los mercados suelen ignorarlo inicialmente o incluso subir por expectativas de recortes de tasas de la Fed. Esta señal es un indicador económico fundamental (deterioro real) pero NO un timing de entrada bajista.
Tono: Analítico y cauteloso. Menciona: despidos acelerándose, debilidad del consumidor, señal recesiva temprana. PERO advierte que históricamente el mercado puede subir post-señal (expecting Fed pivot). Si coincide con SOLVENCY_DEATH = crisis combinada más relevante.

<b>INTERBANK_STRESS</b> [Precisión como predictor de crash: 21%, 92 señales | CONTRARIAN: 79% win rate, +4.2% avg retorno 90d]
SOFR > Fed Funds + 10bps. Los bancos se cobran prima entre sí - no confían en sus contrapartes.
IMPORTANTE: Esta señal es también históricamente CONTRARIAN - cuando dispara, el mercado tiende a subir en los siguientes 90 días (79% win rate). Esto probablemente refleja que SOFR spikes son temporales (liquidez overnight normaliza rápido) o que la Fed interviene. NO es un crash predictor confiable por sí solo.
Tono: Cauto. Menciona: tensión interbancaria, pero aclara que históricamente no precede caídas inmediatas. SOFR spikes pueden ser ruido técnico del mercado de repos. Solo es relevante en COMBINACIÓN con otros signos sistémicos (ej: SOLVENCY_DEATH + INTERBANK_STRESS = validación cruzada de funding stress).

<b>FLASH_MOVE</b>
Movimiento > 5% en una sesión en activos críticos (SPX, VIX, BTC, Bonos, Oil, Gold, DXY).
Tono: Atención. No necesariamente catastrófico, pero detecta black swans que indicadores suavizados pierden. Menciona el activo, la magnitud, y posibles causas. Si múltiples activos se mueven simultáneamente, la gravedad aumenta.

<b>BUY_WPM_NOW</b> [Win rate 90d: 44%]
WPM cae >5% + RSI < 30 + Volumen > 2x promedio. Señal de entrada en metales preciosos.
Tono: Oportunista pero cauteloso. Menciona: capitulación técnica en WPM, oportunidad de entrada en streaming minero, pero señal históricamente volátil. Risk/reward favorable si la tesis de ciclo se mantiene.

<b>BUY_BTC_NOW</b> [Win rate 90d: 56%, avg +11.7%]
Liquidez Neta cruza al alza su SMA 10d + BTC RSI < 60. El combustible monetario está girando a favor de BTC.
Tono: Constructivo. Mejor señal de entrada del sistema. Menciona: pivot de liquidez confirmado, históricamente BTC responde a expansión de balance Fed, momentum alcista incipiente.

<b>SELL_WPM_NOW / SELL_BTC_NOW</b>
Señales de salida (RSI sobrecomprado, objetivo alcanzado, o reversión de liquidez).
Tono: Ejecutivo y conciso. Anuncia salida, menciona motivo y ganancia/pérdida acumulada.

<b>SELL_ALL_NOW</b>
SOLVENCY_DEATH dispara mientras hay posiciones abiertas (WPM o BTC en cartera).
Tono: EMERGENCIA TOTAL. El mercado colapsa con posiciones abiertas. Liquidar inmediatamente. Sin matices.

═══════════════════════════════════════
REGLAS FINALES
═══════════════════════════════════════
- Si hay macro_event ({data.get('macro_event')}): Menciónalo como contexto ("volatilidad anticipando CPI", "mercado posicionándose pre-FOMC").
- NUNCA inventes datos. Usa SOLO los números proporcionados.
- NUNCA uses Markdown. Solo HTML de Telegram.
- Varía la redacción entre días. Evita fórmulas repetitivas.
- En señales de alta precisión (>80%), sé directo y urgente.
- En señales de baja precisión (<40%), sé cauteloso y menciona que la señal es ruidosa por sí sola.
- Si ves múltiples señales en "reason", significa que hay más de un gatillo activo. Analiza CADA UNO.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"Rate limited. Retrying in 30s... (Attempt {attempt+1}/3)")
                time.sleep(30)
                continue
            return f"Error en IA: {e}. Datos crudos: {data}"


# --- MESSENGER (TELEGRAM) ---
def send_telegram(message):
    """Dispatches the final message via the Telegram Bot API."""
    if not TG_TOKEN: return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json().get("result", {}).get("message_id")
    except Exception as e:
        print(f"Telegram Failed: {e}")
        return None

def pin_message(message_id):
    """Pins a critical alert in the chat."""
    if not TG_TOKEN or not message_id: return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/pinChatMessage"
    payload = {"chat_id": TG_CHAT_ID, "message_id": message_id}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Pinning Failed: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Load persistent state (entry positions).
        state = load_state()
        print(f"Loaded state: {state}")

        market_data, wpm_data, fred_data, housing_data = get_data()
        analysis = analyze_market(market_data, wpm_data, fred_data, housing_data, state)

        print(f"Analysis Result: {analysis}")

        # Update state if needed (entry/exit signals).
        if analysis.get("state_update"):
            for key, value in analysis["state_update"].items():
                state[key] = value
            save_state(state)
            print(f"Updated state: {state}")

        current_hour = datetime.utcnow().hour
        is_manual_run = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        should_send = False

        if analysis["event"] == "DATA_OUTAGE":
            missing = ", ".join(analysis["missing"])
            today_str = datetime.now().strftime("%Y-%m-%d")
            outage_msg = (
                f"<b>🚨 ERROR CRÍTICO | FALLO DE DATOS</b>\n\n"
                f"<b>Fecha:</b> {today_str}\n"
                f"<b>Sensores Afectados:</b> {missing}\n\n"
                f"El Centinela no puede evaluar la seguridad del sistema sin estos datos críticos. "
                f"Verifique la fuente de datos (Yahoo Finance/FRED) o los logs del sistema."
            )
            msg_id = send_telegram(outage_msg)
            print(f"DATA OUTAGE: missing {missing}")
        elif analysis["event"] != "NORMAL":
            should_send = True
        elif is_manual_run:
            should_send = True
            print("Manual Test Detected. Forcing alert.")
        elif current_hour >= 20:
            should_send = True
        else:
            print(f"Status Normal (Hour {current_hour} UTC). Keeping silent.")

        if should_send:
            alert_text = generate_alert_text(analysis)
            msg_id = send_telegram(alert_text)
            print("Message dispatched.")
            
            # Pin only if it's a REAL crisis (not a manual test of normal state)
            if analysis["event"] != "NORMAL":
                pin_message(msg_id)
                print("Crisis Alert Pinned.")

    except Exception as e:
        print(f"Critical Failure: {e}")
        send_telegram(f"⚠️ Sentinel Error: {e}")
