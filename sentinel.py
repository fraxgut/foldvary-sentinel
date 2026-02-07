import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_datareader.data as web
import requests
import yfinance as yf
from google import genai

import history_store
from telegram_format import sanitize_telegram_html

# --- CONFIGURATION ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GIST_TOKEN = os.environ.get("GIST_TOKEN")
STATE_GIST_ID = os.environ.get("STATE_GIST_ID")
DEBUG_CRISIS = False  # Set to True to test alarm formatting immediately
HTTP_TIMEOUT_SECONDS = 20
HTTP_MAX_RETRIES = 4
HTTP_BACKOFF_BASE_SECONDS = 2
MARKET_STALE_DAYS = 5
FRED_STALE_DAYS = {
    "BAMLH0A0HYM2": 7,    # HY spread (daily)
    "RRPONTSYD": 7,       # RRP (daily)
    "WALCL": 14,          # Fed balance sheet (weekly)
    "WTREGEN": 7,         # TGA (daily)
    "ICSA": 14,           # Weekly claims
    "SOFR": 7,            # Daily
    "DFF": 7,             # Daily
    "USREC": 45           # Monthly-ish
}
HOUSING_STALE_DAYS = {
    "MORTGAGE30US": 14,   # Weekly
    "HOUST": 45,          # Monthly
    "CSUSHPINSA": 75,     # Monthly with lag
    "DRSFRMACBS": 120     # Quarterly
}
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", os.path.join("output", "history.db"))
HISTORY_GIST_FILENAME = os.environ.get("HISTORY_GIST_FILENAME", "history.db.b64")
HISTORY_RETENTION_DAYS = int(os.environ.get("HISTORY_RETENTION_DAYS", "365"))
RISK_WINDOW_DAYS = int(os.environ.get("RISK_WINDOW_DAYS", os.environ.get("REGIME_WINDOW_DAYS", "14")))
RISK_TREND_DAYS = int(os.environ.get("RISK_TREND_DAYS", os.environ.get("REGIME_TREND_DAYS", "7")))
RISK_MIN_DAYS = int(os.environ.get("RISK_MIN_DAYS", os.environ.get("REGIME_MIN_DAYS", "7")))
CYCLE_LOOKBACK_DAYS = int(os.environ.get("CYCLE_LOOKBACK_DAYS", str(365 * 10)))
CYCLE_TREND_MONTHS = int(os.environ.get("CYCLE_TREND_MONTHS", "12"))
CYCLE_STARTS_Z_MONTHS = int(os.environ.get("CYCLE_STARTS_Z_MONTHS", "60"))

# Initialize the new Client (only if key is present)
client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

_HTML_TAG_RE = re.compile(r"</?[^>]+>")


def _request_with_retries(
    method,
    url,
    *,
    headers=None,
    json_payload=None,
    data=None,
    operation="HTTP request",
    retries=HTTP_MAX_RETRIES,
    timeout=HTTP_TIMEOUT_SECONDS,
    retry_statuses=(429, 500, 502, 503, 504),
):
    retries = max(1, int(retries))
    for attempt in range(retries):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_payload,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt >= retries - 1:
                raise
            wait_time = HTTP_BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(
                f"{operation} network error: {exc}. "
                f"Retrying in {wait_time}s ({attempt + 1}/{retries})"
            )
            time.sleep(wait_time)
            continue

        if response.status_code in retry_statuses and attempt < retries - 1:
            retry_after = response.headers.get("Retry-After")
            wait_time = None
            if retry_after:
                try:
                    wait_time = max(1, int(float(retry_after)))
                except ValueError:
                    wait_time = None
            if wait_time is None:
                wait_time = HTTP_BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(
                f"{operation} returned {response.status_code}. "
                f"Retrying in {wait_time}s ({attempt + 1}/{retries})"
            )
            time.sleep(wait_time)
            continue

        return response

    raise RuntimeError(f"{operation} failed after {retries} attempts.")


def _extract_telegram_error(response):
    try:
        payload = response.json()
    except ValueError:
        return response.text
    return payload.get("description") or response.text


def _telegram_plain_text(message):
    text = message or ""
    text = _HTML_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.strip()
    return text or "Mensaje generado sin formato."


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
            "WAR_PROTOCOL": [],
            "INTERBANK_STRESS": [],
            "LABOUR_SHOCK": [],
            "FLASH_MOVE": []
        },
        "daily_alerts": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sent_events": [],
            "sent_ops_warnings": [],
        }
    }

    if not GIST_TOKEN or not STATE_GIST_ID:
        return default_state

    url = f"https://api.github.com/gists/{STATE_GIST_ID}"
    headers = {"Authorization": f"token {GIST_TOKEN}"}
    try:
        response = _request_with_retries(
            "GET",
            url,
            headers=headers,
            operation="State load",
        )
        response.raise_for_status()
        gist_data = response.json()
    except Exception as e:
        default_state["_state_load_failed"] = True
        print(f"State load error: {e}. Using default state.")
        return default_state

    file_entry = gist_data.get("files", {}).get("state.json")
    if not file_entry or not file_entry.get("content"):
        print("State file missing in gist; initializing default state.")
        return default_state

    try:
        state = json.loads(file_entry["content"])

        # Ensure recent_signals exists (migration for old state files)
        if "recent_signals" not in state:
            state["recent_signals"] = default_state["recent_signals"]
        else:
            for key in default_state["recent_signals"]:
                state["recent_signals"].setdefault(key, [])

        # Ensure daily_alerts exists and is for today (reset if new day)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if "daily_alerts" not in state or state["daily_alerts"].get("date") != today_str:
            state["daily_alerts"] = {
                "date": today_str,
                "sent_events": [],
                "sent_ops_warnings": [],
            }
        else:
            state["daily_alerts"].setdefault("sent_ops_warnings", [])

        return state
    except Exception as e:
        default_state["_state_load_failed"] = True
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
        response = _request_with_retries(
            "PATCH",
            url,
            headers=headers,
            json_payload=payload,
            operation="State save",
        )
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

    if current_date.tzinfo is not None:
        current_date = current_date.replace(tzinfo=None)
    current_day = current_date.date()

    for signal_name in state["recent_signals"]:
        # Filter out dates older than window_days
        state["recent_signals"][signal_name] = [
            date_str for date_str in state["recent_signals"][signal_name]
            if (current_day - datetime.strptime(date_str, "%Y-%m-%d").date()).days <= window_days
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

    if not dates_a or not dates_b:
        return False

    parsed_a = []
    parsed_b = []
    for date_str in dates_a:
        try:
            parsed_a.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue
    for date_str in dates_b:
        try:
            parsed_b.append(datetime.strptime(date_str, "%Y-%m-%d").date())
        except ValueError:
            continue

    if not parsed_a or not parsed_b:
        return False

    for day_a in parsed_a:
        for day_b in parsed_b:
            if abs((day_a - day_b).days) <= window_days:
                return True
    return False


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
            "WAR_PROTOCOL": [],
            "INTERBANK_STRESS": [],
            "LABOUR_SHOCK": [],
            "FLASH_MOVE": []
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

    max_year = max(int(k.split("-")[0]) for k in events.keys())
    calendar_warning = None
    if date.year > max_year:
        calendar_warning = f"Calendario macro desactualizado (último año {max_year})"

    return events.get(date_str, None), calendar_warning


def collect_stale_data(df, fred_df, housing_df, end_date):
    """
    Detects stale series based on last available date by asset class.
    Returns a list of warnings for any series beyond the freshness threshold.
    """
    warnings = []

    def check_series(series, name, max_age_days):
        if series is None or series.dropna().empty:
            return
        last_date = series.dropna().index.max()
        if last_date is None:
            return
        age_days = (end_date.date() - last_date.date()).days
        if age_days > max_age_days:
            warnings.append(f"{name} rezagado (último {last_date.date()})")

    if not df.empty:
        for col in df.columns:
            check_series(df[col], col, MARKET_STALE_DAYS)

    if not fred_df.empty:
        for series_name, max_days in FRED_STALE_DAYS.items():
            if series_name in fred_df.columns:
                check_series(fred_df[series_name], series_name, max_days)

    if not housing_df.empty:
        for series_name, max_days in HOUSING_STALE_DAYS.items():
            if series_name in housing_df.columns:
                check_series(housing_df[series_name], series_name, max_days)

    return warnings


# --- DATA FETCHING ---
def download_with_backoff(tickers, start, end, retries=3):
    """
    Downloads data with exponential backoff (5s, 15s, 30s) to handle API hiccups.
    """
    delays = [5, 15, 30]
    for i in range(retries + 1):
        try:
            df = yf.download(tickers, start=start, end=end, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Download attempt {i+1} failed: {e}")
        
        if i < retries:
            wait = delays[i] if i < len(delays) else 30
            print(f"Retrying in {wait}s...")
            time.sleep(wait)
    return pd.DataFrame()

def normalize_tnx(series):
    """
    Normalize ^TNX units if Yahoo returns yield * 10.
    Heuristic: if median > 20, assume scaled and divide by 10.
    """
    if series is None or series.empty:
        return series
    median_val = series.median(skipna=True)
    if pd.notna(median_val) and median_val > 20:
        return series / 10.0
    return series


def get_data(start_date=None, end_date=None):
    """
    Fetches market data from Yahoo Finance and macro data from FRED.
    Returns close prices, WPM full OHLCV, and FRED macro indicators.
    Implements retry logic and SPY fallback for SPX outages.

    Note: Fetches 365 days (250+ trading days) to support adaptive thresholds.
    """
    cache_dir = os.environ.get("YFINANCE_CACHE_DIR", "/tmp/yfinance-cache")
    try:
        yf.set_tz_cache_location(cache_dir)
    except Exception as e:
        print(f"YFinance cache setup warning: {e}")

    end_date = end_date or datetime.now(timezone.utc)
    if end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
    start_date = start_date or (end_date - timedelta(days=365))

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
        "DX-Y.NYB",   # US Dollar Index
        "WPM",        # Wheaton Precious Metals
        "BTC-USD",    # Bitcoin
    ]

    # Primary Download with Retry
    raw_data = download_with_backoff(tickers, start_date, end_date)
    raw_data = raw_data.ffill()

    # Handle MultiIndex vs Single Level
    if not raw_data.empty:
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close'].copy()
        else:
            data = raw_data.copy()
    else:
        data = pd.DataFrame()

    # Normalize TNX unit scale if needed
    if "^TNX" in data.columns:
        data["^TNX"] = normalize_tnx(data["^TNX"])

    # --- FALLBACK SYSTEM ---
    def process_fallback(target, backup, name):
        """Checks if target is missing and tries to fetch backup."""
        is_missing = True
        if target in data.columns and not data[target].dropna().empty:
            is_missing = False
        
        if is_missing:
            print(f"⚠️ WARNING: {target} ({name}) missing. Attempting fallback to {backup}...")
            backup_data = download_with_backoff([backup], start_date, end_date)
            
            if not backup_data.empty:
                # Extract Close series safely
                if isinstance(backup_data.columns, pd.MultiIndex):
                    # Try to find 'Close' -> 'Ticker'
                    if 'Close' in backup_data.columns and backup in backup_data['Close'].columns:
                        series = backup_data['Close'][backup]
                    else:
                        series = backup_data.iloc[:, 0] # Best guess
                elif 'Close' in backup_data.columns:
                    series = backup_data['Close']
                else:
                    series = backup_data.iloc[:, 0]
                
                # Inject into main dataframe
                data[target] = series
                print(f"✅ FALLBACK SUCCESS: {backup} injected as {target} proxy.")
            else:
                print(f"❌ FALLBACK FAILED: {backup} also unavailable.")

    # Apply Fallbacks for Critical Sensors
    process_fallback("^SPX", "SPY", "S&P 500")        # SPY ETF as proxy for Index
    process_fallback("BTC-USD", "BTC=F", "Bitcoin")   # Futures as proxy for Spot
    process_fallback("CL=F", "BZ=F", "Crude Oil")     # Brent as proxy for WTI
    process_fallback("^TNX", "^TYX", "10Y Yield")     # 30Y Yield as proxy for 10Y
    process_fallback("^VIX", "^VXO", "VIX")           # VXO (OEX Vol) as proxy for VIX
    process_fallback("GC=F", "GLD", "Gold")           # ETF as proxy
    process_fallback("SI=F", "SLV", "Silver")         # ETF as proxy
    process_fallback("HG=F", "CPER", "Copper")        # ETF as proxy
    process_fallback("HRC=F", "SLX", "Steel")         # ETF as proxy
    process_fallback("ITA", "XAR", "Defense")         # ETF as proxy

    # Fetch WPM separately for Volume data with RETRY logic
    try:
        wpm_raw = download_with_backoff(["WPM"], start_date, end_date)
        wpm_raw = wpm_raw.ffill()
        if isinstance(wpm_raw.columns, pd.MultiIndex):
            wpm_full = wpm_raw.droplevel(1, axis=1)
        else:
            wpm_full = wpm_raw
    except Exception as e:
        print(f"WPM Download Error: {e}")
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
    if start_date:
        housing_start = start_date - timedelta(days=CYCLE_LOOKBACK_DAYS)
    else:
        housing_start = end_date - timedelta(days=CYCLE_LOOKBACK_DAYS)
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


def map_risk_level(phase_label):
    """
    Maps phase labels to risk-level labels without using the word "Fase".
    """
    if not phase_label:
        return None
    mapping = {
        "Fase 1 - AUGE": "Nivel 1 - Bajo",
        "Fase 2 - ESPECULACIÓN": "Nivel 2 - Elevado",
        "Fase 3 - QUIEBRE": "Nivel 3 - Crítico",
    }
    if phase_label in mapping:
        return mapping[phase_label]
    return phase_label.replace("Fase", "Nivel")


def compute_cycle_context(housing_df):
    """
    Builds long-cycle context from housing/credit fundamentals.
    Uses multi-year windows and monthly data (not short-term risk regime).
    """
    if housing_df is None or housing_df.empty:
        return {
            "cycle_phase": None,
            "cycle_trend": None,
            "cycle_confidence": None,
        }

    if not isinstance(housing_df.index, pd.DatetimeIndex):
        return {
            "cycle_phase": None,
            "cycle_trend": None,
            "cycle_confidence": None,
        }

    monthly = housing_df.resample("M").last().ffill()
    if monthly.empty:
        return {
            "cycle_phase": None,
            "cycle_trend": None,
            "cycle_confidence": None,
        }

    def _series(name):
        if name not in monthly.columns:
            return None
        series = monthly[name].dropna()
        return series if len(series) >= 2 else None

    starts = _series("HOUST")
    mortgage = _series("MORTGAGE30US")
    prices = _series("CSUSHPINSA")
    delinq = _series("DRSFRMACBS")

    available_series = [s for s in (starts, mortgage, prices, delinq) if s is not None]
    if not available_series:
        return {
            "cycle_phase": None,
            "cycle_trend": None,
            "cycle_confidence": None,
        }

    max_months = max(len(s) for s in available_series)

    def _score_at(pos):
        pressure = 0
        spec = 0
        metrics_used = 0

        if starts is not None:
            starts_hist = starts.iloc[:pos + 1].dropna()
            if len(starts_hist) >= CYCLE_STARTS_Z_MONTHS:
                window = starts_hist.iloc[-CYCLE_STARTS_Z_MONTHS:]
                mean = window.mean()
                std = window.std()
                if std > 0:
                    starts_z = (window.iloc[-1] - mean) / std
                else:
                    starts_z = 0
                metrics_used += 1
                if starts_z < -1.0:
                    pressure += 2
                if starts_z > 0.5:
                    spec += 1
            if len(starts_hist) >= 36:
                ma_short = starts_hist.iloc[-12:].mean()
                ma_long = starts_hist.iloc[-36:].mean()
                if ma_long > 0:
                    starts_trend = (ma_short - ma_long) / ma_long * 100
                    metrics_used += 1
                    if starts_trend < -3.0:
                        pressure += 1
                    if starts_trend > 3.0:
                        spec += 1

        if mortgage is not None:
            mort_hist = mortgage.iloc[:pos + 1].dropna()
            if len(mort_hist) >= 60:
                mort_avg = mort_hist.iloc[-60:].mean()
                gap = mort_hist.iloc[-1] - mort_avg
                metrics_used += 1
                if gap > 0.75:
                    pressure += 1
                if gap <= 0.0:
                    spec += 1

        if delinq is not None:
            del_hist = delinq.iloc[:pos + 1].dropna()
            if len(del_hist) >= 13:
                change = del_hist.iloc[-1] - del_hist.iloc[-13]
                metrics_used += 1
                if change > 0.2:
                    pressure += 1

        if prices is not None:
            price_hist = prices.iloc[:pos + 1].dropna()
            if len(price_hist) >= 13:
                yoy = (price_hist.iloc[-1] / price_hist.iloc[-13] - 1) * 100
                metrics_used += 1
                if yoy < 0:
                    pressure += 1
                if yoy >= 6:
                    spec += 1
            if len(price_hist) >= 7:
                sixm = (price_hist.iloc[-1] / price_hist.iloc[-7] - 1) * 100
                metrics_used += 1
                if sixm >= 3:
                    spec += 1

        cycle_index = pressure + (spec * 0.5)
        return {
            "pressure": pressure,
            "spec": spec,
            "metrics_used": metrics_used,
            "cycle_index": cycle_index,
        }

    last_pos = max_months - 1
    current = _score_at(last_pos)
    if not current or current["metrics_used"] == 0:
        return {
            "cycle_phase": None,
            "cycle_trend": None,
            "cycle_confidence": None,
        }

    pressure = current["pressure"]
    spec = current["spec"]
    if pressure >= 4:
        phase = "Fase 3 - QUIEBRE"
    elif spec >= 2 or pressure >= 3:
        phase = "Fase 2 - ESPECULACIÓN"
    else:
        phase = "Fase 1 - AUGE"

    trend = "Estable"
    if last_pos >= CYCLE_TREND_MONTHS:
        previous = _score_at(last_pos - CYCLE_TREND_MONTHS)
        if previous and previous["metrics_used"] > 0:
            delta = current["cycle_index"] - previous["cycle_index"]
            if delta >= 1:
                trend = "Ascendente"
            elif delta <= -1:
                trend = "Descendente"

    if max_months >= 72 and current["metrics_used"] >= 4:
        confidence = "Alta"
    elif max_months >= 36 and current["metrics_used"] >= 3:
        confidence = "Media"
    else:
        confidence = "Baja"

    return {
        "cycle_phase": phase,
        "cycle_trend": trend,
        "cycle_confidence": confidence,
    }


# --- THE LOGIC CORE ---
def analyse_market(df, wpm_df, fred_df, housing_df, state, end_date=None):
    """
    Analyses raw data against the Foldvary parameters.
    Also checks for exit signals if positions are held.
    """
    end_date = end_date or datetime.now(timezone.utc)

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
    critical = {"us10y": us10y, "spx": spx}
    missing = [k for k, v in critical.items() if v is None]
    if missing:
        return {"event": "DATA_OUTAGE", "missing": missing}

    data_warnings = collect_stale_data(df, fred_df, housing_df, end_date)

    # Safe Defaults for Auxiliary Metrics (Updated for 2026 Baseline)
    # Uses these ONLY if primary AND backup feeds fail completely.
    dxy = get_latest("DX-Y.NYB") or 96.00
    oil = get_latest("CL=F") or 65.00
    gold = get_latest("GC=F") or 5200.00
    silver = get_latest("SI=F") or 100.00
    copper = get_latest("HG=F") or 6.00
    steel = get_latest("HRC=F") or 960.00
    ita = get_latest("ITA") or 230.00
    vix = get_latest("^VIX") or 18.00
    wpm_val = wpm or 150.00 # Fallback for price

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
    labour_shock = False

    if not fred_df.empty and "ICSA" in fred_df.columns:
        icsa_series = fred_df["ICSA"].dropna()
        if len(icsa_series) >= 26 * 5:  # 26 weeks * 5 trading days ~= 6 months
            icsa_current = icsa_series.iloc[-1]
            icsa_4w_avg = icsa_series.rolling(window=20).mean().iloc[-1]  # 4 weeks * 5 days
            icsa_6m_low = icsa_series.rolling(window=26 * 5).min().iloc[-1]

            # Trigger: Adaptive Z-Score (Mean + 2.0 StdDev over 1 year)
            # Replaces arbitrary fixed % thresholds with statistical anomaly detection.
            icsa_threshold, icsa_zscore = calculate_z_score_threshold(icsa_series, window=250, num_std=2.0)

            if icsa_threshold and icsa_4w_avg > icsa_threshold:
                labour_shock = True

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
        # Housing starts collapsing statistically (Z-Score < -1.5) while rates are rising
        # Replaces fixed 15% decline and 6.5% rate with statistical deviation logic.
        
        # Calculate Z-Score for Housing Starts (detect abnormal weakness)
        # Data is daily (ffilled), so 2-year lookback = ~504 trading days
        if len(houst) >= 252:
            houst_window = houst.rolling(window=504) # 2 year lookback for housing cycle
            houst_mean = houst_window.mean().iloc[-1]
            houst_std = houst_window.std().iloc[-1]
            houst_zscore = (houst_current - houst_mean) / houst_std if houst_std > 0 else 0
        else:
            houst_zscore = 0
            
        # Housing Bust: Starts are 1.5 deviations BELOW mean AND Rates > 52-week Avg
        # This confirms "Weak Construction" + "Tightening Credit" relative to recent history
        rate_52w_avg = housing_df["MORTGAGE30US"].rolling(window=252).mean().iloc[-1] if "MORTGAGE30US" in housing_df.columns else 6.0
        
        housing_bust = houst_zscore < -1.5 and mortgage_rate > rate_52w_avg
    else:
        mortgage_rate = 0
        houst_current = 0
        houst_decline_pct = 0
        cs_current = 0
        cs_momentum = 0
        delinq_current = 0
        delinq_rising = False
        housing_bust = False
        houst_zscore = 0
        rate_52w_avg = 0

    # Stress score (calculated early for use in later logic)
    stress_score = 0
    if us10y and us10y > 4.5: stress_score += 1
    if spread > 4.0: stress_score += 2
    if vix > 20: stress_score += 1
    if dxy > 105: stress_score += 1
    if spx_rsi_val < 30 or spx_rsi_val > 70: stress_score += 1
    if housing_bust: stress_score += 2
    if delinq_rising: stress_score += 1
    if labour_shock: stress_score += 2
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

    # PRIORITY 2: DEPRESSION WATCH (systemic combos within 30 days)
    # Top combos from depression proxy analysis
    depression_alert_fired = False
    if check_temporal_combo(state, "SUGAR_CRASH", "INTERBANK_STRESS", window_days=30) and check_temporal_combo(state, "SUGAR_CRASH", "FLASH_MOVE", window_days=30):
        active_triggers.append((
            "DEPRESSION_ALERT",
            "⚠️ VIGILANCIA DEPRESIÓN: Euforia + Estrés Interbancario + Shock de Volatilidad (30d)"
        ))
        stress_score += 3
        depression_alert_fired = True
    elif check_temporal_combo(state, "SUGAR_CRASH", "INTERBANK_STRESS", window_days=30):
        active_triggers.append((
            "DEPRESSION_WATCH",
            "⚠️ VIGILANCIA DEPRESIÓN: Euforia + Estrés Interbancario (30d)"
        ))
        stress_score += 2
        depression_alert_fired = True
    elif check_temporal_combo(state, "SUGAR_CRASH", "LABOUR_SHOCK", window_days=30):
        active_triggers.append((
            "DEPRESSION_WATCH",
            "⚠️ VIGILANCIA DEPRESIÓN: Euforia + Deterioro Laboral (30d)"
        ))
        stress_score += 2
        depression_alert_fired = True

    # PRIORITY 3: TEMPORAL_CRISIS (signals within 30 days of each other)
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
        # Additional check: Bollinger Band Lower Break (Statistical Cheapness)
        # Calculates dynamic lower band (Mean - 2*StdDev) to confirm price is statistically low
        wpm_series = df["WPM"]
        wpm_bb_mean = wpm_series.rolling(window=20).mean().iloc[-1]
        wpm_bb_std = wpm_series.rolling(window=20).std().iloc[-1]
        wpm_lower_band = wpm_bb_mean - (2 * wpm_bb_std)
        
        if wpm_val < wpm_lower_band:
            active_triggers.append(("BUY_WPM_NOW", f"Oportunidad WPM: Ruptura Banda Bollinger Inferior ({wpm_lower_band:.2f}) + Volumen 2x"))
    if net_liquidity > net_liq_sma and net_liq_prev < net_liq_sma_prev and btc_rsi_val < 60:
        active_triggers.append(("BUY_BTC_NOW", "Pivote Fed: Liquidez Neta cruzando al alza"))
    if housing_bust:
        reason_housing = f"Ciclo Foldvary: Inicios Construcción débil (Z-Score {houst_zscore:.1f}σ) + Hipotecas sobre media anual ({rate_52w_avg:.1f}%)"
        if delinq_rising:
            reason_housing += " + Morosidad al alza"
        active_triggers.append(("HOUSING_BUST", reason_housing))

    # LABOUR_SHOCK - unemployment claims spiking
    if labour_shock and icsa_4w_avg and icsa_zscore:
        active_triggers.append(("LABOUR_SHOCK", f"Mercado Laboral: Claims en percentil {icsa_zscore:.1f}σ (Ruptura estadística)"))

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

    # EM_CURRENCY_STRESS - Adaptive Percentile Threshold
    # Replaces fixed 107 with 95th percentile over 1 year
    # Replaces fixed 4.2% US10Y with "Above 250-day Moving Average" (Trend Filter)
    us10y_sma_250 = df["^TNX"].rolling(window=250).mean().iloc[-1] if "^TNX" in df.columns and len(df) >= 250 else 4.2
    
    if dxy_threshold and dxy and dxy > dxy_threshold and us10y and us10y > us10y_sma_250:
        active_triggers.append(("EM_CURRENCY_STRESS", f"Estrés Cambiario EM: DXY ({dxy:.1f}) en percentil {dxy_percentile:.0f}% + US10Y > SMA250 ({us10y_sma_250:.2f}%)"))
    elif not dxy_threshold and dxy and dxy > 107 and us10y and us10y > 4.2:
         # Fallback to fixed threshold if insufficient data
        active_triggers.append(("EM_CURRENCY_STRESS", f"Estrés Cambiario EM: DXY ({dxy:.1f}) > 107 (Umbral fijo)"))

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

    phase_daily = history_store.phase_from_score(stress_score)
    risk_daily_label = map_risk_level(phase_daily)

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

    # Determine Final Event and Reason (prioritize depression watch / combo signals)
    if not active_triggers:
        event = "NORMAL"
        reason = "Revisión Diaria."
    else:
        priority = ["COMBO_CRISIS", "DEPRESSION_ALERT", "DEPRESSION_WATCH", "TEMPORAL_CRISIS"]
        selected = None
        for p in priority:
            for t in active_triggers:
                if t[0] == p:
                    selected = t
                    break
            if selected:
                break

        if selected:
            event = selected[0]
            primary_reason = selected[1]
            other_reasons = [t[1] for t in active_triggers if t[0] != event]
            reason = primary_reason if not other_reasons else primary_reason + " | " + " | ".join(other_reasons)
        elif len(active_triggers) == 1:
            event = active_triggers[0][0]
            reason = active_triggers[0][1]
        else:
            event = "MULTIPLE_CRISIS"
            reasons_list = [t[1] for t in active_triggers]
            reason = "MÚLTIPLES ALERTAS: " + " | ".join(reasons_list)

    # State updates: record entry prices on BUY, clear on SELL (even if multiple signals fire)
    event_set = {evt for evt, _ in active_triggers}
    state_update = {}

    sell_all = "SELL_ALL_NOW" in event_set
    sell_wpm = "SELL_WPM_NOW" in event_set
    sell_btc = "SELL_BTC_NOW" in event_set
    buy_wpm = "BUY_WPM_NOW" in event_set
    buy_btc = "BUY_BTC_NOW" in event_set

    if sell_all:
        state_update["wpm_entry_price"] = None
        state_update["btc_entry_price"] = None
    else:
        if sell_wpm:
            state_update["wpm_entry_price"] = None
        elif buy_wpm:
            state_update["wpm_entry_price"] = wpm_val

        if sell_btc:
            state_update["btc_entry_price"] = None
        elif buy_btc:
            state_update["btc_entry_price"] = btc

    if (buy_wpm and not sell_wpm and not sell_all) or (buy_btc and not sell_btc and not sell_all):
        state_update["last_buy_ts"] = datetime.now(timezone.utc).isoformat()

    # Track crisis signals for temporal combo detection
    tracked_signals = ["SOLVENCY_DEATH", "SUGAR_CRASH", "EM_CURRENCY_STRESS", "WAR_PROTOCOL",
                       "INTERBANK_STRESS", "LABOUR_SHOCK", "FLASH_MOVE"]
    for trigger_event, _ in active_triggers:
        if trigger_event in tracked_signals:
            add_signal_to_history(state, trigger_event, end_date)

    trigger_events = [evt for evt, _ in active_triggers]

    cycle_context = compute_cycle_context(housing_df)
    macro_event, calendar_warning = get_macro_event(end_date)

    return {
        "event": event,
        "reason": reason,
        "stress_level": stress_level,
        "stress_score": stress_score,
        "phase_daily": phase_daily,
        "risk_daily_label": risk_daily_label,
        "cycle_phase": cycle_context.get("cycle_phase"),
        "cycle_trend": cycle_context.get("cycle_trend"),
        "cycle_confidence": cycle_context.get("cycle_confidence"),
        "trigger_events": trigger_events,
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
        "macro_event": macro_event,
        "calendar_warning": calendar_warning,
        "data_warnings": "; ".join(data_warnings) if data_warnings else None,
    }


# --- THE PERSONA (GEMINI) ---
def generate_alert_text(data):
    """
    Sends raw data to Gemini using the new SDK with retry logic.
    """
    if not client:
        return sanitize_telegram_html(f"Error en IA: GEMINI_API_KEY no configurada. Datos crudos: {data}")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    event = data.get("event", "NORMAL")

    # Build clean data payload (exclude internal fields)
    internal_fields = {
        "state_update",
        "trigger_events",
        "stress_score",
        "phase_score",
        "phase_daily",
        "risk_daily_label",
        "regime_phase",
        "regime_score",
        "regime_method",
        "regime_value",
        "regime_trend_score",
        "risk_regime_label",
    }
    market_data = {k: v for k, v in data.items() if k not in internal_fields}

    prompt = f"""Eres el <b>Centinela F2628</b>, un sistema de defensa patrimonial y alerta de crisis macroeconómica. Tu misión es detectar el agotamiento del ciclo de deuda y la inminencia de un colapso sistémico. La función principal es vigilancia de depresión; las señales de trading son auxiliares.

FECHA: {today_str}
EVENTO DETECTADO: {event}
DATOS DEL SISTEMA: {market_data}

═══════════════════════════════════════
MARCO TEÓRICO (Resumen Ejecutivo)
═══════════════════════════════════════
Fase 1 - AUGE: Crédito barato fomenta malinversión.
Fase 2 - ESPECULACIÓN: Burbujas de activos, euforia y búsqueda de rentas.
Fase 3 - QUIEBRE: Colapso de liquidez y recesión estructural.

═══════════════════════════════════════
FORMATO DE RESPUESTA
═══════════════════════════════════════
REGLAS ESTRICTAS:
- PROHIBIDO: Usar la palabra "Foldvary". NUNCA menciones este nombre ni el "Ciclo de Foldvary" en el mensaje público.
- Tono: Profesional, técnico pero masificado (accesible).
- Formato: SOLO HTML de Telegram (<b>, <i>, <u>, saltos de línea).
- Para énfasis, usa etiquetas HTML (<b>, <i>) y nunca ** o *.
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
   {"- <b>Ciclo (Largo Plazo):</b> " + str(data.get('cycle_phase')) if data.get('cycle_phase') else ""}
   {"- <b>Tendencia Ciclo:</b> " + str(data.get('cycle_trend')) if data.get('cycle_trend') else ""}
   {"- <b>Confianza Ciclo:</b> " + str(data.get('cycle_confidence')) if data.get('cycle_confidence') else ""}
   {"- <b>Régimen de Riesgo (" + str(data.get('regime_window_days')) + "d):</b> " + str(data.get('risk_regime_label')) if data.get('risk_regime_label') else ""}
   {"- <b>Tendencia Riesgo:</b> " + str(data.get('regime_trend')) if data.get('regime_trend') else ""}
   {"- <b>Confianza Riesgo:</b> " + str(data.get('regime_confidence')) if data.get('regime_confidence') else ""}
   {"- <b>Riesgo Diario:</b> " + str(data.get('risk_daily_label')) if data.get('risk_daily_label') else ""}
   {"- <b>Vigilancia Depresión:</b> " + ("ALERTA (riesgo sistémico prolongado)" if event == "DEPRESSION_ALERT" else "WATCH (fragilidad estructural)") if event in ("DEPRESSION_ALERT", "DEPRESSION_WATCH") else ""}
   {"- <b>Evento Macro:</b> " + str(data.get('macro_event')) if data.get('macro_event') else ""}
   {"- <b>Calendario Macro:</b> " + str(data.get('calendar_warning')) if data.get('calendar_warning') else ""}
   {"- <b>Datos Rezagados:</b> " + str(data.get('data_warnings')) if data.get('data_warnings') else ""}
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
   - Si LABOUR_SHOCK: menciona en análisis (no hay métrica ICSA en lista principal)
   - Si INTERBANK_STRESS: menciona en análisis (no hay métrica SOFR en lista principal)
   - Si DEPRESSION_ALERT/DEPRESSION_WATCH: subraya <u><b>S&P 500</b></u>, <u><b>VIX</b></u> y <u><b>Spread Crédito HY</b></u> como proxy de fragilidad

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
   Párrafo(s) de análisis. Comienza con la fecha: "{today_str} -". Conecta los datos con la fase del ciclo (Auge, Especulación, o Quiebre). Identifica qué fase estamos transitando. Varía la redacción cada día. Si hay macro_event, úsalo como contexto. Si existe "Ciclo (Largo Plazo)", úsalo como fase principal. Si el "Régimen de Riesgo" o el "Riesgo Diario" difieren, descríbelo como estrés táctico/puntual y NO como cambio de ciclo salvo persistencia. Usa la palabra "fase" SOLO para el ciclo de largo plazo; para riesgo diario utiliza "nivel" o "pulso". Nunca llames "ciclo" al régimen de riesgo.

   IMPORTANTE: NO menciones códigos internos de señales (HOUSING_BUST, SOLVENCY_DEATH, etc.) en el análisis. NUNCA digas "la señal X no se activó". Habla como un economista, no como un script. NUNCA digas "Foldvary".

   GUÍA DE ESTILO Y VOCABULARIO (STRICT COMPLIANCE REQUIRED)
   ---------------------------------------------------------
   Regla de Oro: El lector NO debe saber que eres un script de Python. NO uses nombres de variables ni códigos internos.

   TABLA DE TRADUCCIÓN OBLIGATORIA:
   ❌ CÓDIGO INTERNO (PROHIBIDO) | ✅ LENGUAJE NATURAL (USAR ESTO)
   ------------------------------|--------------------------------
   HOUSING_BUST                  | "Giro del ciclo inmobiliario", "paralización de obras", "crisis de vivienda".
   SOLVENCY_DEATH                | "Estrés de crédito", "cierre del grifo de financiación", "pánico en deuda corporativa".
   SUGAR_CRASH                   | "Trampa alcista", "euforia artificial", "divergencia técnica severa".
   BOND_FREEZE                   | "Pánico en renta fija", "desplome de bonos", "disparada de rendimientos".
   WAR_PROTOCOL                  | "Riesgo de conflicto", "shock de oferta en energía", "incertidumbre geopolítica".
   EM_CURRENCY_STRESS            | "Asfixia de emergentes", "super-dólar destructivo", "crisis de balanza de pagos".
   LABOUR_SHOCK                  | "Deterioro del empleo", "aumento del desempleo", "debilidad laboral".
   INTERBANK_STRESS              | "Desconfianza bancaria", "tensión de liquidez", "estrés en el mercado repo".
   COMBO_CRISIS                  | "Tormenta perfecta", "colapso sistémico simultáneo", "crisis de liquidez y solvencia".
   TEMPORAL_CRISIS               | "Convergencia de riesgos", "deterioro acelerado en múltiples frentes".
   DEPRESSION_WATCH              | "vigilancia de depresión", "riesgo sistémico prolongado".
   DEPRESSION_ALERT              | "alerta de depresión", "riesgo de colapso prolongado".
   FLASH_MOVE                    | "Movimiento violento", "shock de volatilidad", "ajuste repentino".
   BUY_WPM_NOW                   | "Oportunidad en activos reales", "punto de entrada en streaming", "capitulación en metales".
   BUY_BTC_NOW                   | "Giro de liquidez favorable", "expansión monetaria detectada".
   SELL_...                      | "Toma de beneficios", "alerta de salida", "protección de capital".

   EJEMPLOS DE CORRECCIÓN:
   ❌ MAL: "No se activó la señal HOUSING_BUST gracias a los inicios de construcción."
   ✅ BIEN: "El sector inmobiliario se mantiene estable, con los inicios de construcción sosteniendo el ciclo actual."

   ❌ MAL: "Estamos en la Fase 2 del Ciclo Foldvary."
   ✅ BIEN: "Estamos en una fase de especulación del ciclo económico."

   ❌ MAL: "La señal SOLVENCY_DEATH está apagada."
   ✅ BIEN: "Los mercados de crédito operan con normalidad, sin estrés visible en la financiación corporativa."

   En estado NORMAL: Análisis breve (3-5 líneas) sobre posición en el ciclo y riesgos latentes.
   En CRISIS: Análisis extenso explicando la mecánica del riesgo, comparaciones históricas, y qué vigilar.
   En DEPRESSION_ALERT/WATCH: enfatiza horizonte de meses/trimestres, contracción prolongada, y fragilidad de crédito; evita lenguaje de trading táctico.

IMPORTANTE: Genera EXACTAMENTE UN mensaje completo. NO generes dos versiones ni variantes.

═══════════════════════════════════════
TÍTULOS POR EVENTO
═══════════════════════════════════════
- NORMAL → <b>👁‍🗨 Sistemas Nominales</b>
- MULTIPLE_CRISIS → <b>🚨 ALERTA MÚLTIPLE | RIESGO SISTÉMICO</b>
- DEPRESSION_ALERT → <b>🛑 ALERTA DEPRESIÓN | RIESGO SISTÉMICO</b>
- DEPRESSION_WATCH → <b>⚠️ VIGILANCIA DEPRESIÓN | FRAGILIDAD ESTRUCTURAL</b>
- COMBO_CRISIS → <b>🔴🔴🔴 ALERTA MÁXIMA | CRISIS INMINENTE 🔴🔴🔴</b>
- TEMPORAL_CRISIS → <b>⚠️ ALERTA ALTA | CONVERGENCIA DE RIESGOS</b>
- SOLVENCY_DEATH → <b>🚨 ALARMA | ☠️ ESTRÉS DE CRÉDITO</b>
- EM_CURRENCY_STRESS → <b>🚨 ALARMA | 💱 ASFIXIA DE MERCADOS (EM)</b>
- WAR_PROTOCOL → <b>🚨 ALARMA | ⚔️ PROTOCOLO DE CONFLICTO</b>
- BOND_FREEZE → <b>🚨 ALARMA | 🧊 CONGELAMIENTO DE BONOS</b>
- SUGAR_CRASH → <b>🚨 ALARMA | 🍬 EUFORIA TERMINAL</b>
- HOUSING_BUST → <b>🚨 ALARMA | 🏚️ GIRO DEL CICLO INMOBILIARIO</b>
- LABOUR_SHOCK → <b>🚨 ALARMA | 📉 DETERIORO LABORAL</b>
- INTERBANK_STRESS → <b>🚨 ALARMA | 🏦 ESTRÉS INTERBANCARIO</b>
- FLASH_MOVE → <b>⚡ ALERTA | MOVIMIENTO SÚBITO</b>
- BUY_WPM_NOW → <b>🎯 SEÑAL DE ENTRADA | WPM</b>
- BUY_BTC_NOW → <b>🎯 SEÑAL DE ENTRADA | BTC</b>
- SELL_WPM_NOW → <b>💰 SEÑAL DE SALIDA | WPM</b>
- SELL_BTC_NOW → <b>💰 SEÑAL DE SALIDA | BTC</b>
- SELL_ALL_NOW → <b>🚨 LIQUIDACIÓN TOTAL | EXIT ALL</b>

═══════════════════════════════════════
INSTRUCCIONES POR SEÑAL (basadas en evidencia estadística 2011-2026)
═══════════════════════════════════════

<b>COMBO_CRISIS</b> [Precisión Crash: Alta]
Detecta euforia (VIX bajo) + congelamiento de crédito (Spread > 5%). Históricamente precede correcciones severas o estancamiento.
Tono: MÁXIMA ALARMA. La divergencia entre optimismo bursátil y realidad crediticia es insostenible. Riesgo de liquidación forzada.

<b>TEMPORAL_CRISIS</b> [Precisión Crash: Media-Alta]
Convergencia de dos fracturas estructurales en <30 días.
- SUGAR + EM STRESS: Euforia local vs Asfixia global (Dólar). El mercado ignora el riesgo sistémico externo.
- SOLVENCY + WAR: Crédito caro + Shock energético. Estanflación inminente.
Tono: Serio y preventivo. La acumulación de fallas estructurales aumenta la fragilidad del sistema.

<b>DEPRESSION_ALERT / DEPRESSION_WATCH</b> [Señal de riesgo prolongado]
Combinación de euforia bursátil con estrés de financiación y/o deterioro laboral.
Tono: Estructural, de ciclo largo. Advierte sobre probabilidad de contracción profunda y prolongada.
REGLAS ADICIONALES DEPRESIÓN:
- No enmarcar como oportunidad de trading.
- Hablar de restricción crediticia, contracción del consumo, deterioro del empleo y desinversión.
- Usar lenguaje de horizonte largo (trimestres, no días).

<b>WAR_PROTOCOL</b> [Precisión Crash: 64% | Retorno 90d: +0.9%]
La señal geopolítica más efectiva. Petróleo disparado (Z-Score > 2) + Oro arriba + SPX abajo.
Tono: BAJISTA / GRAVE. El mercado está valorando un conflicto real. Históricamente, el SPX sufre o se estanca en los siguientes 90 días.

<b>EM_CURRENCY_STRESS</b> [Precisión Crash: 49% | Retorno 90d: +0.0%]
DXY en percentil 95 + US10Y en tendencia alcista. El "freno de mano" de la economía global.
Tono: ALERTA DE ESTRANGULAMIENTO. Históricamente, el retorno del SPX a 90 días es PLANO (0%). No es necesariamente un crash, sino un techo de mercado. La liquidez global se seca.

<b>BOND_FREEZE</b> [Precisión Crash: 39% | Retorno 90d: +0.9%]
Pánico en bonos (Z-Score > 2).
Tono: PRECAUCIÓN. Señala volatilidad, pero no siempre caída sostenida. A menudo es una oportunidad de compra si la Fed interviene para calmar el mercado de deuda.

<b>SUGAR_CRASH</b> [Precisión Crash: 35%]
Divergencia técnica en máximos.
Tono: MODERADO. Señal de "techo temporal". El mercado está extendido, pero sin un catalizador fundamental (como crédito o guerra), la euforia puede persistir más de lo racional.

<b>SOLVENCY_DEATH</b> [Precisión Crash: 22% | Retorno 90d: +5.0%]
Spread HY > 5%.
Tono: CAUTELOSO Y MATIZADO. Estructuralmente grave, pero por sí sola es una señal "contrarian" a corto plazo (el mercado rebota el 78% de las veces). Indica estrés de fondo que requiere monitoreo, pero no venta de pánico inmediata a menos que se combine con otras señales.

<b>HOUSING_BUST</b> [Señal Estructural de Largo Plazo]
Construcción colapsando (-1.5σ) + Tasas subiendo.
Tono: ANALÍTICO (Ciclo de 18 Años). Confirma que la economía real se está frenando. No esperes un crash bursátil mañana, pero sí una recesión en 12-18 meses. Es el "canario en la mina" del ciclo Foldvary.

<b>LABOUR_SHOCK</b> [WIN RATE CONTRARIAN: 93% | Retorno 90d: +6.0%]
Desempleo disparándose (>2σ).
IMPORTANTE: El mercado SUBE el 93% de las veces tras esta señal.
Tono: ANALÍTICO Y "BULLISH" (por liquidez). La economía real sufre, pero Wall Street celebra esperando el rescate de la Fed (Pivot). No asustes a los usuarios: advierte del daño económico, pero reconoce la probable respuesta alcista de los activos de riesgo.

<b>INTERBANK_STRESS</b> [WIN RATE CONTRARIAN: 79%]
SOFR > Fed Funds.
Tono: TÉCNICO. Tensión de fontanería financiera. Al igual que Labour Shock, suele provocar inyecciones de liquidez que levantan el mercado.

<b>FLASH_MOVE</b>
Movimiento > 5% en una sesión.
Tono: ATENCIÓN. Shock de volatilidad. Analizar según el activo afectado.

<b>BUY_WPM_NOW</b> [Win Rate 90d: 57% | Retorno: +8.5%]
WPM barato (Banda Bollinger) + Pánico.
Tono: OPORTUNIDAD TÁCTICA. Compra contra la tendencia ("cuchillo cayendo"). Históricamente pierde dinero el primer mes, pero rinde fuerte (+8.5%) al trimestre. Paciencia requerida.

<b>BUY_BTC_NOW</b> [Win Rate 90d: 56% | Retorno: +11.7%]
Liquidez Neta subiendo + BTC descansando.
Tono: CONSTRUCTIVO / ALCISTA. La mejor señal de entrada del sistema. Sigue el dinero de la Fed. Históricamente muy rentable (+11.7% trimestral).

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

    max_retries = 8
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return sanitize_telegram_html(response.text)
        except Exception as e:
            err_str = str(e)
            is_retryable = ("429" in err_str or "500" in err_str or "503" in err_str or 
                           "INTERNAL" in err_str or "disconnected" in err_str or "timeout" in err_str)
            
            if attempt < max_retries - 1 and is_retryable:
                wait_time = 30 * (2 ** attempt)  # 30s, 60s, 120s...
                print(f"Gemini error: {err_str[:100]}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            return sanitize_telegram_html(f"Error en IA: {e}. Datos crudos: {data}")


def update_regime_context(analysis):
    """
    Persists daily snapshots in SQLite and enriches the analysis with a multi-day risk regime.
    """
    if not analysis or "stress_score" not in analysis:
        return analysis

    if not HISTORY_DB_PATH:
        return analysis

    ops_warnings = list(analysis.get("ops_warnings") or [])

    try:
        history_load_state = True
        if GIST_TOKEN and STATE_GIST_ID:
            history_load_state = history_store.load_db_from_gist(
                GIST_TOKEN,
                STATE_GIST_ID,
                HISTORY_DB_PATH,
                HISTORY_GIST_FILENAME,
            )
            if history_load_state is False:
                ops_warnings.append("HISTORY_GIST_LOAD_FAILED_SKIP_SAVE")

        history_store.ensure_db(HISTORY_DB_PATH)

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        snapshot = {
            "date": date_str,
            "run_ts": datetime.now(timezone.utc).isoformat(),
            "event": analysis.get("event"),
            "stress_score": analysis.get("stress_score"),
            "stress_level": analysis.get("stress_level"),
            "phase_daily": analysis.get("phase_daily"),
            "phase_score": float(analysis.get("stress_score")) if analysis.get("stress_score") is not None else None,
            "regime_phase": None,
            "regime_score": None,
            "regime_trend": None,
            "regime_confidence": None,
            "spx": analysis.get("spx"),
            "vix": analysis.get("vix"),
            "spread": analysis.get("spread"),
            "us10y": analysis.get("us10y"),
            "dxy": analysis.get("dxy"),
            "net_liq_b": analysis.get("net_liq_b"),
            "gold": analysis.get("gold"),
            "btc": analysis.get("btc_price"),
            "triggers": ",".join(analysis.get("trigger_events") or []),
            "reason": analysis.get("reason"),
        }

        history_store.upsert_daily_snapshot(HISTORY_DB_PATH, snapshot)
        history_store.prune_history(HISTORY_DB_PATH, HISTORY_RETENTION_DAYS)

        history_limit = history_store.regime_history_limit(
            RISK_WINDOW_DAYS,
            RISK_TREND_DAYS,
            RISK_MIN_DAYS,
        )
        history_rows = history_store.fetch_recent_history(HISTORY_DB_PATH, history_limit)
        regime = history_store.compute_regime(history_rows, RISK_WINDOW_DAYS, RISK_TREND_DAYS, RISK_MIN_DAYS)

        analysis.update(regime)
        analysis["risk_regime_label"] = map_risk_level(analysis.get("regime_phase"))
        history_store.update_regime_fields(HISTORY_DB_PATH, date_str, regime)

        if GIST_TOKEN and STATE_GIST_ID:
            if history_load_state is False:
                print("History load failed earlier; skipping gist save to avoid remote overwrite.")
                ops_warnings.append("HISTORY_GIST_SAVE_SKIPPED_DUE_LOAD_FAILURE")
            else:
                save_ok = history_store.save_db_to_gist(
                    GIST_TOKEN,
                    STATE_GIST_ID,
                    HISTORY_DB_PATH,
                    HISTORY_GIST_FILENAME,
                )
                if save_ok is False:
                    ops_warnings.append("HISTORY_GIST_SAVE_FAILED")
    except Exception as exc:
        print(f"History store error: {exc}")
        ops_warnings.append("HISTORY_STORE_PIPELINE_ERROR")

    if ops_warnings:
        analysis["ops_warnings"] = sorted(set(ops_warnings))
    return analysis


# --- MESSENGER (TELEGRAM) ---
def send_telegram(message):
    """Dispatches the final message via the Telegram Bot API."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return None
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = _request_with_retries(
            "POST",
            url,
            json_payload=payload,
            operation="Telegram send (HTML)",
        )
        if response.ok:
            return response.json().get("result", {}).get("message_id")

        description = _extract_telegram_error(response)
        if response.status_code == 400 and "parse entities" in description.lower():
            plain_payload = {
                "chat_id": TG_CHAT_ID,
                "text": _telegram_plain_text(message),
            }
            print("Telegram HTML parse error; retrying as plain text.")
            fallback_response = _request_with_retries(
                "POST",
                url,
                json_payload=plain_payload,
                operation="Telegram send (plain text)",
            )
            fallback_response.raise_for_status()
            return fallback_response.json().get("result", {}).get("message_id")

        response.raise_for_status()
    except Exception as e:
        print(f"Telegram Failed: {e}")
        return None

def pin_message(message_id):
    """Pins a critical alert in the chat."""
    if not TG_TOKEN or not TG_CHAT_ID or not message_id:
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/pinChatMessage"
    payload = {"chat_id": TG_CHAT_ID, "message_id": message_id}
    try:
        response = _request_with_retries(
            "POST",
            url,
            json_payload=payload,
            operation="Telegram pin",
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Pinning Failed: {e}")


def send_ops_warnings(state, warnings, is_manual_run=False):
    if not warnings:
        return
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    if "daily_alerts" not in state:
        state["daily_alerts"] = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "sent_events": [],
            "sent_ops_warnings": [],
        }
    sent_ops = state["daily_alerts"].setdefault("sent_ops_warnings", [])
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    labels = {
        "STATE_LOAD_FAILED_SKIP_SAVE": "No se pudo cargar `state.json` desde gist; guardado remoto omitido para evitar sobreescritura.",
        "HISTORY_GIST_LOAD_FAILED_SKIP_SAVE": "No se pudo cargar la base histórica desde gist.",
        "HISTORY_GIST_SAVE_SKIPPED_DUE_LOAD_FAILURE": "Guardado histórico en gist omitido tras fallo de carga para evitar pérdida de datos.",
        "HISTORY_GIST_SAVE_FAILED": "Falló el guardado de historial en gist.",
        "HISTORY_STORE_PIPELINE_ERROR": "Error interno en el pipeline de historial/regímenes.",
    }

    for warning_code in sorted(set(warnings)):
        if not is_manual_run and warning_code in sent_ops:
            continue
        description = labels.get(warning_code, warning_code)
        ops_msg = (
            f"<b>⚠️ AVISO OPERATIVO | CENTINELA</b>\n\n"
            f"<b>Fecha:</b> {today_str}\n"
            f"<b>Código:</b> {warning_code}\n"
            f"<b>Detalle:</b> {description}\n\n"
            f"Revise logs y sincronización de estado/historial."
        )
        msg_id = send_telegram(ops_msg)
        if msg_id and warning_code not in sent_ops:
            sent_ops.append(warning_code)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # Load persistent state (entry positions).
        state = load_state()
        skip_state_save = bool(state.pop("_state_load_failed", False))
        print(f"Loaded state: {state}")

        market_data, wpm_data, fred_data, housing_data = get_data()
        analysis = analyse_market(market_data, wpm_data, fred_data, housing_data, state)
        analysis = update_regime_context(analysis)

        print(f"Analysis Result: {analysis}")

        # Update state with entry/exit signals first (independent of notification suppression)
        if analysis.get("state_update"):
            for key, value in analysis["state_update"].items():
                state[key] = value

        current_hour = datetime.now(timezone.utc).hour
        is_manual_run = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
        should_send = False
        detected_event = analysis["event"]
        ops_warnings = list(analysis.get("ops_warnings") or [])
        if skip_state_save and GIST_TOKEN and STATE_GIST_ID:
            ops_warnings.append("STATE_LOAD_FAILED_SKIP_SAVE")

        # --- SMART NOTIFICATION LOGIC ---
        if detected_event == "DATA_OUTAGE":
            missing = ", ".join(analysis["missing"])
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            outage_msg = (
                f"<b>🚨 ERROR CRÍTICO | FALLO DE DATOS</b>\n\n"
                f"<b>Fecha:</b> {today_str}\n"
                f"<b>Sensores Afectados:</b> {missing}\n\n"
                f"El Centinela no puede evaluar la seguridad del sistema sin estos datos críticos. "
                f"Verifique la fuente de datos (Yahoo Finance/FRED) o los logs del sistema."
            )
            msg_id = send_telegram(outage_msg)
            print(f"DATA OUTAGE: missing {missing}")
        
        elif detected_event != "NORMAL":
            # Check if this specific alarm was already sent today
            already_sent = detected_event in state["daily_alerts"]["sent_events"]

            if is_manual_run:
                print(f"Manual Run: Forcing dispatch of '{detected_event}' (ignoring cache).")
                should_send = True
            elif already_sent and current_hour < 20:
                print(f"🤐 SUPPRESSED: '{detected_event}' already sent today. Waiting for EOD summary.")
                should_send = False
            else:
                should_send = True
            
            # Record this event as sent if we are sending it (and it wasn't already)
            if should_send and not already_sent:
                state["daily_alerts"]["sent_events"].append(detected_event)

        elif is_manual_run:
            should_send = True
            print("Manual Test Detected. Forcing alert.")
        
        elif current_hour >= 20:
            should_send = True # Always send the EOD heartbeat
        
        else:
            print(f"Status Normal (Hour {current_hour} UTC). Keeping silent.")

        send_ops_warnings(state, ops_warnings, is_manual_run=is_manual_run)

        # Save state (positions + daily alerts history)
        if skip_state_save and GIST_TOKEN and STATE_GIST_ID:
            print("State load failed earlier; skipping state save to avoid remote overwrite.")
        else:
            save_state(state)
            print(f"Updated state: {state}")

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
