# Backtesting Script for Foldvary Sentinel (Scientific Version - Maximalist Report)
# Replays the trigger logic day-by-day over 15+ years of historical data.
# Generates a detailed PDF report with Volatility, Drawdowns, and Statistical Analysis.

import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import time
import numpy as np
import os
from fpdf import FPDF

# --- CONFIGURATION ---
LOOKBACK_YEARS = 15
FORWARD_WINDOWS = [30, 90, 180]
TEMPORAL_WINDOW_DAYS = 30
DEDUP_DAYS = 30
DEPRESSION_FORWARD_DAYS = 365
DEPRESSION_FORWARD_TRADING_DAYS = 252
HIST_WINDOW_TRADING_DAYS = 252 * 5
DEPRESSION_Z_THRESHOLD = 2.0
DEPRESSION_COMPONENTS_REQUIRED = 3

# --- TECHNICAL INDICATORS ---
def rsi(series, period=14):
    if series.empty or len(series) < period:
        return pd.Series([50.0] * len(series), index=series.index)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_z_score_threshold(series, window=250, num_std=2.0):
    if len(series) < window: return None, None
    rolling_mean = series.rolling(window=window).mean().iloc[-1]
    rolling_std = series.rolling(window=window).std().iloc[-1]
    if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std == 0: return None, None
    threshold = rolling_mean + (num_std * rolling_std)
    z_score = (series.iloc[-1] - rolling_mean) / rolling_std
    return threshold, z_score

def calculate_percentile_threshold(series, window=250, percentile=95):
    if len(series) < window: return None, None
    threshold = series.rolling(window=window).quantile(percentile / 100.0).iloc[-1]
    if pd.isna(threshold): return None, None
    recent = series.iloc[-window:]
    pctl = (recent < series.iloc[-1]).sum() / len(recent) * 100
    return threshold, pctl

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

# --- DATA FETCHING ---
def fetch_historical_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)
    print(f"Fetching data from {start_date.date()} to {end_date.date()}...")

    tickers = ["^TNX", "^SPX", "^VIX", "CL=F", "GC=F", "SI=F", "HG=F", "HRC=F", "ITA", "DX-Y.NYB", "WPM", "BTC-USD"]
    
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
        market_df = pd.DataFrame()
        for ticker in tickers:
            try:
                if ticker in raw.columns.levels[0]:
                    market_df[ticker] = raw[ticker]['Close']
                    if ticker == "WPM":
                        market_df["WPM_Volume"] = raw[ticker]['Volume']
                elif 'Close' in raw.columns and ticker in raw['Close'].columns:
                     market_df[ticker] = raw['Close'][ticker]
            except Exception: pass
        market_df = market_df.ffill()
        if "^TNX" in market_df.columns:
            market_df["^TNX"] = normalize_tnx(market_df["^TNX"])
    except Exception as e:
        print(f"YF Error: {e}")
        return None, None, None

    try:
        fred_df = web.DataReader(
            ["BAMLH0A0HYM2", "RRPONTSYD", "WALCL", "WTREGEN", "ICSA", "SOFR", "DFF", "USREC"],
            "fred",
            start_date,
            end_date,
        ).ffill()
    except Exception as e:
        print(f"FRED Error: {e}")
        fred_df = pd.DataFrame()

    try:
        housing_df = web.DataReader(["HOUST", "MORTGAGE30US", "CSUSHPINSA", "DRSFRMACBS"], "fred", start_date, end_date).ffill()
    except Exception as e:
        print(f"FRED Housing Error: {e}")
        housing_df = pd.DataFrame()

    return market_df, fred_df, housing_df

# --- LOGIC ---
def analyse_date(market_df, fred_df, housing_df, idx, history_log):
    if idx < 504:
        return [], history_log, {}
    triggers = []
    details = {}
    
    m_win = market_df.iloc[:idx+1]
    f_win = fred_df.loc[:market_df.index[idx]] if not fred_df.empty else pd.DataFrame()
    h_win = housing_df.loc[:market_df.index[idx]] if not housing_df.empty else pd.DataFrame()
    current_date = market_df.index[idx]

    def get_last(df, col):
        return df[col].iloc[-1] if col in df.columns and not df[col].empty and pd.notna(df[col].iloc[-1]) else None

    us10y = get_last(m_win, "^TNX")
    spx = get_last(m_win, "^SPX")
    btc = get_last(m_win, "BTC-USD")
    wpm = get_last(m_win, "WPM")
    dxy = get_last(m_win, "DX-Y.NYB") or 100
    oil = get_last(m_win, "CL=F") or 70
    gold = get_last(m_win, "GC=F") or 2000
    vix = get_last(m_win, "^VIX") or 15
    
    if us10y is None or spx is None: return [], []

    # Calculated
    btc_rsi = rsi(m_win["BTC-USD"]).iloc[-1] if "BTC-USD" in m_win.columns else 50
    wpm_rsi = rsi(m_win["WPM"]).iloc[-1] if "WPM" in m_win.columns else 50
    us10y_rsi = rsi(m_win["^TNX"]).iloc[-1] if "^TNX" in m_win.columns else 50
    spx_rsi = rsi(m_win["^SPX"]).iloc[-1] if "^SPX" in m_win.columns else 50
    spx_high50 = m_win["^SPX"].rolling(50).max().iloc[-1]
    spx_rsi_high50 = rsi(m_win["^SPX"]).rolling(50).max().iloc[-1]
    wpm_crash = False
    if wpm and "WPM" in m_win.columns and len(m_win) > 1:
        wpm_price_yest = m_win["WPM"].iloc[-2]
        if pd.notna(wpm_price_yest) and wpm_price_yest > 0:
            wpm_crash = wpm < (wpm_price_yest * 0.95)

    # --- SINGLE SIGNALS ---
    if not f_win.empty and "BAMLH0A0HYM2" in f_win.columns and len(f_win) > 3:
        recent = f_win["BAMLH0A0HYM2"].dropna().tail(3)
        if len(recent) == 3 and (recent > 5.0).all(): triggers.append("SOLVENCY_DEATH")

    if spx >= spx_high50 and spx_rsi < spx_rsi_high50 and vix < 13: triggers.append("SUGAR_CRASH")

    ot, oz = calculate_z_score_threshold(m_win["CL=F"], 250, 2.0)
    g_h20 = m_win["GC=F"].rolling(20).max().iloc[-2] if "GC=F" in m_win.columns else 99999
    spx_l20 = m_win["^SPX"].rolling(20).min().iloc[-2]
    if ot and oil > ot and gold > g_h20 and spx < spx_l20: triggers.append("WAR_PROTOCOL")

    dt, dp = calculate_percentile_threshold(m_win["DX-Y.NYB"], 250, 95)
    us10y_sma = m_win["^TNX"].rolling(250).mean().iloc[-1] if len(m_win) > 250 else 4.2
    if dt and dxy > dt and us10y > us10y_sma:
        triggers.append("EM_CURRENCY_STRESS")
    elif not dt and dxy > 107 and us10y > 4.2:
        triggers.append("EM_CURRENCY_STRESS")

    bt, bz = calculate_z_score_threshold(m_win["^TNX"], 250, 2.0)
    if bt and us10y > bt and us10y_rsi > 70: triggers.append("BOND_FREEZE")

    if not h_win.empty and "HOUST" in h_win.columns and len(h_win) > 252:
        houst = h_win["HOUST"].dropna()
        if len(houst) > 252:
            h_mean = houst.rolling(504).mean().iloc[-1]
            h_std = houst.rolling(504).std().iloc[-1]
            h_z = (houst.iloc[-1] - h_mean) / h_std if h_std > 0 else 0
            m_avg = h_win["MORTGAGE30US"].rolling(252).mean().iloc[-1] if "MORTGAGE30US" in h_win.columns else 6.0
            m_cur = get_last(h_win, "MORTGAGE30US") or 6.0
            if h_z < -1.5 and m_cur > m_avg: triggers.append("HOUSING_BUST")

    if "ICSA" in f_win.columns:
        icsa = f_win["ICSA"].dropna()
        if len(icsa) > 250:
            it, iz = calculate_z_score_threshold(icsa, 250, 2.0)
            i_avg = icsa.rolling(20).mean().iloc[-1]
            if it and i_avg > it: triggers.append("LABOUR_SHOCK")
    
    sofr = get_last(f_win, "SOFR"); dff = get_last(f_win, "DFF")
    if sofr and dff and sofr > dff + 0.10: triggers.append("INTERBANK_STRESS")

    if wpm and wpm_crash and "WPM_Volume" in m_win.columns and len(m_win) > 20:
        w_mean = m_win["WPM"].rolling(20).mean().iloc[-1]
        w_std = m_win["WPM"].rolling(20).std().iloc[-1]
        w_lower = w_mean - (2 * w_std)
        w_vol = get_last(m_win, "WPM_Volume")
        w_vol_avg = m_win["WPM_Volume"].rolling(20).mean().iloc[-1]
        if wpm < w_lower and wpm_rsi < 30 and w_vol > (w_vol_avg * 2): triggers.append("BUY_WPM_NOW")

    net_liq = (get_last(f_win, "WALCL") or 0) - ((get_last(f_win, "WTREGEN") or 0) + (get_last(f_win, "RRPONTSYD") or 0))
    if "WALCL" in f_win.columns and len(f_win) > 10:
        nl_s = f_win["WALCL"] - (f_win["WTREGEN"] + f_win["RRPONTSYD"])
        nl_sma = nl_s.rolling(10).mean().iloc[-1]
        nl_prev = nl_s.iloc[-2]; nl_prev_sma = nl_s.rolling(10).mean().iloc[-2]
        if net_liq > nl_sma and nl_prev < nl_prev_sma and btc_rsi < 60: triggers.append("BUY_BTC_NOW")

    # --- FLASH MOVE DETECTOR ---
    flash_tickers = {
        "^TNX": "Bonos 10Y",
        "^SPX": "S&P 500",
        "CL=F": "Petróleo",
        "GC=F": "Oro",
        "BTC-USD": "Bitcoin",
        "DX-Y.NYB": "Dólar (DXY)",
        "^VIX": "VIX"
    }
    if len(m_win) > 1:
        for ticker in flash_tickers:
            if ticker in m_win.columns:
                current_price = m_win[ticker].iloc[-1]
                prev_price = m_win[ticker].iloc[-2]
                if pd.notna(current_price) and pd.notna(prev_price) and prev_price > 0:
                    pct_change = ((current_price - prev_price) / prev_price) * 100
                    if abs(pct_change) > 5.0:
                        triggers.append("FLASH_MOVE")
                        break

    # --- COMPLEX COMBINATIONS ---
    active_recent = set(s for d, s in history_log if (current_date - d).days <= TEMPORAL_WINDOW_DAYS)

    if "SOLVENCY_DEATH" in triggers and "SUGAR_CRASH" in triggers:
        triggers.append("COMBO_CRISIS")

    # Depression watch combos (windowed)
    if ("SUGAR_CRASH" in active_recent and "INTERBANK_STRESS" in active_recent and "FLASH_MOVE" in active_recent):
        triggers.append("DEPRESSION_ALERT")
        details["DEPRESSION_ALERT"] = "Sugar+Interbank+Flash"
    elif "SUGAR_CRASH" in active_recent and "INTERBANK_STRESS" in active_recent:
        triggers.append("DEPRESSION_WATCH")
        details["DEPRESSION_WATCH"] = "Sugar+Interbank"
    elif "SUGAR_CRASH" in active_recent and "LABOUR_SHOCK" in active_recent:
        triggers.append("DEPRESSION_WATCH")
        details["DEPRESSION_WATCH"] = "Sugar+Labour"

    if "SUGAR_CRASH" in active_recent and "EM_CURRENCY_STRESS" in active_recent:
        triggers.append("TEMPORAL_CRISIS")
        details["TEMPORAL_CRISIS"] = "Sugar+EM"
    elif "SOLVENCY_DEATH" in active_recent and "WAR_PROTOCOL" in active_recent:
        triggers.append("TEMPORAL_CRISIS")
        details["TEMPORAL_CRISIS"] = "Solvency+War"

    tracked_signals = {"SOLVENCY_DEATH", "SUGAR_CRASH", "EM_CURRENCY_STRESS", "WAR_PROTOCOL",
                       "INTERBANK_STRESS", "LABOUR_SHOCK", "FLASH_MOVE"}
    for t in triggers:
        if t in tracked_signals:
            history_log.append((current_date, t))

    cutoff = current_date - timedelta(days=TEMPORAL_WINDOW_DAYS * 2)
    new_log = [x for x in history_log if x[0] > cutoff]
    return triggers, new_log, details

def max_drawdown_pct(series):
    if series.empty:
        return None
    start_price = series.iloc[0]
    if pd.isna(start_price) or start_price == 0:
        return None
    min_price = series.min()
    return ((min_price - start_price) / start_price) * 100

def prepare_depression_inputs(market_df, fred_df, housing_df):
    spx = market_df["^SPX"] if "^SPX" in market_df.columns else pd.Series()
    spx_dd_forward = []
    for i in range(len(market_df)):
        end_i = min(i + DEPRESSION_FORWARD_TRADING_DAYS, len(market_df) - 1)
        spx_window = spx.iloc[i:end_i + 1].dropna()
        spx_dd_forward.append(max_drawdown_pct(spx_window))

    icsa_4w = pd.Series()
    if not fred_df.empty and "ICSA" in fred_df.columns:
        icsa_4w = fred_df["ICSA"].dropna().rolling(window=20).mean()

    interbank_spread = pd.Series()
    if not fred_df.empty and "SOFR" in fred_df.columns and "DFF" in fred_df.columns:
        interbank_spread = (fred_df["SOFR"] - fred_df["DFF"]).dropna()

    return {
        "spx_dd_forward": pd.Series(spx_dd_forward, index=market_df.index),
        "icsa_4w": icsa_4w,
        "interbank_spread": interbank_spread,
    }


def compute_depression_outcomes(market_df, fred_df, housing_df, idx, precomp):
    current_date = market_df.index[idx]
    end_date = current_date + timedelta(days=DEPRESSION_FORWARD_DAYS)

    # Equity drawdown proxy (statistical: percentile of historical 12m drawdowns)
    spx_dd_forward = precomp["spx_dd_forward"]
    spx_dd_12m = spx_dd_forward.iloc[idx] if idx < len(spx_dd_forward) else None
    prior_dd = spx_dd_forward.iloc[:idx].dropna().tail(HIST_WINDOW_TRADING_DAYS)
    equity_crash = False
    if spx_dd_12m is not None and not prior_dd.empty:
        dd_threshold = prior_dd.quantile(0.05)
        equity_crash = spx_dd_12m <= dd_threshold

    # Credit spread severity (z-score on level)
    credit_max = None
    credit_z = None
    credit_freeze = False
    if not fred_df.empty and "BAMLH0A0HYM2" in fred_df.columns:
        series = fred_df["BAMLH0A0HYM2"].dropna()
        prior = series.loc[:current_date].tail(HIST_WINDOW_TRADING_DAYS)
        future = series.loc[current_date:end_date]
        if not prior.empty and not future.empty:
            credit_max = future.max()
            mean = prior.mean()
            std = prior.std()
            if std and std > 0:
                credit_z = (credit_max - mean) / std
                credit_freeze = credit_z >= DEPRESSION_Z_THRESHOLD

    # Labour shock severity (z-score on 4w avg claims)
    labor_shock = False
    icsa_max_4w = None
    icsa_z = None
    icsa_4w = precomp["icsa_4w"]
    if not icsa_4w.empty:
        prior = icsa_4w.loc[:current_date].dropna().tail(HIST_WINDOW_TRADING_DAYS)
        future = icsa_4w.loc[current_date:end_date].dropna()
        if not prior.empty and not future.empty:
            icsa_max_4w = future.max()
            mean = prior.mean()
            std = prior.std()
            if std and std > 0:
                icsa_z = (icsa_max_4w - mean) / std
                labor_shock = icsa_z >= DEPRESSION_Z_THRESHOLD

    # Housing bust severity (housing starts z-score + mortgage rate z-score)
    housing_bust = False
    houst_z_min = None
    mortgage_z_max = None
    if not housing_df.empty and "HOUST" in housing_df.columns:
        houst = housing_df["HOUST"].dropna()
        prior_h = houst.loc[:current_date].tail(HIST_WINDOW_TRADING_DAYS)
        future_h = houst.loc[current_date:end_date]
        if not prior_h.empty and not future_h.empty:
            h_mean = prior_h.mean()
            h_std = prior_h.std()
            if h_std and h_std > 0:
                houst_min = future_h.min()
                houst_z_min = (houst_min - h_mean) / h_std

        mortgage = housing_df["MORTGAGE30US"].dropna() if "MORTGAGE30US" in housing_df.columns else pd.Series()
        if not mortgage.empty:
            prior_m = mortgage.loc[:current_date].tail(HIST_WINDOW_TRADING_DAYS)
            future_m = mortgage.loc[current_date:end_date]
            if not prior_m.empty and not future_m.empty:
                m_mean = prior_m.mean()
                m_std = prior_m.std()
                if m_std and m_std > 0:
                    mortgage_z_max = (future_m.max() - m_mean) / m_std

        if houst_z_min is not None and mortgage_z_max is not None:
            housing_bust = (houst_z_min <= -DEPRESSION_Z_THRESHOLD) and (mortgage_z_max >= 1.0)

    # Interbank stress severity (z-score on SOFR - DFF spread)
    interbank_stress = False
    interbank_max_spread = None
    interbank_z = None
    interbank_spread = precomp["interbank_spread"]
    if not interbank_spread.empty:
        prior = interbank_spread.loc[:current_date].dropna().tail(HIST_WINDOW_TRADING_DAYS)
        future = interbank_spread.loc[current_date:end_date].dropna()
        if not prior.empty and not future.empty:
            interbank_max_spread = future.max()
            mean = prior.mean()
            std = prior.std()
            if std and std > 0:
                interbank_z = (interbank_max_spread - mean) / std
                interbank_stress = interbank_z >= DEPRESSION_Z_THRESHOLD

    # NBER recession indicator (context only)
    nber_recession = False
    if not fred_df.empty and "USREC" in fred_df.columns:
        usrec_future = fred_df.loc[current_date:end_date]["USREC"].dropna()
        if not usrec_future.empty:
            nber_recession = usrec_future.max() >= 1

    depression_score = sum([equity_crash, credit_freeze, labor_shock, housing_bust, interbank_stress])
    depression_flag = depression_score >= DEPRESSION_COMPONENTS_REQUIRED

    return {
        "DepressionScore": depression_score,
        "DepressionFlag": int(depression_flag),
        "Depress_NBER_Recession": int(nber_recession),
        "Depress_EquityCrash": int(equity_crash),
        "Depress_CreditFreeze": int(credit_freeze),
        "Depress_LaborShock": int(labor_shock),
        "Depress_HousingBust": int(housing_bust),
        "Depress_Interbank": int(interbank_stress),
        "Depress_SPX_Drawdown_12m": round(spx_dd_12m, 2) if spx_dd_12m is not None else None,
        "Depress_SPX_Drawdown_5p": round(prior_dd.quantile(0.05), 2) if not prior_dd.empty else None,
        "Depress_Credit_Max": round(credit_max, 2) if credit_max is not None else None,
        "Depress_Credit_Z": round(credit_z, 2) if credit_z is not None else None,
        "Depress_ICSA_4w_Max": round(icsa_max_4w, 0) if icsa_max_4w is not None else None,
        "Depress_ICSA_Z": round(icsa_z, 2) if icsa_z is not None else None,
        "Depress_HOUST_Z_Min": round(houst_z_min, 2) if houst_z_min is not None else None,
        "Depress_MORT_Z_Max": round(mortgage_z_max, 2) if mortgage_z_max is not None else None,
        "Depress_Interbank_Spread_Max": round(interbank_max_spread, 3) if interbank_max_spread is not None else None,
        "Depress_Interbank_Z": round(interbank_z, 2) if interbank_z is not None else None,
    }

# --- REPORTING ---
class DetailedPDF(FPDF):
    def header(self):
        # Register Lexend Font
        self.add_font('Lexend', '', 'fonts/LexendDeca-Regular.ttf', uni=True)
        self.add_font('Lexend', 'B', 'fonts/LexendDeca-Bold.ttf', uni=True)
        
        self.set_font('Lexend', 'B', 14)
        self.cell(0, 10, 'Centinela F2628 - Comprehensive Backtest Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Lexend', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_maximalist_report(results, market_df):
    pdf = DetailedPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Global Stats
    start_date = results['Date'].min()
    end_date = results['Date'].max()
    spx_total_return = (market_df["^SPX"].iloc[-1] - market_df["^SPX"].iloc[0]) / market_df["^SPX"].iloc[0] * 100
    
    pdf.set_font("Lexend", "", 10)
    pdf.multi_cell(0, 5, f"Period: {start_date} to {end_date} | Total Trading Days: {len(market_df)}\nBenchmark (SPX) Total Return: {spx_total_return:+.2f}%\n\nBasis: Austro-Georgist Synthesis (18-year Cycle)\nMain Inspiration: Fred Foldvary")
    pdf.ln(5)

    # --- GLOSSARY ---
    pdf.set_font("Lexend", "B", 12)
    pdf.cell(0, 8, "Glossary & Definitions", 0, 1)
    pdf.set_font("Lexend", "", 9)
    glossary = (
        "Win Rate: Percentage of times the asset price INCREASED over the period.\n"
        "Crash Accuracy: Percentage of times the asset price DECREASED (Correct for Crisis Signals).\n"
        "Avg Return: The average percentage change in price.\n"
        "Vol(SD): Standard Deviation of returns (Volatility risk).\n"
        "COMBO_CRISIS: Simultaneous trigger of Solvency Death (Credit) + Sugar Crash (Euphoria).\n"
        "NOTE: For Crisis Signals (Red), a high 'Win Rate' means the signal FAILED (market rose).\n"
        "      A high 'Crash Accuracy' means the signal SUCCEEDED (market fell)."
    )
    pdf.multi_cell(0, 5, glossary)
    pdf.ln(5)

    # --- NBER RECESSION ALIGNMENT ---
    if "Depress_NBER_Recession" in results.columns:
        pdf.set_font("Lexend", "B", 12)
        pdf.cell(0, 8, "NBER Recession Alignment (Context)", 0, 1)
        pdf.set_font("Lexend", "", 9)
        pdf.multi_cell(0, 4, "Share of signals followed by an NBER recession flag within the 12-month window.")
        pdf.ln(2)

        pdf.set_font("Lexend", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(60, 6, "Signal", 1, 0, 'L', 1)
        pdf.cell(25, 6, "Count", 1, 0, 'C', 1)
        pdf.cell(25, 6, "NBER Rate", 1, 1, 'C', 1)
        pdf.set_font("Lexend", "", 8)

        for event in results['Event'].unique():
            sub = results[results['Event'] == event]
            nber_rate = sub["Depress_NBER_Recession"].mean() * 100 if not sub.empty else 0
            pdf.cell(60, 6, str(event), 1, 0, 'L')
            pdf.cell(25, 6, str(len(sub)), 1, 0, 'C')
            pdf.cell(25, 6, f"{nber_rate:.1f}%", 1, 1, 'C')
        pdf.ln(4)

    # --- CLUSTER START SUMMARY (DE-DUPLICATED) ---
    if "IsClusterStart" in results.columns:
        cluster = results[results["IsClusterStart"] == 1]
        if not cluster.empty:
            pdf.set_font("Lexend", "B", 12)
            pdf.cell(0, 8, "Cluster-Start Summary (De-duplicated)", 0, 1)
            pdf.set_font("Lexend", "", 9)
            pdf.multi_cell(0, 4, "Uses only the first signal in each 30-day cluster to reduce autocorrelation. Rates are more conservative.")
            pdf.ln(2)

            pdf.set_font("Lexend", "B", 8)
            pdf.set_fill_color(240, 240, 240)
            cols = ["Signal", "Count", "Crash90", "Win180", "Depression", "NBER"]
            widths = [60, 14, 16, 16, 20, 14]
            for i, c in enumerate(cols):
                pdf.cell(widths[i], 6, c, 1, 0, 'C', 1)
            pdf.ln()

            pdf.set_font("Lexend", "", 8)
            for event in results["Event"].unique():
                sub = cluster[cluster["Event"] == event]
                if sub.empty:
                    continue
                crash90 = (sub["Fwd_90d"] < 0).mean() * 100 if "Fwd_90d" in sub.columns else 0
                win180 = (sub["Fwd_180d"] > 0).mean() * 100 if "Fwd_180d" in sub.columns else 0
                dep_rate = sub["DepressionFlag"].mean() * 100 if "DepressionFlag" in sub.columns else 0
                nber_rate = sub["Depress_NBER_Recession"].mean() * 100 if "Depress_NBER_Recession" in sub.columns else 0
                pdf.cell(widths[0], 6, str(event), 1, 0, 'L')
                pdf.cell(widths[1], 6, str(len(sub)), 1, 0, 'C')
                pdf.cell(widths[2], 6, f"{crash90:.0f}%", 1, 0, 'C')
                pdf.cell(widths[3], 6, f"{win180:.0f}%", 1, 0, 'C')
                pdf.cell(widths[4], 6, f"{dep_rate:.0f}%", 1, 0, 'C')
                pdf.cell(widths[5], 6, f"{nber_rate:.0f}%", 1, 1, 'C')
            pdf.ln(4)

    events = results['Event'].unique()
    
    # Sort events: Crisis first, then Entry
    crisis_events = [e for e in events if "BUY" not in e]
    entry_events = [e for e in events if "BUY" in e]
    
    for category, event_list in [("Crisis Predictors (Bearish)", crisis_events), ("Entry Signals (Bullish)", entry_events)]:
        if not event_list: continue
        
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Lexend", "B", 12)
        pdf.cell(0, 8, category, 1, 1, 'L', 1)
        pdf.ln(2)
        
        for event in event_list:
            pdf.set_font("Lexend", "B", 11)
            pdf.set_text_color(0, 0, 100)
            pdf.cell(0, 8, f"SIGNAL: {event}", 0, 1)
            pdf.set_text_color(0, 0, 0)
            
            sub = results[results['Event'] == event]
            
            # Table Header
            pdf.set_font("Lexend", "B", 8)
            pdf.set_fill_color(240, 240, 240)
            cols = ["Win", "Count", "WinRate", "CrashAcc", "Avg", "Med", "Best", "Worst", "Vol(SD)"]
            w_width = [10, 12, 18, 18, 18, 18, 18, 18, 18]
            
            for i, c in enumerate(cols):
                pdf.cell(w_width[i], 6, c, 1, 0, 'C', 1)
            pdf.ln()
            
            pdf.set_font("Lexend", "", 8)
            
            for w in FORWARD_WINDOWS:
                col_ret = f"Fwd_{w}d"
                data = sub[col_ret].dropna()
                
                if len(data) == 0:
                    pdf.cell(0, 6, "Insufficient Data", 1, 1)
                    continue
                
                count = len(data)
                win_rate = (data > 0).sum() / count * 100
                crash_acc = (data < 0).sum() / count * 100
                avg = data.mean()
                med = data.median()
                best = data.max()
                worst = data.min()
                vol = data.std()
                
                pdf.cell(w_width[0], 6, f"{w}d", 1, 0, 'C')
                pdf.cell(w_width[1], 6, str(count), 1, 0, 'C')
                
                # Color code win rate
                if win_rate > 55: pdf.set_text_color(0, 100, 0)
                else: pdf.set_text_color(0, 0, 0)
                pdf.cell(w_width[2], 6, f"{win_rate:.0f}%", 1, 0, 'C')
                
                # Color code crash acc
                if crash_acc > 55: pdf.set_text_color(150, 0, 0)
                else: pdf.set_text_color(0, 0, 0)
                pdf.cell(w_width[3], 6, f"{crash_acc:.0f}%", 1, 0, 'C')
                pdf.set_text_color(0, 0, 0)
                
                pdf.cell(w_width[4], 6, f"{avg:+.1f}%", 1, 0, 'C')
                pdf.cell(w_width[5], 6, f"{med:+.1f}%", 1, 0, 'C')
                pdf.cell(w_width[6], 6, f"{best:+.1f}%", 1, 0, 'C')
                pdf.cell(w_width[7], 6, f"{worst:+.1f}%", 1, 0, 'C')
                pdf.cell(w_width[8], 6, f"{vol:.1f}%", 1, 0, 'C')
                pdf.ln()
            
            # Qualitative Analysis Block
            pdf.ln(1)
            pdf.set_font("Lexend", "", 8)
            
            fwd90 = sub["Fwd_90d"].dropna()
            if len(fwd90) > 0:
                avg90 = fwd90.mean()
                crash90 = (fwd90 < 0).sum() / len(fwd90) * 100
                
                analysis = "Evaluation: "
                if "BUY" in event:
                    if avg90 > 5: analysis += "STRONG BULLISH. Historically profitable entry point. "
                    elif avg90 > 0: analysis += "MODERATE BULLISH. Positive expectancy but requires patience. "
                    else: analysis += "WEAK/FAILED ENTRY. Historically loses money. "
                else:
                    if crash90 > 60: analysis += "VALID CRISIS INDICATOR. High probability of market decline. "
                    elif crash90 < 20: analysis += "CONTRARIAN INDICATOR. Market consistently RIES after this signal (Bullish). "
                    elif abs(avg90) < 2: analysis += "STAGNATION SIGNAL. Market tends to move sideways (Choppy). "
                    else: analysis += "NOISE / LOW PREDICTIVE VALUE. "
                
                pdf.multi_cell(0, 4, analysis)
            
            pdf.ln(4)

    pdf.output("comprehensive_backtest_report.pdf")
    print("PDF Generated: comprehensive_backtest_report.pdf")

def run_backtest():
    market_df, fred_df, housing_df = fetch_historical_data()
    if market_df is None: return

    print(f"Running Scientific Backtest on {len(market_df)} days...")
    results = []
    history_log = []
    last_seen = {}
    precomp = prepare_depression_inputs(market_df, fred_df, housing_df)

    for idx in range(505, len(market_df)):
        triggers, history_log, details = analyse_date(market_df, fred_df, housing_df, idx, history_log)
        
        if not triggers: continue

        depression_outcomes = compute_depression_outcomes(market_df, fred_df, housing_df, idx, precomp)
        
        for event in triggers:
            asset = "WPM" if "WPM" in event else ("BTC-USD" if "BTC" in event else "^SPX")
            
            entry = market_df[asset].iloc[idx]
            event_detail = details.get(event)
            current_date = market_df.index[idx]
            last_date = last_seen.get(event)
            is_cluster_start = True if last_date is None else (current_date - last_date).days > DEDUP_DAYS
            last_seen[event] = current_date

            res = {
                "Date": current_date.strftime("%Y-%m-%d"),
                "Event": event,
                "EventDetail": event_detail,
                "Asset": asset,
                "IsClusterStart": int(is_cluster_start),
            }
            res.update(depression_outcomes)
            
            if pd.isna(entry) or entry == 0: continue
            
            for w in FORWARD_WINDOWS:
                fut_idx = idx + w
                if fut_idx < len(market_df):
                    ex = market_df[asset].iloc[fut_idx]
                    res[f"Fwd_{w}d"] = ((ex - entry) / entry) * 100
                else:
                    res[f"Fwd_{w}d"] = None
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        df.to_csv("maximalist_backtest.csv", index=False)
        df.to_csv("scientific_backtest.csv", index=False)
        os.makedirs("output", exist_ok=True)
        df.to_csv("output/maximalist_backtest.csv", index=False)
        df.to_csv("output/scientific_backtest.csv", index=False)
        generate_maximalist_report(df, market_df)
    else:
        print("No signals.")

if __name__ == "__main__":
    run_backtest()
