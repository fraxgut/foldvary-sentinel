import os
import sys
from datetime import datetime, timedelta, time as dt_time, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import history_store
import sentinel


def _parse_date(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


def _default_state():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
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
            "FLASH_MOVE": [],
        },
        "daily_alerts": {
            "date": today_str,
            "sent_events": [],
        },
    }


def main():
    days = int(os.environ.get("BACKFILL_DAYS", "365"))
    start_override = _parse_date(os.environ.get("BACKFILL_START_DATE"))
    end_override = _parse_date(os.environ.get("BACKFILL_END_DATE"))

    end_date = end_override or datetime.now(timezone.utc)
    if start_override:
        start_date = start_override
    else:
        start_date = end_date - timedelta(days=days)

    db_path = os.environ.get("HISTORY_DB_PATH", os.path.join("output", "history.db"))
    retention_days = int(os.environ.get("HISTORY_RETENTION_DAYS", "365"))

    sync_gist = os.environ.get("BACKFILL_SYNC_GIST", "false").lower() == "true"
    history_load_state = True

    if sync_gist and sentinel.GIST_TOKEN and sentinel.STATE_GIST_ID:
        history_load_state = history_store.load_db_from_gist(
            sentinel.GIST_TOKEN,
            sentinel.STATE_GIST_ID,
            db_path,
            os.environ.get("HISTORY_GIST_FILENAME", "history.db.b64"),
        )

    history_store.ensure_db(db_path)

    market_data, wpm_data, fred_data, housing_data = sentinel.get_data(
        start_date=start_date,
        end_date=end_date,
    )

    if market_data.empty:
        print("No market data available for backfill window.")
        return

    state = _default_state()

    dates = sorted({idx.date() for idx in market_data.index})
    total = len(dates)

    for idx, day in enumerate(dates, start=1):
        day_dt = datetime.combine(day, dt_time.min)

        df_slice = market_data.loc[:day_dt]
        wpm_slice = wpm_data.loc[:day_dt] if not wpm_data.empty else wpm_data
        fred_slice = fred_data.loc[:day_dt] if not fred_data.empty else fred_data
        housing_slice = housing_data.loc[:day_dt] if not housing_data.empty else housing_data

        analysis = sentinel.analyse_market(
            df_slice,
            wpm_slice,
            fred_slice,
            housing_slice,
            state,
            end_date=day_dt,
        )

        if analysis.get("state_update"):
            for key, value in analysis["state_update"].items():
                state[key] = value

        date_str = day_dt.strftime("%Y-%m-%d")
        snapshot = {
            "date": date_str,
            "run_ts": day_dt.isoformat(),
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

        history_store.upsert_daily_snapshot(db_path, snapshot)

        risk_window = int(os.environ.get("RISK_WINDOW_DAYS", os.environ.get("REGIME_WINDOW_DAYS", "14")))
        risk_trend = int(os.environ.get("RISK_TREND_DAYS", os.environ.get("REGIME_TREND_DAYS", "7")))
        risk_min = int(os.environ.get("RISK_MIN_DAYS", os.environ.get("REGIME_MIN_DAYS", "7")))
        history_limit = history_store.regime_history_limit(
            risk_window,
            risk_trend,
            risk_min,
        )
        history_rows = history_store.fetch_recent_history(db_path, history_limit)
        regime = history_store.compute_regime(
            history_rows,
            risk_window,
            risk_trend,
            risk_min,
        )
        history_store.update_regime_fields(db_path, date_str, regime)

        if idx % 25 == 0 or idx == total:
            print(f"Backfill {idx}/{total}: {date_str} -> {analysis.get('event')}")

    history_store.prune_history(db_path, retention_days)

    if sync_gist and sentinel.GIST_TOKEN and sentinel.STATE_GIST_ID:
        if history_load_state is False:
            print("History load failed earlier; skipping gist save to avoid remote overwrite.")
        else:
            history_store.save_db_to_gist(
                sentinel.GIST_TOKEN,
                sentinel.STATE_GIST_ID,
                db_path,
                os.environ.get("HISTORY_GIST_FILENAME", "history.db.b64"),
            )

    print("Backfill complete.")


if __name__ == "__main__":
    main()
