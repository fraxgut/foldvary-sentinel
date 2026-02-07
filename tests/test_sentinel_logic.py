import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import pandas as pd

import sentinel


def _base_state():
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
            "date": "2026-02-07",
            "sent_events": [],
            "sent_ops_warnings": [],
        },
    }


def _market_df(rows=320):
    idx = pd.date_range("2025-01-01", periods=rows, freq="D")
    lin = np.linspace
    return pd.DataFrame(
        {
            "^TNX": lin(4.00, 4.15, rows),
            "^SPX": lin(5000, 5150, rows),
            "^VIX": np.full(rows, 20.0),
            "CL=F": np.full(rows, 70.0),
            "GC=F": np.full(rows, 2000.0),
            "SI=F": np.full(rows, 25.0),
            "HG=F": np.full(rows, 4.0),
            "HRC=F": np.full(rows, 800.0),
            "ITA": np.full(rows, 200.0),
            "DX-Y.NYB": np.full(rows, 102.0),
            "WPM": np.full(rows, 50.0),
            "BTC-USD": np.full(rows, 40000.0),
        },
        index=idx,
    )


def _wpm_df(index):
    return pd.DataFrame(
        {
            "Volume": np.full(len(index), 1_000_000.0),
            "Close": np.full(len(index), 50.0),
        },
        index=index,
    )


def _fred_df(index, spread=3.0):
    return pd.DataFrame(
        {
            "BAMLH0A0HYM2": np.full(len(index), spread),
            "RRPONTSYD": np.full(len(index), 500.0),
            "WALCL": np.full(len(index), 7_000_000.0),
            "WTREGEN": np.full(len(index), 700_000.0),
            "ICSA": np.full(len(index), 220_000.0),
            "SOFR": np.full(len(index), 5.0),
            "DFF": np.full(len(index), 5.0),
        },
        index=index,
    )


class SentinelLogicTests(unittest.TestCase):
    def test_check_temporal_combo_respects_window_days(self):
        state = _base_state()
        state["recent_signals"]["SUGAR_CRASH"] = ["2026-01-01"]
        state["recent_signals"]["EM_CURRENCY_STRESS"] = ["2026-02-20"]

        self.assertFalse(
            sentinel.check_temporal_combo(
                state,
                "SUGAR_CRASH",
                "EM_CURRENCY_STRESS",
                window_days=30,
            )
        )
        self.assertTrue(
            sentinel.check_temporal_combo(
                state,
                "SUGAR_CRASH",
                "EM_CURRENCY_STRESS",
                window_days=60,
            )
        )

    def test_analyse_market_emergency_exit_updates_state(self):
        market = _market_df()
        wpm = _wpm_df(market.index)
        fred = _fred_df(market.index, spread=3.0)
        # Enforce 3-day solvency confirmation.
        fred.iloc[-3:, fred.columns.get_loc("BAMLH0A0HYM2")] = [5.2, 5.3, 5.4]

        state = _base_state()
        state["wpm_entry_price"] = 40.0

        result = sentinel.analyse_market(
            market,
            wpm,
            fred,
            pd.DataFrame(),
            state,
            end_date=market.index[-1].to_pydatetime(),
        )

        self.assertIn("SELL_WPM_NOW", result["trigger_events"])
        self.assertEqual(result["state_update"]["wpm_entry_price"], None)

    def test_analyse_market_returns_data_outage_for_missing_core_series(self):
        market = _market_df().drop(columns=["^TNX"])
        result = sentinel.analyse_market(
            market,
            _wpm_df(market.index),
            _fred_df(market.index),
            pd.DataFrame(),
            _base_state(),
            end_date=market.index[-1].to_pydatetime(),
        )

        self.assertEqual(result["event"], "DATA_OUTAGE")
        self.assertIn("us10y", result["missing"])

    def test_update_regime_context_adds_ops_warning_on_history_load_failure(self):
        analysis = {
            "event": "NORMAL",
            "stress_score": 1,
            "stress_level": "Bajo",
            "phase_daily": "Fase 1 - AUGE",
            "spx": 5000,
            "vix": 20,
            "spread": 3.0,
            "us10y": 4.0,
            "dxy": 102.0,
            "net_liq_b": 5800.0,
            "gold": 2000.0,
            "btc_price": 40000.0,
            "trigger_events": [],
            "reason": "ok",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "history.db")
            with mock.patch.object(sentinel, "HISTORY_DB_PATH", db_path), \
                 mock.patch.object(sentinel, "GIST_TOKEN", "token"), \
                 mock.patch.object(sentinel, "STATE_GIST_ID", "gist"), \
                 mock.patch.object(sentinel.history_store, "load_db_from_gist", return_value=False), \
                 mock.patch.object(sentinel.history_store, "save_db_to_gist") as mocked_save:
                enriched = sentinel.update_regime_context(dict(analysis))

        self.assertIn("HISTORY_GIST_LOAD_FAILED_SKIP_SAVE", enriched.get("ops_warnings", []))
        self.assertIn("HISTORY_GIST_SAVE_SKIPPED_DUE_LOAD_FAILURE", enriched.get("ops_warnings", []))
        mocked_save.assert_not_called()

    def test_send_ops_warnings_records_sent_codes_on_success(self):
        state = _base_state()
        with mock.patch.object(sentinel, "TG_TOKEN", "token"), \
             mock.patch.object(sentinel, "TG_CHAT_ID", "chat"), \
             mock.patch.object(sentinel, "send_telegram", return_value=123):
            sentinel.send_ops_warnings(state, ["HISTORY_GIST_SAVE_FAILED"], is_manual_run=False)

        self.assertIn("HISTORY_GIST_SAVE_FAILED", state["daily_alerts"]["sent_ops_warnings"])


if __name__ == "__main__":
    unittest.main()
