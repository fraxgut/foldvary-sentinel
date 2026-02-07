import os
import unittest
from unittest import mock

import history_store


class HistoryStoreRegimeTests(unittest.TestCase):
    def test_regime_history_limit_auto_includes_lookbacks_and_min_samples(self):
        with mock.patch.dict(
            os.environ,
            {
                "REGIME_METHOD": "auto",
                "REGIME_Z_LOOKBACK": "180",
                "REGIME_PCT_LOOKBACK": "200",
                "REGIME_AUTO_Z_MIN_SAMPLES": "90",
                "REGIME_AUTO_PCT_MIN_SAMPLES": "220",
            },
            clear=False,
        ):
            limit = history_store.regime_history_limit(
                window_days=14,
                trend_days=7,
                min_days=7,
            )
        self.assertEqual(limit, 220)

    def test_regime_history_limit_absolute_uses_base_window_only(self):
        with mock.patch.dict(os.environ, {"REGIME_METHOD": "absolute"}, clear=False):
            limit = history_store.regime_history_limit(
                window_days=14,
                trend_days=7,
                min_days=7,
            )
        self.assertEqual(limit, 14)

    def test_compute_regime_hysteresis_downgrades_from_phase3(self):
        rows = [
            {"phase_score": 5.0, "regime_phase": "Fase 3 - QUIEBRE"},
            {"phase_score": 1.0, "regime_phase": None},
        ]
        with mock.patch.dict(
            os.environ,
            {
                "REGIME_METHOD": "absolute",
                "PHASE_2_THRESHOLD": "2.0",
                "PHASE_2_EXIT_THRESHOLD": "1.5",
                "PHASE_3_THRESHOLD": "4.0",
                "PHASE_3_EXIT_THRESHOLD": "3.0",
            },
            clear=False,
        ):
            regime = history_store.compute_regime(
                rows,
                window_days=1,
                trend_days=1,
                min_days=1,
            )
        self.assertEqual(regime["regime_phase"], "Fase 1 - AUGE")


if __name__ == "__main__":
    unittest.main()
