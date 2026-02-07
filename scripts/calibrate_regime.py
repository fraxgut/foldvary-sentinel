import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pandas_datareader.data as web

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import history_store


def _get_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _get_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _percentile_rank(values, current_value):
    if not values:
        return None
    count = sum(1 for v in values if v < current_value)
    return (count / len(values)) * 100


def _f1_score(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _best_threshold(values, labels, direction="ge"):
    if not values:
        return None, 0.0

    pairs = sorted(zip(values, labels), key=lambda x: x[0])
    thresholds = sorted({v for v, _ in pairs})

    best_threshold = thresholds[0]
    best_f1 = 0.0

    for threshold in thresholds:
        tp = fp = fn = 0
        for value, label in pairs:
            if value is None:
                continue
            prediction = value >= threshold if direction == "ge" else value <= threshold
            if prediction and label:
                tp += 1
            elif prediction and not label:
                fp += 1
            elif not prediction and label:
                fn += 1
        f1 = _f1_score(tp, fp, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def _validate_thresholds(phase2_enter, phase2_exit, phase3_enter, phase3_exit):
    if phase2_exit > phase2_enter:
        phase2_exit = phase2_enter
    if phase3_exit > phase3_enter:
        phase3_exit = phase3_enter
    if phase3_enter < phase2_enter:
        phase3_enter = phase2_enter
    if phase3_exit < phase2_enter:
        phase3_exit = phase2_enter
    return phase2_enter, phase2_exit, phase3_enter, phase3_exit


def _build_labels(usrec_daily, dates, lookahead_days):
    labels = []
    for day in dates:
        start = day + timedelta(days=1)
        end = day + timedelta(days=lookahead_days)
        window = usrec_daily.loc[start:end]
        label = 1 if not window.empty and window.max() >= 1 else 0
        labels.append(label)
    return labels


def main():
    db_path = os.environ.get("HISTORY_DB_PATH", os.path.join("output", "history.db"))
    output_path = os.environ.get("REGIME_CALIBRATION_PATH", os.path.join("output", "regime_calibration.json"))

    value_type = os.environ.get("CALIBRATION_VALUE_TYPE", "percentile").lower()
    if value_type not in {"percentile", "zscore", "absolute"}:
        print("Invalid CALIBRATION_VALUE_TYPE; use percentile, zscore, or absolute.")
        return

    window_days = _get_int("REGIME_WINDOW_DAYS", 14)
    lookback = _get_int("REGIME_PCT_LOOKBACK", 180) if value_type == "percentile" else _get_int("REGIME_Z_LOOKBACK", 180)
    min_samples = _get_int("CALIBRATION_MIN_SAMPLES", 120)

    lookahead_phase2 = _get_int("CALIBRATION_LOOKAHEAD_PHASE2_DAYS", 365)
    lookahead_phase3 = _get_int("CALIBRATION_LOOKAHEAD_PHASE3_DAYS", 180)

    history = history_store.fetch_history(db_path)
    if not history:
        print("No history available for calibration.")
        return

    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in history]
    scores = [row.get("phase_score") for row in history]

    regime_scores = []
    regime_values = []
    usable_dates = []

    for idx, score in enumerate(scores):
        if score is None:
            continue

        start = max(0, idx - window_days + 1)
        window_scores = [s for s in scores[start:idx + 1] if s is not None]
        if not window_scores:
            continue
        regime_score = sum(window_scores) / len(window_scores)

        sample_start = max(0, idx - lookback + 1)
        sample = [s for s in scores[sample_start:idx + 1] if s is not None]
        if len(sample) < min_samples:
            continue

        if value_type == "percentile":
            regime_value = _percentile_rank(sample, regime_score)
        elif value_type == "zscore":
            mean = sum(sample) / len(sample)
            variance = sum((x - mean) ** 2 for x in sample) / len(sample)
            std = variance ** 0.5
            if std == 0:
                continue
            regime_value = (regime_score - mean) / std
        else:
            regime_value = regime_score

        regime_scores.append(regime_score)
        regime_values.append(regime_value)
        usable_dates.append(dates[idx])

    if len(regime_values) < min_samples:
        print("Not enough samples for calibration after filtering.")
        return

    start_date = min(usable_dates)
    end_date = max(usable_dates) + timedelta(days=max(lookahead_phase2, lookahead_phase3))

    usrec = web.DataReader("USREC", "fred", start_date, end_date).ffill()
    usrec_daily = usrec.resample("D").ffill()["USREC"]

    labels_phase2 = _build_labels(usrec_daily, usable_dates, lookahead_phase2)
    labels_phase3 = _build_labels(usrec_daily, usable_dates, lookahead_phase3)

    phase2_enter, f1_phase2 = _best_threshold(regime_values, labels_phase2, direction="ge")
    phase3_enter, f1_phase3 = _best_threshold(regime_values, labels_phase3, direction="ge")

    phase2_exit, f1_phase2_exit = _best_threshold(regime_values, [1 - v for v in labels_phase2], direction="le")
    phase3_exit, f1_phase3_exit = _best_threshold(regime_values, [1 - v for v in labels_phase3], direction="le")

    phase2_enter, phase2_exit, phase3_enter, phase3_exit = _validate_thresholds(
        phase2_enter,
        phase2_exit,
        phase3_enter,
        phase3_exit,
    )

    payload = {
        "value_type": value_type,
        "phase2_enter": round(float(phase2_enter), 4),
        "phase2_exit": round(float(phase2_exit), 4),
        "phase3_enter": round(float(phase3_enter), 4),
        "phase3_exit": round(float(phase3_exit), 4),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookahead_phase2_days": lookahead_phase2,
        "lookahead_phase3_days": lookahead_phase3,
        "window_days": window_days,
        "lookback_days": lookback,
        "min_samples": min_samples,
        "sample_size": len(regime_values),
        "f1_phase2_enter": round(f1_phase2, 4),
        "f1_phase3_enter": round(f1_phase3, 4),
        "f1_phase2_exit": round(f1_phase2_exit, 4),
        "f1_phase3_exit": round(f1_phase3_exit, 4),
        "note": "Thresholds chosen by maximising F1 for recession/no-recession labels.",
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Calibration saved to", output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
