"""
Run multiple train+evaluate experiments and collect a comparison table.

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --configs configs/exp_baseline.json configs/exp_sensitive.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple experiment configs")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/exp_baseline.json",
            "configs/exp_sensitive.json",
            "configs/exp_conservative.json",
        ],
        help="List of JSON config files",
    )
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print(f"\n[runner] Running: {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    rows: list[dict[str, str]] = []
    for cfg_rel in args.configs:
        cfg_path = (root / cfg_rel).resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        cfg = load_json(cfg_path)
        exp_name = cfg.get("EXPERIMENT_NAME", cfg_path.stem)

        print("\n" + "=" * 70)
        print(f"[runner] Experiment: {exp_name}")
        print("=" * 70)

        run_step([sys.executable, "train.py", "--config", str(cfg_path)], cwd=root)
        run_step([sys.executable, "evaluate.py", "--config", str(cfg_path)], cwd=root)

        summary_path = root / "outputs" / exp_name / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Expected summary not found: {summary_path}")

        summary = load_json(summary_path)
        rows.append(
            {
                "experiment": summary.get("experiment", exp_name),
                "threshold": f"{summary.get('threshold', 0.0):.6f}",
                "anomalies": str(summary.get("num_anomalies", "")),
                "anomaly_rate_percent": f"{summary.get('anomaly_rate_percent', 0.0):.3f}",
                "score_mean": f"{summary.get('score_mean', 0.0):.6f}",
                "score_std": f"{summary.get('score_std', 0.0):.6f}",
                "threshold_method": str(summary.get("threshold_method", "")),
                "alpha": str(summary.get("hyperparameters", {}).get("alpha", "")),
                "beta": str(summary.get("hyperparameters", {}).get("beta", "")),
                "corr_threshold": str(summary.get("hyperparameters", {}).get("corr_threshold", "")),
            }
        )

    out_csv = root / "outputs" / "experiment_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "threshold",
        "anomalies",
        "anomaly_rate_percent",
        "score_mean",
        "score_std",
        "threshold_method",
        "alpha",
        "beta",
        "corr_threshold",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n[runner] All experiments complete")
    print(f"[runner] Comparison CSV -> {out_csv}")


if __name__ == "__main__":
    main()
