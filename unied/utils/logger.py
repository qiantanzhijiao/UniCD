# -*- coding: utf-8 -*-
"""Small training logger: writes a CSV-like log and mirrors to stdout."""
import os
import time
from pathlib import Path


class TrainLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "train_log.csv"
        self.fields = ["epoch", "step", "loss", "det", "cls", "reg", "tebs", "br", "cor", "res", "sec/iter"]
        if not self.metrics_path.exists():
            self._write_header()

    def _write_header(self):
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            f.write(",".join(self.fields) + "\n")

    def log(self, **values):
        line = ",".join(str(values.get(f, "")) for f in self.fields)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        parts = " ".join(f"{f}={values[f]:.4f}" if isinstance(values.get(f), float) else f"{f}={values[f]}" for f in self.fields if f in values)
        print(f"[train] {time.strftime('%H:%M:%S')} {parts}")
