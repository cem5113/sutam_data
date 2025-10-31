#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_all.py
- Girdi: saatlik full-grid parquet (sf_crime_grid_full_labeled.parquet gibi)
- Çıktılar: sf_crime_grid_{3h,8h,1d,1w,1m}.parquet
- Not: aggregate_windows.py ile aynı klasörde çalıştırın
"""

import argparse
import subprocess
from pathlib import Path

FREQS_DEFAULT = ["3H","8H","1D","1W","1M"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Saatlik full-grid parquet")
    p.add_argument("--tz", type=str, default="America/Los_Angeles")
    p.add_argument("--freqs", type=str, default=",".join(FREQS_DEFAULT),
                   help="Virgüllü: 3H,8H,1D,1W,1M")
    p.add_argument("--prefix", type=str, default="sf_crime_grid_", help="Çıktı dosya prefix’i")
    return p.parse_args()

def main():
    args = parse_args()
    freqs = [f.strip() for f in args.freqs.split(",") if f.strip()]
    base = Path(args.input).resolve()
    if not base.exists():
        raise SystemExit(f"Girdi yok: {base}")

    here = Path(__file__).resolve().parent
    aw = here / "aggregate_windows.py"
    if not aw.exists():
        raise SystemExit(f"aggregate_windows.py bulunamadı: {aw}")

    for f in freqs:
        out = base.parent / f"{args.prefix}{f.lower()}.parquet"
        cmd = [
            "python", str(aw),
            "--input", str(base),
            "--out",   str(out),
            "--freq",  f,
            "--tz",    args.tz or ""
        ]
        print("▶️", " ".join(cmd))
        subprocess.check_call([c for c in cmd if c != ""])
        if not out.exists():
            raise SystemExit(f"Üretilemedi: {out}")
        print(f"✅ {out}")

if __name__ == "__main__":
    main()
