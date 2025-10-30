#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_windows.py
- Saatlik full-grid verisini (GEOID × dt) alır
- Günlük (1D) veya 8 saatlik (8H) pencerelere toplar
- Y_label'ı pencere içinde >=1 olay varsa 1 olacak şekilde yeniden kurar
- Leakage-safe rolling priors (28g / 180g) ekler (GEOID × {day_of_week, block_id/None} kırılımı)
- Çıktıyı parquet yazar

Kullanım:
  python aggregate_windows.py \
    --input sf_crime_grid_full_labeled.parquet \
    --out   sf_crime_grid_8h.parquet \
    --freq  8H
  # veya
  python aggregate_windows.py --input ... --out sf_crime_grid_1d.parquet --freq 1D
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

CAL_DROP = {"dt_local"}  # saatlikten kalan, pencerede anlamsız

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="Saatlik full-grid parquet (sf_crime_grid_full_labeled.parquet)")
    p.add_argument("--out", type=Path, required=True,
                   help="Çıktı parquet (örn. sf_crime_grid_8h.parquet / sf_crime_grid_1d.parquet)")
    p.add_argument("--freq", type=str, choices=["1D","8H"], default="1D",
                   help="Pencere frekansı: 1D (günlük) veya 8H (8 saatlik bloklar)")
    p.add_argument("--tz", type=str, default=None,
                   help="Yerel TZ analiz için opsiyonel (takvim kolonları). Örn: America/Los_Angeles")
    return p.parse_args()

def add_cal(df: pd.DataFrame, base_col: str, tz: Optional[str]) -> pd.DataFrame:
    # base_col: t0 (pencere başlangıcı)
    if tz:
        df["t0_local"] = df[base_col].dt.tz_convert(tz)
        bc = "t0_local"
    else:
        bc = base_col
    df["year"] = df[bc].dt.year.astype("int16")
    df["month"] = df[bc].dt.month.astype("int8")
    df["day_of_week"] = df[bc].dt.dayofweek.astype("int8")
    df["hour_start"] = df[bc].dt.hour.astype("int8")
    return df

def make_block_start_8h(dt: pd.Series) -> Tuple[pd.Series, pd.Series]:
    # 0–7, 8–15, 16–23 blokları
    block_id = (dt.dt.hour // 8).astype("int8")
    t0 = dt.dt.floor("D") + pd.to_timedelta(block_id * 8, unit="h")
    t0 = t0.dt.tz_localize("UTC") if t0.dt.tz is None else t0
    return t0, block_id

def prior_rolling(df: pd.DataFrame, window: str, suffix: str,
                  keys: List[str]) -> pd.DataFrame:
    # df: GEOID, t0, Y_label, ... (t0 zaman sırasına göre)
    df = df.sort_values(["GEOID", "t0"]).copy()

    grp_cols = ["GEOID"] + keys
    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("t0").set_index("t0")
        # geçmişe bakan olay sayısı (leakage-safe: shift(1))
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return out.reset_index()

    out = df.groupby(grp_cols, group_keys=False).apply(_roll)
    # pencere başına saat sayısı:
    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1h"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return out

def main():
    args = parse_args()
    inp: Path = args.input
    outp: Path = args.out
    freq = args.freq

    df = pd.read_parquet(inp)
    if "dt" not in df.columns or "GEOID" not in df.columns:
        raise SystemExit("Saatlik grid formatı bekleniyor: dt ve GEOID gerekli.")
    # Temizlik
    for c in CAL_DROP:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Saat serisini garanti et
    df["dt"] = pd.to_datetime(df["dt"], utc=True)

    # Pencere başlangıcı t0, ve 8H ise block_id
    if freq == "1D":
        t0 = df["dt"].dt.floor("D")
        block_id = None
    else:  # 8H
        t0, block_id = make_block_start_8h(df["dt"])

    # Toplanacak sayısal kolonlar:
    # - crime_count: SUM (penceredeki toplam olay)
    # - diğer numerikler: MEAN (daha istikrarlı)
    base_drop = {"GEOID","dt","Y_label"}
    num_cols = [c for c in df.columns
                if (c not in base_drop) and pd.api.types.is_numeric_dtype(df[c])]

    agg_sum = {"crime_count": "sum"} if "crime_count" in df.columns else {}
    other_nums = [c for c in num_cols if c != "crime_count"]

    # Toplama anahtarları
    keys = ["GEOID"]
    if freq == "8H":
        keys += [t0.rename("t0"), block_id.rename("block_id")]
    else:
        keys += [t0.rename("t0")]

    sdf = df.copy()
    sdf["t0"] = t0
    if block_id is not None:
        sdf["block_id"] = block_id

    agg = (
        sdf.groupby(keys, as_index=False)
           .agg(**agg_sum, **{c: ("mean") for c in other_nums})
    )

    # Yeni Y_label: pencerede ≥1 olay var mı?
    if "crime_count" in agg.columns:
        agg["Y_label"] = (agg["crime_count"] > 0).astype("int8")
    else:
        # nadir bir durum: olay sayısı kolonun yoksa, satır sayısından Y yapılabilir;
        # ama full-grid'te crime_count var kabul ediyoruz
        raise SystemExit("crime_count kolonu bekleniyordu.")

    # Takvim kolonları (t0 tabanlı)
    agg["t0"] = pd.to_datetime(agg["t0"], utc=True)
    agg = add_cal(agg, base_col="t0", tz=args.tz)

    # Priors: 28 gün & 180 gün — mevsimsellik anahtarları
    season_keys = ["day_of_week"]
    if "block_id" in agg.columns:
        season_keys.append("block_id")

    agg = prior_rolling(agg, window="28D", suffix="28d", keys=season_keys)
    agg = prior_rolling(agg, window="180D", suffix="180d", keys=season_keys)

    # Sıralama ve yazma
    order_cols = ["GEOID","t0"] + (["block_id"] if "block_id" in agg.columns else [])
    agg = agg.sort_values(order_cols).reset_index(drop=True)

    outp.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(outp, index=False)
    y_rate = float(agg["Y_label"].mean())
    print(f"[OK] Yazıldı: {outp}  satır={len(agg):,}  GEOID={agg['GEOID'].nunique():,}")
    print(f"[INFO] Y_label(1) % ≈ {100*y_rate:.4f} | 0 % ≈ {100*(1-y_rate):.4f}")
    print(f"[INFO] Zaman aralığı: {agg['t0'].min()} → {agg['t0'].max()}")
    if "block_id" in agg.columns:
        print(agg["block_id"].value_counts().sort_index())
        print("[INFO] 8H blokları: 0=[00–08), 1=[08–16), 2=[16–24)")

if __name__ == "__main__":
    main()
