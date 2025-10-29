#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_labels_and_priors_xl.py
- Büyük veriler (≈2M satır, 70 sütun) için optimize etiket + prior üretimi
- 2 geçiş: (1) minimal okuma ile crime_count & Y_label, (2) yan özellikler partiler halinde
- Leakage-safe rolling priors (3m/12m) — geleceğe bakmaz (shift(1))
- Varsayılan: full grid KAPALI (isteğe bağlı aç)
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Yardımcılar
# -------------------------
def _season_from_month(m: int) -> str:
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5):  return "spring"
    if m in (6, 7, 8):  return "summer"
    return "fall"

def _norm_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    elif "geoid" in df.columns:
        df["GEOID"] = df["geoid"].astype(str)
    else:
        raise ValueError("GEOID/geoid kolonu bulunamadı.")
    return df

def _to_dt(df: pd.DataFrame) -> pd.Series:
    # saatlik zemin: datetime -> floor('H') (UTC)
    if "datetime" in df.columns:
        s = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif {"date","time"} <= set(df.columns):
        s = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=True)
    elif "event_hour" in df.columns:
        s = pd.to_datetime(df["event_hour"], errors="coerce", utc=True)
    elif "received_time" in df.columns:
        s = pd.to_datetime(df["received_time"], errors="coerce", utc=True)
    else:
        raise ValueError("datetime veya (date+time) veya event_hour/received_time kolonu gerekli.")
    if s.isna().all():
        raise ValueError("Zaman kolonları çözümlenemedi (hepsi NaT).")
    return s.dt.floor("H")

def _add_calendar(df: pd.DataFrame, tz: str | None) -> pd.DataFrame:
    if tz:
        df["dt_local"] = df["dt"].dt.tz_convert(tz)
        base = "dt_local"
    else:
        base = "dt"
    df["year"] = df[base].dt.year.astype("int16")
    df["month"] = df[base].dt.month.astype("int8")
    df["day_of_week"] = df[base].dt.dayofweek.astype("int8")
    df["hour"] = df[base].dt.hour.astype("int8")
    df["season"] = df["month"].map(_season_from_month).astype("category")
    return df

def _build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    dt_min = df["dt"].min().floor("H")
    dt_max = df["dt"].max().ceil("H")
    all_hours = pd.date_range(dt_min, dt_max, freq="H", tz="UTC")
    geoids = df["GEOID"].dropna().astype(str).unique()
    grid = (
        pd.MultiIndex.from_product([geoids, all_hours], names=["GEOID","dt"])
        .to_frame(index=False)
        .sort_values(["GEOID","dt"])
        .reset_index(drop=True)
    )
    return grid

def _prior_rolling(df: pd.DataFrame, window: str, suffix: str,
                   keys: Tuple[str,str,str]=("day_of_week","hour","season")) -> pd.DataFrame:
    df = df.sort_values(["GEOID","dt"]).copy()
    grp_cols = ["GEOID", *keys]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("dt").set_index("dt")
        cnt = g["Y_label"].rolling(window=window).sum().shift(1).fillna(0.0)
        g[f"prior_cnt_{suffix}"] = cnt.to_numpy().astype("float32")
        return g.reset_index()

    out = df.groupby(grp_cols, group_keys=False).apply(_roll)
    hours_in_window = float(pd.Timedelta(window) / pd.Timedelta("1H"))
    out[f"prior_p_{suffix}"] = (out[f"prior_cnt_{suffix}"] / hours_in_window).astype("float32")
    return out

def _downcast_numeric(df: pd.DataFrame, prefer_float32=True) -> pd.DataFrame:
    # basit downcast: int -> smallest, float -> float32
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    if prefer_float32:
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
    return df

# -------------------------
# Geçiş-1: minimal okuma (crime_count & Y_label)
# -------------------------
def pass1_build_cc(input_path: Path, tz: str | None) -> pd.DataFrame:
    # Sadece GEOID + zaman + (varsa crime_count) kolonlarını oku
    minimal_cols = ["GEOID","geoid","datetime","date","time","event_hour","received_time","crime_count"]
    usecols = None  # parquet için 'columns' paramı kullanacağız
    # Parquet ise kolon seçerek oku:
    if input_path.suffix.lower() == ".parquet":
        # pyarrow backend kolon filtresi çalışır
        raw = pd.read_parquet(input_path, columns=[c for c in minimal_cols if c])  # type: ignore
    else:
        raw = pd.read_csv(input_path, usecols=[c for c in minimal_cols if c in pd.read_csv(input_path, nrows=0).columns])
    raw = _norm_geoid(raw)
    raw["dt"] = _to_dt(raw)
    raw = raw.dropna(subset=["GEOID","dt"]).copy()
    raw["dt"] = raw["dt"].dt.tz_convert("UTC")

    if "crime_count" in raw.columns:
        cc = (raw[["GEOID","dt","crime_count"]]
              .groupby(["GEOID","dt"], as_index=False)["crime_count"].sum())
        cc["crime_count"] = cc["crime_count"].astype("int32")
    else:
        cc = (raw.groupby(["GEOID","dt"], as_index=False)
                  .size().rename(columns={"size":"crime_count"}))
        cc["crime_count"] = cc["crime_count"].astype("int16")

    cc["Y_label"] = (cc["crime_count"] > 0).astype("int8")
    cc = _add_calendar(cc, tz=tz)
    cc = _downcast_numeric(cc)
    return cc

# -------------------------
# Geçiş-2: yan özellikleri saatliğe indir ve ekle
# -------------------------
def pass2_merge_side_features(input_path: Path,
                              base_df: pd.DataFrame,
                              side_cols: List[str],
                              batch: int = 12) -> pd.DataFrame:
    # Numeric ve mevcut olan kolonları partiler halinde işle
    # Not: event-level ise saatliğe 'mean' ile indiriyoruz.
    if input_path.suffix.lower() == ".parquet":
        # Parquet'te kolon seçerek partileme: listeleri dilimleyip tekrar tekrar oku
        cols_present = pd.read_parquet(input_path, columns=None).columns.tolist()  # schema için hızlı okuma
    else:
        cols_present = pd.read_csv(input_path, nrows=0).columns.tolist()

    side_cols = [c for c in side_cols if c in cols_present]
    if not side_cols:
        return base_df

    keep_id = []
    for c in ["GEOID","geoid","datetime","date","time","event_hour","received_time"]:
        if c in cols_present: keep_id.append(c)

    # partili okuma
    for i in range(0, len(side_cols), batch):
        part = side_cols[i:i+batch]
        cols = list(set(keep_id + part))
        if input_path.suffix.lower() == ".parquet":
            chunk = pd.read_parquet(input_path, columns=cols)
        else:
            chunk = pd.read_csv(input_path, usecols=[c for c in cols if c in cols_present])

        chunk = _norm_geoid(chunk)
        chunk["dt"] = _to_dt(chunk)
        chunk = chunk.dropna(subset=["GEOID","dt"]).copy()
        chunk["dt"] = chunk["dt"].dt.tz_convert("UTC")

        # sadece numeric yan kolonları al
        numerics = [c for c in part if c in chunk.columns]
        if not numerics:
            continue

        side_hourly = (chunk[["GEOID","dt", *numerics]]
                       .groupby(["GEOID","dt"], as_index=False)
                       .mean(numeric_only=True))
        side_hourly = _downcast_numeric(side_hourly)
        base_df = base_df.merge(side_hourly, on=["GEOID","dt"], how="left")

    return base_df

# -------------------------
# Ana akış
# -------------------------
def run(input_path: Path,
        out_parquet: Path,
        tz: str | None,
        no_full_grid: bool,
        side_cols: List[str]):

    # Geçiş-1: minimal okuma → cc & Y
    df = pass1_build_cc(input_path, tz=tz)

    # (Opsiyonel) full grid
    if not no_full_grid:
        grid = _build_full_grid(df[["GEOID","dt","crime_count"]])
        df = grid.merge(df, on=["GEOID","dt"], how="left")
        df["crime_count"] = df["crime_count"].fillna(0).astype("int16")
        df["Y_label"] = df["Y_label"].fillna(0).astype("int8")
        df = _add_calendar(df, tz=tz)  # grid sonrası tekrar sağlamlaştır

    # Geçiş-2: yan özellikler
    if side_cols:
        df = pass2_merge_side_features(input_path, df, side_cols, batch=12)

    # Leakage-safe priors
    df = _prior_rolling(df, window="90D", suffix="3m")
    df = _prior_rolling(df, window="365D", suffix="12m")

    # Sırala, downcast et ve yaz
    df = df.sort_values(["GEOID","dt"]).reset_index(drop=True)
    df = _downcast_numeric(df)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[OK] Yazıldı: {out_parquet}  (satır: {len(df):,}, GEOID: {df['GEOID'].nunique():,})")


def parse_args():
    p = argparse.ArgumentParser(description="Y_label + leakage-safe prior (büyük veri için optimize)")
    p.add_argument("--input", type=Path, required=True,
                   help="Girdi dosyası (.parquet veya .csv) — ör: fr_crime_10.parquet")
    p.add_argument("--out", type=Path, default=Path("sf_crime_grid_full_labeled.parquet"),
                   help="Çıktı Parquet yolu")
    p.add_argument("--tz", type=str, default=None,
                   help="Yerel saat dilimi (örn: America/Los_Angeles). Yoksa UTC kabul edilir.")
    p.add_argument("--no-full-grid", action="store_true",
                   help="Tam GEOID×saat gridini OLUŞTURMA (önerilen).")
    default_side = [
        "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
        "911_geo_hr_last3d","911_geo_hr_last7d","311_request_count",
        "hr_cnt","daily_cnt",
        "wx_tavg","wx_tmin","wx_tmax","wx_prcp","wx_temp_range","wx_is_rainy","wx_is_hot_day",
        "poi_total_count","poi_risk_score",
        "poi_count_300m","poi_risk_300m","poi_count_600m","poi_risk_600m","poi_count_900m","poi_risk_900m",
        "distance_to_police","distance_to_government_building",
        "bus_stop_count","train_stop_count","population",
    ]
    p.add_argument("--side-cols", nargs="*", default=default_side,
                   help="Saatliğe indirgenecek yan özellik kolonları (numeric).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        out_parquet=args.out,
        tz=args.tz,
        no_full_grid=args.no_full_grid,
        side_cols=args.side_cols,
    )
