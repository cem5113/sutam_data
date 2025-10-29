#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_labels_and_priors_xl.py  (REVİZE)
- Girdi: fr_crime_09.parquet veya fr_crime_10.parquet (tek dosya, CSV de olabilir)
- Çıktı: sf_crime_grid_full_labeled.parquet (+ opsiyonel paketleme)

Ne yapar?
  1) GEOID×saat bazında crime_count ve Y_label üretir (event-level ise sayar; crime_count varsa toplar)
  2) Leakage-safe rolling priors (3m/12m): (GEOID, day_of_week, hour, season) kırılımında geçmişe bakar
  3) İstenirse hazır risky_hours.parquet ve metrics_stacking_ohe.parquet dosyalarını doğrular, out_dir'e kopyalar
  4) İstenirse hepsini tek ZIP içinde paketler (artifact)

Kullanım:
  python make_labels_and_priors_xl.py \
    --input fr_crime_10.parquet \
    --out outputs/sf_crime_grid_full_labeled.parquet \
    --tz America/Los_Angeles \
    --no-full-grid \
    --out-dir outputs \
    --package-zip outputs/fr-crime-outputs-parquet.zip
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Yol / FS yardımcıları
# =========================
HERE = Path(__file__).resolve().parent

def resolve_path(p: Path) -> Path:
    """Dosya yolunu sağlamlaştır: mutlak değilse önce CWD, sonra script klasörü."""
    p = Path(p)
    if p.is_absolute():
        return p
    # Önce çalışma dizini
    cand1 = Path.cwd() / p
    if cand1.exists():
        return cand1
    # Sonra script dizini
    cand2 = HERE / p
    if cand2.exists():
        return cand2
    # Bulunamadı → anlatımlı hata
    msg = [
        f"[HATA] Girdi bulunamadı: {p}",
        f"  Çalışma dizini: {Path.cwd()}",
        f"  Script dizini : {HERE}",
        "  Çalışma dizinindeki dosyalar (ilk 50):"
    ]
    try:
        listing = sorted([q.name for q in Path.cwd().iterdir()])[:50]
        msg.append("   - " + "\n   - ".join(listing))
    except Exception:
        pass
    raise FileNotFoundError("\n".join(msg))


# =========================
# Yardımcı fonksiyonlar
# =========================
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

def _coerce_tz_aware(s: pd.Series) -> pd.Series:
    """Naive datetime yakalanırsa UTC olarak işaretle; aware ise olduğu gibi bırak."""
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s

def _to_dt(df: pd.DataFrame) -> pd.Series:
    # saatlik zemin: datetime -> floor('H')
    s = None
    if "datetime" in df.columns:
        s = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    elif {"date","time"} <= set(df.columns):
        s = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=False)
    elif "event_hour" in df.columns:
        s = pd.to_datetime(df["event_hour"], errors="coerce", utc=False)
    elif "received_time" in df.columns:
        s = pd.to_datetime(df["received_time"], errors="coerce", utc=False)
    if s is None:
        raise ValueError("datetime veya (date+time) veya event_hour/received_time kolonu gerekli.")
    if s.isna().all():
        raise ValueError("Zaman kolonları çözümlenemedi (hepsi NaT).")
    s = _coerce_tz_aware(s).dt.tz_convert("UTC")
    return s.dt.floor("H")

def _add_calendar(df: pd.DataFrame, tz: Optional[str]) -> pd.DataFrame:
    base = "dt"
    if tz:
        df["dt_local"] = df["dt"].dt.tz_convert(tz)
        base = "dt_local"
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
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    if prefer_float32:
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
    return df

def _safe_read_parquet_columns(p: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    # Tek noktadan okuma; pyarrow varsayılanı yeterli
    return pd.read_parquet(p, columns=columns)


# =========================
# Geçiş-1: minimal okuma (crime_count & Y_label)
# =========================
def pass1_build_cc(input_path: Path, tz: Optional[str]) -> pd.DataFrame:
    minimal_cols = ["GEOID","geoid","datetime","date","time","event_hour","received_time","crime_count"]
    if input_path.suffix.lower() == ".parquet":
        raw = _safe_read_parquet_columns(input_path, columns=[c for c in minimal_cols if c])
    else:
        header = pd.read_csv(input_path, nrows=0).columns.tolist()
        usecols = [c for c in minimal_cols if c in header]
        raw = pd.read_csv(input_path, usecols=usecols)

    raw = _norm_geoid(raw)
    raw["dt"] = _to_dt(raw)
    raw = raw.dropna(subset=["GEOID","dt"]).copy()

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


# =========================
# Geçiş-2: yan özellikleri saatliğe indir ve ekle
# =========================
def pass2_merge_side_features(input_path: Path,
                              base_df: pd.DataFrame,
                              side_cols: List[str],
                              batch: int = 12) -> pd.DataFrame:
    # Şema için hızlı okuma (parquet'te tüm kolonları alma maliyeti kabul edilebilir)
    if input_path.suffix.lower() == ".parquet":
        cols_present = _safe_read_parquet_columns(input_path, columns=None).columns.tolist()
    else:
        cols_present = pd.read_csv(input_path, nrows=0).columns.tolist()

    side_cols = [c for c in side_cols if c in cols_present]
    if not side_cols:
        return base_df

    keep_id = [c for c in ["GEOID","geoid","datetime","date","time","event_hour","received_time"] if c in cols_present]

    for i in range(0, len(side_cols), batch):
        part = side_cols[i:i+batch]
        cols = list(set(keep_id + part))
        if input_path.suffix.lower() == ".parquet":
            chunk = _safe_read_parquet_columns(input_path, columns=cols)
        else:
            chunk = pd.read_csv(input_path, usecols=[c for c in cols if c in cols_present])

        chunk = _norm_geoid(chunk)
        chunk["dt"] = _to_dt(chunk)
        chunk = chunk.dropna(subset=["GEOID","dt"]).copy()

        numerics = [c for c in part if c in chunk.columns]
        if not numerics:
            continue

        side_hourly = (chunk[["GEOID","dt", *numerics]]
                       .groupby(["GEOID","dt"], as_index=False)
                       .mean(numeric_only=True))
        side_hourly = _downcast_numeric(side_hourly)
        base_df = base_df.merge(side_hourly, on=["GEOID","dt"], how="left")

    return base_df


# =========================
# Risky & Metrics doğrulama / kopyalama / paketleme
# =========================
def _validate_and_copy_optional(p: Optional[Path], out_dir: Path, tag: str) -> Optional[Path]:
    if p is None:
        return None
    p = resolve_path(p)
    if not p.exists():
        print(f"[WARN] {tag} bulunamadı: {p}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / p.name
    if p.resolve() != dst.resolve():
        shutil.copy2(p, dst)
    print(f"[OK] {tag} kopyalandı → {dst}")
    return dst

def _sanity_check_risky_hours(p: Path):
    try:
        df = pd.read_parquet(p)
    except Exception:
        print(f"[WARN] risky_hours okunamadı: {p}")
        return
    lower = {c.lower(): c for c in df.columns}
    must_have = {"geoid","hour","risk_score"}
    if not must_have.issubset(set(lower.keys())):
        print(f"[WARN] risky_hours şeması beklenenden farklı (geoid/hour/risk_score aranır): {list(df.columns)[:10]}...")
    else:
        print(f"[OK] risky_hours şeması uygun görünüyor ({p.name}).")

def _sanity_check_metrics(p: Path):
    try:
        df = pd.read_parquet(p)
    except Exception:
        print(f"[WARN] metrics dosyası okunamadı: {p}")
        return
    maybe_cols = [c for c in ["model","metric","value","fold","timestamp"] if c in df.columns]
    print(f"[OK] metrics satır: {len(df):,}  kolonlar: {maybe_cols}")

def _package_zip(zip_path: Path, files: List[Path]):
    zip_path = resolve_path(zip_path) if not zip_path.is_absolute() else zip_path
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f and Path(f).exists():
                z.write(f, arcname=Path(f).name)
    print(f"[OK] Paket oluşturuldu: {zip_path} (içerik: {[Path(f).name for f in files if f and Path(f).exists()]})")


# =========================
# Ana akış
# =========================
def run(input_path: Path,
        out_parquet: Path,
        tz: Optional[str],
        no_full_grid: bool,
        side_cols: List[str],
        risky_hours_path: Optional[Path],
        metrics_path: Optional[Path],
        out_dir: Optional[Path],
        package_zip: Optional[Path]):

    input_path = resolve_path(input_path)
    out_parquet = resolve_path(out_parquet) if not out_parquet.is_absolute() else out_parquet
    if out_dir:
        out_dir = resolve_path(out_dir) if not out_dir.is_absolute() else out_dir
    if package_zip:
        package_zip = resolve_path(package_zip) if not package_zip.is_absolute() else package_zip

    print(f"[INFO] Girdi: {input_path}")
    print(f"[INFO] Çıktı: {out_parquet}")

    # 1) Geçiş-1: minimal okuma → cc & Y
    df = pass1_build_cc(input_path, tz=tz)

    # 2) (Opsiyonel) full grid
    if not no_full_grid:
        grid = _build_full_grid(df[["GEOID","dt","crime_count"]])
        df = grid.merge(df, on=["GEOID","dt"], how="left")
        df["crime_count"] = df["crime_count"].fillna(0).astype("int16")
        df["Y_label"] = df["Y_label"].fillna(0).astype("int8")
        df = _add_calendar(df, tz=tz)

    # 3) Geçiş-2: yan özellikler
    if side_cols:
        df = pass2_merge_side_features(input_path, df, side_cols, batch=12)

    # 4) Leakage-safe priors
    df = _prior_rolling(df, window="90D",  suffix="3m")
    df = _prior_rolling(df, window="365D", suffix="12m")

    # 5) Sırala, downcast et ve yaz
    df = df.sort_values(["GEOID","dt"]).reset_index(drop=True)
    df = _downcast_numeric(df)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[OK] Yazıldı: {out_parquet}  (satır: {len(df):,}, GEOID: {df['GEOID'].nunique():,})")

    # 6) Risky & Metrics (opsiyonel): doğrula + out_dir'e kopyala
    packaged_files: List[Path] = [out_parquet]
    if out_dir:
        if risky_hours_path:
            dst_risky = _validate_and_copy_optional(risky_hours_path, out_dir, "risky_hours")
            if dst_risky:
                _sanity_check_risky_hours(dst_risky)
                packaged_files.append(dst_risky)
        if metrics_path:
            dst_metrics = _validate_and_copy_optional(metrics_path, out_dir, "metrics")
            if dst_metrics:
                _sanity_check_metrics(dst_metrics)
                packaged_files.append(dst_metrics)

    # 7) Paketleme (opsiyonel)
    if package_zip:
        _package_zip(package_zip, packaged_files)


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Y_label + leakage-safe priors + (opsiyonel) risky/metrics paketleme")
    p.add_argument("--input", type=Path, required=True, help="fr_crime_09.parquet / fr_crime_10.parquet (veya CSV)")
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

    # Opsiyonel: hazır dosyalar
    p.add_argument("--risky-hours", type=Path, default=None,
                   help="Hazır risky_hours.parquet yolu (opsiyonel)")
    p.add_argument("--metrics", type=Path, default=None,
                   help="Hazır metrics_stacking_ohe.parquet yolu (opsiyonel)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Opsiyonel kopyalama klasörü (risky & metrics buraya kopyalanır)")
    p.add_argument("--package-zip", type=Path, default=None,
                   help="Hepsini tek ZIP olarak paketle (örn: outputs/fr-crime-outputs-parquet.zip)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        out_parquet=args.out,
        tz=args.tz,
        no_full_grid=args.no_full_grid,
        side_cols=args.side_cols,
        risky_hours_path=args.risky_hours,
        metrics_path=args.metrics,
        out_dir=args.out_dir,
        package_zip=args.package_zip,
    )
