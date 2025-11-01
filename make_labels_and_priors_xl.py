#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_labels_and_priors_xl.py  (REVIZE — DAILY / HOURLY MODLU)
- Girdi : fr_crime_09.parquet veya CSV (event-level ya da önceden agregat)
- Çıktı : sf_crime_grid_full_labeled.parquet (default: GÜNLÜK, GEOID×t0)
- Modlar:
    * daily (varsayılan): 24 saatlik pencereler, GEOID×gün full-grid
    * hourly: Saatlik full-grid (eski davranış)

Ne yapar?
  1) (daily) GEOID×gün crime_count SUM ve Y_label=(sum>0); full-grid (eksik günler 0)
     (hourly) GEOID×saat count; full-grid
  2) Yan değişkenleri aynı granülaritede birleştirir:
       - *_count, crime_count → SUM
       - diğer numerikler → MEAN
  3) Leakage-safe priors (90g/365g): geçmişe bakan rolling + shift(1)
       - daily: anahtarlar (GEOID, day_of_week)
       - hourly: anahtarlar (GEOID, day_of_week, hour, season)
  4) (Ops.) risky_hours & metrics dosyalarını doğrular ve kopyalar
  5) (Ops.) Tüm çıktıların ZIP paketi
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------------------------
# Yol yardımcıları
# ---------------------------
def resolve_out_path(p: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p)

def resolve_path(p: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    cand1 = Path.cwd() / p
    if cand1.exists():
        return cand1
    cand2 = HERE / p
    if cand2.exists():
        return cand2
    raise FileNotFoundError(f"Girdi bulunamadı: {p} | cwd={Path.cwd()} | here={HERE}")

# ---------------------------
# Baz yardımcılar
# ---------------------------
def _season_from_month(m: int) -> str:
    if m in (12, 1, 2): return "winter"
    if m in (3, 4, 5):  return "spring"
    if m in (6, 7, 8):  return "summer"
    return "fall"

def _norm_geoid(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    if "GEOID" in cols:
        df["GEOID"] = df["GEOID"].astype(str)
    elif "geoid" in cols:
        df["GEOID"] = df["geoid"].astype(str)
        df = df.drop(columns=["geoid"])
    else:
        raise ValueError("GEOID/geoid kolonu bulunamadı.")
    return df.loc[:, ~df.columns.duplicated()].copy()

def _coerce_tz_aware(s: pd.Series) -> pd.Series:
    if s.dt.tz is None:
        return s.dt.tz_localize("UTC")
    return s

def _to_dt(df: pd.DataFrame) -> pd.Series:
    # Olası kolonlar: datetime | (date+time) | event_hour | received_time
    s = None
    if "datetime" in df.columns:
        s = pd.to_datetime(df["datetime"], errors="coerce", utc=False)
    elif {"date","time"} <= set(df.columns):
        s = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=False)
    elif "event_hour" in df.columns:
        s = pd.to_datetime(df["event_hour"], errors="coerce", utc=False)
    elif "received_time" in df.columns:
        s = pd.to_datetime(df["received_time"], errors="coerce", utc=False)
    if s is None or s.isna().all():
        raise ValueError("Zaman için datetime/(date+time)/event_hour/received_time gerekli.")
    s = _coerce_tz_aware(s).dt.tz_convert("UTC")
    return s.dt.floor("1h")

def _add_calendar(df: pd.DataFrame, base_col: str, tz: Optional[str]) -> pd.DataFrame:
    base = base_col
    if tz:
        df[f"{base_col}_local"] = df[base_col].dt.tz_convert(tz)
        base = f"{base_col}_local"
    df["year"]        = df[base].dt.year.astype("int16")
    df["month"]       = df[base].dt.month.astype("int8")
    df["day_of_week"] = df[base].dt.dayofweek.astype("int8")
    if base_col == "dt":  # sadece saatlikte anlamlı
        df["hour"]    = df[base].dt.hour.astype("int8")
        df["season"]  = df["month"].map(_season_from_month).astype("category")
    return df

def _downcast_numeric(df: pd.DataFrame, prefer_float32=True) -> pd.DataFrame:
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    if prefer_float32:
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")
    return df

def _safe_read_parquet_columns(p: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if columns is None:
        df = pd.read_parquet(p)
        return df.loc[:, ~df.columns.duplicated()].copy()
    try:
        df = pd.read_parquet(p, columns=columns)
        return df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(p)
            actual = list(schema.names)
            lower  = {c.lower(): c for c in actual}
            resolved = []
            for c in columns:
                if c in actual: resolved.append(c)
                elif c.lower() in lower: resolved.append(lower[c.lower()])
            if not resolved:
                df = pd.read_parquet(p)
                return df.loc[:, ~df.columns.duplicated()].copy()
            df = pd.read_parquet(p, columns=list(dict.fromkeys(resolved)))
            return df.loc[:, ~df.columns.duplicated()].copy()
        except Exception:
            df = pd.read_parquet(p)
            return df.loc[:, ~df.columns.duplicated()].copy()

# ---------------------------
# Grid kurucular
# ---------------------------
def _build_full_grid_hourly(df: pd.DataFrame) -> pd.DataFrame:
    dt_min = df["dt"].min().floor("1h")
    dt_max = df["dt"].max().ceil("1h")
    hours = pd.date_range(dt_min, dt_max, freq="1h", tz="UTC")
    geoids = df["GEOID"].dropna().astype(str).unique()
    grid = pd.MultiIndex.from_product([geoids, hours], names=["GEOID","dt"]).to_frame(index=False)
    return grid.sort_values(["GEOID","dt"]).reset_index(drop=True)

def _build_full_grid_daily(df: pd.DataFrame) -> pd.DataFrame:
    d0_min = df["d0"].min().floor("D")
    d0_max = df["d0"].max().ceil("D")
    days = pd.date_range(d0_min, d0_max, freq="D", tz="UTC")
    geoids = df["GEOID"].dropna().astype(str).unique()
    grid = pd.MultiIndex.from_product([geoids, days], names=["GEOID","d0"]).to_frame(index=False)
    return grid.sort_values(["GEOID","d0"]).reset_index(drop=True)

# ---------------------------
# Prior hesaplayıcı
# ---------------------------
def _prior_rolling(df: pd.DataFrame, time_col: str, window: str, suffix: str, keys: List[str]) -> pd.DataFrame:
    """
    Sızıntısız prior:
      - rolling(..., closed='left')  → mevcut t0/dt pencereye girmez
      - prior_cnt_* = penceredeki Y_label toplamı
      - prior_p_*   = prior_cnt_* / penceredeki gözlem sayısı (daily→gün, hourly→saat)
    """
    # Zorunlu kolonlar
    if time_col not in df.columns:
        raise RuntimeError(f"prior: '{time_col}' kolonu yok.")
    if "GEOID" not in df.columns:
        raise RuntimeError("prior: 'GEOID' kolonu yok.")
    if "Y_label" not in df.columns:
        # Koruyucu: yoksa crime_count'tan türet
        if "crime_count" in df.columns:
            df = df.copy()
            df["Y_label"] = (pd.to_numeric(df["crime_count"], errors="coerce").fillna(0) > 0).astype("int8")
        else:
            raise RuntimeError("prior: 'Y_label' (veya crime_count) yok.")

    # Hazırlık
    df = df.copy()
    df["GEOID"] = df["GEOID"].astype(str)
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(["GEOID", time_col])

    keys = [k for k in (keys or []) if k in df.columns]
    grp_cols = ["GEOID", *keys]

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(time_col).set_index(time_col)
        y = pd.to_numeric(g["Y_label"], errors="coerce").fillna(0.0)

        # BUGÜN HARİÇ → closed='left'
        cnt = y.rolling(window=window, closed="left").sum()
        obs = y.rolling(window=window, closed="left").count()
        p   = cnt / obs.replace(0, np.nan)

        out = g.copy()
        out[f"prior_cnt_{suffix}"] = cnt.astype("float32").to_numpy()
        out[f"prior_p_{suffix}"]   = p.astype("float32").to_numpy()
        return out.reset_index()

    # pandas >=2.2: include_groups paramı var; yoksa fallback
    try:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll, include_groups=False)
    except TypeError:
        out = df.groupby(grp_cols, group_keys=False).apply(_roll)

    # Başlangıç dönemindeki NaN oranları 0’a çek
    out[f"prior_cnt_{suffix}"] = out[f"prior_cnt_{suffix}"].fillna(0.0).astype("float32")
    out[f"prior_p_{suffix}"]   = out[f"prior_p_{suffix}"].fillna(0.0).astype("float32")

    return out
   
# ---------------------------
# Pass1: temel cc (hourly) + (daily)ye indirgeme
# ---------------------------
def pass1_build_cc(input_path: Path, tz: Optional[str], granularity: str) -> pd.DataFrame:
    minimal = ["GEOID","geoid","datetime","date","time","event_hour","received_time","crime_count"]
    if input_path.suffix.lower() == ".parquet":
        raw = _safe_read_parquet_columns(input_path, columns=minimal)
    else:
        header = pd.read_csv(input_path, nrows=0).columns.tolist()
        usecols = [c for c in minimal if c in header]
        raw = pd.read_csv(input_path, usecols=usecols)

    raw = _norm_geoid(raw)
    raw["dt"] = _to_dt(raw)  # saatlik time stamp

    # Saatlik bazda crime_count
    if "crime_count" in raw.columns:
        hourly = (raw[["GEOID","dt","crime_count"]]
                  .groupby(["GEOID","dt"], as_index=False)["crime_count"].sum())
        hourly["crime_count"] = hourly["crime_count"].astype("int32")
    else:
        hourly = (raw.groupby(["GEOID","dt"], as_index=False)
                  .size().rename(columns={"size":"crime_count"}))
        hourly["crime_count"] = hourly["crime_count"].astype("int16")

    if granularity == "hourly":
        hourly["Y_label"] = (hourly["crime_count"] > 0).astype("int8")
        hourly = _add_calendar(hourly, base_col="dt", tz=tz)
        return _downcast_numeric(hourly)

    # DAILY: dt → d0 (gün başı UTC), günlük SUM ve Y_label
    hourly["d0"] = hourly["dt"].dt.floor("D")
    daily = (hourly.groupby(["GEOID","d0"], as_index=False)
             .agg(crime_count=("crime_count","sum")))
    daily["Y_label"] = (daily["crime_count"] > 0).astype("int8")

    # Takvim (günlükte hour/season yok)
    # t0 = d0 (çıktıda t0 olarak kullanacağız)
    daily = daily.rename(columns={"d0":"t0"})
    daily = _add_calendar(daily, base_col="t0", tz=tz)
    return _downcast_numeric(daily)

# ---------------------------
# Pass2: yan değişkenleri birleştir (granülerliğe göre)
# ---------------------------
def _agg_rule(col: str) -> str:
    # isim sezgisi: *_count / crime_count → SUM, diğerleri → MEAN
    lc = col.lower()
    if lc.endswith("_count") or lc == "crime_count":
        return "sum"
    return "mean"

def pass2_merge_side_features(input_path: Path,
                              base_df: pd.DataFrame,
                              side_cols: List[str],
                              granularity: str,
                              batch: int = 12) -> pd.DataFrame:
    # Hangi kolonlar mevcut?
    if input_path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        cols_present = set(pq.read_schema(input_path).names)
    else:
        cols_present = set(pd.read_csv(input_path, nrows=0).columns.tolist())

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

        if granularity == "hourly":
            side = (chunk[["GEOID","dt", *numerics]]
                    .groupby(["GEOID","dt"], as_index=False)
                    .mean(numeric_only=True))
            key_cols = ["GEOID","dt"]
        else:
            # daily: dt → t0 (gün)
            chunk["t0"] = chunk["dt"].dt.floor("D")
            agg_map = {c: _agg_rule(c) for c in numerics}
            side = (chunk[["GEOID","t0", *numerics]]
                    .groupby(["GEOID","t0"], as_index=False)
                    .agg(agg_map))
            key_cols = ["GEOID","t0"]

        side = _downcast_numeric(side)
        base_df = base_df.merge(side, on=key_cols, how="left")

    return base_df

# ---------------------------
# Opsiyoneller: kopya & paket
# ---------------------------
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
        lower = {c.lower() for c in df.columns}
        if {"geoid","hour","risk_score"} <= lower:
            print(f"[OK] risky_hours şeması uygun ({p.name}).")
        else:
            print(f"[WARN] risky_hours beklenen şemada değil: {list(df.columns)[:10]}...")
    except Exception:
        print(f"[WARN] risky_hours okunamadı: {p}")

def _sanity_check_metrics(p: Path):
    try:
        df = pd.read_parquet(p)
        maybe = [c for c in ["model","metric","value","fold","timestamp"] if c in df.columns]
        print(f"[OK] metrics satır={len(df):,} kolonlar={maybe}")
    except Exception:
        print(f"[WARN] metrics dosyası okunamadı: {p}")

def _package_zip(zip_path: Path, files: List[Path]):
    zip_path = resolve_out_path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f and Path(f).exists():
                z.write(f, arcname=Path(f).name)
    print(f"[OK] Paket: {zip_path}")

# ---------------------------
# ÇALIŞTIR
# ---------------------------
def run(input_path: Path,
        out_parquet: Path,
        tz: Optional[str],
        side_cols: List[str],
        risky_hours_path: Optional[Path],
        metrics_path: Optional[Path],
        out_dir: Optional[Path],
        package_zip: Optional[Path],
        granularity: str):

    input_path = resolve_path(input_path)
    out_parquet = resolve_out_path(out_parquet)
    if out_dir:
        out_dir = resolve_out_path(out_dir)
    if package_zip:
        package_zip = resolve_out_path(package_zip)

    print(f"[INFO] Girdi: {input_path}")
    print(f"[INFO] Mod  : {granularity.upper()}  (daily → GEOID×t0, hourly → GEOID×dt)")
    print(f"[INFO] Çıktı: {out_parquet}")

    # 1) minimal okuma → cc & Y
    base = pass1_build_cc(input_path, tz=tz, granularity=granularity)

    # 2) FULL GRID kur
    if granularity == "hourly":
        grid = _build_full_grid_hourly(base[["GEOID","dt","crime_count"]])
        df = grid.merge(base, on=["GEOID","dt"], how="left")
        df["crime_count"] = df["crime_count"].fillna(0).astype("int16")
        df["Y_label"]     = df["Y_label"].fillna(0).astype("int8")
        df = _add_calendar(df, base_col="dt", tz=tz)
        time_col = "dt"
        prior_keys = ["day_of_week","hour","season"]
    else:
        # daily: base zaten GEOID×t0; full day grid kur
        tmp = base.rename(columns={"t0":"d0"})
        grid = _build_full_grid_daily(tmp[["GEOID","d0","crime_count"]])
        df = grid.merge(tmp, on=["GEOID","d0"], how="left").rename(columns={"d0":"t0"})
        df["crime_count"] = df["crime_count"].fillna(0).astype("int32")
        df["Y_label"]     = df["Y_label"].fillna(0).astype("int8")
        df = _add_calendar(df, base_col="t0", tz=tz)
        time_col = "t0"
        prior_keys = ["day_of_week"]  # günlükte hour/season yok

    # 3) Yan değişkenleri birleştir
    if side_cols:
        df = pass2_merge_side_features(input_path, df, side_cols, granularity=granularity, batch=12)

    # 4) Priors (sızıntısız)
   df = _prior_rolling(df, time_col=time_col, window="90D",  suffix="3m",  keys=prior_keys)
   df = _prior_rolling(df, time_col=time_col, window="365D", suffix="12m", keys=prior_keys)

    # 5) Yaz (çıktı şeması: GEOID × t0 (UTC) + numerikler)
    df = df.sort_values(["GEOID", time_col]).reset_index(drop=True)
    df = _downcast_numeric(df)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[OK] Yazıldı: {out_parquet}  (satır: {len(df):,}, GEOID: {df['GEOID'].nunique():,})")

    # 6) Y dağılımı
    y_counts = df["Y_label"].value_counts(dropna=False).reindex([1,0], fill_value=0)
    total = int(y_counts.sum()); y1 = int(y_counts.get(1,0)); y0 = int(y_counts.get(0,0))
    p1 = (100.0 * y1 / total) if total else 0.0
    p0 = (100.0 * y0 / total) if total else 0.0
    print(f"[STATS] Y=1: {y1:,} ({p1:.2f}%) | Y=0: {y0:,} ({p0:.2f}%) | Toplam: {total:,}")

    stats_df = pd.DataFrame({"Y_label":[1,0], "Count":[y1,y0], "Percent(%)":[round(p1,4), round(p0,4)]})
    stats_csv_path = out_parquet.with_name("y_label_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[OK] Y_label dağılımı → {stats_csv_path}")

    # 7) Ops: risky & metrics kopyala
    packaged_files: List[Path] = [out_parquet]
    if out_dir:
        if risky_hours_path:
            dst_r = _validate_and_copy_optional(risky_hours_path, out_dir, "risky_hours")
            if dst_r:
                _sanity_check_risky_hours(dst_r)
                packaged_files.append(dst_r)
        if metrics_path:
            dst_m = _validate_and_copy_optional(metrics_path, out_dir, "metrics")
            if dst_m:
                _sanity_check_metrics(dst_m)
                packaged_files.append(dst_m)

    # 8) Ops: paket
    if package_zip:
        _package_zip(package_zip, packaged_files)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Y_label + FULL GRID (daily/hourly) + leakage-safe priors")
    p.add_argument("--input", type=Path, required=True, help="fr_crime_09.parquet (veya CSV)")
    p.add_argument("--out",   type=Path, default=Path("sf_crime_grid_full_labeled.parquet"),
                   help="Çıktı Parquet yolu")
    p.add_argument("--tz",    type=str, default=None, help="Yerel TZ (örn: America/Los_Angeles)")
    p.add_argument("--granularity", type=str, choices=["daily","hourly"], default="daily",
                   help="Çıktı granülaritesi (varsayılan: daily)")
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
                   help="Yan özellik kolonları (numeric; daily=SUM/MEAN sezgisel).")
    p.add_argument("--risky-hours", type=Path, default=None, help="risky_hours.parquet (ops)")
    p.add_argument("--metrics",     type=Path, default=None, help="metrics_stacking_ohe.parquet (ops)")
    p.add_argument("--out-dir",     type=Path, default=None, help="Opsiyonel kopyalama klasörü")
    p.add_argument("--package-zip", type=Path, default=None, help="Hepsini ZIPle")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    run(
        input_path=a.input,
        out_parquet=a.out,
        tz=a.tz,
        side_cols=a.side_cols,
        risky_hours_path=a.risky_hours,
        metrics_path=a.metrics,
        out_dir=a.out_dir,
        package_zip=a.package_zip,
        granularity=a.granularity,
    )
