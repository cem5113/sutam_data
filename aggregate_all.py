name: Make Labels + Train (SUTAM)

on:
  workflow_dispatch:
    inputs:
      persist:
        description: "Çıktılar nasıl saklansın?"
        type: choice
        options: [artifact, commit, none]
        default: artifact
      granularity:
        description: "Hangi model(ler) eğitilsin?"
        type: choice
        options: [hourly, 8h, 1d, all]
        default: hourly
      balanced:
        description: "Saatlik modeli (yalnızca bayrak) dengele"
        type: boolean
        default: false
      windows:
        description: "Multi-window frekansları (virgülle): 3H,8H,1D,1W,1M"
        default: "3H,8H,1D,1W,1M"
      horizon:
        description: "Forecast ufku (örn: 72h, 30d, 12w)"
        default: "72h"
      topk:
        description: "Şehir geneli Top-K GEOID"
        default: "50"

permissions:
  contents: write
  actions: read

jobs:
  build_and_train:
    runs-on: ubuntu-latest
    env:
      TZ: "America/Los_Angeles"
      DATA_DIR: .
      OUT_FILE: sf_crime_grid_full_labeled.parquet
      OUT_ZIP:  fr-crime-outputs-parquet.zip

    steps:
      - name: Checkout repo (with LFS)
        uses: actions/checkout@v4
        with:
          lfs: true

      - name: Show workspace
        run: |
          echo "PWD=$(pwd)"
          ls -lah

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: requirements.txt

      - name: Install dependencies
        run: |
          python -m pip install -U pip
          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          else
            pip install numpy pandas pyarrow tqdm scipy \
                        scikit-learn xgboost lightgbm imbalanced-learn \
                        joblib matplotlib
          fi

      - name: Detect input parquet & optional files
        id: detect
        run: |
          set -euo pipefail
          prefer_10="$(/usr/bin/find "${DATA_DIR}" -maxdepth 2 -type f -name 'fr_crime_10.parquet' -print -quit)" || true
          prefer_09="$(/usr/bin/find "${DATA_DIR}" -maxdepth 2 -type f -name 'fr_crime_09.parquet' -print -quit)" || true
          if [ -n "${prefer_10}" ]; then INPUT="${prefer_10}"; elif [ -n "${prefer_09}" ]; then INPUT="${prefer_09}"; else echo "❌ fr_crime_10.parquet / fr_crime_09.parquet yok"; exit 1; fi
          echo "input=${INPUT}" >> "$GITHUB_OUTPUT"
          echo "✅ INPUT=${INPUT}"

          RH="$(/usr/bin/find "${DATA_DIR}" -maxdepth 2 -type f \( -name 'risk_hourly.parquet' -o -name 'risky_hours.parquet' \) -print -quit)" || true
          MET="$(/usr/bin/find "${DATA_DIR}" -maxdepth 2 -type f -name 'metrics_stacking_ohe.parquet' -print -quit)" || true
          [ -n "${RH}" ] && echo "risky_hours=${RH}" >> "$GITHUB_OUTPUT"
          [ -n "${MET}" ] && echo "metrics=${MET}"     >> "$GITHUB_OUTPUT"

      - name: Build labels + priors (+ package)
        run: |
          set -euo pipefail
          INPUT="${{ steps.detect.outputs.input }}"
          OUT_PQ="${DATA_DIR}/${OUT_FILE}"
          OUT_ZIP="${DATA_DIR}/${OUT_ZIP}"

          EXTRA_ARGS=()
          [ -n "${{ steps.detect.outputs.risky_hours }}" ] && EXTRA_ARGS+=( --risky-hours "${{ steps.detect.outputs.risky_hours }}" )
          [ -n "${{ steps.detect.outputs.metrics }}" ]     && EXTRA_ARGS+=( --metrics     "${{ steps.detect.outputs.metrics }}" )

          echo "▶️  make_labels_and_priors_xl.py"
          python -u make_labels_and_priors_xl.py \
            --input       "${INPUT}" \
            --out         "${OUT_PQ}" \
            --tz          America/Los_Angeles \
            --out-dir     "${DATA_DIR}" \
            --package-zip "${OUT_ZIP}" \
            "${EXTRA_ARGS[@]}"

          test -s "${OUT_PQ}" || { echo "❌ Output parquet yok: ${OUT_PQ}"; exit 1; }
          [ -f "${DATA_DIR}/y_label_stats.csv" ] && { echo "Y_label stats:"; cat "${DATA_DIR}/y_label_stats.csv" || true; }

      # ---- 8H / 1D agregasyon (opsiyonel – eski adım)
      - name: Aggregate to 8H and/or 1D
        if: ${{ github.event.inputs.granularity == '8h' || github.event.inputs.granularity == '1d' || github.event.inputs.granularity == 'all' }}
        run: |
          set -euo pipefail
          INPUT="${{ env.DATA_DIR }}/${{ env.OUT_FILE }}"
          if [ "${{ github.event.inputs.granularity }}" = "8h" ] || [ "${{ github.event.inputs.granularity }}" = "all" ]; then
            echo "▶️  aggregate_windows.py (8H)"
            python -u aggregate_windows.py --input "${INPUT}" --out "sf_crime_grid_8h.parquet" --freq 8H --tz America/Los_Angeles
          fi
          if [ "${{ github.event.inputs.granularity }}" = "1d" ] || [ "${{ github.event.inputs.granularity }}" = "all" ]; then
            echo "▶️  aggregate_windows.py (1D)"
            python -u aggregate_windows.py --input "${INPUT}" --out "sf_crime_grid_1d.parquet" --freq 1D --tz America/Los_Angeles
          fi

      # ---- Multi-window agregasyon (3H/8H/1D/1W/1M) – dinamik hizalı
      - name: Aggregate multi-windows (3H/8H/1D/1W/1M)
        run: |
          set -euo pipefail
          echo "▶️  aggregate_all.py (${ { github.event.inputs.windows } })"
          python -u aggregate_all.py \
            --input "${{ env.DATA_DIR }}/${{ env.OUT_FILE }}" \
            --freqs "${{ github.event.inputs.windows }}" \
            --tz America/Los_Angeles

      # ---- Saatlik eğitim (tek script)
      - name: Train hourly model
        if: ${{ github.event.inputs.granularity == 'hourly' || github.event.inputs.granularity == 'all' }}
        run: |
          set -euo pipefail
          echo "▶️  train_hourly_model.py (balanced=${{ github.event.inputs.balanced }})"
          python -u train_hourly_model.py

      # ---- 8H eğitim (aynı script, input parametre veriyoruz)
      - name: Train 8H model
        if: ${{ github.event.inputs.granularity == '8h' || github.event.inputs.granularity == 'all' }}
        run: |
          set -euo pipefail
          test -s sf_crime_grid_8h.parquet || { echo "8H dosyası yok, eğitim atlandı."; exit 0; }
          echo "▶️  train_hourly_model.py (8H)"
          python -u train_hourly_model.py --input sf_crime_grid_8h.parquet --freq 8H --tz America/Los_Angeles

      # ---- 1D eğitim
      - name: Train 1D model
        if: ${{ github.event.inputs.granularity == '1d' || github.event.inputs.granularity == 'all' }}
        run: |
          set -euo pipefail
          test -s sf_crime_grid_1d.parquet || { echo "1D dosyası yok, eğitim atlandı."; exit 0; }
          echo "▶️  train_hourly_model.py (1D)"
          python -u train_hourly_model.py --input sf_crime_grid_1d.parquet --freq 1D --tz America/Los_Angeles

      # ---- Multi-window eğitim (3H/8H/1D/1W/1M)
      - name: Train multi-windows (3H/8H/1D/1W/1M)
        run: |
          set -euo pipefail
          echo "▶️  train_multi_windows.py"
          python -u train_multi_windows.py \
            --dir "." \
            --prefix "sf_crime_grid_" \
            --freqs "${{ github.event.inputs.windows }}" \
            --undersample "0.0"

      # ---- Forecast (multi-window) — çalıştırma zamanına göre hizalı
      - name: Run risk_forecast (multi-window predictions)
        run: |
          set -euo pipefail
          echo "▶️  risk_forecast.py (freq=auto, horizon=${{ github.event.inputs.horizon }}, topk=${{ github.event.inputs.topk }})"
          python -u risk_forecast.py \
            --freq auto \
            --horizon "${{ github.event.inputs.horizon }}" \
            --topk "${{ github.event.inputs.topk }}"
          echo "✅ Forecasts oluşturuldu:"
          ls -lh forecasts || true

      - name: Upload artifacts (if selected)
        if: ${{ github.event.inputs.persist == 'artifact' }}
        uses: actions/upload-artifact@v4
        with:
          name: sutam-model-outputs
          path: |
            ${{ env.DATA_DIR }}/${{ env.OUT_FILE }}
            ${{ env.DATA_DIR }}/${{ env.OUT_ZIP }}
            ${{ env.DATA_DIR }}/y_label_stats.csv
            sf_crime_grid_8h.parquet
            sf_crime_grid_1d.parquet
            models/*.joblib
            reports/*.json
            reports/*.csv
            forecasts/*.csv
            forecasts/*.json
          if-no-files-found: warn
          retention-days: 14

      - name: Commit outputs (if selected)
        if: ${{ github.event.inputs.persist == 'commit' }}
        run: |
          set -e
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add "${DATA_DIR}/${OUT_FILE}" || true
          git add "${DATA_DIR}/${OUT_ZIP}"  || true
          [ -f "${DATA_DIR}/y_label_stats.csv" ] && git add "${DATA_DIR}/y_label_stats.csv" || true
          [ -f sf_crime_grid_8h.parquet ] && git add sf_crime_grid_8h.parquet || true
          [ -f sf_crime_grid_1d.parquet ] && git add sf_crime_grid_1d.parquet || true
          git add models/*.joblib 2>/dev/null || true
          git add reports/* 2>/dev/null || true
          git add forecasts/* 2>/dev/null || true
          if git diff --cached --quiet; then
            echo "No changes to commit."
            exit 0
          fi
          git commit -m "chore: labels+priors, multi-window aggregation/training & forecasts"
          git push origin "${GITHUB_REF_NAME:-$(git rev-parse --abbrev-ref HEAD)}"

      - name: Job summary
        run: |
          set -e
          {
            echo "## Make Labels + Train — Özet"
            echo "- Input: \`${{ steps.detect.outputs.input }}\`"
            echo "- Granularity: \`${{ github.event.inputs.granularity }}\`"
            echo "- Balanced(hourly flag): \`${{ github.event.inputs.balanced }}\`"
            echo "- Windows: \`${{ github.event.inputs.windows }}\`"
            echo "- Forecast: horizon=\`${{ github.event.inputs.horizon }}\`, topk=\`${{ github.event.inputs.topk }}\`"
            [ -n "${{ steps.detect.outputs.risky_hours }}" ] && echo "- risky_hours: \`${{ steps.detect.outputs.risky_hours }}\`"
            [ -n "${{ steps.detect.outputs.metrics }}" ] && echo "- metrics: \`${{ steps.detect.outputs.metrics }}\`"
            echo "- Output Parquet: \`${{ env.DATA_DIR }}/${{ env.OUT_FILE }}\`"
            echo "- Output ZIP: \`${{ env.DATA_DIR }}/${{ env.OUT_ZIP }}\`"
            [ -f "${{ env.DATA_DIR }}/y_label_stats.csv" ] && { echo ""; echo "### Y_label dağılımı"; tail -n +1 "${{ env.DATA_DIR }}/y_label_stats.csv"; }
            echo ""; ls -lh models 2>/dev/null || true
            echo ""; ls -lh reports 2>/dev/null || true
            echo ""; ls -lh forecasts 2>/dev/null || true
          } >> "$GITHUB_STEP_SUMMARY"
