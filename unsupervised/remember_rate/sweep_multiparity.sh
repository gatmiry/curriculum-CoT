#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/mnt/task_runtime/myenv/bin/python"
PLOTS_DIR="${ROOT_DIR}/plots/multiparity"
PLOT_DATA_DIR="${ROOT_DIR}/plot_data/multiparity"

mkdir -p "$PLOTS_DIR" "$PLOT_DATA_DIR"

"$PYTHON_BIN" "${ROOT_DIR}/train_multiparity.py" \
  --plots_dir "$PLOTS_DIR" \
  --plot_data_dir "$PLOT_DATA_DIR"




