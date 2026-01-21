#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/mnt/task_runtime/myenv/bin/python"
PLOTS_DIR="${ROOT_DIR}/plots/multitarget"
PLOT_DATA_DIR="${ROOT_DIR}/plot_data/multitarget"

mkdir -p "$PLOTS_DIR" "$PLOT_DATA_DIR"

"$PYTHON_BIN" "${ROOT_DIR}/train_multitarget.py" \
  --plots_dir "$PLOTS_DIR" \
  --plot_data_dir "$PLOT_DATA_DIR"



