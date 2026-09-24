#!/usr/bin/env bash
# Run a marimo notebook top to bottom as a script, one notebook at a time on this machine.
#   tools/run_notebook.sh ../01_optimization_fundamentals.py
# Uses the notebook's inline (PEP 723) dependencies through `uv run --script`.
set -euo pipefail
nb="$(realpath "$1")"
lockdir="$HOME/.cache/contact-slides"; mkdir -p "$lockdir"
exec 9>"$lockdir/notebook.lock"
if ! flock -n 9; then echo "[waiting for the notebook lock]" >&2; flock 9; fi
cd "$(dirname "$nb")"
MPLBACKEND=Agg timeout 1200 nice -n 10 uv run --quiet --script "$nb" && echo "OK: $(basename "$nb") ran top to bottom"
