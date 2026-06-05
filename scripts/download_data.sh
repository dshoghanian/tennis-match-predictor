#!/usr/bin/env bash
# Download Jeff Sackmann ATP/WTA singles match files (1991-2026) into data/raw/<tour>/.
# Sources: https://github.com/JeffSackmann/tennis_atp , https://github.com/JeffSackmann/tennis_wta
# (CC BY-NC-SA 4.0). Usage: download_data.sh [atp|wta]   (no arg = both tours)
set -euo pipefail

ROOT="$(dirname "$0")/.."
TOURS=("atp" "wta")
if [ "${1:-}" != "" ]; then TOURS=("$1"); fi

for tour in "${TOURS[@]}"; do
  raw_dir="$ROOT/data/raw/$tour"
  base_url="https://raw.githubusercontent.com/JeffSackmann/tennis_${tour}/master"
  mkdir -p "$raw_dir"
  for year in $(seq 1991 2026); do
    out="$raw_dir/${tour}_matches_${year}.csv"
    if [ -f "$out" ]; then
      echo "skip  $tour $year (exists)"
      continue
    fi
    echo "fetch $tour $year"
    curl -fsSL "$base_url/${tour}_matches_${year}.csv" -o "$out" \
      || { echo "warn  $tour $year not available"; rm -f "$out"; }
  done
done
echo "Done. Files in data/raw/<tour>/"
