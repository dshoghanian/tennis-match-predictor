#!/usr/bin/env bash
# Download Jeff Sackmann ATP singles match files (1991-2026) into data/raw/.
# Source: https://github.com/JeffSackmann/tennis_atp (CC BY-NC-SA 4.0)
set -euo pipefail

RAW_DIR="$(dirname "$0")/../data/raw"
BASE_URL="https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
mkdir -p "$RAW_DIR"

for year in $(seq 1991 2026); do
  url="$BASE_URL/atp_matches_${year}.csv"
  out="$RAW_DIR/atp_matches_${year}.csv"
  if [ -f "$out" ]; then
    echo "skip  $year (exists)"
    continue
  fi
  echo "fetch $year"
  curl -fsSL "$url" -o "$out" || { echo "warn  $year not available"; rm -f "$out"; }
done
echo "Done. Files in $RAW_DIR"
