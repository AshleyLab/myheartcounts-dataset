#!/usr/bin/env bash
# Pre-flight: peek INSIDE each GDrive archive (without downloading it whole) to
# confirm the tar's top-level structure matches what 02_extract.sh expects.
# Streams only the first part of each multi-part gzip and lists the first members.
# Run on the DTN (needs internet). Informational + WARN; does not download data.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"
module load system rclone

RID=(--drive-root-folder-id "$GDRIVE_FOLDER_ID")

peek() { # $1=archive relpath (a part-00 for multi-part, or the single file)  $2=expected substring  $3=label
  echo "[$3] $1"
  local out
  out=$(rclone cat "${RID[@]}" "$GDRIVE_REMOTE:$1" 2>/dev/null | tar -tz 2>/dev/null | head -5 || true)
  if [ -z "$out" ]; then
    echo "  WARN: could not list entries (auth? wrong path? rclone not configured?)"
    return 0
  fi
  echo "$out" | sed 's/^/    /'
  if echo "$out" | grep -q -- "$2"; then
    echo "  PASS: top-level matches expected '$2'"
  else
    echo "  WARN: expected '$2' not found in first entries — 02_extract may place files wrongly. Review before building."
  fi
}

echo "=== Archive structure pre-flight (expected vs actual top-level paths) ==="
peek "archives/hdf5_sharable_2026_full.tar.gz.part-00" ".h5"               "hdf5 (flat <user>.h5)"
peek "archives/daily_hf_full.tar.gz.part-00"           "daily_hf/"         "daily_hf"
peek "archives/daily_hourly_hf_full.tar.gz"            "daily_hourly_hf/"  "daily_hourly_hf"
peek "archives/hourly_trajectory_full.tar.gz"          "hourly_trajectory/" "hourly_trajectory"
peek "archives/minute_trajectory_full.tar.gz.part-00"  "minute_trajectory/" "minute_trajectory"

echo ""
echo "If all show PASS, 02_extract.sh's targets are correct. Any WARN -> inspect that"
echo "archive's layout and adjust 02_extract.sh's -C target before the full download/build."
