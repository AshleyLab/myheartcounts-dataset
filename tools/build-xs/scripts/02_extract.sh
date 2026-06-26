#!/usr/bin/env bash
# Reconstruct multi-part archives and extract the full tree into $EXTRACT.
# Heavy I/O -> run inside the SLURM job (03_build.sbatch), not on a login node.
# $EXTRACT can be redirected (e.g. to $L_SCRATCH) by exporting EXTRACT before calling.
set -euo pipefail
shopt -s nullglob              # missing globs expand to empty (handled explicitly below)
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"

A="$FULL/archives"
echo "Extract target: $EXTRACT"
mkdir -p "$EXTRACT/data/hdf5" "$EXTRACT/data/processed"

# 1) small in-place files (copy verbatim; filtered later). Required -> let set -e catch absence.
cp -f "$FULL/README.md" "$FULL/normalization_stats.json" "$FULL/task_feature_exclusions.json" "$EXTRACT/"
for sub in splits labels processed forecasting_sample_index; do
  if [ -d "$FULL/data/$sub" ]; then
    mkdir -p "$EXTRACT/data/$sub"
    cp -rf "$FULL/data/$sub/." "$EXTRACT/data/$sub/"
  else
    echo "ERROR: required source dir missing: $FULL/data/$sub" >&2; exit 1
  fi
done

# 2) big archives -> canonical dirs. Required: a fully-missing archive is fatal.
extract_multi() { # $1=prefix  $2=target
  local prefix="$1" target="$2"
  local parts=( "${prefix}".part-* )
  if [ ${#parts[@]} -gt 0 ]; then
    # version-sort to be robust to unpadded suffixes (part-2 vs part-10)
    IFS=$'\n' read -r -d '' -a parts < <(printf '%s\n' "${parts[@]}" | sort -V && printf '\0')
    echo "  extracting $(basename "$prefix") (${#parts[@]} parts) -> $target"
    cat "${parts[@]}" | tar -xz -C "$target"
  elif [ -f "${prefix}" ]; then
    echo "  extracting $(basename "$prefix") -> $target"
    tar -xzf "${prefix}" -C "$target"
  else
    echo "ERROR: required archive missing: ${prefix}[.part-*]" >&2; exit 1
  fi
}

extract_multi "$A/hdf5_sharable_2026_full.tar.gz" "$EXTRACT/data/hdf5"
extract_multi "$A/daily_hf_full.tar.gz"           "$EXTRACT/data/processed"
extract_multi "$A/daily_hourly_hf_full.tar.gz"    "$EXTRACT/data/processed"
extract_multi "$A/hourly_trajectory_full.tar.gz"  "$EXTRACT/data"
extract_multi "$A/minute_trajectory_full.tar.gz"  "$EXTRACT/data"

echo "Extraction done. Sanity:"
for d in data/hdf5 data/processed/daily_hf data/processed/daily_hourly_hf data/hourly_trajectory data/minute_trajectory; do
  if [ -d "$EXTRACT/$d" ]; then printf "  %-32s %s entries\n" "$d" "$(ls -1 "$EXTRACT/$d" | wc -l)"; else echo "  MISSING $d"; exit 1; fi
done
