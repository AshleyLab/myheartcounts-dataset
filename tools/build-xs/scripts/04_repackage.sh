#!/usr/bin/env bash
# Assemble the final upload tree $UPLOAD from the filtered staged tree $STAGE:
#  - copy the small in-place files (top-level + data/{splits,labels,processed*,forecasting_sample_index})
#  - tar the 5 big dataset dirs into $UPLOAD/archives/*_xs.tar.gz
# Mirrors the OpenMHC-Full structure. XS is small -> single tarballs. Missing inputs are fatal.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"

rm -rf "$UPLOAD"
mkdir -p "$UPLOAD/archives" "$UPLOAD/data"

# top-level small files (required)
cp -f "$STAGE/README.md" "$STAGE/normalization_stats.json" "$STAGE/task_feature_exclusions.json" "$UPLOAD/"

# data/ small files (exclude the big dataset dirs that live under processed/)
for sub in splits labels processed forecasting_sample_index; do
  [ -d "$STAGE/data/$sub" ] || { echo "ERROR: missing $STAGE/data/$sub" >&2; exit 1; }
  mkdir -p "$UPLOAD/data/$sub"
  if [ "$sub" = "processed" ]; then
    rsync -a --exclude '/daily_hf/' --exclude '/daily_hourly_hf/' "$STAGE/data/$sub/" "$UPLOAD/data/$sub/"
  else
    rsync -a "$STAGE/data/$sub/" "$UPLOAD/data/$sub/"
  fi
done

# big dataset dirs -> archives (tar from inside parent so paths are canonical names). Missing = fatal.
tar_dir() { # $1=parent  $2=dirname  $3=outname
  [ -d "$1/$2" ] || { echo "ERROR: missing dataset dir $1/$2" >&2; exit 1; }
  echo "  tarring $2 -> archives/$3"
  tar -czf "$UPLOAD/archives/$3" -C "$1" "$2"
}
tar_dir "$STAGE/data"            hdf5              hdf5_sharable_2026_xs.tar.gz
tar_dir "$STAGE/data/processed"  daily_hf          daily_hf_xs.tar.gz
tar_dir "$STAGE/data/processed"  daily_hourly_hf   daily_hourly_hf_xs.tar.gz
tar_dir "$STAGE/data"            hourly_trajectory hourly_trajectory_xs.tar.gz
tar_dir "$STAGE/data"            minute_trajectory minute_trajectory_xs.tar.gz

# sanity: exactly 5 openable tarballs
ntar=$(ls -1 "$UPLOAD"/archives/*_xs.tar.gz 2>/dev/null | wc -l)
[ "$ntar" -eq 5 ] || { echo "ERROR: expected 5 tarballs, found $ntar" >&2; exit 1; }
for t in "$UPLOAD"/archives/*_xs.tar.gz; do tar -tzf "$t" >/dev/null || { echo "ERROR: bad tarball $t" >&2; exit 1; }; done

echo "Upload tree assembled at $UPLOAD:"
find "$UPLOAD" -type f -printf '%10s  %p\n' | sort -k2
du -sh "$UPLOAD"
