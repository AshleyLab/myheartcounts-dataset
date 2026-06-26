#!/usr/bin/env bash
# Snapshot the GDrive OpenMHC-Full listing and diff against the previous snapshot
# to confirm the source is FROZEN before building. Run on DTN or login (light).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"
module load system rclone

STAMP="$(date +%Y%m%d_%H%M%S)"
NEW="$BUILD/manifest.$STAMP.json"
PREV="$(ls -1t "$BUILD"/manifest.*.json 2>/dev/null | head -1 || true)"

echo "Listing OpenMHC-Full by folder-id $GDRIVE_FOLDER_ID (recursive, with sizes/hashes)..."
rclone lsjson -R --hash --drive-root-folder-id "$GDRIVE_FOLDER_ID" "$GDRIVE_REMOTE:" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(sorted([{k:f.get(k) for k in ('Path','Size','ModTime','Hashes')} for f in d], key=lambda x:x['Path']), indent=2))" \
  > "$NEW"

NFILES=$(python3 -c "import json;print(len(json.load(open('$NEW'))))")
TOTAL=$(python3 -c "import json;print(sum(f['Size'] for f in json.load(open('$NEW')) if f.get('Size',-1)>0))")
echo "Snapshot: $NEW  ($NFILES entries, $(numfmt --to=iec $TOTAL 2>/dev/null || echo $TOTAL bytes))"

if [ -n "$PREV" ]; then
  echo "Diff vs previous snapshot: $PREV"
  if diff -q "$PREV" "$NEW" >/dev/null; then
    echo "RESULT: IDENTICAL to previous snapshot -> source looks FROZEN. Safe to build."
  else
    echo "RESULT: CHANGED since previous snapshot -> NOT frozen. Review diff below:"
    diff "$PREV" "$NEW" || true
    echo "Re-run later until two consecutive snapshots match before building."
  fi
else
  echo "RESULT: first snapshot taken. Re-run later; build only when two snapshots match."
fi
ln -sf "$NEW" "$BUILD/manifest.latest.json"
