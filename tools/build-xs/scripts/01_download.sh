#!/usr/bin/env bash
# Download the full OpenMHC-Full folder from GDrive to $FULL. Run on the DTN
# (dtn.sherlock.stanford.edu) — compute nodes have no outbound internet.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"
module load system rclone

case "$(hostname)" in
  *dtn*) : ;;  # good
  sh*-ln*|*login*) echo "WARNING: looks like a login node. Prefer the DTN for a 38 GB pull." ;;
esac

mkdir -p "$FULL"
echo "Downloading OpenMHC-Full (folder-id $GDRIVE_FOLDER_ID) -> $FULL (~38 GB)..."
rclone copy --drive-root-folder-id "$GDRIVE_FOLDER_ID" -P \
  --transfers 4 --checkers 8 --drive-chunk-size 64M \
  "$GDRIVE_REMOTE:" "$FULL"

echo "Verifying against source..."
rclone check --drive-root-folder-id "$GDRIVE_FOLDER_ID" --one-way "$GDRIVE_REMOTE:" "$FULL"
echo "Download complete. Tree:"
find "$FULL" -maxdepth 2 -type d | sort
du -sh "$FULL"
