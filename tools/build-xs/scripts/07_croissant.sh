#!/usr/bin/env bash
# Fetch the Croissant metadata for the dataset from Dataverse's native exporter.
# Run AFTER files + citation metadata exist on the draft. Run on the DTN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"
[ -f "$TOKEN_FILE" ] || { echo "Missing $TOKEN_FILE"; exit 1; }
TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
OUT="$BUILD/openmhc-xs.croissant.json"

echo "Requesting Croissant export for $DOI ..."
HTTP=$(curl -s -o "$OUT" -w "%{http_code}" -H "X-Dataverse-key:$TOKEN" \
  "$DATAVERSE_BASE/api/datasets/export?exporter=croissant&persistentId=$DOI") \
  || { echo "curl transport failed (network/DTN?). Re-run on the DTN."; exit 2; }
echo "HTTP $HTTP -> $OUT"
if [ "$HTTP" != "200" ]; then
  echo "Exporter not ready (common on a brand-new draft until the version is finalized)."
  echo "Options: (1) re-run after metadata is filled / version published;"
  echo "         (2) generate locally with mlcroissant or adapt MHC-benchmark croissant_metadata_and_rai/build_croissant.py."
  exit 2
fi
python3 -c "import json;d=json.load(open('$OUT'));print('Croissant OK:',d.get('name'),'| fields:',len(d.get('recordSet',[])) if isinstance(d.get('recordSet'),list) else 'n/a')"
