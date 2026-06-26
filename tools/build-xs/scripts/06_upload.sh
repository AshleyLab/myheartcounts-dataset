#!/usr/bin/env bash
# Upload the OpenMHC-XS tree to the Harvard Dataverse draft via the Native API.
# Run on the DTN (needs internet). Requires $TOKEN_FILE with your API token.
# Leaves the dataset as a DRAFT — does NOT publish.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/config.sh"

[ -f "$TOKEN_FILE" ] || { echo "Missing $TOKEN_FILE (chmod 600). See RUNBOOK §prereqs."; exit 1; }
TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
API="$DATAVERSE_BASE/api/datasets/:persistentId"
DRYRUN="${DRYRUN:-0}"   # set DRYRUN=1 to print actions without uploading

echo "Confirming draft $DOI is reachable and writable..."
STATE=$(curl -fsS -H "X-Dataverse-key:$TOKEN" "$API/?persistentId=$DOI" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['data']['latestVersion']['versionState'])") \
  || { echo "Cannot read draft (token/DOI/permissions?). Aborting."; exit 1; }
echo "  versionState=$STATE"

# Files already on the draft (dir|name) — skip them so partial/re-run uploads are idempotent.
EXIST=$(curl -fsS -H "X-Dataverse-key:$TOKEN" "$API/?persistentId=$DOI" | python3 -c "import sys,json
d=json.load(sys.stdin)
for f in d['data']['latestVersion'].get('files',[]):
    print((f.get('directoryLabel','') or '')+'|'+f['dataFile']['filename'])")
echo "  already present: $(printf '%s' "$EXIST" | grep -c . ) file(s) — will skip those"

# Upload every file under $UPLOAD, preserving subfolders via directoryLabel.
count=0; skipped=0
while IFS= read -r -d '' f; do
  rel="${f#$UPLOAD/}"
  dir="$(dirname "$rel")"; [ "$dir" = "." ] && dir=""
  if printf '%s\n' "$EXIST" | grep -qxF "${dir}|$(basename "$rel")"; then
    skipped=$((skipped+1)); echo "skip (already on draft): $rel"; continue
  fi
  # omit directoryLabel for root files (empty string is not portably accepted)
  jsonData=$(python3 -c "import json,sys
o={'description':'OpenMHC-XS (5% subset, 593 users)','categories':['Data']}
d=sys.argv[1]
if d: o['directoryLabel']=d
print(json.dumps(o))" "$dir")
  count=$((count+1))
  echo "[$count] upload $rel  (dir='${dir:-/}')"
  if [ "$DRYRUN" = "1" ]; then continue; fi
  curl -fsS -H "X-Dataverse-key:$TOKEN" -X POST \
    -F "file=@${f}" -F "jsonData=${jsonData}" \
    "$API/add?persistentId=$DOI" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('   ->',d.get('status'),d.get('data',{}).get('files',[{}])[0].get('label',d.get('message','')))" \
    || { echo "   UPLOAD FAILED for $rel — see RUNBOOK (large files may need DVUploader / S3 direct upload)."; exit 1; }
  sleep 1
done < <(find "$UPLOAD" -type f -print0 | sort -z)

echo "Uploaded $count new file(s), skipped $skipped already-present, to draft $DOI. Review at:"
echo "  $DATAVERSE_BASE/dataset.xhtml?persistentId=$DOI&version=DRAFT"
echo "Publishing is manual/your call:"
echo "  curl -H X-Dataverse-key:\$TOKEN -X POST \"$API/actions/:publish?persistentId=$DOI&type=major\""
