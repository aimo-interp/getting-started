#!/bin/bash
# Run the thought-anchor curves (resample.py case) for every case in cases.jsonl, pinned to Crusoe.
# Reruns resume. Higher efforts get a larger token cap, since a cut-off continuation counts as wrong.
#   bash solutions/anchors/thought_anchors/run_all.sh          (PARALLEL cases at a time, default 3)
set -o pipefail
cd "$(dirname "$0")/../../.."
set -a; [ -f .env ] && . ./.env; set +a
HERE=solutions/anchors/thought_anchors
LOGS=data/thought_anchors_crusoe/logs
mkdir -p "$LOGS"

python - <<'EOF' | xargs -P "${PARALLEL:-3}" -L 1 bash -c '
  name=$0; effort=$1
  python '"$HERE"'/resample.py case --case "$name" --max-tokens "$2" --workers "${WORKERS:-24}" \
    > '"$LOGS"'/"$name".txt 2>&1 && echo "done $name" || echo "FAILED $name (see '"$LOGS"'/$name.txt)"'
import json
caps = {"low": 12000, "medium": 32768, "high": 49152}
for line in open("solutions/anchors/thought_anchors/cases.jsonl"):
    c = json.loads(line)
    print(c["case"], c["effort"], caps[c["effort"]])
EOF
