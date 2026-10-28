#!/bin/bash
# Prints cv_score (all cases) then AUC and the organiser-only score (guard inputs).
cd "$(dirname "$0")/../.."
uv run cvbench/traj.py 2>/dev/null | grep -E "^(seg_traj|METRIC|AUC)"
uv run cvbench/traj.py --score-sources train-main-v2 2>/dev/null | grep -E "^METRIC" | sed 's/METRIC/HELDOUT/'
