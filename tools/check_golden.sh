#!/usr/bin/env bash
# =============================================================================
# check_golden.sh — compare CURRENT code's output against the golden baseline.
#
# Run this AFTER ANY change (refactor, backend split, etc.). It re-runs the
# same fixed-seed cells and diffs them against golden/. It NEVER modifies
# golden/ — the answer key stays frozen.
#
#   "ALL MATCH"  -> your change preserved behavior. Safe.
#   "MISMATCH"   -> your change altered behavior. Inspect the printed diff.
#
# NOTE: this exact-match check is for CPU-only refactors (Phases 1-3), where
# output must be bit-identical. The GPU kernel will need a tolerance compare
# instead (float reductions reorder), not this script.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

TARGETS="nguyen1 nguyen2 nguyen3"
SEEDS="0 1 2"
GOLD=golden
TMP=$(mktemp -d)

if [ ! -d "$GOLD" ]; then
  echo "No golden/ baseline found. Generate it first: tools/make_golden.sh"
  exit 1
fi

echo "Building lgp_run..."
make lgp_run >/dev/null

fail=0
for t in $TARGETS; do
  for s in $SEEDS; do
    ./lgp_run "datasets/${t}_train.csv" "datasets/${t}_test.csv" "$t" "$s" \
      "$TMP/results.csv" "$TMP/prog_${t}_s${s}.txt" "$TMP/hist_${t}_s${s}.csv" >/dev/null

    for pair in "hist:csv" "prog:txt"; do
      kind=${pair%%:*}; ext=${pair##*:}
      f="${kind}_${t}_s${s}.${ext}"
      if diff -q "$GOLD/$f" "$TMP/$f" >/dev/null 2>&1; then
        echo "  OK    $f"
      else
        echo "  FAIL  $f   <-- output changed vs golden"
        fail=1
      fi
    done
  done
done

rm -rf "$TMP"
echo
if [ "$fail" -eq 0 ]; then
  echo "ALL MATCH — current code reproduces the golden baseline."
else
  echo "MISMATCH — current code differs from golden. Your change altered behavior."
  exit 1
fi
