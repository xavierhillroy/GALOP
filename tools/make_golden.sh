#!/usr/bin/env bash
# =============================================================================
# make_golden.sh — generate the GOLDEN BASELINE (the "answer key").
#
# Run this ONCE, on code you trust (today's CPU reference implementation).
# It records the exact output of several fixed-seed runs into golden/.
#
# You do NOT run this routinely. Re-run it only when you have DELIBERATELY
# decided that new behavior is the correct new reference (rare). To check a
# change against the baseline, use check_golden.sh instead.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where it's called from

# Which (target, seed) cells make up the baseline. More cells = stronger
# anchor. These all have datasets/<target>_{train,test}.csv present.
TARGETS="nguyen1 nguyen2 nguyen3"
SEEDS="0 1 2"
GOLD=golden

echo "Building lgp_run..."
make lgp_run >/dev/null

# Start clean so stale files can't linger in the baseline.
rm -rf "$GOLD"
mkdir -p "$GOLD"

for t in $TARGETS; do
  for s in $SEEDS; do
    echo "  golden: $t seed $s"
    ./lgp_run "datasets/${t}_train.csv" "datasets/${t}_test.csv" "$t" "$s" \
      /tmp/golden_results_scratch.csv \
      "$GOLD/prog_${t}_s${s}.txt" \
      "$GOLD/hist_${t}_s${s}.csv" >/dev/null
  done
done

# Compact fingerprint manifest: one sha256 line per file. Handy for comparing
# across machines/repos without shipping the files. check_golden.sh doesn't
# need this (it uses diff), but `sha256sum -c golden/SHA256SUMS` works too.
sha256sum "$GOLD"/*.csv "$GOLD"/*.txt > "$GOLD/SHA256SUMS"

echo
echo "Golden baseline written to $GOLD/  ($(ls "$GOLD" | wc -l) files)."
echo "Commit it and tag the commit:  git add golden && git commit && git tag golden-baseline"
