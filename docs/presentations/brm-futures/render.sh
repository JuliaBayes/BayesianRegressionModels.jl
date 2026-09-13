#!/usr/bin/env bash
set -euo pipefail

deck_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$deck_dir/../../.." && pwd)
output_dir="$repo_root/docs/src/public/decks"
html="$output_dir/brm-futures.html"
pdf="$output_dir/brm-futures.pdf"

command -v quarto >/dev/null
chrome=$(command -v chrome-headless-shell || command -v google-chrome)
mkdir -p "$output_dir"

(
  cd "$deck_dir"
  quarto render brm-futures.qmd \
    --to revealjs \
    --output brm-futures.html
)
mv "$deck_dir/brm-futures.html" "$html"

"$chrome" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=10000 \
  --print-to-pdf="$pdf" \
  "file://$html?print-pdf"

test -s "$html"
test -s "$pdf"
