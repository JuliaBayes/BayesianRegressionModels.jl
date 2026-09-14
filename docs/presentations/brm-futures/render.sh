#!/usr/bin/env bash
set -euo pipefail

deck_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$deck_dir/../../.." && pwd)
output_dir="$repo_root/docs/src/public/decks"
html="$output_dir/brm-futures.html"
pdf="$output_dir/brm-futures.pdf"

command -v quarto >/dev/null
command -v pdfinfo >/dev/null
command -v rg >/dev/null
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

slide_count=$(rg -c '^## ' "$deck_dir/brm-futures.qmd")
notes_count=$(rg -c '^::: \{\.notes\}' "$deck_dir/brm-futures.qmd")
pdf_pages=$(pdfinfo "$pdf" | awk '/^Pages:/ {print $2}')

test "$slide_count" -eq 17
test "$notes_count" -eq "$slide_count"
test "$pdf_pages" -eq 18

if rg -q 'integration gate|Case artifact is intentionally gated|Awaiting reproducible results panel' \
  "$deck_dir/brm-futures.qmd"; then
  echo "deck still contains an adaptive-centering integration placeholder" >&2
  exit 1
fi

if rg -q '(src|href)="brm-futures_files/' "$html"; then
  echo "render is not self-contained: brm-futures_files reference found" >&2
  exit 1
fi

printf 'verified\tslides=%s\tnotes=%s\tpages=%s\tself_contained=yes\n' \
  "$((slide_count + 1))" "$notes_count" "$pdf_pages"
