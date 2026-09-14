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
command -v curl >/dev/null
command -v perl >/dev/null
command -v sha256sum >/dev/null
chrome=$(command -v chrome-headless-shell || command -v google-chrome)
mkdir -p "$output_dir"

mathjax_url='https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg-full.js'
mathjax_sha256='a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6'
mathjax_bundle=$(mktemp "${TMPDIR:-/tmp}/brm-mathjax.XXXXXX")
typeset_dump=$(mktemp "${TMPDIR:-/tmp}/brm-typeset.XXXXXX")
typeset_profile=$(mktemp -d /tmp/kb-deck-profile.XXXXXX)
cleanup_render() {
  find "$mathjax_bundle" -maxdepth 0 -type f -delete
  find "$typeset_dump" -maxdepth 0 -type f -delete
  if command -v kb-clean-temp-dir >/dev/null; then
    kb-clean-temp-dir "$typeset_profile"
  else
    julia --startup-file=no -e 'rm(ARGS[1]; recursive=true, force=true)' \
      "$typeset_profile"
  fi
}
trap cleanup_render EXIT

curl -fsSL --retry 3 "$mathjax_url" -o "$mathjax_bundle"
printf '%s  %s\n' "$mathjax_sha256" "$mathjax_bundle" | sha256sum -c -

(
  cd "$deck_dir"
  quarto render brm-futures.qmd \
    --to revealjs \
    --output brm-futures.html
)
mv "$deck_dir/brm-futures.html" "$html"
perl "$deck_dir/embed-mathjax.pl" "$html" "$mathjax_bundle"

source_math_count=$(rg -o 'class="math (inline|display)"' "$html" | wc -l)
"$chrome" \
  --headless=new \
  --disable-gpu \
  --hide-scrollbars \
  --no-sandbox \
  --host-resolver-rules='MAP * ~NOTFOUND' \
  --user-data-dir="$typeset_profile" \
  --window-size=1600,900 \
  --virtual-time-budget=10000 \
  --dump-dom \
  "file://$html?transition=none" > "$typeset_dump"
perl "$deck_dir/freeze-mathjax.pl" "$typeset_dump" "$html"
frozen_math_count=$(rg -o '<mjx-container' "$html" | wc -l)
test "$frozen_math_count" -eq "$source_math_count"

perl "$deck_dir/print-pdf.pl" \
  "$chrome" "$html" "$pdf" 18 "$typeset_profile"

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

if rg -q "<(script|img)[^>]+src=['\"]https?://|<link[^>]+href=['\"]https?://|url\\(['\"]?https?://" \
  "$html"; then
  echo "render is not self-contained: external runtime resource found" >&2
  exit 1
fi

if ! rg -q '<mjx-container' "$html"; then
  echo "render is missing its frozen SVG math" >&2
  exit 1
fi

if rg -q '<script[^>]+id="MathJax-script"|RevealMath\.MathJax3\(\)' "$html"; then
  echo "render retained its build-time MathJax runtime" >&2
  exit 1
fi

printf 'verified\tslides=%s\tnotes=%s\tpages=%s\tself_contained=yes\n' \
  "$((slide_count + 1))" "$notes_count" "$pdf_pages"
