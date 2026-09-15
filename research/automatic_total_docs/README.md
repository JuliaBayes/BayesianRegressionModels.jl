# Automatic-total documentation acceptance

The three pages render saved results; this check samples no models.

- Four model examples passed 12 checks, including required generated Stan and automatic total-block discovery (`x44d4T`, exit 0).
- The retained-flat-prior regression passed 12 density/gradient checks (`J2B2Ee`, exit 0).
- `render_total_pages.jl` expands the actual fitted BRM declarations via the production backend-comparison helper.
- The first fixture omitted `docs/package.json` and therefore missed the math dependency (`ovy5EU`, exit 1 after all Julia model panes passed). Copying the existing repository package file fixed the fixture; the VitePress build completed (`OgrncA`, exit 0). The corrected render driver includes this copy.
- `check_total_pages.mjs` owns its temporary HTTP server, isolated browser profile and Chrome child. The browser check finished with exit 0 (`jHjRJx`), closed its own resources, verified 15/15/14+14 table rows, all four model panes, working Stan-tab selection, no broken images or equation/browser errors, client-side navigation, and 390-pixel layouts with no document overflow.
- `browser/` retains screenshots and measured DOM state. `panes/` retains the exact generated authoring/StanBlocks/Stan/Turing text checked before rendering.

These fixtures record the producer's exact scratch paths. To repeat elsewhere, update `repo`, `root` and the preview output paths to that checkout and use the repository docs environment. The browser driver needs Node's WebSocket support (`--experimental-websocket` with the recorded Node 20 artifact) and provisioned Chrome. Only the existing repo `docs/package.json` supplies frontend dependencies. No preview listener remains after the completed driver.
