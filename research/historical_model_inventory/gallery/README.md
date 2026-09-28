# Historical inventory gallery

This directory is an isolated runtime consumer of `../model_matrix.tsv`. The
single mounted `HistoricalInventory` semantic graph owns the matrix rows, four
filter parameters, their evaluated option domains, the filtered rows, the card
tree, and its GET operation. `HistoricalInventoryGalleryApp` only mounts that
graph through `semantic_app`; it has no second declaration, manual route/form
mirror, `AppData`/`AppContext` layer, or DynamicObjects-versus-HTMXObjects
shadow model.

The row evidence remains authoritative. The descriptor-to-semantic-card
adapter is marked experimental everywhere it appears and cannot promote an
unsupported or unresolved validation tier.

The validation filter and primary badge use the current audited/inferred
translation tier. Each card shows the exact-metadata tier separately;
historical example stamps are never treated as current capability receipts.

Validate without opening a network listener:

```sh
julia --project=web-macro research/historical_model_inventory/gallery/validate.jl
```

The browser route uses HTMXObjects' standard runtime and default operation
policy: ordinary navigation returns the framework's HTMX-enabled page shell,
whose load operation mounts the complete semantic surface. Subsequent filter
submissions use the same authoritative graph and its inferred option domains.

Serve explicitly when desired:

```sh
julia --project=web-macro research/historical_model_inventory/gallery/serve.jl 8127 127.0.0.1
```

The optional third serving argument selects another compatible matrix path.
`served_smoke.tsv` records the exact Strato2 commit, matrix/source hashes,
listener URL, restart, HTTP statuses, and observed card/filter counts.

## Service operation (strato2)

The `historical-brm-gallery.service` user unit serves this gallery on strato2
(`WorkingDirectory` the `BayesianRegressionModels.jl` checkout,
`julia --startup-file=no --project=web-macro
research/historical_model_inventory/gallery/serve.jl 8129 127.0.0.1`).

The `web-macro` environment develops the shared checkouts under
`~/github/nsiccha/`; its `Manifest.toml` is host-local (gitignored) while
those checkouts advance on every remote attach. When a checkout moves under
the manifest, the service fails at boot with

    ArgumentError: Package <Pkg> does not have <Dep> in its dependencies

(2026-09-24: `HTMXObjects` without `Dates`, against a manifest resolved
2026-07-29). Recover in the service checkout:

```sh
cd ~/github/nsiccha/BayesianRegressionModels.jl
julia --startup-file=no --project=web-macro -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="/home/n/github/nsiccha/MutatingFunctions.jl"), Pkg.PackageSpec(path="/home/n/github/nsiccha/OutputSignatures.jl")]); Pkg.resolve()'
git diff --quiet -- web-macro/Project.toml  # must be clean: the entries are committed
julia --startup-file=no --project=web-macro research/historical_model_inventory/gallery/validate.jl
```

A bare `Pkg.resolve()` is not enough when the drift spans an unregistered
dependency: Julia 1.10 ignores `[sources]`, so unregistered packages must be
developed from local checkouts (here `MutatingFunctions` and
`OutputSignatures`, hard dependencies of `BayesianRegressionModels` since
0.2.1, committed to `web-macro/Project.toml` for exactly this reason).
`web-macro/Project.toml` carries no comments: Pkg rewrites that file and
would strip them, dirtying the service checkout.

`validate.jl` currently reports 8 stale lazy-shell failures on top of a
healthy serve (todo `2026-09-25T11-10-01-587-0ms5fwg` tracks the update to
HTMXObjects' single-render contract); until it lands, the recovery gate is
the remaining 64 assertions plus a live boot with 200s on `/` and a
`source_fidelity=confirmed` filter at 154 cards.

The unit carries `StartLimitIntervalSec=300` / `StartLimitBurst=5` so a boot
failure stops the unit instead of restart-churning; after a successful
recovery, `systemctl --user enable --now historical-brm-gallery.service` and
append the smoke row to `served_smoke.tsv`.
