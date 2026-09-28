@dynamicstruct struct AppData
    __status__ = initialize_progress!(:state; description="BRM pipeline")
    examples_dir = joinpath(dirname(@__DIR__), "examples")

    dataset = Dataset()

    # Stan compile concurrency control: per-.stan-path locks for the
    # "ensure file written + compile_model" sequence (without these, two
    # requests for the same model `make` concurrently into the same .so
    # target and corrupt it — subsequent dlopen segfaults Julia, no
    # stack, no OOM, just process gone), plus a global lock that
    # serializes ALL compiles across paths (gallery-opening N cards
    # spawns N parallel `make` jobs at -O3-scale memory peaks, which
    # OOM-killed the 3.8 GB strato box). Holding the global lock around
    # `compile_model` caps the box at one compile in flight; StanModel
    # construction is left outside the lock (cheap dlopen).
    @struct stan_locks = begin
        # Keyed by absolute .stan path. Entries are never reaped; one per
        # distinct compiled model. `get!(ReentrantLock, per_path, path)`
        # lazily creates.
        per_path = ThreadsafeDict{String,ReentrantLock}()
        global_lock = ReentrantLock()
    end
    # Per-(formula, namespace) cache of the rendered auto-PPC HTML element.
    # The underlying Stan fit is already memoized via polling_fetchindex /
    # @memo! on `r.stan.*`, but `build_ppc_section` itself is non-trivial
    # (DataFrames -> Vega-Lite specs); caching the resulting h.div node
    # turns repeated card opens into pure HTML re-emits.
    ppc_html_cache = ThreadsafeDict{Tuple{UInt,Symbol},Any}()

    namespace_from(label) = isempty(strip(label)) ? :default :
        Symbol(lowercase(first(split(strip(label), r"[\s:\-]+"))))

    # One run of the @brm pipeline for a given (text, namespace) pair. Every
    # stage is a derived property -- accessing `.brmi` triggers parse + eval,
    # `.benches` triggers the benchmark loop, etc. Unused branches don't
    # compute. Safety is enforced at `wrapped`, which refuses to produce Julia
    # code for an unsafe formula.
    # TODO(DO): investigate whether indexed inline structs (@include) should
    # accept default values for index params. Today `@include run(text, namespace=:default)`
    # errors with "index param must be a Symbol" because the default becomes
    # a `:kw` Expr. Unclear if there's a sensible meaning for such defaults at
    # all (cache keying, forwarding, etc.) or if we should just continue
    # requiring bare Symbol params.
    @struct run(text, namespace) = begin
        formula = Formula(text)
        df = dataset.df
        container = dataset.container(namespace)

        wrapped = begin
            formula.is_safe || throw(formula.violation)
            _brm(text; df=container)
        end
        brmi = eval(wrapped)

        # ── cimpl branch ──
        vbrmi = VBRMI(brmi)
        dim   = LogDensityProblems.dimension(vbrmi)
        x0    = randn(Xoshiro(0), dim)
        ldp   = string(LogDensityProblems.logdensity(vbrmi, x0))
        grad  = FiniteDifferences.grad(
            central_fdm(5, 1),
            Base.Fix1(LogDensityProblems.logdensity, vbrmi),
            x0,
        )[1]

        tol    = 1e-8
        n_dead = count(<=(tol) ∘ abs, grad)
        dead   = findall(<=(tol) ∘ abs, grad)

        benches = vcat(
            [
                "logdensity (total)" => @be(randn(dim), LogDensityProblems.logdensity($vbrmi, _)),
                "lprior!"            => @be(randn(dim), lprior!($vbrmi, _)),
            ],
            [
                "  lprior!($group_key[$i] $(part))" => @be(randn(nparams(part)), lprior!($part, _))
                for (group_key, parts) in pairs(vbrmi.meta.blocks)
                for (i, part) in enumerate(parts)
            ],
            [
                "llikelihood!($key)" => @be(llikelihood!($m))
                for (key, m) in pairs(vbrmi.meta.materialized)
            ],
        )

        # ── stan branch ──
        sbbrmi = SBBRMI(brmi; mod=@__MODULE__)

        # Everything Stan-related bundled under `r.stan.*`. Nested access via
        # step_chain's tuple-path specs (e.g. `(:stan, :src)`).
        @struct stan = begin
            src = stan_code(sbbrmi)
            # Hash-keyed cache path so identical Stan source reuses the same
            # .stan (and co-located .so) across requests. `BridgeStan.compile_model`
            # invokes make, which skips when the .so is newer than the .stan —
            # so a cache hit resolves in milliseconds instead of re-running
            # the C++ build.
            # Hash folds in the make-args so changing compile flags (e.g.
            # adding STAN_THREADS=true) naturally routes to a fresh path and
            # triggers a rebuild — BridgeStan doesn't support reloading a
            # previously-dlopened .so, so we can't reuse old binaries.
            # `O=0` overrides Stan's default `-O3` (set in
            # stan_math/make/compiler_flags as `O ?= 3`). Templated Stan
            # headers blow up g++'s memory at -O3 -- ~2 GB peak per
            # invocation, which OOM-kills julia on the 3.8 GB strato box.
            # -O0 cuts that ~3x at the cost of slower model run-time
            # (acceptable for a pipeline-explorer demo). The hash includes
            # make_args so old -O3 .so caches are not reused.
            _make_args = ["O=0", "STAN_THREADS=true"]
            file = begin
                p = joinpath(tempdir(), "brm_stan",
                             string(hash((src, _make_args))) * ".stan")
                mkpath(dirname(p))
                p
            end
            # Serialize file-write + compile per .stan path. Two threads
            # racing into the same `make` invocation produced a half-written
            # `.so` whose subsequent dlopen segfaulted Julia (the make race
            # is independently real -- see commit message of the original
            # lock fix). The lock is held only for the file-write +
            # compile_model call; StanModel construction is left unguarded
            # because per-task ReentrantLocks combined with progress-yields
            # appeared to deadlock the chain in practice.
            lib = Base.lock(get!(ReentrantLock, stan_locks.per_path, file)) do
                @info "[stan] before compile_model" file
                isfile(file) || write(file, src)
                # Hold the global compile lock only across the actual `make`
                # invocation -- per-path lock above already short-circuits
                # cache hits cheaply, so the global queue only fills up with
                # genuine first-time compiles.
                rv = Base.lock(stan_locks.global_lock) do
                    BridgeStan.compile_model(file; make_args=_make_args)
                end
                @info "[stan] after compile_model" file
                rv
            end
            # SB's `stan_data` walks the SlicModel → StanModel tracing which
            # auto-declares `_n` / `_m` sizes for every vector / matrix, then
            # `bridgestan_data` JSON-serializes with Stan's column-major
            # matrix convention.
            data     = StanBlocks.stan.bridgestan_data(StanBlocks.stan_data(sbbrmi.model))
            instance = begin
                @info "[stan] before StanModel(instance)"
                rv = BridgeStan.StanModel(lib, data)
                @info "[stan] after StanModel(instance)"
                rv
            end
            dim      = BridgeStan.param_unc_num(instance)
            # Fixed-rng narrow-normal init — deterministic, cache-friendly.
            init     = 0.1 .* randn(Xoshiro(42), dim)

            # Smoke-test evaluation: log density at the init params. Forces the
            # model-loaded + data-bound path without running Pathfinder.
            log_density = BridgeStan.log_density(instance, init)

            # Synthetic-data generation: sample N unconstrained parameter draws
            # from a narrow zero-mean normal, then `param_constrain` each with
            # `include_tp` + `include_gq` so the output matrix also carries
            # transformed parameters and generated-quantities (the synthetic
            # outcomes `y_sim` live in the GQ block when the model defines
            # one).
            @struct generated = begin
                n     = 50
                unc   = 0.1 .* randn(Xoshiro(44), dim, n)
                names = BridgeStan.param_names(instance; include_tp=true, include_gq=true)
                # `value` is the constrained-draws matrix (m × n where m
                # includes p+tp+gq); access via `generated.value` from
                # outside.
                value = constrain_draws(unc, instance; rng_seed=45)
                dfs   = dfs_from_constrained(value, names)
                df         = dfs.long
                wide_df    = dfs.wide
                summary_df = dfs.summary
            end
            # One row per base parameter name (before any `.`), with the
            # number of indexed entries and Stan-block classification:
            # P = parameter, TP = transformed parameter, GQ = generated
            # quantity. Classification is derived from BridgeStan's
            # param_names called with/without include_tp / include_gq.
            param_shapes_df = begin
                p_names   = BridgeStan.param_names(instance)
                p_tp_names = BridgeStan.param_names(instance; include_tp=true)
                tp_set = Set(setdiff(p_tp_names, p_names))
                gq_set = Set(setdiff(generated.names, p_tp_names))
                classify(name) =
                    name in tp_set ? "TP" : name in gq_set ? "GQ" : "P"
                splits     = [split(nm, '.', limit=2) for nm in generated.names]
                base_names = [String(first(s)) for s in splits]
                kinds      = [classify(nm) for nm in generated.names]
                uniq_base  = unique(base_names)
                uniq_kind  = [kinds[findfirst(==(b), base_names)] for b in uniq_base]
                DataFrame(
                    param = uniq_base,
                    kind = uniq_kind,
                    n_indices = [count(==(p), base_names) for p in uniq_base],
                )
            end
            # Ground-truth overlays for plot scatter/vrule layers. `df` uses
            # constrained-with-TP/GQ parameter names (matches
            # `generated.df`'s long-format layout); `unc_df` uses the
            # unconstrained subset (matches `posterior.long_df` on the fit
            # path so the scatter overlay joins correctly).
            @struct truth = begin
                df = begin
                    truth_col = view(generated.value, :, fit.draw_idx)
                    splits     = [split(nm, '.', limit=2) for nm in generated.names]
                    base_names = [String(first(s)) for s in splits]
                    parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
                    indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
                    DataFrame(
                        param = base_names,
                        index = indices,
                        truth = collect(truth_col),
                    )
                end
                unc_df = begin
                    unc_names = BridgeStan.param_unc_names(fit.instance)
                    splits     = [split(nm, '.', limit=2) for nm in unc_names]
                    base_names = [String(first(s)) for s in splits]
                    parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
                    indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
                    DataFrame(
                        param = base_names,
                        index = indices,
                        truth = fit.truth_unc,
                    )
                end
            end
            # Simulation-based calibration setup: pick one draw from the prior
            # predictive `generated` matrix, extract every `*_gen[.i.j]` entry,
            # and fold them back into the Stan data dict as their `*` (observed)
            # counterparts. The resulting `fit_instance` shares the compiled
            # .so with `instance` but is bound to this synthetic observed data,
            # so Pathfinder / full warmup samples `p(theta | y_sim)` and should
            # recover the ground-truth `generated_unc[:, fit_draw_idx]`.
            # Simulation-based-calibration fit setup. `draw_idx`-th column of
            # the prior-predictive `generated` matrix is folded back into the
            # Stan data dict as the "observed" data (via `*_gen → *` renames),
            # then a fresh `instance` is bound to that synthetic data and used
            # by `pathfinder` / `warmup_fit` to recover the ground-truth
            # `generated.unc[:, draw_idx]`.
            @struct fit = begin
                draw_idx = 1
                truth_unc = collect(view(generated.unc, :, draw_idx))
                data_dict = begin
                    base = StanBlocks.stan_data(sbbrmi.model)
                    col  = view(generated.value, :, draw_idx)
                    groups = Dict{Symbol, Vector{Tuple{Vector{Int}, Float64}}}()
                    for (i, name) in enumerate(generated.names)
                        m = match(r"^(.+)_gen(?:\.(.+))?$", name)
                        isnothing(m) && continue
                        base_name = Symbol(m.captures[1])
                        idx_str   = m.captures[2]
                        idxs = isnothing(idx_str) ? Int[] :
                               [Base.parse(Int, s) for s in split(idx_str, ".")]
                        push!(get!(Vector{Tuple{Vector{Int}, Float64}}, groups, base_name),
                              (idxs, col[i]))
                    end
                    # Preserve the original's element type so integer-valued data
                    # vars (Bernoulli outcomes, counts) stay integers after the
                    # override — Stan's int-typed data variables reject Float64s
                    # even when the values are exact (1.0, 0.0, …).
                    overrides = Dict{Symbol, Any}()
                    for (name, entries) in groups
                        orig = base[name]
                        if length(entries) == 1 && isempty(entries[1][1])
                            overrides[name] = convert(typeof(orig), entries[1][2])
                        else
                            out = similar(orig)  # preserves eltype
                            for (idxs, v) in entries
                                out[idxs...] = v  # implicit convert to eltype(out)
                            end
                            overrides[name] = out
                        end
                    end
                    merge(base, overrides)
                end
                instance = begin
                    @info "[stan] before StanModel(fit.instance)"
                    rv = BridgeStan.StanModel(lib,
                        StanBlocks.stan.bridgestan_data(data_dict))
                    @info "[stan] after StanModel(fit.instance)"
                    rv
                end
            end

            # IP: Pathfinder init (fast, no MCMC). The `progress=__status__`
            # hook lets Treebars nest the maxiters subtree under whatever
            # node called `fetchindex!(status, …, pathfinder, instance, init)`.
            "Pathfinder(maxiters=$maxiters)"
            pathfinder(instance, init; rng=Xoshiro(42), maxiters=100) = begin
                @info "[stan] entering pathfinder" maxiters
                rv = initialize_mcmc(StanProblem(instance; nan_on_error=true), init; rng, progress=__status__, maxiters)
                @info "[stan] leaving pathfinder"
                rv
            end
            # IP: full Stan + WarmupHMC fit. Same progress-hooking pattern as
            # `pathfinder` above. Returns a rich NamedTuple with
            # `.posterior_position`, `.ess`, `.n_divergent_samples`, etc.
            # Named `warmup_fit` (not `posterior_warmup`) so the bare name
            # doesn't share the `posterior_*` prefix with the `posterior`
            # bundle below — would otherwise re-trip the prefix lint.
            "WarmupHMC(n_draws=$n_draws)"
            warmup_fit(instance, init; rng=Xoshiro(42), n_draws=200) = begin
                @info "[stan] entering warmup_fit" n_draws
                rv = adaptive_warmup_mcmc(rng, StanProblem(instance; nan_on_error=true); init, n_draws, progress=__status__)
                @info "[stan] leaving warmup_fit"
                rv
            end

            # All posterior-fit derivations (pathfinder draws + warmup draws +
            # their long/wide/summary DFs and constrained-with-TP/GQ variants).
            # Resolves the @error-level "17 posterior_* siblings + bare
            # posterior" lint by giving every member a home inside `@include
            # posterior`. The nested `@include warmup` further bundles the
            # warmup-specific subset (8 props sharing the warmup_* prefix).
            @struct posterior = begin
                # Gaussian-approximation draws from Pathfinder. Reads the IP
                # via `@memo!` so if `compute_steps` already computed it with
                # progress nesting, we get the cached value for free.
                pathfinder = begin
                    pf = @memo! __parent__.pathfinder(__parent__.fit.instance, __parent__.init)
                    pf.position .+ pf.scale * randn(Xoshiro(43), __parent__.dim, 200)
                end
                # Default value. Switch the alias to `warmup.draws` to promote
                # the full fit, or expose a toggle via a @param later.
                value = pathfinder
                # For fit plots we only want the unconstrained parameters (the
                # raw sampler output) — constrained/TP/GQ adds lots of derived
                # noise (y_gen, y_likelihood) that drowns out the actual
                # parameter posterior. Keep prior-predictive plots on the full
                # constrained set; fit plots use `param_unc_names` here.
                unc_names = BridgeStan.param_unc_names(__parent__.fit.instance)
                dfs       = dfs_from_constrained(value, unc_names)
                long_df    = dfs.long
                wide_df    = dfs.wide
                summary_df = dfs.summary
                # Constrained-with-TP/GQ variant — needed for PPCs that
                # consume the linear-predictor TP (`loc` / `log_rate` /
                # `log_odds`). The base long_df above intentionally excludes
                # TPs/GQs to keep the generic plot tabset focused on raw
                # parameters.
                constrained_full = constrain_draws(value, __parent__.fit.instance; rng_seed=46)
                full_long_df = dfs_from_constrained(
                    constrained_full,
                    BridgeStan.param_names(__parent__.fit.instance; include_tp=true, include_gq=true),
                ).long

                @struct warmup = begin
                    # Parallel set for the warmup+MCMC path. The IP cache is
                    # warmed by `fetchindex!` in `compute_steps`; `@memo!`
                    # hits it here.
                    draws = (@memo! __parent__.__parent__.warmup_fit(__parent__.__parent__.fit.instance, __parent__.__parent__.init)).posterior_position
                    dfs   = dfs_from_constrained(draws, __parent__.unc_names)
                    long_df    = dfs.long
                    wide_df    = dfs.wide
                    summary_df = dfs.summary
                    diagnostics = begin
                        w = @memo! __parent__.__parent__.warmup_fit(__parent__.__parent__.fit.instance, __parent__.__parent__.init)
                        (; w.n_divergent_samples, ess=w.ess)
                    end
                    constrained_full = constrain_draws(draws, __parent__.__parent__.fit.instance; rng_seed=47)
                    full_long_df = dfs_from_constrained(
                        constrained_full,
                        BridgeStan.param_names(__parent__.__parent__.fit.instance; include_tp=true, include_gq=true),
                    ).long
                end
            end
        end

        # Stage-named aliases. `step_chain` / `compute_steps` extract step
        # outputs by looking up these names via `getproperty`.
        parse     = formula.raw
        transform = (; formula.transformed, formula.alllocals)
        wrap      = wrapped
    end

    # Per-request bundle for a (label, formula) pair. Pure construction; the
    # routes side handles persistence before invoking this.
    @struct context(label, formula) = begin
        namespace = namespace_from(label)
        run = __parent__.run(formula, namespace)
        sb_repro_html = begin
            compile_out = try
                run.stan.lib
                "(compile succeeded -- lib at `$(run.stan.lib)`)"
            catch e
                sprint(showerror, e)
            end
            h.div(
                h.h1("StanBlocks bug report"),
                h.h2("Formula"),
                h.pre(formula),
                h.h2("SlicModel body"),
                h.p(h.code("r.sbbrmi.model.model")),
                h.pre(run.sbbrmi.model.model),
                h.h2("Generated Stan source"),
                h.p(h.code("r.stan.src")),
                h.pre(run.stan.src),
                h.h2("BridgeStan compile output"),
                h.p(h.code("r.stan.lib")),
                h.pre(compile_out),
            )
        end
    end

    # Expanded DAG of step chains — one NamedTuple per stage target, keyed by
    # step names in dependency order. Built incrementally via `merge`: each
    # stage's chain is its parent's chain plus the one step it adds. Fork
    # points are visible in the `merge` calls (e.g. `slic_model = merge(brmi,
    # …)` forks off of `brmi`, parallel to `vbrmi`). Inner values are one of:
    #   - Symbol                       → `r.<sym>` (top-level property)
    #   - Tuple{Vararg{Symbol}}        → nested access, e.g. `(:stan, :src)` → `r.stan.src`
    #   - NamedTuple of (Symbol|Tuple) → bundle, keys preserved in the result.
    # Top-level key = stage name. Callers either iterate (stage list, indicators)
    # or index by name (`step_chain(:brmi)`).
    step_chain = begin
        parse            = (; parse=:parse)
        transform        = merge(parse,      (; transform=:transform))
        wrap             = merge(transform,  (; wrap=:wrap))
        brmi             = merge(wrap,       (; brmi=:brmi))
        vbrmi            = merge(brmi,       (; vbrmi=(; vbrmi=:vbrmi, dim=:dim, ldp=:ldp, grad=:grad, n_dead=:n_dead, dead=:dead)))
        bench            = merge(vbrmi,      (; bench=:benches))
        slic_model       = merge(brmi,       (; slic_model=:sbbrmi))
        stan_code        = merge(slic_model, (; stan_code=(:stan, :src)))
        stan_compile     = merge(stan_code,  (; stan_compile=(; file=(:stan, :file), lib=(:stan, :lib))))
        stan_instantiate = merge(stan_compile, (; stan_instantiate=(; instance=(:stan, :instance), dim=(:stan, :dim), init=(:stan, :init))))
        stan_eval        = merge(stan_instantiate, (; stan_eval=(:stan, :log_density)))
        stan_shapes      = merge(stan_eval,  (; stan_shapes=(:stan, :param_shapes_df)))
        stan_generate    = merge(stan_shapes, (; stan_generate=(; long=(:stan, :generated, :df), wide=(:stan, :generated, :wide_df), summary=(:stan, :generated, :summary_df), truth=(:stan, :truth, :df))))
        # Pathfinder / full warmup are computed via `fetchindex!` in
        # `compute_steps` (special-cased below by step name) so the IP's
        # progress subtree attaches to the step's phase. Chain-level specs
        # read the resulting cached values back out as plain properties.
        stan_fit_pathfinder = merge(stan_generate, (; stan_fit_pathfinder=(; long=(:stan, :posterior, :long_df), wide=(:stan, :posterior, :wide_df), summary=(:stan, :posterior, :summary_df), full_long=(:stan, :posterior, :full_long_df), truth=(:stan, :truth, :unc_df))))
        stan_fit_warmup     = merge(stan_fit_pathfinder, (; stan_fit_warmup=(; long=(:stan, :posterior, :warmup, :long_df), wide=(:stan, :posterior, :warmup, :wide_df), summary=(:stan, :posterior, :warmup, :summary_df), full_long=(:stan, :posterior, :warmup, :full_long_df), diagnostics=(:stan, :posterior, :warmup, :diagnostics), truth=(:stan, :truth, :unc_df))))
        (; parse, transform, wrap, brmi, vbrmi, bench, slic_model, stan_code, stan_compile,
           stan_instantiate, stan_eval, stan_shapes, stan_generate, stan_fit_pathfinder, stan_fit_warmup)
    end

    # Fetch target for `polling_fetchindex`. Pre-enumerates each step in the
    # requested chain as a pending progress child (so the whole pipeline is
    # visible up front as dim "pending" nodes), then runs each step under its
    # phase — `with_prepared_progress` handles start/finalize/fail around the
    # property access. Heavy work runs in polling_fetchindex's background task
    # via DO's lazy property cascading. Returns a NamedTuple keyed by step
    # names plus `:data` (the synthetic-data frame for the pipeline top pin).
    "Pipeline($name)"
    compute_steps(text, namespace, name::Symbol) = begin
        r = run(text, namespace)
        chain = step_chain[name]
        phases = [prepare_progress!(__status__; description=string(k)) for k in keys(chain)]
        # Resolve each spec against `r`:
        #   Symbol          → `r.<sym>`
        #   Tuple of Symbol → nested path `r.<a>.<b>...`
        #   NamedTuple      → bundle, recurse per value.
        resolve(s::Symbol) = getproperty(r, s)
        resolve(p::Tuple{Vararg{Symbol}}) = foldl(getproperty, p; init=r)
        resolve(b::NamedTuple) = map(resolve, b)
        vals = map(pairs(chain), phases) do (step_name, spec), phase
            with_prepared_progress(phase) do progress
                if step_name === :stan_fit_pathfinder
                    # Warm the pathfinder IP cache under this phase's progress,
                    # then resolve the (long, wide, summary) bundle (which
                    # reads back the cached value via `@memo!`).
                    fetchindex!(progress, r.stan.pathfinder, r.stan.fit.instance, r.stan.init)
                    resolve(spec)
                elseif step_name === :stan_fit_warmup
                    # Warm the warmup IP cache under this phase's progress,
                    # then resolve the (long, wide, summary, diagnostics)
                    # bundle (which reads back the cached value via `@memo!`).
                    fetchindex!(progress, r.stan.warmup_fit, r.stan.fit.instance, r.stan.init)
                    resolve(spec)
                else
                    resolve(spec)
                end
            end
        end
        # Stages render most-recent-first; synthetic data pinned at the top.
        merge((; data=r.df),
              NamedTuple{reverse(keys(chain))}(Tuple(reverse(vals))))
    end

    """
    `record_gallery(record_dir, record_base)` — IP. Drives every gallery
    URL through `compute_steps` to terminal state under per-path
    progress phases, then dumps the rendered HTML for each via
    `HTMXObjects.record!` into `record_dir`. Cached by
    `(record_dir, record_base)`; re-runs on `force=true` from the
    `polling_fetchindex` caller.
    """
    record_gallery(record_dir::String, record_base::String) = begin
        # Build the per-item path list. Same iteration as the gallery
        # view; recording covers shell + per-card endpoints. Confidential
        # client-project examples (`<prefix>-*`) are skipped when their
        # gitignored `<prefix>-ext.jl` didn't load -- e.g. the docs CI lacks
        # the ext, so the dataset extras would fail to load and
        # `compute_steps` would die. Loaded prefixes come from
        # `CONFIDENTIAL_EXT_PREFIXES`, so no project name is hardcoded here.
        examples_dir_local = examples_dir
        loaded_ext_prefixes = CONFIDENTIAL_EXT_PREFIXES
        items = let xs = []
            isdir(examples_dir_local) || mkpath(examples_dir_local)
            files = sort(filter(endswith(".jl"),
                                readdir(examples_dir_local; join=true));
                         by=mtime, rev=true)
            for path in files
                slug = replace(basename(path), r"\.jl$" => "")
                _g = findfirst(p -> startswith(slug, "$p-"), loaded_ext_prefixes)
                (_g !== nothing &&
                 !isfile(joinpath(dirname(@__DIR__), "src",
                                  "$(loaded_ext_prefixes[_g])-ext.jl"))) && continue
                lines = readlines(path)
                header = Dict{String,String}()
                i = 1
                while i <= length(lines)
                    m = match(r"^# (\w+):\s*(.*)$", lines[i])
                    m === nothing && break
                    header[m[1]] = m[2]
                    i += 1
                end
                # Skip headers' `#= ... =#` markdown body
                if i <= length(lines) && strip(lines[i]) == "#="
                    i += 1
                    while i <= length(lines) && strip(lines[i]) != "=#"
                        i += 1
                    end
                    i <= length(lines) && (i += 1)
                end
                formula_text = strip(join(lines[i:end], '\n'))
                isempty(formula_text) && continue
                label = get(header, "label", basename(path))
                push!(xs, (; slug, label, formula=String(formula_text)))
            end
            xs
        end

        # Static paths -- gallery shell, library, plus per-card content.
        static_paths = ["/pipeline/gallery", "/library"]
        card_paths   = ["/pipeline/gallery/card/$(it.slug)" for it in items]
        all_paths    = vcat(static_paths, card_paths)
        phases = [prepare_progress!(__status__; description=p) for p in all_paths]

        # Pre-warm compute_steps for each item so that when record! hits
        # the URL the response is the cached terminal HTML rather than a
        # polling fragment. Lookup uses the same cache key as the route.
        for (it, phase) in zip(items, phases[length(static_paths)+1:end])
            with_prepared_progress(phase) do _
                ns = context(it.label, it.formula).namespace
                fetchindex!(__status__, compute_steps,
                            it.formula, ns, :stan_fit_pathfinder)
            end
        end

        isdir(record_dir) || mkpath(record_dir)
        # `HTMXObjects.record!` re-`route!`s the app with recording closures;
        # the live registration is clobbered while this loop runs, so
        # restore via plain `route!(app)` in `finally`. Fresh AppContext
        # instance per record run -- each is per-request anyway.
        app = AppContext()
        HTMXObjects.route!(app; record_dir, record_base)
        router = HTMXObjects.ROUTER
        try
            for (path, phase) in zip(static_paths, phases[1:length(static_paths)])
                with_prepared_progress(phase) do _
                    HTMXObjects._drive_record_path(router, path, Pair{String,String}[])
                    HTMXObjects._drive_record_path(router, path, ["HX-Request" => "true"])
                end
            end
            # Card paths were progress-warmed above; this loop just dumps.
            for path in card_paths
                HTMXObjects._drive_record_path(router, path, Pair{String,String}[])
                HTMXObjects._drive_record_path(router, path, ["HX-Request" => "true"])
            end
        finally
            HTMXObjects.route!(app)
        end

        n_html = 0; n_other = 0
        for (root, _, fs) in walkdir(record_dir)
            for f in fs
                ext = lowercase(splitext(f)[2])
                ext == ".html" ? (n_html += 1) : (n_other += 1)
            end
        end
        (; n_html, n_other, n_paths=length(all_paths), n_items=length(items),
           record_dir, record_base)
    end
end

APPDATA = AppData()
