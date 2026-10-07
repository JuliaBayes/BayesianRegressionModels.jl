# Random-effect R2D2M2 emission consumes the same resolved allocation as Stan.
# Only the target-language emission differs: every budget, reference, empirical
# variance and derived scale is an explicit ordinary RKPPL graph quantity.
function _rk_ast_r2d2m2_scale!(definitions, taken, reference, phi, r2, share,
        variance=1.0)
    _rk_ast_statistical_call!(definitions, taken, :brm_r2d2m2_scale,
        reference, phi, r2, share, variance; kernel=true)
end

function _rk_ast_r2d2m2_variance!(statements, taken, stem, column)
    center = _rk_ast_fresh_name(string(stem, "_centered"), taken)
    variance = _rk_ast_fresh_name(string(stem, "_variance"), taken)
    n = Expr(:call, :length, column)
    mean = Expr(:call, :/, Expr(:call, :sum, column), n)
    push!(statements, Expr(:(=), center, Expr(:call, :.-, column, mean)))
    push!(statements, Expr(:(=), variance,
        Expr(:call, :/, Expr(:call, :sum, Expr(:call, :.*, center, center)),
            Expr(:call, :-, n, 1))))
    variance
end

# A named factor-level indicator, read several times by its variance.
function _rk_ast_r2d2m2_indicator!(definitions, statements, taken, stem, term, position)
    indicator = _rk_ast_fresh_name(string(stem, "_indicator"), taken)
    push!(statements, Expr(:(=), indicator,
        _rk_ast_factor_indicator!(definitions, taken, term, position)))
    indicator
end

function _rk_ast_r2d2m2_population!(definitions, statements, taken, predictors, joint,
        references, phi, r2, population_priors)
    for (target, allocation) in joint.predictors
        predictor = only(p for p in predictors if p.name === target)
        reference = references[allocation.ref_index]
        design_scales = Dict{Symbol,Vector{Union{Nothing,Symbol}}}()
        for (index, label) in enumerate(allocation.labels)
            share = allocation.share_idx[index]
            share == 0 && continue
            term = only(t for t in predictor.terms if t.addressee === label ||
                (haskey(t.options, :labels) && label in t.options.labels))
            position = term.kind === :factor ?
                only(findall(==(label), term.options.labels)) : nothing
            stem = string(target, "_", label, "_r2d2")
            column = position === nothing ? only(term.columns) :
                _rk_ast_r2d2m2_indicator!(definitions, statements, taken, stem, term, position)
            variance = _rk_ast_r2d2m2_variance!(statements, taken, stem, column)
            scale = _rk_ast_fresh_name(string(target, "_", label, "_r2d2_scale"), taken)
            push!(statements, Expr(:(=), scale,
                _rk_ast_r2d2m2_scale!(definitions, taken, reference, phi, r2, share, variance)))
            if position === nothing
                population_priors[(target, term.addressee)] = (:Normal, (0.0, scale))
            else
                scales = get!(design_scales, term.addressee) do
                    Union{Nothing,Symbol}[nothing for _ in term.options.labels]
                end
                scales[position] = scale
            end
        end
        for (address, scales) in design_scales
            any(isnothing, scales) && error(
                "RK backend: R2D2 allocation omits a factor design column")
            population_priors[(target, address)] = (:Normal, (0.0, Expr(:vect, scales...)))
        end
        for contrast in allocation.cats
            contrast.n_contrasts == 0 && continue
            term = only(t for t in predictor.terms if t.kind === :factor &&
                t.addressee === contrast.address)
            columns = if haskey(term.options, :index)
                [_rk_ast_r2d2m2_indicator!(definitions, statements, taken,
                    string(target, "_", contrast.address, "_r2d2_", position), term, position)
                    for position in eachindex(term.options.labels)]
            else
                col = only(term.columns)
                K = contrast.n_contrasts + 1
                positions = [i for i in 1:K if i != term.options.drop]
                [Expr(:call, :brm_dummy, col,
                    Expr(:ref, Expr(:call, :levels, col), i)) for i in positions]
            end
            length(columns) == contrast.n_contrasts || error(
                "RK backend: R2D2 contrast order differs from the shared allocation")
            scales = Symbol[]
            for (index, column) in enumerate(columns)
                stem = string(target, "_", contrast.address, "_r2d2_", index)
                variance = _rk_ast_r2d2m2_variance!(statements, taken, stem, column)
                scale = _rk_ast_fresh_name(string(stem, "_scale"), taken)
                push!(statements, Expr(:(=), scale,
                    _rk_ast_r2d2m2_scale!(definitions, taken, reference, phi, r2,
                        contrast.phi_start + index, variance)))
                push!(scales, scale)
            end
            population_priors[(target, term.addressee)] =
                (:Normal, (0.0, Expr(:vect, scales...)))
        end
    end
end

# Budget names follow SBBRMI's `<draws>_r2d2_<group>_R2` / `_phi` spelling;
# `tau` names the derived scale vector the group component receives.
function _rk_ast_ranef_r2d2!(definitions, statements, bucket, tau, bindings, taken,
        predictors, population_priors; stem_base=tau)
    decomposition = bucket.decomposition
    scales = Any[nothing for _ in bucket.margins]
    for (index, group) in enumerate(decomposition.groups)
        stem = string(stem_base, "_r2d2_", index)
        r2 = _rk_ast_fresh_name(string(stem, "_R2"), taken)
        phi = _rk_ast_fresh_name(string(stem, "_phi"), taken)
        prior = _rk_ast_declared_prior(group.r2_prior, bindings, taken)
        push!(statements, Expr(:call, :~, r2,
            Expr(:call, :restricted, prior, 0.0, 1.0)))
        n = length(group.indices) + group.n_extra
        push!(statements, Expr(:call, :~, phi,
            Expr(:call, :Dirichlet, Expr(:vect, fill(group.alpha, n)...))))
        references = Dict{Int,Any}()
        for (local_index, margin) in enumerate(group.indices)
            reference = group.references[local_index]
            if reference === nothing
                reference = _rk_ast_fresh_name(string(stem, "_ref_", margin), taken)
                push!(statements, Expr(:call, :~, reference,
                    _rk_ast_positive_prior(nothing, bindings, taken)))
            end
            references[margin] = reference
            scale = _rk_ast_fresh_name(string(stem, "_sd_", margin), taken)
            push!(statements, Expr(:(=), scale,
                _rk_ast_r2d2m2_scale!(definitions, taken, reference, phi, r2, local_index)))
            scales[margin] = scale
        end
        decomposition.joint === nothing ||
            _rk_ast_r2d2m2_population!(definitions, statements, taken, predictors,
                decomposition.joint, references, phi, r2, population_priors)
    end
    # Partial ICC allocations retain a sampled Normal kernel on each
    # unclaimed positive scale; no unused SD parameters are introduced.
    for margin in eachindex(scales)
        scales[margin] === nothing || continue
        free = _rk_ast_fresh_name(string(stem_base, "_r2d2_free_tau_", margin), taken)
        push!(statements, Expr(:call, :~, free,
            _rk_ast_positive_prior(nothing, bindings, taken)))
        scales[margin] = free
    end
    push!(statements, Expr(:(=), tau, Expr(:vect, scales...)))
end
