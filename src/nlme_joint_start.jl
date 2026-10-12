# Semantic NLME estimates -> a complete unconstrained Stan point for the same BRMI.
# Keep the NLME partition and the RK/SB pairing as the two sources of truth.

"""
    BRMNLMEJointMap
    brm_nlme_joint_map(rk::RKBRMI, sb::SBBRMI; path, kwargs...)

Prepare a bridge from natural subject effects and an NLME covariance to the
complete unconstrained coordinates of a compiled BridgeStan model. BRM
compiles `sb` itself at `path`, so the returned `stan_problem` is the exact
target whose coordinates are converted. Pass that problem to the sampler.
Both backends must lower the same BRMI instance. The NLME view must cover every RK coordinate,
and the semantic RK/Stan transport must pair every coordinate on both sides.
Shared latent variables may arrive in the fitted population vector or in an
explicit full packed `shared_point` from the RK model. Student-t mixing weights,
centered Stan random effects, and
other incomplete pairings are refused rather than silently initialized.
"""
struct BRMNLMEJointMap{R<:RKBRMI,S<:SBBRMI,P,M}
    rk::R
    sb::S
    stan_problem::P
    stan_model::M
    view::BRMNLMEView
    transport::BRMCoordinateTransport
    blocks::Vector{NamedTuple}
end

function brm_nlme_joint_map(rk::RKBRMI, sb::SBBRMI; path, kwargs...)
    parent(rk) === parent(sb) || error(
        "brm_nlme_joint_map: build both backends from the same BRMI instance")
    view = brm_nlme_view(rk)
    any(==(:subject_mixing), view.role) && _brm_nlme_error(
        "brm_nlme_joint_map: Student-t mixing coordinates are not natural " *
        "Gaussian subject effects")
    known = (:population, :scalar, :subject_scale, :subject_correlation,
        :subject_effect, :shared_latent)
    extra = [(view.coordinates[i], role) for (i, role) in enumerate(view.role)
        if role ∉ known]
    isempty(extra) || _brm_nlme_error(
        "brm_nlme_joint_map: coordinates $(extra) require validated values " *
        "outside theta, sigma, omega and eta")
    problem = stan_instantiate(sb; path, kwargs...)
    stan_model = problem.model
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan_model))
    blocks = NamedTuple[]
    for record in _brm_nlme_records(rk.plan)
        record.kind === :ranef || continue
        key = (; group=record.group, id=record.id)
        indices = findall(m -> isequal(m.block, key), view.block_margins)
        length(indices) == length(record.margins) &&
            [(view.block_margins[i].predictor, view.block_margins[i].coefficient)
                for i in indices] == collect(record.margins) || _brm_nlme_error(
            "brm_nlme_joint_map: random-effect block $key does not match the " *
            "NLME margin order")
        scales = BRMCoordinatePair[]
        for margin in record.margins
            matches = [p for p in transport.pairs
                if p.address.kind === :ranef_scale &&
                   p.address.group === record.group && isequal(p.address.id, record.id) &&
                   p.address.margin == margin]
            length(matches) == 1 || _brm_nlme_error(
                "brm_nlme_joint_map: random-effect scale for $key / $margin " *
                "has $(length(matches)) semantic Stan matches")
            push!(scales, only(matches))
        end
        factors = [f for f in transport.correlations
            if f.group === record.group && isequal(f.id, record.id)]
        length(factors) == (isnothing(record.L) ? 0 : 1) || _brm_nlme_error(
            "brm_nlme_joint_map: correlation factor for $key is not paired")
        push!(blocks, (; record, indices, scales,
            factor=isempty(factors) ? nothing : only(factors)))
    end
    length(blocks) == length(unique(m.block for m in view.block_margins)) ||
        _brm_nlme_error("brm_nlme_joint_map: NLME random-effect blocks are not unique")
    BRMNLMEJointMap(rk, sb, problem, stan_model, view, transport, blocks)
end

function _brm_nlme_joint_subject_order(view::BRMNLMEView, subjects)
    length(subjects) == length(view.levels) || throw(DimensionMismatch(
        "brm_nlme_joint_start: $(length(subjects)) subject labels for " *
        "$(length(view.levels)) model levels"))
    order = Int[]
    for level in view.levels
        matches = findall(s -> isequal(s, level), subjects)
        length(matches) == 1 || _brm_nlme_error(
            "brm_nlme_joint_start: model level $(repr(level)) occurs " *
            "$(length(matches)) times in the supplied subject labels")
        push!(order, only(matches))
    end
    order
end

function _brm_nlme_joint_scale(values, record, K)
    scales = if isnothing(record.scales)
        value = _brm_rk_declaration_value(values, record.scale)
        value isa Number ? [value] : vec(collect(value))
    else
        [_brm_rk_declaration_value(values, name) for name in record.scales]
    end
    length(scales) == K || _brm_nlme_error(
        "brm_nlme_joint_values: random-effect scale has $(length(scales)) " *
        "values for $K margins")
    all(x -> x isa Real && isfinite(x) && x > 0, scales) || _brm_nlme_error(
        "brm_nlme_joint_values: a random-effect scale is not finite and positive")
    Float64.(scales)
end

"""
    brm_nlme_joint_values(map::BRMNLMEJointMap, stan_u)

Read a complete Stan unconstrained point back as
`(; theta, sigma, omega, eta, subjects, shared)`. `theta` contains ordinary
population coordinates; `shared` contains the names and unconstrained values
of shared latent coordinates in RK order, whether they were fitted or fixed.
`omega` is the covariance matrix
of natural effects; columns of `eta` follow `subjects` in the NLME view's
grouping-level order. The inverse uses the validated semantic transport and
the actual RK constrained scale/correlation values.
"""
function brm_nlme_joint_values(map::BRMNLMEJointMap, stan_u::AbstractVector)
    t, view = map.transport, map.view
    length(stan_u) == length(t.stan_names) || throw(DimensionMismatch(
        "brm_nlme_joint_values: $(length(stan_u)) Stan coordinates for " *
        "$(length(t.stan_names)) compiled parameters"))
    all(isfinite, stan_u) || _brm_nlme_error(
        "brm_nlme_joint_values: Stan coordinates must be finite")
    u = brm_stan_point_to_rk(t, stan_u)
    values = _rk_constrained_values(map.rk, u)
    nη = length(view.block_margins)
    omega = zeros(Float64, nη, nη)
    eta = zeros(Float64, nη, length(view.levels))
    for block in map.blocks
        (; record, indices) = block
        K = length(indices)
        scale = _brm_nlme_joint_scale(values, record, K)
        L = isnothing(record.L) ? Matrix{Float64}(I, K, K) :
            Matrix{Float64}(_brm_rk_declaration_value(values, record.L))
        size(L) == (K, K) || _brm_nlme_error(
            "brm_nlme_joint_values: correlation factor has size $(size(L)), " *
            "expected ($K, $K)")
        A = Diagonal(scale) * L
        omega[indices, indices] = A * A'
        for i in eachindex(view.levels)
            z = u[view.subject_coordinates[i][indices]]
            eta[indices, i] = A * z
        end
    end
    shared = findall(==(:shared_latent), view.role)
    (; theta=u[findall(==(:population), view.role)],
       sigma=u[findall(==(:scalar), view.role)], omega, eta,
       subjects=copy(view.levels),
       shared=(; names=view.coordinates[shared], values=u[shared]))
end

"""
    brm_nlme_joint_start(map::BRMNLMEJointMap;
        theta, sigma, omega, eta, subjects, shared_point=nothing)

Convert one NLME estimate into the *same model's* complete BridgeStan
`init_positions`. `theta` and `sigma` are the NLME model's unconstrained
vectors; `omega` is its natural-effect covariance matrix; `eta[:, j]` is
the natural effect of the explicitly named `subjects[j]`. Subject columns
may be reordered, but every model level must occur exactly once. Covariance
between independent NLME blocks must be zero.
If the NLME model estimated shared latent coordinates, pass its complete `theta`
vector in the model's packed order, including those coordinates. For a fit that
held them fixed, pass the shorter population `theta` and the same full packed
`shared_point` supplied to `brm_nlme_model`. Only the shared entries of that
point are copied. The two forms cannot be combined.

Returns `(; stan, fixed, free, checked)`. `stan` is the full unconstrained
vector. `fixed` names and indexes the population, scalar, shared and covariance
coordinates; `free` names and indexes standardized subject coordinates.
Both sets include their values and partition the full vector. `checked`
is BRM's physical RK/Stan coordinate comparison. The function also inverts
the constructed point and checks the supplied theta, sigma, omega, eta and
shared values.
No caller input is mutated.
"""
function brm_nlme_joint_start(map::BRMNLMEJointMap; theta, sigma, omega, eta,
        subjects, shared_point=nothing)
    t, view = map.transport, map.view
    θ = Vector{Float64}(theta)
    σ = Vector{Float64}(sigma)
    Ω = Matrix{Float64}(omega)
    H = Matrix{Float64}(eta)
    p = findall(==(:population), view.role)
    s = findall(==(:scalar), view.role)
    shared = findall(==(:shared_latent), view.role)
    estimated = sort!(vcat(p, shared))
    fitted_shared = !isempty(shared) && length(θ) == length(estimated)
    length(θ) in (length(p), length(estimated)) || throw(DimensionMismatch(
        "brm_nlme_joint_start: theta has $(length(θ)) entries, expected " *
        "$(length(p)) with shared_point or $(length(estimated)) with estimated shared coordinates"))
    fitted_shared && shared_point !== nothing && _brm_nlme_error(
        "brm_nlme_joint_start: estimated shared coordinates in theta and " *
        "shared_point cannot both supply the same values")
    length(σ) == length(s) || throw(DimensionMismatch(
        "brm_nlme_joint_start: sigma has $(length(σ)) entries, expected $(length(s))"))
    nη, nsubjects = length(view.block_margins), length(view.levels)
    size(Ω) == (nη, nη) || throw(DimensionMismatch(
        "brm_nlme_joint_start: omega has size $(size(Ω)), expected ($nη, $nη)"))
    size(H) == (nη, nsubjects) || throw(DimensionMismatch(
        "brm_nlme_joint_start: eta has size $(size(H)), expected ($nη, $nsubjects)"))
    all(isfinite, θ) && all(isfinite, σ) && all(isfinite, Ω) && all(isfinite, H) ||
        _brm_nlme_error("brm_nlme_joint_start: all estimate coordinates must be finite")
    issymmetric(Ω) || _brm_nlme_error(
        "brm_nlme_joint_start: omega must be symmetric")
    order = _brm_nlme_joint_subject_order(view, subjects)
    u = zeros(Float64, length(view.coordinates))
    if !isempty(shared) && !fitted_shared
        shared_point === nothing && _brm_nlme_error(
            "brm_nlme_joint_start: $(length(shared)) shared latent coordinate(s) " *
            "need `shared_point`, a full packed unconstrained point from this RK model")
        length(shared_point) == length(u) || throw(DimensionMismatch(
            "brm_nlme_joint_start: shared_point has length $(length(shared_point)), " *
            "expected $(length(u))"))
        u[shared] .= shared_point[shared]
        all(isfinite, u[shared]) || _brm_nlme_error(
            "brm_nlme_joint_start: shared_point contains a nonfinite shared coordinate")
    end
    u[fitted_shared ? estimated : p] = θ
    u[s] = σ
    # The compiled model supplies the inverse transform for its correlation
    # factor. Start from its valid unit point, then replace only the constrained
    # covariance carriers and invert the whole physical parameter vector.
    covariance = NamedTuple[]
    used = falses(nη)
    for block in map.blocks
        (; indices) = block
        used[indices] .= true
        other = setdiff(1:nη, indices)
        all(iszero, Ω[indices, other]) || _brm_nlme_error(
            "brm_nlme_joint_start: omega couples independent NLME blocks")
        localΩ = Matrix(Ω[indices, indices])
        factor = cholesky(Symmetric(localΩ); check=false)
        issuccess(factor) || _brm_nlme_error(
            "brm_nlme_joint_start: an omega block is not positive definite")
        A = Matrix(factor.L)
        scales = sqrt.(diag(localΩ))
        Lcorr = Diagonal(1 ./ scales) * A
        for (i, j) in enumerate(order)
            u[view.subject_coordinates[i][indices]] = A \ H[indices, j]
        end
        push!(covariance, (; block, scales, Lcorr))
    end
    all(used) || _brm_nlme_error(
        "brm_nlme_joint_start: an NLME subject margin has no covariance block")
    stan0 = brm_rk_point_to_stan(t, u)
    constrained = BridgeStan.param_constrain(map.stan_model, stan0)
    names = String.(BridgeStan.param_names(map.stan_model))
    pos = Dict(name => i for (i, name) in enumerate(names))
    length(pos) == length(names) || _brm_nlme_error(
        "brm_nlme_joint_start: Stan constrained parameter names are not unique")
    function put!(name, value)
        i = get(pos, name, nothing)
        i === nothing && _brm_nlme_error(
            "brm_nlme_joint_start: Stan constrained coordinate `$name` is absent")
        constrained[i] = value
    end
    for (; block, scales, Lcorr) in covariance
        for (pair, scale) in zip(block.scales, scales)
            pair.relation in (:identity, :exp) || _brm_nlme_error(
                "brm_nlme_joint_start: unsupported scale relation $(pair.relation)")
            put!(pair.stan_value, pair.relation === :exp ? log(scale) : scale)
        end
        isnothing(block.factor) && continue
        carrier = block.factor.stan_carrier
        for i in axes(Lcorr, 1), j in axes(Lcorr, 2)
            put!(string(carrier, ".", i, ".", j), Lcorr[i, j])
        end
    end
    stan = BridgeStan.param_unconstrain(map.stan_model, constrained)
    roundtrip = brm_nlme_joint_values(map, stan)
    isapprox(roundtrip.theta, u[p]; atol=1e-9, rtol=1e-9) &&
        isapprox(roundtrip.sigma, σ; atol=1e-9, rtol=1e-9) &&
        isapprox(roundtrip.omega, Ω; atol=1e-9, rtol=1e-9) &&
        isapprox(roundtrip.eta, H[:, order]; atol=1e-9, rtol=1e-9) &&
        isapprox(roundtrip.shared.values, u[shared]; atol=1e-9, rtol=1e-9) ||
        _brm_nlme_error("brm_nlme_joint_start: NLME-to-Stan round trip disagrees")
    checked = brm_check_coordinate_transport(t, map.rk, map.stan_model,
        brm_stan_point_to_rk(t, stan))
    fixed_indices = Int[]
    free_indices = Int[]
    for (j, k) in enumerate(t.permutation)
        if view.role[k] === :subject_effect
            push!(free_indices, j)
        else
            push!(fixed_indices, j)
        end
    end
    coords(indices) = (; indices, names=t.stan_names[indices], values=stan[indices])
    (; stan, fixed=coords(fixed_indices), free=coords(free_indices), checked)
end
