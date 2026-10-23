# test/car_term.jl — semantic and executable contract for the areal term
# `car(area; edges=(from, to), type=:icar | :bym2)`: an intrinsic CAR field over
# an adjacency graph, alone or in the BYM2 convolution.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra: pinv, diag
using Distributions: Normal, Beta, Uniform, logpdf
import StanBlocks.stan: transpiles
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
import ReactiveKernelsPPLExamples

const CAR_RUNTIME = get(ENV, "BRM_CAR_RUNTIME", "1") != "0"
const CAR_CACHE = joinpath(tempdir(), "brm-car-term")
const BS = StanBlocks.BridgeStan

# Areas "a".."h". Component {a, b, c, d, h}: a–b, b–c, c–d, a–c, d–h; component
# {e, f}: e–f; "g" is an island (rows, no neighbours); "h" has no rows.
const EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
car_df(; area=["a", "b", "c", "d", "e", "f", "g", "a", "e"], y=zeros(length(area)),
       from=EDGES[1], to=EDGES[2]) = (; area, y, from, to)

function car_model(df=car_df(); type=:bym2)
    type === :bym2 ? (@brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        effect(mu, Intercept) ~ Normal(0, 1)
        y ~ Normal(mu, 1.0)
    end) : (@brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar)
        effect(mu, Intercept) ~ Normal(0, 1)
        y ~ Normal(mu, 1.0)
    end)
end

stan(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=@__MODULE__).model)

# Independent scaling factor: geometric mean of diag(pinv(Q)) of one component.
function laplacian(n, pairs)
    Q = zeros(n, n)
    for (i, j) in pairs
        Q[i, i] += 1; Q[j, j] += 1; Q[i, j] -= 1; Q[j, i] -= 1
    end
    Q
end
geomean(x) = exp(sum(log, x) / length(x))

@testset "car is a direct areal term: one intercept, the graph as data" begin
    sb = SBBRMI(car_model(); mod=@__MODULE__)
    @test popcoefnames(car_model(), :mu) == [:Intercept]
    code = StanBlocks.stan_code(sb.model)
    for name in ("brm_icar_lpdf", "car_mu_area_sigma", "car_mu_area_rho",
                 "car_mu_area_phi", "car_mu_area_theta")
        @test occursin(name, code)
    end
    # sorted areas a..h = 1..8; structured areas a,b,c,d,h (component 1) then e,f
    @test sb.data[:car_mu_area_n_nodes] == 8
    @test sb.data[:car_mu_area_n_struct] == 7
    @test sb.data[:car_mu_area_struct_idx] == [1, 2, 3, 4, 6, 7, 0, 5]
    @test sb.data[:car_mu_area_comp_start] == [1, 6]
    @test sb.data[:car_mu_area_comp_size] == [5, 2]
    @test sb.data[:car_mu_area_node1] == [1, 2, 3, 1, 4, 6]
    @test sb.data[:car_mu_area_node2] == [2, 3, 4, 3, 5, 7]
    @test sb.data[:car_mu_area_row_idx] == [1, 2, 3, 4, 5, 6, 7, 1, 5]
    # scaling factors: the two-area component has pinv(Q) = Q / 4, so s = 1/4
    inv_scale = sb.data[:car_mu_area_inv_scale]
    @test inv_scale[[5, 6]] ≈ [4.0, 4.0]
    @test inv_scale[7] == 0.0
    big = geomean(diag(pinv(laplacian(5, [(1, 2), (2, 3), (3, 4), (1, 3), (4, 5)]))))
    @test inv_scale[[1, 2, 3, 4, 8]] ≈ fill(1 / big, 5)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "car type=:icar has no mixing share and no unstructured noise" begin
    icar = car_model(; type=:icar)
    sb = SBBRMI(icar; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("car_mu_area_phi", code)
    @test !occursin("car_mu_area_rho", code)
    @test !occursin("car_mu_area_theta", code)
    @test !haskey(sb.data, :car_mu_area_inv_scale)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    df = car_df()
    mixed = @brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar)
        mixing(:, car(area)) ~ Beta(2, 2)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "has none" SBBRMI(mixed; mod=@__MODULE__)
end

@testset "car scale and mixing share are addressable" begin
    df = car_df()
    configured = @brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        sd(:, car(area)) ~ Normal(0.0, 0.5)
        mixing(mu, car(area)) ~ Beta(2, 2)
        y ~ Normal(mu, 1.0)
    end
    specs = term_priors(configured)
    @test Set(s.class for s in specs) == Set((:term_sd, :term_mixing))
    @test all(s.term === Symbol("car(area)") for s in specs)
    code = stan(configured)
    @test occursin("real<lower=0.0> car_mu_area_sigma", code)
    @test occursin("car_mu_area_sigma ~ normal(0.0, 0.5)", code)
    @test occursin("real<lower=0.0, upper=1.0> car_mu_area_rho", code)
    @test occursin("car_mu_area_rho ~ beta(2", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test_throws "mixing share bounds" stan(@brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        mixing(:, car(area)) ~ Uniform(0.2, 1.5)
        y ~ Normal(mu, 1.0)
    end)
end

@testset "car labels: numeric codes compare by value, kinds must agree" begin
    ints = car_df(; area=[1, 2, 3, 4, 5, 6, 7, 1, 5],
                  from=[1, 2, 3, 1, 4, 5], to=[2, 3, 4, 3, 8, 6])
    floats = merge(ints, (; from=Float64.(ints.from), to=Float64.(ints.to)))
    a = SBBRMI(car_model(ints); mod=@__MODULE__)
    b = SBBRMI(car_model(floats); mod=@__MODULE__)
    for key in (:car_mu_area_node1, :car_mu_area_node2, :car_mu_area_struct_idx,
                :car_mu_area_row_idx, :car_mu_area_inv_scale)
        @test a.data[key] == b.data[key]
    end
    @test_throws "mix" SBBRMI(car_model(merge(ints, (; from=string.(ints.from)))); mod=@__MODULE__)
    @test_throws "not an integer code" SBBRMI(car_model(merge(floats, (; from=floats.from .+ 0.5))); mod=@__MODULE__)
end

@testset "car refusals name the problem" begin
    @test_throws "self-loop" SBBRMI(car_model(car_df(; from=["a", "a"], to=["b", "a"])); mod=@__MODULE__)
    @test_throws "listed twice" SBBRMI(car_model(car_df(; from=["a", "b"], to=["b", "a"])); mod=@__MODULE__)
    @test_throws "unequal lengths" SBBRMI(car_model(car_df(; from=["a", "b"], to=["b"])); mod=@__MODULE__)
    @test_throws "no edges" SBBRMI(car_model(car_df(; from=String[], to=String[])); mod=@__MODULE__)
    df = car_df()
    @test_throws "needs `type=`" SBBRMI((@brm df begin
        mu ~ 1 + car(area; edges=(from, to))
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "not supported" SBBRMI((@brm df begin
        mu ~ 1 + car(area; edges=(from, to), type=:leroux)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    @test_throws "needs `edges=`" SBBRMI((@brm df begin
        mu ~ 1 + car(area; type=:icar)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
end

@testset "car replay: frozen graph, rows of fitted areas only; fresh rebuilds" begin
    sb = SBBRMI(car_model(); mod=@__MODULE__)
    new = car_df(; area=["h", "g", "a"])   # h had no training rows
    frozen = reprocess(sb, new)
    @test frozen.data[:car_mu_area_row_idx] == [8, 7, 1]
    for key in (:car_mu_area_node1, :car_mu_area_struct_idx, :car_mu_area_inv_scale)
        @test frozen.data[key] == sb.data[key]
    end
    @test StanBlocks.stan_code(frozen.model) == StanBlocks.stan_code(sb.model)
    # the edge columns are not needed to replay a frozen graph
    @test reprocess(sb, (; area=["a"], y=[0.0])).data[:car_mu_area_row_idx] == [1]
    @test_throws "not an area of the fitted graph" reprocess(sb, car_df(; area=["z"]))
    fresh = reprocess(sb, car_df(; area=["a", "b"], from=["a"], to=["b"]);
                      freeze_constants=false)
    @test fresh.data[:car_mu_area_n_nodes] == 2
    @test fresh.data[:car_mu_area_inv_scale] ≈ [4.0, 4.0]
    d = brm_descriptor(sb)
    @test any(o -> o.logical === :car_mu_area, d.outputs)
    described = brm_description(d)
    text = brm_description_markdown(described)
    @test occursin("BYM2", text)
    @test occursin("intrinsic CAR", text)
    @test described.complete
end

# Julia oracle of the whole BYM2 model density (propto=false, Jacobian on).
function car_oracle(df, q; intercept, sigma, rho, theta, phi)
    nodes = sort(unique(vcat(df.area, df.from, df.to)))
    k(label) = findfirst(==(label), nodes)
    struct_nodes = ["a", "b", "c", "d", "h", "e", "f"]   # phi order
    comp = Dict("a" => 1, "b" => 1, "c" => 1, "d" => 1, "h" => 1, "e" => 2, "f" => 2)
    sizes = [5, 2]
    s1 = geomean(diag(pinv(laplacian(5, [(1, 2), (2, 3), (3, 4), (1, 3), (4, 5)]))))
    scale = [s1, 0.25]
    effect = Dict{String,Float64}()
    for (p, label) in enumerate(struct_nodes)
        effect[label] = sigma * (sqrt(rho / scale[comp[label]]) * phi[p] +
                                 sqrt(1 - rho) * theta[k(label)])
    end
    effect["g"] = sigma * theta[k("g")]
    pos = Dict(l => p for (p, l) in enumerate(struct_nodes))
    icar = -0.5 * sum((phi[pos[a]] - phi[pos[b]])^2 for (a, b) in zip(df.from, df.to))
    s2z = logpdf(Normal(0, 0.001 * 5), sum(phi[1:5])) + logpdf(Normal(0, 0.001 * 2), sum(phi[6:7]))
    prior = logpdf(Normal(0, 1), intercept) + logpdf(Normal(0, 1), sigma) + log(sigma) +
            logpdf(Beta(0.5, 0.5), rho) + log(rho * (1 - rho)) +
            sum(logpdf.(Normal(0, 1), theta)) + icar + s2z
    lik = sum(logpdf(Normal(intercept + effect[a], 1.0), y) for (a, y) in zip(df.area, df.y))
    prior + lik
end

instantiate(sb) = begin
    code = StanBlocks.stan_code(sb.model)
    isdir(CAR_CACHE) || mkpath(CAR_CACHE)
    StanBlocks.stan_instantiate(sb.model; path=joinpath(CAR_CACHE, string(hash(code)) * ".stan"))
end
index_of(names, name) = only(findall(==(name), names))
indices_of(names, prefix) = findall(startswith(prefix * "."), names)

@testset "car BridgeStan density equals the Julia oracle; coordinates resolve" begin
    if CAR_RUNTIME
        df = car_df(; y=[0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9])
        sb = SBBRMI(car_model(df); mod=@__MODULE__)
        problem = instantiate(sb)
        sm = problem.model
        names = String.(BS.param_names(sm))
        @test LogDensityProblems.dimension(problem) == 1 + 2 + 8 + 7
        q = zeros(LogDensityProblems.dimension(problem))
        theta = collect(range(-1.0, 1.0; length=8))
        phi = [0.4, -0.3, 0.2, 0.1, -0.4, 0.25, -0.25]
        q[index_of(names, "car_mu_area_sigma")] = log(0.7)
        q[index_of(names, "car_mu_area_rho")] = log(0.6 / 0.4)
        q[indices_of(names, "car_mu_area_theta")] = theta
        q[indices_of(names, "car_mu_area_phi")] = phi
        intercept_i = only(i for (i, n) in enumerate(names) if !startswith(n, "car_mu_area"))
        q[intercept_i] = 0.15
        lp = BS.log_density(sm, q; propto=false, jacobian=true)
        @test lp ≈ car_oracle(df, q; intercept=0.15, sigma=0.7, rho=0.6, theta, phi) atol=1e-8
        lp2, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp2) && all(isfinite, gradient)
        cnames = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        d = brm_descriptor(sb)
        coords(p) = brm_term_coordinates(d, :mu, cnames; term=:car_mu_area, parameter=p).coordinates
        @test length(coords(:sd)) == 1
        @test length(coords(:mixing)) == 1
        @test length(coords(:structured)) == 7
        @test length(coords(:unstructured)) == 8
        effects = constrained[coords(:effects)]
        @test length(effects) == 8
        @test effects[7] ≈ 0.7 * theta[7]                        # island g
        value = constrained[coords(:value)]
        @test value ≈ effects[sb.data[:car_mu_area_row_idx]]
    else
        @info "Skipping BridgeStan car runtime gate (BRM_CAR_RUNTIME=0)"
        @test true
    end
end

# The posteriordb BYM2 reference program (Morris et al. 2019; the S-sweep's
# `bym2_s`), written independently of BRM: one soft sum-to-zero, one scaling
# factor.
StanBlocks.@deffun begin
    @lpxf twin_icar_lpdf(phi::vector[N], node1::int[E], node2::int[E], s2z_scale::real)::real = begin
        acc = 0.0
        for e in 1:E
            d = phi[node1[e]] - phi[node2[e]]
            acc += d * d
        end
        -0.5 * acc + normal_lpdf(sum(phi), 0.0, s2z_scale)
    end
    twin_icar_lpdfs(phi::vector[N], node1::int[E], node2::int[E], s2z_scale::real)::real =
        twin_icar_lpdf(phi, node1, node2, s2z_scale)
    twin_icar_rng(vector[N], node1::int[E], node2::int[E], s2z_scale::real)::vector[N] = begin
        out::vector[N]
        for i in 1:N
            out[i] = normal_rng(0.0, 1.0)
        end
        out
    end
end

@testset "car BYM2 equals the posteriordb reference program on NYC traffic data" begin
    if CAR_RUNTIME
        M = ReactiveKernelsPPLExamples.Bym2OffsetOnlyExample
        N = length(M.BYM2_Y)
        nyc = (; area=collect(1:N), y=M.BYM2_Y, log_E=log.(M.BYM2_E),
                 from=M.BYM2_NODE1, to=M.BYM2_NODE2)
        brmi = @brm nyc begin
            log(lambda) ~ 1 + offset(log_E) + car(area; edges=(from, to), type=:bym2)
            effect(lambda, Intercept) ~ Normal(0, 1)
            y ~ Poisson(lambda)
        end
        sb = SBBRMI(brmi; mod=@__MODULE__)
        # the term is named after its linear predictor (`log_lambda` for a linked one)
        prefix = only(String(k)[1:end-length("_comp_size")] for k in keys(sb.data)
                      if endswith(String(k), "_comp_size"))
        @test startswith(prefix, "car_") && endswith(prefix, "_area")
        @test sb.data[Symbol(prefix, "_comp_size")] == [N]          # one connected graph
        @test sb.data[Symbol(prefix, "_struct_idx")] == 1:N
        scaling = 1 / first(sb.data[Symbol(prefix, "_inv_scale")])
        # BRM's scaling factor is the exact generalized-inverse value; posteriordb's
        # comes from the case study's perturbed inverse `inv(Q + 1e-8 max(D) I)`
        # conditioned on sum(phi) = 0, about 3e-5 lower on this graph.
        exact = geomean(diag(pinv(laplacian(N, zip(M.BYM2_NODE1, M.BYM2_NODE2)))))
        @test scaling ≈ exact rtol=1e-10
        @test scaling ≈ M.BYM2_SCALING_FACTOR rtol=1e-4
        twin = StanBlocks.@slic (; y=M.BYM2_Y, node1=M.BYM2_NODE1, node2=M.BYM2_NODE2,
                log_E=log.(M.BYM2_E), scaling_factor=scaling, s2z_scale=0.001 * N, N=N) begin
            beta0 ~ normal(0, 1)
            sigma ~ normal(0, 1; lower=0.0)
            rho ~ beta(0.5, 0.5; lower=0.0, upper=1.0)
            theta::vector[N] ~ normal(0, 1)
            phi::vector[N] ~ twin_icar(node1, node2, s2z_scale)
            convolved = sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi
            eta = log_E + beta0 + convolved * sigma
            y ~ poisson_log(eta)
        end
        isdir(CAR_CACHE) || mkpath(CAR_CACHE)
        tm = StanBlocks.stan_instantiate(twin; path=joinpath(CAR_CACHE,
            string(hash(StanBlocks.stan_code(twin))) * ".stan")).model
        bm = instantiate(sb).model
        tnames, bnames = String.(BS.param_names(tm)), String.(BS.param_names(bm))
        @test length(tnames) == length(bnames) == 2N + 3
        for point in 1:3
            theta = sin.(point .+ (1:N)) ./ 2
            phi = cos.(point .* (1:N)) ./ 3
            tq, bq = zeros(2N + 3), zeros(2N + 3)
            tq[index_of(tnames, "beta0")] = 0.2point
            tq[index_of(tnames, "sigma")] = log(0.4point)
            tq[index_of(tnames, "rho")] = 0.3 - 0.2point
            tq[indices_of(tnames, "theta")] = theta
            tq[indices_of(tnames, "phi")] = phi
            bq[only(i for (i, n) in enumerate(bnames) if !startswith(n, prefix))] = 0.2point
            bq[index_of(bnames, prefix * "_sigma")] = log(0.4point)
            bq[index_of(bnames, prefix * "_rho")] = 0.3 - 0.2point
            bq[indices_of(bnames, prefix * "_theta")] = theta
            bq[indices_of(bnames, prefix * "_phi")] = phi
            @test BS.log_density(bm, bq; propto=false, jacobian=true) ≈
                  BS.log_density(tm, tq; propto=false, jacobian=true) rtol=1e-12
        end
    else
        @info "Skipping the NYC reference comparison (BRM_CAR_RUNTIME=0)"
        @test true
    end
end

@testset "car on RKBRMI: the same law as SBBRMI at transported coordinates" begin
    if CAR_RUNTIME
        y = [0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9]
        for type in (:bym2, :icar)
            brmi = car_model(car_df(; y); type)
            rk, problem = consumer_problem(brmi)
            sb = SBBRMI(brmi; mod=@__MODULE__)
            stan = instantiate(sb)
            stan_names = BS.param_unc_names(stan.model)
            transport = brm_coordinate_transport(rk, sb, stan_names)
            @test length(transport) == length(stan_names)
            for u in (zeros(length(transport)),
                      collect(range(-0.43, 0.51; length=length(transport))))
                stan_u = brm_rk_point_to_stan(transport, u)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                stan_gradient = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, stan_gradient;
                    propto=false, jacobian=true)
                @test value ≈ stan_value + transport.logdensity_offset atol=1e-9 rtol=1e-12
                @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=1e-8 rtol=1e-10
            end
        end
    else
        @info "Skipping the RKBRMI car parity gate (BRM_CAR_RUNTIME=0)"
        @test true
    end
end

@testset "car on RKBRMI: frozen replay reads the fitted graph by area label" begin
    if CAR_RUNTIME
        y = [0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9]
        rk = RKBRMI(car_model(car_df(; y)))
        mu_at(backend, u) = begin
            bound = rk_bound_data(backend)
            query = ReactiveKernels.prepare(backend.model.spec;
                have=(:unconstrained, keys(bound)...), want=(:mu,), bound)
            value = Base.invokelatest(query, u)
            value isa Tuple ? only(value) : value
        end
        u = collect(range(-0.4, 0.5; length=length(coordinate_names(rk.model.layout))))
        train = mu_at(rk, u)
        # rows of areas a, g (island), e; the edge vectors are not needed
        replay = reprocess(rk, (; area=["a", "g", "e"], y=zeros(3)))
        @test mu_at(replay, u) ≈ train[[1, 7, 5]] atol=1e-12 rtol=1e-12
        # the graph is the fit's even when the replay rows miss areas (here the
        # island `g`, which only the training rows name)
        @test mu_at(reprocess(rk, (; area=["e", "a"], y=zeros(2))), u) ≈ train[[5, 1]] atol=1e-12 rtol=1e-12
        @test_throws "not an area of the fitted graph" reprocess(rk, (; area=["z"], y=[0.0]))
        # Not built yet: a data-free fitted state does not carry the graph
        # (`brm_fitted_state` names the term); replays use the training fit.
        @test_broken (brm_fitted_state(rk); true)
    else
        @info "Skipping the RKBRMI car replay gate (BRM_CAR_RUNTIME=0)"
        @test true
    end
end

@testset "car on RKBRMI: the design binder binds what a replay binds" begin
    if CAR_RUNTIME
        y = [0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9]
        rk = RKBRMI(car_model(car_df(; y)))
        binder = rk_design_binder(rk)
        # `h` names an area without training rows; the second design of the
        # same kind is bound into the first one's program
        for design in ((; area=["a", "b"], y=zeros(2)), (; area=["e", "g", "h"], y=zeros(3)))
            @test binder(design) == rk_bound_data(reprocess(rk, design))
        end
    else
        @info "Skipping the RKBRMI car binder gate (BRM_CAR_RUNTIME=0)"
        @test true
    end
end

@testset "car optional priors drop on models without the term" begin
    df = car_df()
    plain = @brm df begin
        mu ~ 1
        optional(mixing(:, car(area))) ~ Beta(2, 2)
        optional(sd(:, car(area))) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1.0)
    end
    bare = @brm df begin
        mu ~ 1
        y ~ Normal(mu, 1.0)
    end
    @test stan(plain) == stan(bare)
end

println("car_term.jl: all testsets passed")
