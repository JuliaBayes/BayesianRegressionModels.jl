# test/plate_nested_grouping.jl — a per-subject linear predictor whose random
# effects span several groupings, read with the plate index (snag
# `plate-predictor-c996ec8d`).
#
# Districts nested in regions, one row per district:
#
#     alpha ~ 1 + x + (1 | region | region_g) + (1 | district | district_g)
#
# `district_g` lists one unique label per row, so it labels the plate's cells;
# the region effect is an ordinary term already added into each row's
# `alpha[d]`, as `(1 | region) + (1 | district)` is in brms. The program must
# equal the spelling it replaces: a separate region predictor on a region
# frame, added in the cell by index.
#
# Run: julia --startup-file=no --project=test test/plate_nested_grouping.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const ND = 12
const REGIONS = ["R1", "R2", "R3", "R4"]
const NESTED = let
    region_g = [REGIONS[1 + (k % 4)] for k in 1:ND]
    # Unsorted labels: cells stay in row order.
    district_g = ["D$(lpad(k, 2, '0'))" for k in [5, 2, 11, 8, 1, 12, 3, 9, 6, 10, 4, 7]]
    (; region_g, district_g,
       x=[0.1 * ((k * 7) % 5) - 0.2 for k in 1:ND],
       log_exposure=[[0.2 + 0.01k, 0.25, 0.3 - 0.01k] for k in 1:ND],
       period=[[1, 2, 3] for _ in 1:ND],
       count_r=[[(k + t) % 4 + 1 for t in 1:3] for k in 1:ND],
       count_s=[[(2k + t) % 3 for t in 1:3] for k in 1:ND],
       period_g=[1, 2, 3],
       region_frame=REGIONS,
       region_pos=[findfirst(==(r), REGIONS) for r in region_g])
end

# Two predictors share the region and district blocks, as in a multi-margin
# small-area model; a period effect on its own frame is read by index.
nested_model(data) = @brm data begin
    eta ~ 0 + (1 | per | period_g)
    alpha_r ~ 1 + x + (1 | region | region_g) + (1 | district | district_g)
    alpha_s ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    inv_sqrt_phi ~ Exponential(1)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ NegativeBinomial2(
            exp.(log_exposure[d] .+ alpha_r[d] .+ eta[period[d]]),
            1 / (inv_sqrt_phi * inv_sqrt_phi))
    end
    @plate for d in eachindex(alpha_s)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_s[d]))
    end
end

# The consumer shape the fix replaces: the region effect as separate
# predictors over a region frame, added in every cell by index.
indexed_model(data) = @brm data begin
    eta ~ 0 + (1 | per | period_g)
    nu_r ~ 0 + (1 | region | region_frame)
    nu_s ~ 0 + (1 | region | region_frame)
    alpha_r ~ 1 + x + (1 | district | district_g)
    alpha_s ~ 1 + (1 | district | district_g)
    inv_sqrt_phi ~ Exponential(1)
    @plate for d in eachindex(alpha_r)
        count_r[d] ~ NegativeBinomial2(
            exp.(log_exposure[d] .+ alpha_r[d] .+ nu_r[region_pos[d]] .+ eta[period[d]]),
            1 / (inv_sqrt_phi * inv_sqrt_phi))
    end
    @plate for d in eachindex(alpha_s)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha_s[d] .+ nu_s[region_pos[d]]))
    end
end

# RK coordinate name => Stan unconstrained name, by the BRM naming scheme:
# `<block>.z.<level>.<margin>` is level-major in `<block>_z_flat`.
function stan_names_for(rk_names, stan_names)
    K = Dict{String,Int}()
    for n in string.(rk_names)
        m = match(r"^(.*)\.tau\.(\d+)$", n)
        m === nothing || (K[m[1]] = max(get(K, m[1], 0), parse(Int, m[2])))
    end
    map(rk_names) do n
        s = string(n)
        stan = if (m = match(r"^(.*)\.z\.(\d+)\.(\d+)$", s)) !== nothing
            "$(m[1])_z_flat.$((parse(Int, m[2]) - 1) * K[m[1]] + parse(Int, m[3]))"
        elseif (m = match(r"^(.*)\.(tau|L|beta_pop)\.(\d+)$", s)) !== nothing
            "$(m[1])_$(m[2]).$(m[3])"
        elseif (m = match(r"^(.*)_Intercept$", s)) !== nothing
            "pop_$(m[1])_beta_pop.1"
        else
            s
        end
        @test stan in stan_names
        n => stan
    end
end

@stestset "an outer grouping beside the per-row grouping labels cells by the per-row one" begin
    data = NESTED
    saved = deepcopy(data)
    brmi = nested_model(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_r_cell] == ND
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    # Both predictors stay parameters, sliced per cell.
    @test !haskey(sb.data, :alpha_r) && !haskey(sb.data, :alpha_s)
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok

    stan = consumer_stan(brmi, "plate-nested-grouping")
    backend, problem = consumer_problem(brmi)
    rk_names = coordinate_names(backend.model.layout)
    mapping = stan_names_for(rk_names, BridgeStan.param_unc_names(stan.model))
    n = length(rk_names)
    for u in (zeros(n), collect(range(-0.3, 0.3; length=n)), [0.1 * sin(3k) for k in 1:n])
        check_consumer_stan(problem, stan, mapping, backend, u)
    end

    # The same density as the index-read spelling: only the region block's
    # grouping column is named differently.
    indexed = consumer_stan(indexed_model(data), "plate-nested-grouping-indexed")
    names = replace.(BridgeStan.param_unc_names(stan.model),
        "region_region_g" => "region_region_frame")
    indexed_names = BridgeStan.param_unc_names(indexed.model)
    @test sort(names) == sort(indexed_names)
    # `indexed_u[order[i]] = u[i]`
    order = [findfirst(==(name), indexed_names) for name in names]
    for u in (zeros(n), collect(range(-0.4, 0.4; length=n)), [0.2 * cos(2k) for k in 1:n])
        @test BridgeStan.log_density(stan.model, u; propto=false) ≈
            BridgeStan.log_density(indexed.model, u[invperm(order)]; propto=false) atol=1e-10
    end
    @test isequal(data, saved)
end

# Several `|ID|` blocks over the one per-row grouping are one grouping.
two_blocks_model(data) = @brm data begin
    alpha ~ 1 + (1 | p | district_g) + (0 + x | q | district_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end

@stestset "several |ID| blocks over one grouping" begin
    sb = SBBRMI(two_blocks_model(NESTED); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    @test RKBRMI(two_blocks_model(NESTED)) isa RKBRMI
end

# refused: no grouping column lists one label per row, so none can label a cell
# (the one-row-per-subject contract; long-format data is grouped first).
const HALVES = merge(NESTED, (; half_g=repeat(["H1", "H2", "H3", "H4", "H5", "H6"], 2)))
no_row_grouping_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | half | half_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end

# refused: one cell's predictors describe different subjects.
disagreeing_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    beta ~ 1 + (1 | region | region_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d] .+ beta[d]))
    end
end

@stestset "groupings that cannot label the cells are refused" begin
    message = "grouping columns `region_g`, `half_g` of per-subject linear predictor " *
        "`alpha` must list one non-missing unique subject per row"
    @test_throws message SBBRMI(no_row_grouping_model(HALVES); mod=@__MODULE__,
        total_groups=())
    @test_throws message RKBRMI(no_row_grouping_model(HALVES))
    @test_throws "disagree on their grouping" SBBRMI(disagreeing_model(NESTED);
        mod=@__MODULE__, total_groups=())
    @test_throws "disagree on subject grouping" RKBRMI(disagreeing_model(NESTED))
end

# One district per region: both columns list one label per row and label the
# same rows in the same order.
const ONE_PER_REGION = let rows = [1, 2, 3, 4]
    (; region_g=NESTED.region_g[rows], district_g=NESTED.district_g[rows],
       log_exposure=NESTED.log_exposure[rows], count_s=NESTED.count_s[rows],
       ev_district=NESTED.district_g[[1, 1, 2, 3, 3, 3, 4]],
       ev_x=[0.1, 0.2, -0.1, 0.3, 0.0, 0.2, -0.2],
       ev_y=[0.5, 0.4, 0.1, 0.7, 0.2, 0.3, -0.1])
end

tie_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    @plate for d in eachindex(alpha)
        count_s[d] ~ Poisson(exp.(log_exposure[d] .+ alpha[d]))
    end
end

tie_ragged_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    sigma ~ Exponential(1)
    ragged(ev_x, ev_district)
    ragged(ev_y, ev_district)
    @plate for d in eachindex(alpha)
        ev_y[d] ~ Normal(alpha[d] .+ ev_x[d], sigma)
    end
end

tie_named_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    sigma ~ Exponential(1)
    ragged(ev_x, ev_district => district_g)
    ragged(ev_y, ev_district => district_g)
    @plate for d in eachindex(alpha)
        ev_y[d] ~ Normal(alpha[d] .+ ev_x[d], sigma)
    end
end

@stestset "several per-row groupings: labels decide only where rows join by label" begin
    data = ONE_PER_REGION
    # No label join: either column gives the same cells, count and order.
    sb = SBBRMI(tie_model(data); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_count_s_cell] == 4
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
    @test RKBRMI(tie_model(data)) isa RKBRMI
    # refused: event rows join cells by label, and both columns could label
    # them; a guessed column would join the wrong cells silently whenever their
    # labels overlap.
    message = "grouping columns `region_g`, `district_g` each list one unique label per row"
    @test_throws message SBBRMI(tie_ragged_model(data); mod=@__MODULE__, total_groups=())
    @test_throws message RKBRMI(tie_ragged_model(data))
    named = SBBRMI(tie_named_model(data); mod=@__MODULE__, total_groups=())
    @test named.data[:kernel_nsub_ev_y_cell] == 4
    @test StanBlocks.stanc_check(StanBlocks.stan_code(named.model); warn_pedantic=false).ok
    @test RKBRMI(tie_named_model(data)) isa RKBRMI
end

# A flat response observed against the plate result joins it by the per-row
# grouping's labels.
const NESTED_EVENTS = merge(NESTED, (;
    ev_district=NESTED.district_g[[1, 1, 2, 5, 5, 5, 9, 12]],
    ev_x=[0.1, 0.2, -0.1, 0.3, 0.0, 0.2, -0.2, 0.4],
    ev_y=[0.5, 0.4, 0.1, 0.7, 0.2, 0.3, -0.1, 0.6]))
flat_response_model(data) = @brm data begin
    alpha ~ 1 + (1 | region | region_g) + (1 | district | district_g)
    sigma ~ Exponential(1)
    ragged(ev_x, ev_district)
    @plate for d in eachindex(alpha)
        loc[d] = alpha[d] .+ ev_x[d]
    end
    ragged(ev_y, ev_district) ~ Normal(loc, sigma)
end

@stestset "a flat response joins the plate result by the per-row grouping" begin
    sb = SBBRMI(flat_response_model(NESTED_EVENTS); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_loc] == ND
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end
