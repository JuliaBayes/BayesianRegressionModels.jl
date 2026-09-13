Turing.@model function _brm_r2d2_population(
        design, share_indices, r2_prior, alpha, total_scale, fallback_priors)
    n_columns = size(design, 2)
    length(share_indices) == n_columns || error("R2D2 share/design mismatch")
    length(fallback_priors) == n_columns || error("R2D2 prior/design mismatch")
    n_shares = maximum(share_indices; init=0)
    n_shares > 0 || error("R2D2 population allocation has no eligible columns")

    R2 ~ r2_prior
    phi ~ Distributions.Dirichlet(fill(alpha, n_shares))
    if isnothing(total_scale)
        tau_bsv ~ _brm_constrained_kernel(Normal(0, 1); lower=0)
    else
        tau_bsv = total_scale
    end

    beta = Vector{Real}(undef, n_columns)
    for column in 1:n_columns
        share = share_indices[column]
        if share == 0
            beta[column] ~ fallback_priors[column]
        else
            x = @view design[:, column]
            xbar = sum(x) / length(x)
            varx = sum(abs2(value - xbar) for value in x) / (length(x) - 1)
            varx > 0 || error("R2D2 allocated design column has zero variance")
            scale = sqrt(phi[share] * R2 * tau_bsv^2 / varx)
            beta[column] ~ Normal(0, scale)
        end
    end
    (; beta, residual_scale=sqrt((1 - R2) * tau_bsv^2), R2, phi, tau_bsv)
end

Turing.@model function _brm_r2d2_joint(designs, coefficient_shares,
        margin_shares, r2_prior, alpha, reference_scale, fallback_priors)
    n_shares = maximum((maximum(s; init=0) for s in coefficient_shares); init=0)
    n_shares = max(n_shares,
        maximum((maximum(s; init=0) for s in margin_shares); init=0))
    n_shares > 0 || error("joint R2D2 allocation has no shares")
    length(designs) == length(coefficient_shares) == length(fallback_priors) ||
        error("joint R2D2 predictor allocation mismatch")

    R2 ~ r2_prior
    phi ~ Distributions.Dirichlet(fill(alpha, n_shares))
    odds = R2 / (1 - R2)
    betas = Vector{Any}(undef, length(designs))
    scales = Vector{Any}(undef, length(margin_shares))
    for i in eachindex(designs)
        design, shares = designs[i], coefficient_shares[i]
        beta = Vector{Real}(undef, size(design, 2))
        for column in axes(design, 2)
            share = shares[column]
            if share == 0
                beta[column] ~ fallback_priors[i][column]
            else
                x = @view design[:, column]
                xbar = sum(x) / length(x)
                varx = sum(abs2(value - xbar) for value in x) / (length(x)-1)
                varx > 0 || error("joint R2D2 allocated design column has zero variance")
                beta[column] ~ Normal(0, reference_scale * sqrt(phi[share] * odds / varx))
            end
        end
        betas[i] = beta
    end
    for i in eachindex(margin_shares)
        scales[i] = reference_scale .* sqrt.(phi[margin_shares[i]] .* odds)
    end
    (; betas, scales, R2, phi)
end
