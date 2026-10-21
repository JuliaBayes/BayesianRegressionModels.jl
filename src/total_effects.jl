# Exact Gaussian integration of population coefficients into group totals.
# A maps population coefficients onto the random-effect design and W holds each
# group's value of every absorbed coefficient's column: group g's total of
# random column c has mean sum_p A[c,p] * W[g,p] * beta[p]. W is 1 for a
# coefficient matched to a random column; a group-level predictor (a column
# constant within each group) carries its per-group value. Work is linear in
# the number of groups for fixed coefficient dimension.
StanBlocks.@deffun begin
    brm_total_precision(tau::vector[k], A::matrix[k,p], W::matrix[j,p],
                        precision::vector[p])::matrix[p,p] = begin
        G = crossprod(W)
        out = diag_matrix(precision)
        for a in 1:p
            for b in 1:p
                for c in 1:k
                    out[a,b] += G[a,b] * A[c,a] * A[c,b] / square(tau[c])
                end
            end
        end
        out
    end

    # Each group's mean total: row g is A * (W[g,:]' .* beta).
    brm_total_location(W::matrix[j,p], A::matrix[k,p], beta::vector[p])::matrix[j,k] =
        diag_post_multiply(W, beta) * A'

    brm_total_conditional_mean(total::matrix[j,k], tau::vector[k],
            A::matrix[k,p], W::matrix[j,p], location::vector[p], precision::vector[p],
            Q::matrix[p,p])::vector[p] = begin
        WT = W' * total
        natural = precision .* location
        for a in 1:p
            for c in 1:k
                natural[a] += A[c,a] * WT[a,c] / square(tau[c])
            end
        end
        mdivide_left_spd(Q, natural)
    end

    @lpxf brm_total_lpdf(total::matrix[j,k], tau::vector[k],
            A::matrix[k,p], W::matrix[j,p], location::vector[p], precision::vector[p])::real = begin
        Q = brm_total_precision(tau, A, W, precision)
        beta = brm_total_conditional_mean(total, tau, A, W, location, precision, Q)
        mu = brm_total_location(W, A, beta)
        quadratic = 0.
        lp = -0.5 * (j * k - p) * 1.8378770664093453 - j * sum(log(tau))
        for c in 1:k
            for g in 1:j
                quadratic += square(total[g,c] - mu[g,c]) / square(tau[c])
            end
        end
        for a in 1:p
            if precision[a] > 0.
                lp += 0.5 * (log(precision[a]) - 1.8378770664093453)
                quadratic += precision[a] * square(beta[a] - location[a])
            end
        end
        # Center the quadratic at its minimizer instead of subtracting two
        # potentially enormous uncentered sums of squares.
        lp - 0.5 * (log_determinant(Q) + quadratic)
    end

    brm_total_recover_rng(total::matrix[j,k], tau::vector[k],
            A::matrix[k,p], W::matrix[j,p], location::vector[p], precision::vector[p])::vector[p] = begin
        Q = brm_total_precision(tau, A, W, precision)
        beta = brm_total_conditional_mean(total, tau, A, W, location, precision, Q)
        multi_normal_rng(beta, inverse_spd(Q))
    end

    brm_total_deviations(total::matrix[j,k], mu::matrix[j,k])::matrix[j,k] = total - mu

    brm_total_rng(matrix[j,k], tau::vector[k], A::matrix[k,p], W::matrix[j,p],
                  location::vector[p], precision::vector[p])::matrix[j,k] = begin
        @stan_assert min(precision) > 0.
        beta = to_vector(normal_rng(location, rep_vector(1., p) ./ sqrt(precision)))
        mu = brm_total_location(W, A, beta)
        total = rep_matrix(0., j, k)
        for c in 1:k
            total[:,c] = to_vector(normal_rng(mu[:,c], tau[c]))
        end
        total
    end
end
