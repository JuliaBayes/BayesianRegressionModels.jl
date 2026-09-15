# Exact Gaussian integration of population coefficients into group totals.
# A maps population coefficients onto the random-effect design: X = Z * A.
# Work is linear in the number of groups for fixed coefficient dimension.
StanBlocks.@deffun begin
    brm_total_precision(tau::vector[k], A::matrix[k,p], precision::vector[p],
                        n_groups::int)::matrix[p,p] = begin
        out = diag_matrix(precision)
        for a in 1:p
            for b in 1:p
                for c in 1:k
                    out[a,b] += n_groups * A[c,a] * A[c,b] / square(tau[c])
                end
            end
        end
        out
    end

    brm_total_mean(total::matrix[j,k])::vector[k] = begin
        out = rep_vector(0., k)
        for c in 1:k
            out[c] = sum(total[:,c]) / j
        end
        out
    end

    brm_total_conditional_mean(total::matrix[j,k], tau::vector[k],
            A::matrix[k,p], location::vector[p], precision::vector[p],
            Q::matrix[p,p])::vector[p] = begin
        average = brm_total_mean(total)
        natural = precision .* location
        for a in 1:p
            for c in 1:k
                natural[a] += j * A[c,a] * average[c] / square(tau[c])
            end
        end
        mdivide_left_spd(Q, natural)
    end

    @lpxf brm_total_lpdf(total::matrix[j,k], tau::vector[k],
            A::matrix[k,p], location::vector[p], precision::vector[p])::real = begin
        Q = brm_total_precision(tau, A, precision, j)
        average = brm_total_mean(total)
        beta = brm_total_conditional_mean(total, tau, A, location, precision, Q)
        residual = average - A * beta
        quadratic = 0.
        lp = -0.5 * (j * k - p) * 1.8378770664093453 - j * sum(log(tau))
        for c in 1:k
            quadratic += j * square(residual[c]) / square(tau[c])
            for g in 1:j
                quadratic += square(total[g,c] - average[c]) / square(tau[c])
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
            A::matrix[k,p], location::vector[p], precision::vector[p])::vector[p] = begin
        Q = brm_total_precision(tau, A, precision, j)
        beta = brm_total_conditional_mean(total, tau, A, location, precision, Q)
        multi_normal_rng(beta, inverse_spd(Q))
    end

    brm_total_deviations(total::matrix[j,k], mu::vector[k])::matrix[j,k] = begin
        out = total
        for c in 1:k
            out[:,c] = total[:,c] - rep_vector(mu[c],j)
        end
        out
    end

    brm_total_rng(matrix[j,k], tau::vector[k], A::matrix[k,p],
                  location::vector[p], precision::vector[p])::matrix[j,k] = begin
        @stan_assert min(precision) > 0.
        beta = normal_rng(location, rep_vector(1., p) ./ sqrt(precision))
        mu = A * beta
        total = rep_matrix(0., j, k)
        for c in 1:k
            total[:,c] = normal_rng(rep_vector(mu[c], j), tau[c])
        end
        total
    end
end

