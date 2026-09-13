# Reference indexing follows the distribution's mathematical value shape.
# A simplex/vector or covariance factor is a whole model value at each row.
_brm_parameter_reference_axis(prior::ExprColumn) =
    _brm_parameter_reference_axis(getf(prior), prior)
function _brm_parameter_reference_axis(_constructor, prior)
    shape = _brm_distribution_shape(prior)
    isnothing(shape) || first(shape) === Distributions.Univariate ? :scalar : :whole
end
_brm_parameter_reference_axis(::typeof(LKJCovarianceFactor), _prior) = :whole

_brm_distribution_shape(::typeof(truncated), args) =
    _brm_distribution_shape(first(args))
_brm_distribution_shape(::typeof(censored), args) =
    _brm_distribution_shape(first(args))
_brm_distribution_shape(::typeof(weighted), args) =
    _brm_distribution_shape(first(args))

_brm_prior_constructor(::typeof(truncated)) = true
_brm_prior_constructor(::typeof(censored)) = true

# This is a backend extension seam; the common program retains the marker and
# its complete prior call until the concrete backend chooses its geometry.
function _brm_turing_covariance_prior end
function _brm_turing_horseshoe_prior end
