Turing.Bijectors.bijector(d::BRM._BRMThresholdPrior{false}) = identity
Turing.Bijectors.bijector(d::BRM._BRMThresholdPrior{true}) =
    iszero(length(d)) ? identity : Turing.Bijectors.inverse(Turing.Bijectors.OrderedBijector())
