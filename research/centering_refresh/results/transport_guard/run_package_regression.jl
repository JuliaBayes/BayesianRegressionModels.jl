using Test, WarmupHMC, Random, LinearAlgebra, Statistics, LogDensityProblems, Distributions
using DifferentiationInterface, Enzyme
println("TESTED_PACKAGE=",pkgdir(WarmupHMC));flush(stdout)
root=only(ARGS)
include(joinpath(root,"test/test_problems.jl"))
include(joinpath(root,"test/active_reparametrization_state.jl"))
println("ONLINE_STATE_TESTS_COMPLETE")
