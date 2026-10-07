# test/rk_planning_specialization.jl — RK planning compiles once per corpus.
#
# Run: julia --project=test test/rk_planning_specialization.jl
#
# A BRMI's operation NamedTuple spells every formula name and expression tree,
# so any planner that specializes on it compiles again for each model of a
# corpus. Members of this public synthetic family differ only in operation
# names. After a warm member, emitting another member's RK artifact and
# building its kernel must not compile a BRM method, or a generic method over
# BRM values, whose signature names that member's operations. Only the
# constructors of the model's own values may. ReactiveKernels' kernel types
# legitimately carry argument names and are outside this contract. The family
# uses the value route: a caller-owned submodel, an ordinary callable
# assignment, grouped and population predictors and a log-scale predictor.
using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels

module SpecializationFamilySource
import BayesianRegressionModels: _rk_submodel_rhs!, getargs, getkwargs, name

function shifted_curve end
native_shifted_curve(x, location, shift) = location .+ x .* shift

function _rk_submodel_rhs!(definitions, statements, data, bindings,
        target::Symbol, ::typeof(shifted_curve), rhs)
    x = only(getargs(rhs))
    key = Symbol(target, :_curve_x)
    data[key] = copy(parent(parent(x)))
    location, shift = name(getkwargs(rhs).location), name(getkwargs(rhs).shift)
    helper = Symbol(target, :_native_curve)
    reader = Symbol(target, :_reader)
    push!(bindings, helper => native_shifted_curve)
    push!(definitions, :(function $reader(x, location, shift)
        return $helper(x, location, shift)
    end))
    push!(statements, :($target = $reader($key, $location, $shift)))
    :done
end
end

scaled_sum(curve, scale) = curve .* scale

const FAMILY_DATA = (;
    x=[-0.4, 0.2, 0.7, 1.1, -0.9, 0.3],
    z=[0.5, -0.1, 0.8, -0.6, 0.2, 0.4],
    g=["a", "b", "a", "c", "b", "c"],
    y=[0.1, -0.2, 0.3, 0.6, -0.5, 0.2])

# Every operation and data column name carries `zqspec<index>`, the marker
# searched for in the compile dump.
function define_member(index)
    tag(base) = Symbol(base, :_zqspec, index)
    x, z, g, y = tag(:x), tag(:z), tag(:g), tag(:y)
    a, shift, curve, gain, reads, scale = tag(:a), tag(:shift), tag(:curve),
        tag(:gain), tag(:reads), tag(:scale)
    builder = Symbol(:specialization_member_, index)
    Core.eval(@__MODULE__, quote
        function $builder(df)
            @brm df begin
                $a ~ 1 + $z + (1 | $g)
                $shift ~ Normal(0, 0.4)
                $curve ~ SpecializationFamilySource.shifted_curve($x;
                    location=$a, shift=$shift)
                $gain ~ Normal(1, 0.2)
                $reads = scaled_sum($curve, $gain)
                log($scale) ~ 1 + $z
                $y ~ Normal($reads, $scale)
            end
        end
    end)
    data = NamedTuple{(x, z, g, y)}(values(FAMILY_DATA))
    getfield(@__MODULE__, builder), data
end

# The model's own values are constructed with its own types; nothing else may
# carry them. Methods owned by this test file are the harness, not BRM.
const INHERENT_CONSTRUCTOR = r"^\"Tuple\{Type\{(NamedTuple\{|BayesianRegressionModels\.(BRMI|RKBRMI)\{)"
const HARNESS = r"^\"Tuple\{(typeof\(Main\.|Main\.var\")"

Base.@nospecializeinfer function compiled_during(@nospecialize(f))
    path, io = mktemp()
    ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), io.handle)
    value = try
        Base.invokelatest(f)
    finally
        ccall(:jl_dump_compiles, Cvoid, (Ptr{Cvoid},), C_NULL)
        close(io)
    end
    lines = [last(split(line, '\t'; limit=2)) for line in eachline(path)]
    rm(path)
    value, lines
end

Base.@nospecializeinfer function member_stage(@nospecialize(builder), data, index)
    brmi = builder(data)
    artifact = BRM.emit_rk_artifact(brmi; case_id="specialization-member-$index")
    backend = RKBRMI(brmi)
    (; brmi, artifact, backend)
end

@testset "distinct-name members reuse RK planning compilation" begin
    warm_builder, warm_data = define_member(1)
    warm = Base.invokelatest(member_stage, warm_builder, warm_data, 1)
    @test warm.artifact.plan isa BRM._RKValuePlan
    @test any(p -> last(p) === SpecializationFamilySource.native_shifted_curve,
        warm.artifact.bindings)

    for index in (2, 3)
        builder, data = define_member(index)
        member, lines = compiled_during(() -> member_stage(builder, data, index))
        marker = "_zqspec$index"
        specialized = filter(line -> occursin(marker, line), lines)
        # Positive control: the dump records this member's own construction.
        @test any(line -> occursin(INHERENT_CONSTRUCTOR, line), specialized)
        planning = filter(line -> occursin("BayesianRegressionModels", line) &&
            !occursin(INHERENT_CONSTRUCTOR, line) && !occursin(HARNESS, line),
            specialized)
        isempty(planning) || foreach(line -> println("SPECIALIZED ", line), planning)
        @test isempty(planning)
        # The reused compilation emits the same program up to the names.
        rename(ex) = replace(string(ex), "_zqspec$index" => "_zqspec1",
            "specialization-member-$index" => "specialization-member-1")
        @test rename(member.artifact.ast) == string(warm.artifact.ast)
        @test map(rename, member.artifact.defs) == map(string, warm.artifact.defs)
        @test length(coordinate_names(member.backend.model.layout)) ==
            length(coordinate_names(warm.backend.model.layout))
    end
end
