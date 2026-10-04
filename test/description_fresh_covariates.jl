using Test, BayesianRegressionModels, Distributions
import StanBlocks
include(joinpath(@__DIR__,"covariate_resampling_fixtures.jl"))
@testset "canonical fresh covariates retain actual description output roles" begin
 for (builder,selected) in ((FRESH_SCALAR_BUILDER,[:x,:z]),(FRESH_JOINT_BUILDER,[:x]))
  fitted=SBBRMI(builder(FRESH_SCALAR_DATA);mod=@__MODULE__,total_groups=())
  future=reprocess(fitted,FRESH_SCALAR_DATA;resample_covariates=selected)
  code=stan_code(future); frozen=deepcopy(future.data)
  d=brm_descriptor(future)
  r=brm_description(d)
  @test r.complete
  fresh=filter(c->c.provenance.observation_role===:covariate_draw,r.components)
  @test !isempty(fresh)
  @test all(c->c.provenance.observed_entries==0 && c.provenance.missing_entries==0,fresh)
  @test all(c->c.provenance.generated_entries==5*length(c.provenance.observation_sources),fresh)
  observations=filter(c->c.kind===:observation,fresh)
  @test !isempty(observations)
  @test all(c->any(p->startswith(p,"`"*string(c.provenance.owner)*"` is redrawn"),r.prose),observations)
  @test all(p->!occursin("observed entries remain conditioned data",p) &&
      !occursin("Observed coordinates stay fixed",p),r.prose)
  @test StanBlocks.stanc_check(code;warn_pedantic=false).ok
  @test isequal(future.data,frozen) && stan_code(future)==code
  for o in d.outputs
   o.role===:covariate_draw || continue
   @test any(c->any(p->p.name===o.name && p.role===o.role && p.logical===o.logical,c.outputs),brm_description_components(r))
  end
 end
end
