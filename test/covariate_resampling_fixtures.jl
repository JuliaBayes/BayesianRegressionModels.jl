const FRESH_SCALAR_DATA=(;x=Union{Missing,Float64}[.2,missing,.8,missing,1.1],
    z=Union{Missing,Float64}[missing,.3,.5,missing,.7],u=[-.5,-.2,.1,.4,.8],
    subject=[1,1,2,2,3],y=[.1,-.2,.3,.5,.2])
const FRESH_SCALAR_BUILDER=@brm begin
    mx ~ Normal(0,1)
    sx ~ LogNormal(0,.3)
    mi(x) ~ Normal(mx,sx)
    zloc=.2+.3*x
    zscale=exp(.1+.2*x)
    mi(z) ~ LogNormal(zloc,zscale)
    physical=exp(x)
    log_physical=log(physical)
    mu ~ 1+standardize(x)+center(log(z))+physical+(1|subject)
    y ~ Normal(mu,1)
end
const FRESH_JOINT_BUILDER=@brm begin
    L ~ LKJCovarianceFactor(2;scale_prior=Exponential(1))
    xloc ~ 1+u
    zloc ~ 1+u
    mi([x,z]) ~ MvNormalCholesky([xloc,zloc],L)
    physical=exp(x)
    log_physical=log(physical)
    mu ~ 1+standardize(physical)+z+(1|subject)
    y ~ Normal(mu,1)
end
