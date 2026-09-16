# Capture RBesT's exact gMAP Stan data and program for one case, without sampling.
# Usage: Rscript capture.R RBEST_CLONE OUT CASE VARIANT   (VARIANT legacy | s2z; RBEST_LIBRARY must match)
# Requires RBEST_LIBRARY (RBesT built from RBEST_CLONE) and PUPIL_BRMS_LIBRARY (cmdstanr, jsonlite, digest).
for(v in c("PUPIL_BRMS_LIBRARY","RBEST_LIBRARY")) for(p in rev(strsplit(Sys.getenv(v),":")[[1]])) if(nzchar(p)) .libPaths(c(p,.libPaths()))
suppressPackageStartupMessages({library(RBesT);library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==4L)
clone <- normalizePath(args[[1]]);out <- normalizePath(args[[2]],mustWork=FALSE);case <- args[[3]];variant <- args[[4]]
stopifnot(variant %in% c("legacy","s2z"),!dir.exists(out));dir.create(out,recursive=TRUE)
sha <- system2("git",c("-C",shQuote(clone),"rev-parse","HEAD"),stdout=TRUE)
# legacy: RBesT 1.11-0 at main; s2z: pull request 64 (issue-prep-1-12-0) built from RBEST_LIBRARY.
expected <- if(variant=="legacy") "3d5fa1dda74f9d68360094a11983f32dd1b61c10" else "a5acbbc39c2cd620f0549159aa7c1991d4c28d6a"
stopifnot(sha==expected,identical(readLines(file.path(clone,"DESCRIPTION")),readLines(file.path(find.package("RBesT"),"DESCRIPTION"))[seq_along(readLines(file.path(clone,"DESCRIPTION")))]) ||
  packageVersion("RBesT")==(if(variant=="legacy") "1.11.0" else "1.12.0"))
if(variant=="s2z") options(RBesT.MC.s2z=TRUE) else if(!is.null(getOption("RBesT.MC.s2z"))) stop("legacy capture must use the 1.11 library")
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)));here <- dirname(script)
support <- file.path(here,"..","pupil_scale_totals","support");source(file.path(support,"native_tools.R"))
# Documented calls (R/AS.R, R/crohn.R); chains = 0 builds the object and its Stan data only.
map <- if(case=="AS") {
  data(AS,package="RBesT")
  ref <- read.delim(file.path(here,"reference","datasets","AS.tsv"),check.names=FALSE)
  stopifnot(identical(as.character(AS$study),ref$study),all(AS$n==ref$n),all(AS$r==ref$r))
  gMAP(cbind(r,n-r) ~ 1 | study,family=binomial,data=AS,tau.dist="HalfNormal",tau.prior=1,beta.prior=2,chains=0)
} else if(case=="crohn") {
  data(crohn,package="RBesT")
  ref <- read.delim(file.path(here,"reference","datasets","crohn.tsv"),check.names=FALSE)
  stopifnot(identical(as.character(crohn$study),ref$study),all(crohn$n==ref$n),all(crohn$y==ref$y))
  gMAP(cbind(y,y.se) ~ 1 | study,family=gaussian,data=transform(crohn,y.se=88/sqrt(n)),weights=n,
       tau.dist="HalfNormal",tau.prior=44,beta.prior=cbind(0,88),chains=0)
} else stop("Unknown case")
dataL <- map$fit.data
stopifnot(dataL$ncp==1,dataL$mX==1,dataL$tau_prior_dist==0,dataL$re_dist==0,dataL$n_tau_strata==1,dataL$prior_PD==0)
if(variant=="s2z") stopifnot(dataL$use_s2z==1,dataL$has_intercept==1) else stopifnot(is.null(dataL$use_s2z))
H <- dataL$H
# Helmert basis of PR 64 (zero_sum_basis in gMAP.stan): Q'Q = I, 1'Q = 0.
zero_sum_basis <- function(J) { Q <- matrix(0,J,J-1); for(k in seq_len(J-1)) { s <- 1/sqrt(k*(k+1)); Q[seq_len(k),k] <- s; Q[k+1,k] <- -k*s }; Q }
# Empirical link-scale per-trial values, the same formula model.jl uses for BRM initial values.
theta_hat <- if(case=="AS") log((dataL$r+0.5)/(dataL$r_n-dataL$r+0.5)) else dataL$y
beta_hat <- mean(theta_hat);tau_hat <- max(sd(theta_hat),0.05)
g <- dataL$beta_raw_guess;t <- dataL$tau_raw_guess
for(ncp in c(1L,0L)) {
  d <- dataL;d$ncp <- ncp
  label <- if(ncp==1L) "ncp" else "cp"
  write_stan_json(d,file.path(out,paste0("data-",label,".json")))
  # xi_eta is indexed by RBesT's group index (alphabetical study factor), not by data row.
  if(variant=="legacy") {
    xi_row <- if(ncp==1L) (theta_hat-beta_hat)/tau_hat else (theta_hat-g[1,1])/g[2,1]
    xi_eta <- numeric(dataL$n_groups);xi_eta[dataL$group_index] <- xi_row
    init <- list(beta_raw=array((beta_hat-g[1,1])/g[2,1],1),tau_raw=array((log(tau_hat)-t[1])/t[2],1),xi_eta=xi_eta)
  } else {
    # s2z: sampled alpha = beta + mean(eps) = mean(theta_hat); xi_eta are the J-1 Helmert coordinates of the
    # centered effects, scaled by tau (ncp) or by the intercept guess sd (cp); xi_abar is data-free.
    eps <- numeric(dataL$n_groups);eps[dataL$group_index] <- theta_hat-beta_hat
    Q <- zero_sum_basis(dataL$n_groups);xi_eta <- as.vector(t(Q) %*% eps)/(if(ncp==1L) tau_hat else g[2,1])
    init <- list(beta_raw=array((beta_hat-g[1,1])/g[2,1],1),tau_raw=array((log(tau_hat)-t[1])/t[2],1),xi_eta=xi_eta,xi_abar=array(0,1))
  }
  write_stan_json(init,file.path(out,paste0("init-",label,".json")))
}
write_json(list(beta=beta_hat,tau=tau_hat,theta=theta_hat,group_index=dataL$group_index,beta_raw_guess=g,tau_raw_guess=t),
  file.path(out,"init-physical.json"),digits=NA,auto_unbox=TRUE,pretty=TRUE)
# Stan program: the two #include lines are comment-only license headers (README); drop them for a
# standalone file, keep everything else byte for byte.
src <- readLines(file.path(clone,"inst","stan","gMAP.stan"))
stopifnot(sum(grepl("^#include",src))==2L,all(grepl("license|copyright",src[grepl("^#include",src)])))
clean <- src[!grepl("^#include",src)]
writeLines(clean,file.path(out,"clean.stan"))
# Counter-instrumented copy: declarations in a new functions block, tick at the start of
# transformed parameters, target += tick at the start of model, count at the end of generated quantities.
s <- counter_statements("beta_raw[1]")
ins <- function(code,pattern,line,after=TRUE) { i <- grep(pattern,code,fixed=TRUE);stopifnot(length(i)==1L)
  if(after) append(code,line,after=i) else append(code,line,after=i-1L) }
# Pull request 64 already has a functions block (zero_sum_basis); declare the counter inside it.
instr <- if(any(clean=="functions {")) ins(clean,"functions {",s[1:2]) else c("functions {",s[1:2],"}",clean)
instr <- ins(instr,"transformed parameters {",s[[3]])
instr <- ins(instr,"model {",s[[4]])
gq_end <- max(grep("^}$",instr));instr <- append(instr,s[[5]],after=gq_end-1L)
writeLines(instr,file.path(out,"instrumented.stan"))
nonblank <- function(x) { x <- trimws(strsplit(x,"\n")[[1]]);x[nzchar(x)] }
stopifnot(identical(nonblank(strip_counter(paste(instr,collapse="\n"),"beta_raw[1]")),
  nonblank(paste(if(any(clean=="functions {")) clean else c("functions {","}",clean),collapse="\n"))))
write_json(list(rbest_version=as.character(packageVersion("RBesT")),rbest_sha=sha,case=case,H=H,
  clean_sha256=digest(file.path(out,"clean.stan"),algo="sha256",file=TRUE),
  data_ncp_sha256=digest(file.path(out,"data-ncp.json"),algo="sha256",file=TRUE),
  control=list(adapt_delta=if(variant=="s2z") 0.95 else 0.99,stepsize=0.01,max_treedepth=20,warmup=2000,iter=6000,thin=4,chains=4,init=1),
  variant=variant,
  formula=deparse(map$formula),family=map$family$family),
  file.path(out,"capture.json"),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))
cat("RBEST_CAPTURE_COMPLETE",case,variant,H,"\n")
