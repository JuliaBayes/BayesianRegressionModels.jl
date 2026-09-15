# Rscript s2z_native.R CMDSTAN_PATH NEW_OUTPUT INIT_JSON [s2z_cp|s2z_ncp|s2z_auto|all]
extra_library <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra_library)) .libPaths(c(extra_library,.libPaths()))
suppressPackageStartupMessages({library(brms);library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=3L)
set_cmdstan_path(args[[1]])
out <- normalizePath(args[[2]],mustWork=FALSE)
stopifnot(!dir.exists(out));dir.create(out,recursive=TRUE)
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)))
here <- dirname(script)
source(file.path(here,"native_tools.R"))
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
stopifnot(identical(readLines(file.path(dirname(find.package("brms")),"brms-source-sha.txt")),pin))
load(file.path(here,"reference","df_pupil_complete.rda"))
base_init <- fromJSON(args[[3]])
init <- list(theta_s2z=c(base_init$Intercept,base_init$b),
  b_sigma=base_init$b_sigma,Intercept_sigma=base_init$Intercept_sigma,
  sd_1=base_init$sd_1,z_s2z_1=rep(0,38),udf_b_s2z_1=1/3)
formulas <- list(
  s2z_cp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=TRUE)),sigma ~ subj),
  s2z_ncp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=FALSE)),sigma ~ subj),
  s2z_auto=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center="auto")),sigma ~ subj))
selected <- if(length(args)>=4L && args[[4]]!="all") args[[4]] else names(formulas)
stopifnot(all(selected %in% names(formulas)))
options(brms.normalize=TRUE)
for(label in selected) {
  dest <- file.path(out,label);dir.create(dest)
  for(sub in c("sampling","precursor","counter-receipts","compiled")) dir.create(file.path(dest,sub))
  options(cmdstanr_write_stan_file_dir=file.path(dest,"compiled"))
  Sys.setenv(PUPIL_GRAD_COUNTER_DIR=file.path(dest,"counter-receipts"))
  for(name in c("s2z_native.R","native_tools.R","native_gradient_counter.hpp"))
    stopifnot(file.copy(file.path(here,name),file.path(dest,name)))
  formula <- formulas[[label]]
  cmdstanr::write_stan_json(init,file.path(dest,"init.json"))
  control <- if(label=="s2z_auto") autocenter_control(pilot_args=list(
    output_dir=file.path(dest,"precursor"),save_single_paths=TRUE,sig_figs=17)) else NULL
  fit <- brm(formula,data=df_pupil_complete,backend="cmdstanr",algorithm="sampling",
    chains=1,cores=1,iter=3000,warmup=1000,seed=1,normalize=TRUE,
    init=list(init),control=list(adapt_delta=0.8,max_treedepth=10),center_control=control,
    stanvars=counter_stanvars("theta_s2z[1]"),stan_model_args=list(
      user_header=file.path(dest,"native_gradient_counter.hpp"),stanc_options=list("allow-undefined"=TRUE)),
    output_dir=file.path(dest,"sampling"),output_basename="pupil",save_warmup=TRUE,
    sig_figs=17,refresh=500,silent=0)
  saveRDS(fit,file.path(dest,"fit.rds"))
  code <- stancode(fit)
  writeLines(code,file.path(dest,"instrumented.stan"))
  writeLines(strip_counter(code,"theta_s2z[1]"),file.path(dest,"clean.stan"))
  cmdstanr::write_stan_json(standata(fit),file.path(dest,"resolved-data.json"))
  if(label=="s2z_auto") {
    saveRDS(fit$autocenter,file.path(dest,"autocenter.rds"))
    capture.output(centering_weights(fit),file=file.path(dest,"centering-weights.txt"))
    capture.output(fit$autocenter$diagnostics,file=file.path(dest,"autocenter-diagnostics.txt"))
  }
  csv <- list.files(file.path(dest,"sampling"),pattern="\\.csv$",full.names=TRUE)
  costs <- collect_native_csv(csv,file.path(dest,"counter-receipts"),dest)
  if(label!="s2z_auto") stopifnot(costs$precursor_gradient_calls==0)
  write_json(list(brms_sha=pin,cmdstan=as.character(cmdstan_version()),
    cmdstanr=as.character(packageVersion("cmdstanr")),arm=label,
    target="pupil Student-t intercept, independent REs, numeric-subject sigma regression",
    seed=1,chains=1,warmup=1000,draws=2000,adapt_delta=0.8,max_treedepth=10,
    initialization="pooled total coefficients, zero contrasts, OLS group SDs and log residual scale; same physical point for every S2Z arm",
    source_sha256=digest(file.path(dest,"clean.stan"),algo="sha256",file=TRUE)),
    file.path(dest,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
  capture.output(sessionInfo(),file=file.path(dest,"R-session.txt"))
  cat("S2Z_NATIVE_COMPLETE\t",label,"\t",dest,"\n");flush.console()
}
