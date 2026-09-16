# Native RBesT arm under CmdStan with the gradient counter.
# Usage: Rscript native.R CMDSTAN CAPTURE_DIR OUT ARM   with ARM in rbest_ncp rbest_cp stan_ncp stan_cp (legacy capture)
#        or s2z_ncp s2z_cp (pull-request-64 capture; RBesT control adapt_delta 0.95)
# rbest_*: RBesT's own sampler control (adapt_delta 0.99, stepsize 0.01, max_treedepth 20, warmup 2000).
# stan_*:  CmdStan defaults (adapt_delta 0.8, max_treedepth 10, warmup 1000).  One chain, seed 1,
# 10,000 retained draws, thin 1, matched empirical initialization from capture.R.
for(v in c("PUPIL_BRMS_LIBRARY","RBEST_LIBRARY")) for(p in rev(strsplit(Sys.getenv(v),":")[[1]])) if(nzchar(p)) .libPaths(c(p,.libPaths()))
suppressPackageStartupMessages({library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==4L)
set_cmdstan_path(args[[1]]);cap <- normalizePath(args[[2]]);out <- normalizePath(args[[3]],mustWork=FALSE);arm <- args[[4]]
stopifnot(arm %in% c("rbest_ncp","rbest_cp","stan_ncp","stan_cp","s2z_ncp","s2z_cp"),!dir.exists(out))
param <- sub("^.*_","",arm);rbest <- !startsWith(arm,"stan")
capture <- fromJSON(file.path(cap,"capture.json"))
stopifnot(capture$variant==(if(startsWith(arm,"s2z")) "s2z" else "legacy"))
ctrl <- capture$control
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)));here <- dirname(script)
support <- file.path(here,"..","pupil_scale_totals","support");source(file.path(support,"native_tools.R"))
for(sub in c("sampling","counter-receipts","compiled")) dir.create(file.path(out,sub),recursive=TRUE)
Sys.setenv(PUPIL_GRAD_COUNTER_DIR=file.path(out,"counter-receipts"))
stopifnot(file.copy(file.path(support,"native_gradient_counter.hpp"),file.path(out,"native_gradient_counter.hpp")))
stopifnot(file.copy(file.path(cap,"instrumented.stan"),file.path(out,"compiled","gmap.stan")))
model <- cmdstan_model(file.path(out,"compiled","gmap.stan"),user_header=file.path(out,"native_gradient_counter.hpp"),
  stanc_options=list("allow-undefined"=TRUE))
warmup <- if(rbest) 2000L else 1000L;draws <- 10000L;max_depth <- if(rbest) 20L else 10L
fit <- model$sample(data=file.path(cap,paste0("data-",param,".json")),init=file.path(cap,paste0("init-",param,".json")),
  seed=1,chains=1,parallel_chains=1,iter_warmup=warmup,iter_sampling=draws,thin=1,
  adapt_delta=if(rbest) ctrl$adapt_delta else 0.8,step_size=if(rbest) ctrl$stepsize else NULL,max_treedepth=max_depth,
  save_warmup=TRUE,sig_figs=17,refresh=2000,output_dir=file.path(out,"sampling"),output_basename="gmap")
costs <- collect_native_csv(list.files(file.path(out,"sampling"),pattern="\\.csv$",full.names=TRUE),
  file.path(out,"counter-receipts"),out,warmup=warmup,draws=draws,max_depth=max_depth)
stopifnot(costs$precursor_gradient_calls==0)
write_json(list(arm=arm,parametrization=param,variant=capture$variant,case=capture$case,rbest_version=capture$rbest_version,rbest_sha=capture$rbest_sha,
  seed=1,chains=1,warmup=warmup,draws=draws,thin=1,adapt_delta=if(rbest) ctrl$adapt_delta else 0.8,
  step_size=if(rbest) ctrl$stepsize else "cmdstan default",max_treedepth=max_depth,
  cmdstan=as.character(cmdstan_version()),cmdstanr=as.character(packageVersion("cmdstanr")),
  clean_sha256=capture$clean_sha256,data_sha256=digest(file.path(cap,paste0("data-",param,".json")),algo="sha256",file=TRUE)),
  file.path(out,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))
cat("RBEST_NATIVE_COMPLETE",capture$case,arm,"\n")
