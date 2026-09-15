# Native Stan arm: CMDSTAN OUTPUT GROUPING HIERARCHY LABEL INIT_JSON AUDIT_JSON
extra <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra)) .libPaths(c(extra,.libPaths()))
suppressPackageStartupMessages({library(brms);library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==7L)
set_cmdstan_path(args[[1]])
out <- normalizePath(args[[2]],mustWork=FALSE);grouping <- args[[3]];hierarchy <- args[[4]];label <- args[[5]]
audit <- fromJSON(args[[7]])
stopifnot(audit$status=="passed",audit$grouping==grouping,audit$hierarchy==hierarchy,!dir.exists(out))
dir.create(out,recursive=TRUE)
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)))
here <- dirname(script);support <- file.path(here,"..","pupil_scale_totals","support")
source(file.path(support,"native_tools.R"))
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
stopifnot(identical(readLines(file.path(dirname(find.package("brms")),"brms-source-sha.txt")),pin))
sites <- as.data.frame(fromJSON(file.path(here,"reference","pm_sites.json")));sites$region <- sites[[grouping]]
K <- if(hierarchy=="intercept_only") 1L else 2L;J <- length(unique(sites$region))
re <- if(K==1L) "1" else "1 + log_sat"
forms <- list(
  ordinary_ncp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," || region)"))),
  s2z_cp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region,cor=FALSE,s2z=TRUE,center=TRUE))"))),
  s2z_ncp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region,cor=FALSE,s2z=TRUE,center=FALSE))"))),
  s2z_auto=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region,cor=FALSE,s2z=TRUE,center='auto'))"))))
stopifnot(label %in% names(forms))
base <- fromJSON(args[[6]])
init <- list(Intercept=base$Intercept,b=as.numeric(base$b),sd_1=as.numeric(base$sd_1),
  sigma=base$sigma,z_1=matrix(unlist(base$z_1),nrow=K,byrow=TRUE))
anchor <- "Intercept"
if(label!="ordinary_ncp") {
  init <- list(sd_1=as.numeric(base$sd_1),sigma=base$sigma,z_s2z_1=rep(0,K*(J-1)),udf_b_s2z_1=1/3)
  if(K==1L) {
    init$theta_s2z_active <- c(base$Intercept);init$fixed_s2z <- as.numeric(base$b)
    anchor <- "theta_s2z_active[1]"
  } else {
    init$theta_s2z <- c(base$Intercept,as.numeric(base$b));anchor <- "theta_s2z[1]"
  }
}
for(sub in c("sampling","precursor","counter-receipts","compiled")) dir.create(file.path(out,sub))
options(cmdstanr_write_stan_file_dir=file.path(out,"compiled"),brms.normalize=TRUE)
Sys.setenv(PUPIL_GRAD_COUNTER_DIR=file.path(out,"counter-receipts"))
stopifnot(file.copy(file.path(support,"native_gradient_counter.hpp"),file.path(out,"native_gradient_counter.hpp")))
stopifnot(file.copy(script,file.path(out,"native.R")))
write_stan_json(init,file.path(out,"init.json"))
control <- if(label=="s2z_auto") autocenter_control(pilot_args=list(
  output_dir=file.path(out,"precursor"),save_single_paths=TRUE,sig_figs=17)) else NULL
fit <- brm(forms[[label]],data=sites,backend="cmdstanr",algorithm="sampling",chains=1,cores=1,
  iter=3000,warmup=1000,seed=1,normalize=TRUE,init=list(init),
  control=list(adapt_delta=.8,max_treedepth=10),center_control=control,
  stanvars=counter_stanvars(anchor),stan_model_args=list(user_header=file.path(out,"native_gradient_counter.hpp"),
    stanc_options=list("allow-undefined"=TRUE)),output_dir=file.path(out,"sampling"),
  output_basename="air",save_warmup=TRUE,sig_figs=17,refresh=500,silent=0)
saveRDS(fit,file.path(out,"fit.rds"))
code <- stancode(fit);writeLines(code,file.path(out,"instrumented.stan"))
writeLines(strip_counter(code,anchor),file.path(out,"clean.stan"))
write_stan_json(standata(fit),file.path(out,"resolved-data.json"))
if(label=="s2z_auto") {
  saveRDS(fit$autocenter,file.path(out,"autocenter.rds"))
  capture.output(centering_weights(fit),file=file.path(out,"centering-weights.txt"))
}
costs <- collect_native_csv(list.files(file.path(out,"sampling"),pattern="\\.csv$",full.names=TRUE),
  file.path(out,"counter-receipts"),out)
if(label!="s2z_auto") stopifnot(costs$precursor_gradient_calls==0)
write_json(list(brms_sha=pin,grouping=grouping,hierarchy=hierarchy,arm=label,seed=1,
  chains=1,warmup=1000,draws=2000,cmdstan=as.character(cmdstan_version()),
  cmdstanr=as.character(packageVersion("cmdstanr")),source_sha256=digest(file.path(out,"clean.stan"),algo="sha256",file=TRUE)),
  file.path(out,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))
cat("AIR_NATIVE_COMPLETE ",grouping," ",hierarchy," ",label,"\n")
