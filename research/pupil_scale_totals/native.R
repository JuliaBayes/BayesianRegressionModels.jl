# Rscript native.R CMDSTAN_PATH OUTPUT LABEL INIT_JSON AUDIT_JSON
extra <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra)) .libPaths(c(extra,.libPaths()))
suppressPackageStartupMessages({library(brms);library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE);stopifnot(length(args)==5L)
stopifnot(fromJSON(args[[5]])$status=="passed")
set_cmdstan_path(args[[1]])
out <- normalizePath(args[[2]],mustWork=FALSE)
stopifnot(!dir.exists(out));dir.create(out,recursive=TRUE)
label <- args[[3]]
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)))
here <- dirname(script);old <- file.path(here,"..","pupil_total_effects")
source(file.path(old,"native_tools.R"))
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
stopifnot(identical(readLines(file.path(dirname(find.package("brms")),"brms-source-sha.txt")),pin))
load(file.path(old,"reference","df_pupil_complete.rda"))
base <- fromJSON(args[[4]])
# Keep vectors of length one as vectors in both R and exported Stan JSON.
ordinary <- list(b=as.numeric(base$b),Intercept=base$Intercept,
  Intercept_sigma=base$Intercept_sigma,sd_1=as.numeric(base$sd_1),sd_2=as.numeric(base$sd_2),
  z_1=matrix(unlist(base$z_1),nrow=2,byrow=TRUE),z_2=matrix(0,nrow=1,ncol=20))
stopifnot(all(ordinary$z_1==0))
s2z <- list(theta_s2z=c(base$Intercept,as.numeric(base$b)),
  theta_s2z_sigma=c(base$Intercept_sigma),sd_1=as.numeric(base$sd_1),sd_2=as.numeric(base$sd_2),
  z_s2z_1=rep(0,38),z_s2z_2=rep(0,19),udf_b_s2z_1=1/3,udf_b_s2z_sigma_1=1/3)
forms <- list(
  ordinary_ncp=bf(p_size ~ load + (load || subj),sigma ~ (1|subj)),
  s2z_cp=bf(p_size ~ load + (load|gr(subj,cor=FALSE,s2z=TRUE,center=TRUE)),
    sigma ~ (1|gr(subj,s2z=TRUE,center=TRUE))),
  s2z_ncp=bf(p_size ~ load + (load|gr(subj,cor=FALSE,s2z=TRUE,center=FALSE)),
    sigma ~ (1|gr(subj,s2z=TRUE,center=FALSE))),
  s2z_auto=bf(p_size ~ load + (load|gr(subj,cor=FALSE,s2z=TRUE,center="auto")),
    sigma ~ (1|gr(subj,s2z=TRUE,center="auto"))))
stopifnot(label %in% names(forms))
for(sub in c("sampling","precursor","counter-receipts","compiled")) dir.create(file.path(out,sub))
options(cmdstanr_write_stan_file_dir=file.path(out,"compiled"),brms.normalize=TRUE)
Sys.setenv(PUPIL_GRAD_COUNTER_DIR=file.path(out,"counter-receipts"))
stopifnot(file.copy(file.path(old,"native_gradient_counter.hpp"),file.path(out,"native_gradient_counter.hpp")))
stopifnot(file.copy(script,file.path(out,"native.R")))
stopifnot(file.copy(file.path(old,"native_tools.R"),file.path(out,"native_tools.R")))
init <- if(label=="ordinary_ncp") ordinary else s2z
write_stan_json(init,file.path(out,"init.json"))
control <- if(label=="s2z_auto") autocenter_control(pilot_args=list(
  output_dir=file.path(out,"precursor"),save_single_paths=TRUE,sig_figs=17)) else NULL
anchor <- if(label=="ordinary_ncp") "Intercept" else "theta_s2z[1]"
fit <- brm(forms[[label]],data=df_pupil_complete,backend="cmdstanr",algorithm="sampling",
  chains=1,cores=1,iter=3000,warmup=1000,seed=1,normalize=TRUE,init=list(init),
  control=list(adapt_delta=.8,max_treedepth=10),center_control=control,
  stanvars=counter_stanvars(anchor),stan_model_args=list(user_header=file.path(out,"native_gradient_counter.hpp"),
    stanc_options=list("allow-undefined"=TRUE)),output_dir=file.path(out,"sampling"),
  output_basename="pupil4",save_warmup=TRUE,sig_figs=17,refresh=500,silent=0)
saveRDS(fit,file.path(out,"fit.rds"))
code <- stancode(fit)
writeLines(code,file.path(out,"instrumented.stan"))
writeLines(strip_counter(code,anchor),file.path(out,"clean.stan"))
write_stan_json(standata(fit),file.path(out,"resolved-data.json"))
if(label=="s2z_auto") {
  saveRDS(fit$autocenter,file.path(out,"autocenter.rds"))
  capture.output(centering_weights(fit),file=file.path(out,"centering-weights.txt"))
}
csv <- list.files(file.path(out,"sampling"),pattern="\\.csv$",full.names=TRUE)
costs <- collect_native_csv(csv,file.path(out,"counter-receipts"),out)
if(label!="s2z_auto") stopifnot(costs$precursor_gradient_calls==0)
write_json(list(brms_sha=pin,source_post=4,arm=label,seed=1,chains=1,warmup=1000,draws=2000,
  cmdstan=as.character(cmdstan_version()),cmdstanr=as.character(packageVersion("cmdstanr")),
  initialization="common pooled total coefficients, zero deviations, subject-OLS group SDs, both mixture precisions one",
  source_sha256=digest(file.path(out,"clean.stan"),algo="sha256",file=TRUE)),
  file.path(out,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))
cat("PUPIL4_NATIVE_COMPLETE ",label,"\n")
