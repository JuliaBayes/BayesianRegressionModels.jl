# Exact exported brms target, sampled with native CmdStan NUTS.
# Rscript native_stan.R CMDSTAN_PATH NEW_OUTPUT gaussian|student_mixture INIT_JSON
suppressPackageStartupMessages({library(cmdstanr);library(jsonlite);library(digest)})
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L, args[[3]] %in% c("gaussian","student_mixture"))
set_cmdstan_path(args[[1]])
out <- normalizePath(args[[2]],mustWork=FALSE)
stopifnot(!dir.exists(out))
dir.create(out,recursive=TRUE)
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)))
here <- dirname(script)
prior <- args[[3]]
source <- file.path(here,"reference",if(prior=="gaussian")
    "pupil-uncorrelated-gaussian.stan" else "pupil-uncorrelated-student.stan")
code <- paste(readLines(source),collapse="\n")
original <- code
changes <- list(
  c("functions {","functions {\n  real pupil_record_gradient(real x);\n  real pupil_gradient_count();"),
  c("transformed parameters {","transformed parameters {\n  real pupil_counter_tick = pupil_record_gradient(Intercept);"),
  c("model {","model {\n  target += pupil_counter_tick;"),
  c("generated quantities {","generated quantities {\n  real pupil_gradients = pupil_gradient_count();"))
for(change in changes) {
  stopifnot(length(gregexpr(change[[1]],code,fixed=TRUE)[[1]])==1L,
            grepl(change[[1]],code,fixed=TRUE))
  code <- sub(change[[1]],change[[2]],code,fixed=TRUE)
}
clean <- code
for(change in rev(changes)) clean <- sub(change[[2]],change[[1]],clean,fixed=TRUE)
stopifnot(identical(clean,original))
target <- file.path(out,"pupil-counted.stan")
writeLines(code,target)
stopifnot(file.copy(source,file.path(out,"pupil-original.stan")))
stopifnot(file.copy(file.path(here,"reference","standata.json"),file.path(out,"data.json")))
stopifnot(file.copy(args[[4]],file.path(out,"init.json")))
header <- file.path(out,"native_gradient_counter.hpp")
stopifnot(file.copy(file.path(here,"native_gradient_counter.hpp"),header))
stopifnot(file.copy(script,file.path(out,"native_stan.R")))
receipts <- file.path(out,"counter-receipts");dir.create(receipts)
Sys.setenv(PUPIL_GRAD_COUNTER_DIR=receipts)
model <- cmdstan_model(target,user_header=header,stanc_options=list("allow-undefined"=TRUE))
fit <- model$sample(data=file.path(out,"data.json"),init=file.path(out,"init.json"),
  chains=1,parallel_chains=1,iter_warmup=1000,iter_sampling=2000,
  seed=1,adapt_delta=0.8,max_treedepth=10,save_warmup=TRUE,sig_figs=17,
  output_dir=out,output_basename="pupil",refresh=500)
saveRDS(fit,file.path(out,"fit.rds"))
csv <- fit$output_files()
stopifnot(length(csv)==1L)
x <- read.csv(csv,comment.char="#",check.names=FALSE)
stopifnot(nrow(x)==3000L,all(is.finite(x$pupil_gradients)),all(diff(x$pupil_gradients)>=0))
sampling <- x[1001:3000,,drop=FALSE]
increments <- x$pupil_gradients[1001:3000]-x$pupil_gradients[1000:2999]
stopifnot(all(increments==sampling$n_leapfrog__+1))
processes <- do.call(rbind,lapply(list.files(receipts,pattern="\\.tsv$",full.names=TRUE),read.delim))
stopifnot(sum(processes$gradient_evaluations)==tail(x$pupil_gradients,1))
costs <- data.frame(sampling_gradients=sum(increments),
  all_gradient_calls=tail(x$pupil_gradients,1),warmup_and_initial_gradients=x$pupil_gradients[[1000]],
  divergences=sum(sampling$divergent__),max_depth_hits=sum(sampling$treedepth__>=10),
  sampling_rows=2000,warmup_rows=1000)
write.table(costs,file.path(out,"gradient_counts.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
write.table(sampling,file.path(out,"sampling.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
write_json(list(brms_target_sha256=digest(source,algo="sha256",file=TRUE),
  cmdstan=as.character(cmdstan_version()),cmdstanr=as.character(packageVersion("cmdstanr")),
  intercept_prior=prior,algorithm="native Stan NUTS, diagonal metric",seed=1,chains=1,
  iter_warmup=1000,iter_sampling=2000,adapt_delta=0.8,max_treedepth=10,
  initialization="same physical OLS point supplied to WarmupHMC, without its Pathfinder initialization",
  counter_scope="reverse-mode target calls reaching first transformed-parameter statement",
  counter_identity="sampling increments equal leapfrogs+1, checked on every retained transition"),
  file.path(out,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
capture.output(sessionInfo(),file=file.path(out,"R-session.txt"))
print(costs);cat("NATIVE_COMPLETE\t",out,"\n")
