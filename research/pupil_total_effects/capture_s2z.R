extra_library <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra_library)) .libPaths(c(extra_library,.libPaths()))
suppressPackageStartupMessages({library(brms);library(jsonlite)})
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
receipt <- file.path(dirname(find.package("brms")),"brms-source-sha.txt")
stopifnot(identical(readLines(receipt),pin))
out <- commandArgs(trailingOnly=TRUE)[[1]]
stopifnot(!dir.exists(out));dir.create(out,recursive=TRUE)
load("research/pupil_total_effects/reference/df_pupil_complete.rda")
formulas <- list(
  s2z_ncp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=FALSE)),sigma ~ subj),
  s2z_cp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=TRUE)),sigma ~ subj),
  s2z_auto=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center="auto")),sigma ~ subj))
for(label in names(formulas)) {
  formula <- formulas[[label]]
  writeLines(make_stancode(formula,data=df_pupil_complete,normalize=TRUE),file.path(out,paste0(label,".stan")))
  cmdstanr::write_stan_json(make_standata(formula,data=df_pupil_complete),
      file.path(out,paste0(label,".json")))
  capture.output(formula,file=file.path(out,paste0(label,".formula.txt")))
  cat("CAPTURED\t",label,"\n")
}
