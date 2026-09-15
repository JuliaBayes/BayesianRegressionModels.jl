# Capture the original AIR priors for each grouping and supported independent
# hierarchy. This performs no fitting and never removes a population effect.
extra <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra)) .libPaths(c(extra,.libPaths()))
suppressPackageStartupMessages({library(brms);library(cmdstanr);library(jsonlite);library(digest)})
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)))
here <- dirname(script)
data_path <- file.path(here,"reference","pm_sites.json")
stopifnot(digest(data_path,algo="sha256",file=TRUE)=="8eed2b16c17fd1501d616dda0c983e57d44cc2f5552b21fbcad3c09e864507cd")
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
stopifnot(identical(readLines(file.path(dirname(find.package("brms")),"brms-source-sha.txt")),pin))
sites <- as.data.frame(fromJSON(data_path));stopifnot(nrow(sites)==6003)
out <- commandArgs(trailingOnly=TRUE)[[1]]
for(group in c("cluster_region","cluster_log_region","super_region")) {
  sites$region <- sites[[group]]
  for(hierarchy in c("intercept_only","independent")) {
    re <- if(hierarchy=="intercept_only") "1" else "1 + log_sat"
    forms <- list(
      ordinary_ncp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," || region)"))),
      s2z_ncp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region, cor=FALSE, s2z=TRUE, center=FALSE))"))),
      s2z_cp=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region, cor=FALSE, s2z=TRUE, center=TRUE))"))),
      s2z_auto=bf(as.formula(paste0("log_pm25 ~ log_sat + (",re," | gr(region, cor=FALSE, s2z=TRUE, center='auto'))"))))
    dest <- file.path(out,group,hierarchy);dir.create(dest,recursive=TRUE,showWarnings=FALSE)
    for(label in names(forms)) {
      f <- forms[[label]]
      writeLines(make_stancode(f,data=sites,normalize=TRUE),file.path(dest,paste0(label,".stan")))
      write_stan_json(make_standata(f,data=sites),file.path(dest,paste0(label,".json")))
      write.table(get_prior(f,data=sites),file.path(dest,paste0(label,"-priors.tsv")),sep="\t",row.names=FALSE,quote=FALSE)
    }
    write_json(list(brms_sha=pin,grouping=group,hierarchy=hierarchy,
      observations=nrow(sites),groups=length(unique(sites$region)),
      group_sizes=as.integer(table(sites$region)),population_effects="intercept and slope retained",
      source_data_revision="2afb605f81cf0bdadc6476df865e0337aaacf183"),
      file.path(dest,"provenance.json"),auto_unbox=TRUE,pretty=TRUE)
    cat("AIR_CAPTURED ",group," ",hierarchy,"\n")
  }
}
