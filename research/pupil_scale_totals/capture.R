# Exact post-4 source, with the requested independent mean random effects.
extra <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if(nzchar(extra)) .libPaths(c(extra,.libPaths()))
suppressPackageStartupMessages({library(brms);library(cmdstanr);library(jsonlite)})
pin <- "73cf607889879cb2a55f50b88d8141d76ff43279"
stopifnot(identical(readLines(file.path(dirname(find.package("brms")),"brms-source-sha.txt")),pin))
out <- commandArgs(trailingOnly=TRUE)[[1]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
load("research/pupil_total_effects/reference/df_pupil_complete.rda")
stopifnot(nrow(df_pupil_complete)==2228, is.integer(df_pupil_complete$subj),
          is.integer(df_pupil_complete$load))
forms <- list(
  ordinary_ncp=bf(p_size ~ load + (load || subj), sigma ~ (1 | subj)),
  ordinary_cp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,center=TRUE)),
                sigma ~ (1 | gr(subj,center=TRUE))),
  s2z_cp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=TRUE)),
           sigma ~ (1 | gr(subj,s2z=TRUE,center=TRUE))),
  s2z_ncp=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center=FALSE)),
            sigma ~ (1 | gr(subj,s2z=TRUE,center=FALSE))),
  s2z_auto=bf(p_size ~ load + (load | gr(subj,cor=FALSE,s2z=TRUE,center="auto")),
             sigma ~ (1 | gr(subj,s2z=TRUE,center="auto"))))
for(label in names(forms)) {
  f <- forms[[label]]
  writeLines(make_stancode(f,data=df_pupil_complete,normalize=TRUE),
             file.path(out,paste0(label,".stan")))
  write_stan_json(make_standata(f,data=df_pupil_complete),
                  file.path(out,paste0(label,".json")))
  write.table(get_prior(f,data=df_pupil_complete),
              file.path(out,paste0(label,"-priors.tsv")),sep="\t",row.names=FALSE,quote=FALSE)
  capture.output(f,file=file.path(out,paste0(label,"-formula.txt")))
  cat("CAPTURED ",label,"\n")
}
write.csv(df_pupil_complete,file.path(out,"pupil.csv"),row.names=FALSE)
write_json(list(brms_sha=pin,brms_version=as.character(packageVersion("brms")),
  post="https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/4",
  data_revision="d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c",rows=2228,subjects=20,
  subject_ids=sort(unique(df_pupil_complete$subj)),
  simplification="independent mean random intercepts and slopes; all original priors retained"),
  file.path(out,"provenance.json"),pretty=TRUE,auto_unbox=TRUE)
