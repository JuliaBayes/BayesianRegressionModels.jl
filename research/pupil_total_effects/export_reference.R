# Run from the repository root. This exports source semantics, not Stan fits.
extra_library <- Sys.getenv("PUPIL_BRMS_LIBRARY")
if (nzchar(extra_library)) .libPaths(c(extra_library, .libPaths()))
library(brms)
load("research/pupil_total_effects/reference/df_pupil_complete.rda")
stopifnot(is.integer(df_pupil_complete$subj), is.integer(df_pupil_complete$load))
original <- bf(p_size ~ load + (load | subj), sigma ~ subj)
modified <- bf(p_size ~ load + (load || subj), sigma ~ subj)
prior <- set_prior("normal(5651.9, 2026.1)", class="Intercept")
root <- "research/pupil_total_effects/reference"
writeLines(make_stancode(original, data=df_pupil_complete),
           file.path(root,"post3-original.stan"))
writeLines(make_stancode(modified, data=df_pupil_complete, prior=prior),
           file.path(root,"pupil-uncorrelated-gaussian.stan"))
writeLines(make_stancode(modified, data=df_pupil_complete),
           file.path(root,"pupil-uncorrelated-student.stan"))
write.table(get_prior(modified, data=df_pupil_complete, prior=prior),
            file.path(root,"priors.tsv"), sep="\t", row.names=FALSE, quote=FALSE)
data <- make_standata(modified, data=df_pupil_complete, prior=prior)
stopifnot(ncol(data$X_sigma)==2, data$N_1==20, data$M_1==2)
jsonlite::write_json(data, file.path(root,"standata.json"), auto_unbox=TRUE, digits=NA)
description <- packageDescription("brms")
jsonlite::write_json(list(brms_version=as.character(packageVersion("brms")),
    brms_remote_sha=description$RemoteSha, rows=nrow(df_pupil_complete), groups=20,
    subj_class=class(df_pupil_complete$subj), load_class=class(df_pupil_complete$load),
    mean_load=mean(df_pupil_complete$load), mean_subj=mean(df_pupil_complete$subj),
    source_revision="d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c",
    source_url="https://github.com/bnicenboim/bcogsci/blob/d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c/data/df_pupil_complete.rda"),
    file.path(root,"provenance.json"), auto_unbox=TRUE, pretty=TRUE, digits=NA)
cat("Exported original and modified brms targets; sigma uses numeric subject IDs.\n")
