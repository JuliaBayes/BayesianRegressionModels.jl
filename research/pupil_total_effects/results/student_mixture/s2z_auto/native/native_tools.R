counter_statements <- function(anchor) c(
  "real pupil_record_gradient(real x);", "real pupil_gradient_count();",
  paste0("real pupil_counter_tick = pupil_record_gradient(",anchor,");"),
  "target += pupil_counter_tick;", "real pupil_gradients = pupil_gradient_count();")

counter_stanvars <- function(anchor) {
  s <- counter_statements(anchor)
  brms::stanvar(scode=paste(s[1:2],collapse="\n"),block="functions") +
    brms::stanvar(scode=s[[3]],block="tparameters",position="start") +
    brms::stanvar(scode=s[[4]],block="model",position="start") +
    brms::stanvar(scode=s[[5]],block="genquant",position="end")
}

strip_counter <- function(code,anchor) {
  for(s in counter_statements(anchor)) {
    stopifnot(grepl(s,code,fixed=TRUE))
    code <- gsub(s,"",code,fixed=TRUE)
  }
  stopifnot(!grepl("pupil_counter_tick|pupil_record_gradient|pupil_gradient_count",code))
  code
}

collect_native_csv <- function(csv,receipts,out,warmup=1000L,draws=2000L) {
  stopifnot(length(csv)==1L)
  x <- read.csv(csv,comment.char="#",check.names=FALSE)
  stopifnot(nrow(x)==warmup+draws,all(is.finite(x$pupil_gradients)),all(diff(x$pupil_gradients)>=0))
  indices <- (warmup+1L):(warmup+draws)
  sampling <- x[indices,,drop=FALSE]
  increments <- x$pupil_gradients[indices]-x$pupil_gradients[indices-1L]
  stopifnot(all(increments==sampling$n_leapfrog__+1))
  processes <- do.call(rbind,lapply(list.files(receipts,pattern="\\.tsv$",full.names=TRUE),read.delim))
  main_cost <- tail(x$pupil_gradients,1)
  workflow_cost <- sum(processes$gradient_evaluations)
  stopifnot(workflow_cost>=main_cost)
  costs <- data.frame(sampling_gradients=sum(increments),all_gradient_calls=main_cost,
    warmup_and_initial_gradients=x$pupil_gradients[[warmup]],
    precursor_gradient_calls=workflow_cost-main_cost,workflow_gradient_calls=workflow_cost,
    divergences=sum(sampling$divergent__),max_depth_hits=sum(sampling$treedepth__>=10),
    sampling_rows=draws,warmup_rows=warmup)
  write.table(costs,file.path(out,"gradient_counts.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
  write.table(sampling,file.path(out,"sampling.tsv"),sep="\t",row.names=FALSE,quote=FALSE)
  print(costs)
  costs
}
