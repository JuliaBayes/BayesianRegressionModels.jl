# ---- preprocessing-constant provenance (decision nr3v8n A) ------------------
# Each Category-A transform (zscale/standardize/center/factor/mo/s/t2/gp/hsgp),
# interval-censored predictor, and element-wise `protect`/implicit-fn fallback
# computes data-derived values in Julia at construct-time and lands only the
# transformed/split result in `data`. To support `reprocess`/`restan_data` on a
# new DataFrame we record, per EMITTED data key, how to regenerate it:
#   kind         -- which transform (:zscale/:standardize/:center/:factor/:mo/
#                   :spline/:tensor_spline/:gp/:hsgp/:protect/:interaction/
#                   :categorical_outcome/:ordinal_outcome/
#                   :ordinal_threshold_predictor/:interval_censored_predictor)
#   const_       -- the fitted constant: (μ,σ) / μ / level-vector / TPS basis /
#                   tensor-spline margins/centers / exact-GP axis metadata /
#                   HSGP (μ,L,K,c) / categorical
#                   outcome levels / nothing (protect)
#   raw_ref      -- the source: a column-node tree (zscale/center/standardize/
#                   protect, re-materialised by shared expression replay) or a
#                   column NAME Symbol (factor/mo/spline) or axis-name Tuple
#                   (gp/hsgp)
#   dim_coupled  -- true when a fitted level set drives parameter dimension
struct PreprocEntry
    kind::Symbol
    const_::Any
    raw_ref::Any
    dim_coupled::Bool
end

# Preserve the established type identity while exposing a common preparation
# spelling to backend-independent consumers.
const _BRMPreprocEntry = PreprocEntry
