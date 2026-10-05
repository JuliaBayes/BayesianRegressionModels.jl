# These signatures consume syntax, rather than specializing on each model's
# expression tree. Generate their native code in the package image so the first
# artifact request can reuse it. No data, user callable or numerical kernel is
# executed here; emitted numerical code keeps its existing specialization.
if ccall(:jl_generating_output, Cint, ()) == 1
    precompile(_brm_backend_context, (BRMI,))
    precompile(_brm_prepare_program, (BRMI,))
    precompile(_brm_prepare_model, (BRMI,))
    precompile(_brm_rk_unselected_plan, (BRMI,))
    precompile(_brm_rk_plan, (BRMI,))
    precompile(_rk_emit_ast, (_RKStructuralPlan,))
    precompile(_rk_emit_ast, (_RKValuePlan,))
end
