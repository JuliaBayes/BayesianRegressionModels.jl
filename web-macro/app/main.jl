using BRMMacroWeb

begin
    BRMMacroWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8121
    BRMMacroWeb.serve(; host="0.0.0.0", revise=nothing, port, async=true)
end
