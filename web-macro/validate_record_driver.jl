#!/usr/bin/env julia

# Offline smoke for the gallery record driver (`record_gallery` in
# web-macro/src/app_data.jl): route a fresh AppContext with a recording
# config and prove the registered router resolves the record paths.
# Run from the repository root:
#   julia --project=web-macro web-macro/validate_record_driver.jl
#
# No listener is opened and no Stan program runs: handlers are looked up
# via `HTTP.Handlers.gethandler`, never invoked. This pins the exact call
# shape `record_gallery` uses (`route!(app; record_dir, record_base)` +
# `HTMXObjects.ROUTER`), which broke when upstream removed `CONTEXT`.

cd(mktempdir())

using Test
using HTMXObjects # re-exports HTTP

include(joinpath(@__DIR__, "src", "BRMMacroWeb.jl"))

@testset "record driver" begin
    app = BRMMacroWeb.AppContext()
    record_dir = mktempdir()
    HTMXObjects.route!(app; record_dir, record_base="")
    router = HTMXObjects.ROUTER
    for path in ("/pipeline/gallery", "/library")
        request = HTTP.Request("GET", path, Pair{String,String}[])
        handler = first(HTTP.Handlers.gethandler(router, request))
        @test handler !== HTTP.Handlers.default404
    end
    @test isdefined(HTMXObjects, :_drive_record_path)
    HTMXObjects.route!(app)
end

println("record driver smoke: ok")
