#!/usr/bin/env julia

# Offline validation: no listener is opened and no external service is used.

using Test
using DynamicObjects
using HTMXObjects

include(joinpath(@__DIR__, "HistoricalInventoryGallery.jl"))
using .HistoricalInventoryGallery

@testset "historical inventory gallery" begin
    graph = build_gallery()
    surface = graph.surface
    physical_rows = length(readlines(DEFAULT_MATRIX_PATH)) - 1

    @test length(surface.rows) == physical_rows == 359
    @test length(unique(row["row_key"] for row in surface.rows)) == physical_rows
    @test length(filtered_rows(graph)) == physical_rows

    for property in (
        :source_fidelity,
        :family_provenance,
        :translation_route,
        :validation_tier,
    )
        domain = property_options(surface, property)
        records = option_records(domain)
        @test records !== nothing
        @test first(records).value == "all"
        @test first(records).label == "All (359)"
        @test getproperty(surface, property) in domain
    end

    @test Set(getproperty.(option_records(
        property_options(surface, :translation_route)), :value)) ==
        Set(["all", "ordinary_brm", "stanblocks_plate", "brm_kernel"])
    @test Set(getproperty.(option_records(
        property_options(surface, :validation_tier)), :value)) ==
        union(Set(["all"]), Set(row["inferred_capability_tier"] for row in surface.rows))

    confirmed = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH, source_fidelity="confirmed")
    @test length(filtered_rows(confirmed)) == 154
    exact = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH,
        validation_tier="bridgestan-finite-density-gradient")
    @test length(filtered_rows(exact)) == 180 == count(
        row -> row["inferred_capability_tier"] == "bridgestan-finite-density-gradient",
        surface.rows,
    )
    refreshed_keys = Set([
        "mcelreath:chimpanzees_intercept",
        "mcelreath:chimpanzees_slopes",
        "mcelreath:moralizing_gods",
        "kruschke:recall_conditions",
        "kruschke:recall_pooled",
    ])
    refreshed = filter(row -> row["row_key"] in refreshed_keys, surface.rows)
    @test length(refreshed) == length(refreshed_keys)
    @test all(row ->
        row["inferred_capability_tier"] == "bridgestan-finite-density-gradient" &&
        row["descriptor"] == "pass" && row["stanc"] == "pass" &&
        row["bridgestan_instantiate"] == "pass" &&
        row["gradient_finite"] == "true" && isempty(row["failure_stage"]),
        refreshed,
    )
    beta_binomial_keys = Set([
        "brms:cbpp_beta_binomial",
        "mcelreath:ucbadmit_beta_binomial",
    ])
    beta_binomial = filter(row -> row["row_key"] in beta_binomial_keys, surface.rows)
    @test length(beta_binomial) == length(beta_binomial_keys)
    @test all(row ->
        row["inferred_translation_status"] == "ready" &&
        row["inferred_capability_tier"] == "bridgestan-finite-density-gradient" &&
        occursin("BetaBinomial2", row["inferred_current_brm_body"]) &&
        row["family_adapter_validation_sha"] ==
            "98d54fb413e3994eb3e4c9ea76d659cddce433c5" &&
        row["descriptor"] == "pass" && row["stanc"] == "pass" &&
        row["bridgestan_instantiate"] == "pass" &&
        row["gradient_finite"] == "true" && isempty(row["failure_stage"]),
        beta_binomial,
    )
    student_zip_keys = Set([
        "kruschke:income_famsize",
        "kruschke:guber1999_base",
        "kruschke:guber1999_complement",
        "kruschke:guber1999_interaction",
        "bambi:zip_mu",
        "bambi:zip_psi",
        "bambi:plot_comp_zip",
    ])
    student_zip = filter(row -> row["row_key"] in student_zip_keys, surface.rows)
    @test length(student_zip) == length(student_zip_keys)
    @test all(row ->
        row["inferred_translation_status"] == "ready" &&
        row["inferred_capability_tier"] == "bridgestan-finite-density-gradient" &&
        row["descriptor"] == "pass" && row["stanc"] == "pass" &&
        row["bridgestan_instantiate"] == "pass" &&
        row["gradient_finite"] == "true" && isempty(row["failure_stage"]),
        student_zip,
    )
    # The psi-submodel card shares its partner's joint probe rather than
    # carrying a second direct compilation.
    psi = only(filter(row -> row["row_key"] == "bambi:zip_psi", surface.rows))
    mu = only(filter(row -> row["row_key"] == "bambi:zip_mu", surface.rows))
    @test psi["probe_evidence_kind"] == "inherited-identical-probe"
    @test psi["probe_evidence_from_row"] == mu["row_index"]
    @test psi["probe_id"] == mu["probe_id"]
    combined = HistoricalInventory(;
        matrix_path=DEFAULT_MATRIX_PATH,
        translation_route="ordinary_brm", validation_tier="unsupported")
    @test all(row["semantic_route"] == "ordinary_brm" &&
              row["inferred_capability_tier"] == "unsupported"
              for row in filtered_rows(combined))

    cards = [row_card(row) for row in surface.rows]
    @test length(cards) == physical_rows
    @test all(card -> card isa SemanticCard, cards)
    @test all(card -> any(child -> child isa SemanticFields, card.children), cards)
    @test all(card -> any(child -> child isa SemanticStatus, card.children), cards)
    @test all(card -> any(child -> child isa SemanticDisclosure, card.children), cards)

    unresolved_indices = findall(row ->
        row["inferred_translation_status"] != "ready", surface.rows)
    @test length(unresolved_indices) == count(
        row -> row["inferred_translation_status"] != "ready", surface.rows)
    @test all(index -> any(child -> child isa SemanticUnavailable,
                          cards[index].children), unresolved_indices)

    first_card_html = repr("text/html", first(cards))
    first_card_md = repr("text/markdown", first(cards))
    @test occursin(first(surface.rows)["row_key"], first_card_html)
    @test occursin(first(surface.rows)["row_key"], first_card_md)
    @test occursin("experimental", lowercase(first_card_md))
    @test occursin("```julia", first_card_md)

    inventory = semantic_inventory(surface.rows, length(surface.rows))
    inventory_md = repr("text/markdown", inventory)
    @test all(row -> occursin(row["row_key"], inventory_md), surface.rows)
    @test length(findall("### ", inventory_md)) == physical_rows

    semantic_html = repr("text/html", semantic_app(
        surface;
        title="Historical Bayesian regression models",
        render_operation=HistoricalInventoryGallery._gallery_semantic_operation,
    ))
    for name in ("source_fidelity", "family_provenance",
                 "translation_route", "validation_tier")
        @test occursin("name=\"$name\"", semantic_html)
    end
    @test occursin("hx-get=\"/surface/\"", semantic_html)
    @test !occursin("AppData", inventory_md)
    @test !occursin("AppContext", inventory_md)

    # Exercise the registered HTTP route entirely in process. This opens no
    # socket, but reproduces HTMXObjects' standard browser-shell -> HTMX
    # fragment sequence and proves that later filters reach the same
    # authoritative graph.
    route!(graph)
    drive(path; headers=Pair{String,String}[]) = begin
        request = HTTP.Request("GET", path, headers)
        handler = first(HTTP.Handlers.gethandler(
            HTMXObjects.CONTEXT[].service.router, request))
        handler(request)
    end
    try
        browser_headers = [
            "Accept" => "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        ]
        initial = drive("/"; headers=browser_headers)
        initial_body = String(initial.body)
        @test initial.status == 200
        @test occursin("text/html", HTTP.header(initial, "Content-Type", ""))
        @test occursin(
            "https://cdn.jsdelivr.net/npm/htmx.org@2.0.8/dist/htmx.min.js",
            initial_body,
        )
        @test !occursin("var htmx=function()", initial_body)
        @test isempty(findall(
            "<article class=\"htmxo-semantic-card-body\"", initial_body))
        @test occursin("data-htmxo-operation-load", initial_body)
        @test occursin("hx-trigger=\"load\"", initial_body)
        @test occursin("hx-get=\"/\"", initial_body)

        prefixed = drive("/"; headers=vcat(browser_headers, [
            "X-Forwarded-Prefix" => "/p/HistoricalBRM",
        ]))
        prefixed_body = String(prefixed.body)
        @test prefixed.status == 200
        @test isempty(findall(
            "<article class=\"htmxo-semantic-card-body\"", prefixed_body))
        @test occursin("data-htmxo-operation-load", prefixed_body)
        @test occursin("hx-trigger=\"load\"", prefixed_body)
        @test occursin("hx-get=\"/p/HistoricalBRM/\"", prefixed_body)

        full = drive("/"; headers=vcat(browser_headers, ["HX-Request" => "true"]))
        full_body = String(full.body)
        @test full.status == 200
        @test occursin("text/html", HTTP.header(full, "Content-Type", ""))
        @test length(findall(
            "<article class=\"htmxo-semantic-card-body\"", full_body)) == 359
        @test all(name -> occursin("name=\"$name\"", full_body),
                  ("source_fidelity", "family_provenance",
                   "translation_route", "validation_tier"))

        fidelity = drive(
            "/surface/?source_fidelity=confirmed&family_provenance=all" *
            "&translation_route=all&validation_tier=all";
            headers=vcat(browser_headers, [
                "HX-Request" => "true",
                "X-Forwarded-Prefix" => "/p/HistoricalBRM",
            ]),
        )
        fidelity_body = String(fidelity.body)
        @test fidelity.status == 200
        @test length(findall(
            "<article class=\"htmxo-semantic-card-body\"",
            fidelity_body)) == 154
    finally
        terminate()
    end
end

println("offline gallery validation: ok")
