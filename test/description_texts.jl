# Description language: `brm_description(d; texts=description_texts(:de))` writes
# every BRM text in German and the Markdown follows the description's language.
using Test, BayesianRegressionModels, Distributions
using BayesianRegressionModels: aweights

const TEXT_DATA=(;x=[-2.0,-0.5,0.7,2.5,3.0,4.2],g=[1,1,2,2,3,3],
    y=[0.4,0.1,1.1,1.7,2.4,2.8],z=[1.4,1.1,2.1,2.7,3.4,3.8],
    w=[1.,2.,1.,3.,2.,1.],z_upper=[1.9,1.6,2.6,3.2,3.9,4.3])
# Areas a..h: component {a, b, c, d, h}, component {e, f}, island g.
const AREA_DATA=(;area=["a","b","c","d","e","f","g","a","e"],y=zeros(9),
    from=["a","b","c","a","d","e"],to=["b","c","d","c","h","f"])

# Synthetic models reaching the main description paths: random-effect blocks
# and covariance factors, prepared columns, structured terms, R2D2 allocation,
# observation wrappers, imputation, multi-membership, areal fields and a
# coverage gap.
const TEXT_MODELS=(
    hierarchical=@brm(TEXT_DATA, begin
        sigma ~ Exponential(2)
        mu ~ 1 + zscale(x) + (1 + x | shared | g)
        effect(mu, zscale_x) ~ Normal(0, 0.4)
        y ~ Normal(mu, sigma)
    end),
    structured=@brm(TEXT_DATA, begin
        mu ~ 1 + mo(g) + hsgp(x;k=3)
        length_scale(:,hsgp(x)) ~ Uniform(0.8,2.0)
        y ~ LocationScale(mu,addprop(mu,0.5,0.2),TDist(5.0))
    end),
    allocation=@brm(TEXT_DATA, begin
        mu ~ 1+x+(1|g)
        effect(mu,:) ~ r2d2(R2=Beta(2,3),tau_bsv=0.5)
        y ~ Normal(mu,1.0)
    end),
    evidence=@brm(TEXT_DATA, begin
        mu ~ 1 + x
        y ~ weighted(Normal(mu,1.5),aweights(w))
        z ~ interval_censored(Normal(mu,1.0);upper=z_upper)
    end),
    wrappers=@brm(TEXT_DATA, begin
        mu ~ 1 + x
        y ~ censored(Normal(mu,1.0);lower=0.0,upper=3.0)
        z ~ truncated(Normal(mu,1.0);lower=0.0)
    end),
    smooth=@brm((;x=collect(range(-2.,4.;length=12)),y=sin.(range(-2.,4.;length=12))), begin
        mu ~ 0 + s(x)
        y ~ Normal(mu,1.0)
    end),
    imputation=@brm(merge(TEXT_DATA,(;x=Union{Missing,Float64}[-2.,missing,.7,2.5,missing,4.2])), begin
        mi(x) ~ Normal(0,1)
        squared=x*x
        mu ~ 1 + center(x) + squared
        y ~ Normal(mu,1.0)
    end),
    membership=@brm((;g1=["a","a","b"],g2=["b","c","c"],w1=[2.,1.,0.],w2=[1.,1.,3.],y=[.2,.5,.7]), begin
        mu ~ 1+(1|mm(g1,g2;weights=(w1,w2)))
        y ~ Normal(mu,1.0)
    end),
    bym2=@brm(AREA_DATA, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 1.0)
    end),
    icar=@brm(AREA_DATA, begin
        mu ~ 1 + car(area; edges=(from, to), type=:icar)
        y ~ Normal(mu, 1.0)
    end),
    horseshoe=@brm(TEXT_DATA, begin
        mu ~ 1+x
        effect(mu,x) ~ Horseshoe(local_scale=0.5,global_scale=0.2)
        y ~ Normal(mu,1.0)
    end),
)
const TEXT_DESCRIPTORS=map(m->brm_descriptor(SBBRMI(m;mod=@__MODULE__,total_groups=())),TEXT_MODELS)

const EN=description_texts(:en)
const DE=description_texts(:de)
const PLACEHOLDER=r"(?<![A-Za-z\\_^])\{([A-Za-z_]+)\}"
_page(r)=join((brm_description_markdown(r),r.prose...,r.equations...,
    (get(n,:meaning,"") for n in r.notation)...,r.diagnostics...),"\n")

@testset "English is the default and the language is recorded" begin
    for d in TEXT_DESCRIPTORS
        default=brm_description(d)
        explicit=brm_description(d;texts=EN)
        @test brm_description_markdown(default)==brm_description_markdown(explicit)
        @test default.prose==explicit.prose && default.equations==explicit.equations
        @test brm_description_texts(default).locale===:en
        german=brm_description(d;texts=DE)
        @test german.model_id==default.model_id
        @test map(c->c.id,brm_description_components(german))==map(c->c.id,brm_description_components(default))
        @test brm_description_texts(german)===DE
        @test all(c->brm_description_texts(c)===DE,brm_description_components(german))
        @test all(p->brm_description_texts(p)===DE,german.priors)
    end
    @test !brm_description(TEXT_DESCRIPTORS.horseshoe).complete
end

@testset "German descriptions have no English fallback" begin
    @test keys(DE.texts)==keys(EN.texts)
    pages=map(d->_page(brm_description(d;texts=DE)),TEXT_DESCRIPTORS)
    english_pages=map(d->_page(brm_description(d)),TEXT_DESCRIPTORS)
    used=Set{Symbol}()
    for name in keys(EN.texts)
        en=EN.texts[name]; de=DE.texts[name]
        en==de && continue
        # Sentence-sized pieces of an English text never reach a German page.
        for chunk in split(en,PLACEHOLDER)
            occursin(' ',chunk) && length(chunk)>=10 || continue
            @test !any(p->occursin(chunk,p),pages)
        end
        if startswith(string(name),"math_")
            for wrapper in ("\\mathrm{","\\text{","\\operatorname{")
                @test !any(p->occursin(wrapper*en*"}",p),pages)
            end
        end
        # Record which German texts the models exercised.
        any(chunk->length(chunk)>=4 && any(p->occursin(chunk,p),pages),split(de,PLACEHOLDER)) && push!(used,name)
    end
    @info "German texts exercised by the synthetic models" used=length(used) total=length(keys(EN.texts))
    hierarchical=brm_description_markdown(brm_description(TEXT_DESCRIPTORS.hierarchical;texts=DE))
    @test startswith(hierarchical,"Vollständige Modellbeschreibung.\n")
    @test occursin("\nEffektive A-priori-Verteilungen:\n\n| Logischer Parameter | Verteilung und Träger | A-priori-Verzeichnis |",hierarchical)
    @test occursin("\nNotation:\n",hierarchical) && occursin("\nAchse: ",hierarchical)
    @test occursin("**K1.** ",hierarchical)
    gaps=brm_description_markdown(brm_description(TEXT_DESCRIPTORS.horseshoe;texts=DE))
    @test startswith(gaps,"Unvollständige Modellbeschreibung: Es bleiben semantische Abdeckungslücken.\n")
    @test occursin("\nAbdeckungslücken:\n\n- Keine wissenschaftliche Beschreibung für ",gaps)
    @test any(p->occursin("\\operatorname{Exponential}_{\\mathrm{Rate}}",p) ||
        occursin("\\operatorname{Exponential}_{\\mathrm{Skala}}",p),pages)
end

@testset "hooks write in the description's language" begin
    hook=Normal=>c->BRMDescriptionFragment(;covers=(c.id,),
        prose=(brm_description_texts(c).locale===:de ? "Eigene Beobachtungsdichte." : "Own observation density.",))
    d=TEXT_DESCRIPTORS.hierarchical
    @test "Eigene Beobachtungsdichte." in brm_description(d;hooks=(hook,),texts=DE).prose
    @test "Own observation density." in brm_description(d;hooks=(hook,)).prose
end

@testset "texts are replaced by keyword and validated" begin
    custom=description_texts(:de;effective_priors="A-priori-Annahmen:",
        math_no_density="(keine \\textit{weitere} Dichte)")
    @test custom.texts.effective_priors=="A-priori-Annahmen:"
    @test occursin("\nA-priori-Annahmen:\n",brm_description_markdown(brm_description(TEXT_DESCRIPTORS.hierarchical;texts=custom)))
    @test repr(DE)=="description_texts(:de)"
    @test repr(custom)=="description_texts(:de; 2 replaced texts)"
    # refused: an unknown locale must not fall back to English silently (decision 0szq845)
    @test_throws ArgumentError description_texts(:fr)
    # refused: a misspelled text name would be ignored and leave the default in place
    @test_throws ArgumentError description_texts(:de;effective_prior="A-priori:")
    # refused: dropping a placeholder would silently omit a model value from the page
    @test_throws ArgumentError description_texts(:de;coverage_gap="Lücke bei {id}")
    # refused: an unknown placeholder has no value to fill in
    @test_throws ArgumentError description_texts(:de;complete_description="Fertig {extra}.")
end
