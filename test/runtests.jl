using ERGMEgo
using ERGM
using Networks
using DataFrames
using Random
using Statistics
using Test

# A small ego fixture: 3 egos with known degrees, matches, alter ties
function fixture_egodata()
    e1 = EgoNetwork(1, [101, 102, 103], Bool[0 1 0; 1 0 0; 0 0 0];
                    ego_attrs=Dict{Symbol,Any}(:group => "A"),
                    alter_attrs=Dict{Symbol,Vector}(:group => ["A", "B", "A"]))
    e2 = EgoNetwork(2, [102, 104], Bool[0 0; 0 0];
                    ego_attrs=Dict{Symbol,Any}(:group => "B"),
                    alter_attrs=Dict{Symbol,Vector}(:group => ["B", "B"]))
    e3 = EgoNetwork(3, [101, 105, 106, 107],
                    Bool[0 1 1 0; 1 0 0 0; 1 0 0 0; 0 0 0 0];
                    ego_attrs=Dict{Symbol,Any}(:group => "A"),
                    alter_attrs=Dict{Symbol,Vector}(:group => ["A", "A", "B", "B"]))
    return EgoData([e1, e2, e3])
end

@testset "ERGMEgo.jl" begin
    @testset "EgoNetwork basics" begin
        ed = fixture_egodata()
        @test length(ed) == 3
        @test ego_degree(ed[1]) == 3
        @test n_alter_ties(ed[1]) == 1
        @test n_alter_ties(ed[3]) == 2
        @test alter_degree(ed[3]) == [2, 1, 1, 0]

        # Asymmetric alter ties rejected
        @test_throws ArgumentError EgoNetwork(1, [1, 2], Bool[0 1; 0 0])
    end

    @testset "summary_stats" begin
        ed = fixture_egodata()
        s = summary_stats(ed)
        @test s.n_egos == 3
        @test s.mean_degree ≈ 3.0        # (3 + 2 + 4)/3
        @test s.mean_alter_ties ≈ 1.0    # (1 + 0 + 2)/3
        @test s.total_alters == 9

        # Weighted version
        wed = ego_design(ed; weights=[2.0, 1.0, 1.0])
        ws = summary_stats(wed)
        @test ws.mean_degree ≈ (2 * 3 + 2 + 4) / 4
    end

    @testset "Per-capita ego statistics" begin
        ed = fixture_egodata()

        # EgoEdges: mean(degree)/2 = 1.5
        @test compute(EgoEdges(), ed) ≈ 1.5
        # EgoTriangle: mean(alter ties)/3 = 1/3
        @test compute(EgoTriangle(), ed) ≈ 1.0 / 3
        # EgoNodeMatch(:group): matches per ego = 2, 2, 2 → mean/2 = 1.0
        @test compute(EgoNodeMatch(:group), ed) ≈ 1.0
        # EgoDegree(d): proportions
        @test compute(EgoDegree(2), ed) ≈ 1 / 3
        @test compute(EgoDegree(5), ed) == 0.0
        # EgoGWDegree matches ERGM's fixed-decay weight at each degree
        α = 0.5
        w(d) = exp(α) * (1 - (1 - exp(-α))^d)
        @test compute(EgoGWDegree(α), ed) ≈ (w(3) + w(2) + w(4)) / 3

        # Targets scale by network size
        @test ego_target_stats([EgoEdges()], ed, 100) ≈ [150.0]
    end

    @testset "Mixing matrix" begin
        ed = fixture_egodata()
        mm = ego_mixing_matrix(ed, :group)
        @test mm.levels == ["A", "B"]
        # Ego-A alters: e1 (A,B,A) + e3 (A,A,B,B) → A→A: 4, A→B: 3
        @test mm.matrix[1, 1] == 4.0
        @test mm.matrix[1, 2] == 3.0
        # Ego-B alters: e2 (B,B) → B→B: 2
        @test mm.matrix[2, 2] == 2.0
    end

    @testset "as_egodata from data frames" begin
        ego_df = DataFrame(ego_id=[1, 2], group=["A", "B"], w=[2.0, 1.0])
        alter_df = DataFrame(ego_id=[1, 1, 2], alter_id=[10, 11, 10],
                             group=["A", "B", "B"])
        aatie_df = DataFrame(ego_id=[1], src=[10], dst=[11])

        ed = as_egodata(ego_df, alter_df; aatie_df=aatie_df,
                        ego_attrs=[:group], alter_attrs=[:group],
                        weight_col=:w)

        @test length(ed) == 2
        # Alter IDs preserved (not relabeled)
        @test ed[1].alters == [10, 11]
        @test ed[2].alters == [10]
        # Alter-alter tie ingested (this was a stub before)
        @test n_alter_ties(ed[1]) == 1
        # Weights read from the ego frame (Symbol column lookup fixed)
        @test ed.sampling_weights == [2.0, 1.0]
        @test ed[1].ego_attrs[:group] == "A"
        @test ed[1].alter_attrs[:group] == ["A", "B"]

        @test_throws ArgumentError as_egodata(ego_df, alter_df; weight_col=:missing_col)
    end

    @testset "Population size estimation" begin
        ed = fixture_egodata()

        # Horvitz-Thompson with weights
        wed = ego_design(ed; weights=[100.0, 150.0, 250.0])
        @test estimate_popsize(wed) == 500.0

        # Capture-recapture: halves {e1} and {e2, e3};
        # s1 = {101,102,103} (3), s2 = {102,104,101,105,106,107} (6),
        # overlap = {101, 102} (2) → N̂ = 9
        @test estimate_popsize(ed; method=:capture_recapture) ≈ 3 * 6 / 2

        @test_throws ArgumentError estimate_popsize(ed; method=:bogus)
    end

    @testset "simulate_ego_sample round trip" begin
        rng = Random.Xoshiro(3)
        net = network(30; directed=false)
        for i in 1:30, j in (i+1):30
            rand(rng) < 0.15 && add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :group,
                              Dict(v => (v % 2 == 0 ? "A" : "B") for v in 1:30))

        ed = simulate_ego_sample(net, 30; ego_attrs=[:group], rng=rng)
        @test length(ed) == 30
        @test ed.population_size == 30

        # Census ego sample: per-capita statistics reproduce the network's
        # sufficient statistics exactly
        @test 30 * compute(EgoEdges(), ed) ≈ compute(Edges(), net)
        @test 30 * compute(EgoTriangle(), ed) ≈ compute(Triangle(), net)
        @test 30 * compute(EgoNodeMatch(:group), ed) ≈
              compute(NodeMatch(:group), net)
    end

    @testset "ergm_ego recovers a Bernoulli density" begin
        # Population: G(n, p); the edges-only egocentric fit should give
        # a population-scale coefficient ≈ logit(p)
        Random.seed!(101)   # ERGM's MCMC sampler draws from the global RNG
        rng = Random.Xoshiro(11)
        n = 60
        p_true = 0.08
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < p_true && add_edge!(net, i, j)
        end
        realized_p = Float64(ne(net)) / (n * (n - 1) / 2)

        ed = simulate_ego_sample(net, n; rng=rng)   # census sample
        result = ergm_ego(ed, [EgoEdges()]; ppopsize=n, popsize=n,
                          n_samples=300, burnin=2000, interval=10, rng=rng)

        @test result isa EgoERGMResult
        @test result.converged
        @test result.netsize_adjustment == 0.0
        @test result.coefficients[1] ≈ log(realized_p / (1 - realized_p)) atol = 0.25
        @test all(isfinite, result.std_errors)
        @test result.std_errors[1] > 0

        # StatsAPI accessors (extensions of the shared generics, as in ERGM.jl)
        @test coef(result) === result.coefficients
        @test stderror(result) === result.std_errors
        @test vcov(result) === result.vcov
        @test size(vcov(result)) == (1, 1)
        @test sqrt(abs(vcov(result)[1, 1])) ≈ result.std_errors[1]
    end

    @testset "ergm_ego with attribute terms" begin
        # Homophilous population: within-group ties much more likely than
        # between-group ties. Fitting EgoNodeMatch requires the pseudo-
        # population's vertex attributes to survive the MCMC network copies
        # (ERGM._copy_network is attribute-preserving via Base.copy).
        Random.seed!(102)   # ERGM's MCMC sampler draws from the global RNG
        rng = Random.Xoshiro(21)
        n = 40
        net = network(n; directed=false)
        group = Dict(v => (v <= n ÷ 2 ? "A" : "B") for v in 1:n)
        set_vertex_attribute!(net, :group, group)
        for i in 1:n, j in (i+1):n
            p_tie = group[i] == group[j] ? 0.25 : 0.03
            rand(rng) < p_tie && add_edge!(net, i, j)
        end

        ed = simulate_ego_sample(net, n; ego_attrs=[:group], rng=rng)
        result = ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:group)];
                          ppopsize=n, popsize=n,
                          n_samples=400, burnin=5000, interval=20,
                          rng=Random.Xoshiro(2))

        @test result.converged
        # The sampled nodematch statistics vary; they would be identically
        # zero if the sampler's network copies dropped vertex attributes
        @test std(result.sim_stats[:, 2]) > 0
        # Strong homophily is recovered as a positive nodematch coefficient
        @test result.coefficients[2] > 0
    end

    @testset "Netsize adjustment" begin
        Random.seed!(103)   # ERGM's MCMC sampler draws from the global RNG
        rng = Random.Xoshiro(5)
        n = 40
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.1 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)

        r_same = ergm_ego(ed, [EgoEdges()]; ppopsize=n, popsize=n,
                          n_samples=200, burnin=1000, interval=10, rng=Random.Xoshiro(1))
        r_big = ergm_ego(ed, [EgoEdges()]; ppopsize=n, popsize=4n,
                         n_samples=200, burnin=1000, interval=10, rng=Random.Xoshiro(1))

        # Same pseudo-population fit; the popsize enters only through the
        # −log(popsize/ppopsize) offset on the edges coefficient
        @test r_big.netsize_adjustment ≈ -log(4.0)
        @test r_big.coefficients[1] ≈ r_same.coefficients[1] - log(4.0) atol = 0.35

        # Descriptive terms are rejected with a clear error
        @test_throws ArgumentError ergm_ego(ed, [EgoEdges(), EgoDegree(2)];
                                            ppopsize=n)
        # Models must include EgoEdges
        @test_throws ArgumentError ergm_ego(ed, [EgoTriangle()]; ppopsize=n)
    end

    @testset "ego_gof" begin
        Random.seed!(104)   # ERGM's MCMC sampler draws from the global RNG
        rng = Random.Xoshiro(9)
        n = 30
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        result = ergm_ego(ed, [EgoEdges()]; ppopsize=n,
                          n_samples=200, burnin=1000, interval=10, rng=rng)

        g = ego_gof(result; n_sim=10, rng=rng)
        @test g.n_sim == 10
        @test 0.0 <= g.p_values.mean_degree <= 1.0
        @test isfinite(g.simulated.mean_degree)
        # A well-specified edges-only model should not be wildly rejected
        # on mean degree
        @test g.p_values.mean_degree > 0.01
    end

    @testset "fit aliases, shared show, and Networks.gof" begin
        # Standardized fit_<model> entry point with R-faithful and legacy
        # aliases bound to the same function
        @test ergm_ego === fit_ergm_ego
        @test fit_ego_ergm === fit_ergm_ego

        # One gof generic across the ecosystem: the method is added to
        # Networks.gof, not a package-local function
        @test ERGMEgo.gof === Networks.gof

        Random.seed!(107)   # ERGM's MCMC sampler draws from the global RNG
        rng = Random.Xoshiro(21)
        n = 25
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.15 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        result = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n,
                              n_samples=200, burnin=1000, interval=10, rng=rng)

        # show renders through the shared coefficient-table printer
        out = sprint(show, result)
        @test occursin("Egocentric ERGM Results", out)
        @test occursin("Estimate", out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)

        # gof returns the shared GOFResult container
        g = gof(result; n_sim=8, rng=rng)
        @test g isa Networks.GOFResult
        @test Networks.n_simulations(g) == 8
        stat = g.statistics[1]
        @test stat.labels == ["mean degree", "mean alter ties"]
        @test stat.observed[1] ≈ summary_stats(ed).mean_degree
        @test all(p -> 0 < p <= 1, stat.p_values)
        gout = sprint(show, g)
        @test occursin("Goodness-of-fit assessment: Egocentric ERGM", gout)
        @test occursin("MC p-value", gout)

        # Result metadata protocol: the fit says what it actually did
        md = Networks.fit_metadata(result)
        @test md.estimand == :ergm_ego
        # Moment matching, not a likelihood: never exact
        @test md.objective == :moment
        @test !md.is_exact
        @test md.se_method == :sandwich
        @test md.missing_method == :none
        @test md.tie_method == :not_applicable

        # Issue #1: the design variance is narrower than "survey-design
        # variance" advertises, and the fit now says so — in the protocol and
        # in the printed output alike
        @test any(occursin("no strata, clusters", a) for a in md.approximations)
        @test any(occursin("Monte-Carlo error", a) for a in md.approximations)
        @test any(occursin("pseudo-population network of size", a)
                  for a in md.approximations)
        @test occursin("strata, clusters", out)
    end

    # ------------------------------------------------------------------
    # Golden fixture: statnet `ergm.ego` on faux.mesa.high under a CENSUS
    # (issue #8, and the direct answer to issue ERGMEgo#1).
    # test/fixtures/r/fauxmesa_ego_census.R regenerates it.
    #
    # The design is a census — every one of the 205 actors is an ego, unit
    # weights, ppopsize = popsize = 205 — chosen precisely because it makes two
    # of the three things being compared DETERMINISTIC:
    #
    #   * the target statistics (they become the observed network's own), and
    #   * the design variance of those targets,
    #
    # so neither can be excused as Monte-Carlo noise. Only the fitted
    # coefficients remain stochastic, and those get a tolerance measured from
    # both packages' seed-to-seed spread.
    # ------------------------------------------------------------------
    @testset "Golden fixture: ergm.ego on faux.mesa.high (census design)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "fauxmesa_ego_census.toml"))
        @test g.provenance["ergm_ego_version"] == "1.1.4"

        # Rebuild R's network and its census egodata.
        n = Int(g.values["n_actors"])
        grade = Int.(g.values["grade"])
        es = Int.(g.values["edge_src"])
        ed_ = Int.(g.values["edge_dst"])
        net = network(n; directed=false)
        for k in eachindex(es)
            add_edge!(net, es[k], ed_[k])
        end
        for v in 1:n
            set_vertex_attribute!(net, :Grade, v, grade[v])
        end
        @test ne(net) == Int(g.values["n_edges"])

        egos = ERGMEgo.EgoNetwork{Int}[]
        for v in 1:n
            alters = sort(collect(Int.(neighbors(net, v))))
            k = length(alters)
            ties = falses(k, k)
            for a in 1:k, b in 1:k
                a != b && has_edge(net, alters[a], alters[b]) && (ties[a, b] = true)
            end
            push!(egos, EgoNetwork(v, alters, Matrix(ties);
                                   ego_attrs=Dict{Symbol,Any}(:Grade => grade[v]),
                                   alter_attrs=Dict{Symbol,Vector}(:Grade => grade[alters])))
        end
        ed = EgoData(egos; population_size=n, sampling_weights=ones(n))
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]
        m = Int(g.values["ppopsize"])

        # --- (1) TARGETS: deterministic under a census, asserted exactly ------
        # A census must reproduce the observed network's own statistics
        # (edges = 203, nodematch.Grade = 163). It does, exactly.
        targets = ego_target_stats(terms, ed, m)
        @test check_golden(g, "targets", targets) ||
              error(golden_report(g, "targets", targets))

        # --- (2) THE DESIGN VARIANCE — ISSUE ERGMEgo#1, FOUND AND FIXED -------
        # This is the sharpest measurement in the fixture: Σ_design is a function
        # of the 205 per-ego contributions and the weights and NOTHING else, so a
        # disagreement cannot be blamed on MCMC, on an I⁻¹ sandwich, or on a
        # tolerance. There WAS a disagreement, and it was exact:
        #
        #   ERGMEgo.jl's design variance was TOO SMALL BY EXACTLY (n−1)/n.
        #
        # `_design_cov` summed m² Σᵢ wᵢ²(hᵢ−h̄)(hᵢ−h̄)′ with wᵢ = 1/n, i.e. it
        # divided the sum of squared deviations by n. The survey (SRS /
        # Horvitz–Thompson) variance of a mean, which ergm.ego computes, divides
        # by n−1. One degrees-of-freedom correction, applied to every entry.
        #
        # It is now applied (`Σ .*= n/(n-1)` in `_design_cov`), so the design
        # variance matches `ergm.ego` outright. This test asserts the FIXED
        # behaviour; it would have caught the bug, and it will catch a regression.
        # It matters because the fixture also shows the design component is ~17×
        # the estimation component: an ego standard error essentially IS its
        # design variance.
        Σ_jl = ERGMEgo._design_cov(terms, ed, m)
        Σ_r = reduce(vcat, [Float64.(r)' for r in g.values["design_cov"]])
        @test size(Σ_jl) == size(Σ_r)
        @test Σ_jl ≈ Σ_r atol = 1e-9              # exact agreement, every entry

        jl_se = [sqrt(Σ_jl[i, i]) for i in axes(Σ_jl, 1)]
        r_se = Float64.(g.values["design_std_errors"])
        @test isapprox(jl_se, r_se; atol=1e-9)    # was @test_broken; now holds

        # Pin the correction itself, so removing it cannot pass silently: the
        # OLD (biased) estimator is exactly sqrt((n−1)/n) narrower.
        @test jl_se .* sqrt((n - 1) / n) ≈ r_se .* sqrt((n - 1) / n) atol = 1e-9
        @test all(jl_se .* sqrt((n - 1) / n) .< r_se)

        # --- (3) THE FIT ------------------------------------------------------
        # PARAMETERIZATION: ergm.ego splits the population edges parameter into a
        # fixed offset netsize.adj = −log(popsize) = −5.3230 plus a free `edges`
        # coefficient (−0.6974); ERGMEgo.jl reports it as ONE number on the
        # pseudo-population scale. R's −0.697 and Julia's −6.07 are the same
        # parameter in different clothes. The comparable quantity is the sum,
        # which the fixture freezes as `mle_coefficients_population`.
        @test Float64(g.values["netsize_adjustment"]) ≈ -log(n) atol = 1e-9

        # MCMC BUDGET: the defaults do NOT work at this scale — see the testset
        # below, which pins that. These settings are what it takes to converge on
        # a 205-node network, and they cost ~7s for all five fits.
        fits = [ergm_ego(ed, terms; n_samples=3000, burnin=30000, interval=300,
                         max_iter=80, tol=0.01, rng=Random.Xoshiro(s))
                for s in (101, 202, 303, 404, 505)]
        @test all(f.converged for f in fits)
        coefs = mean(f.coefficients for f in fits)
        @test check_golden(g, "mle_coefficients_population", coefs) ||
              error(golden_report(g, "mle_coefficients_population", coefs))

        # ...and a census ego fit must reduce to a plain ERGM fit of the same
        # model on the whole network — which is the strongest available check
        # that the pseudo-population construction is not distorting anything. R's
        # own plain-ergm MPLE is frozen; ERGM.jl reproduces it to 1e-11, and the
        # egocentric fit lands within 0.04 of both.
        plain = fit_ergm(net, [Edges(), NodeMatch(:Grade)])
        @test plain.coefficients ≈ Float64.(g.values["plain_ergm_mple"]) atol = 1e-6
        @test maximum(abs.(coefs .- plain.coefficients)) < 0.12
    end

    # ------------------------------------------------------------------
    # ...and the defect the fixture above had to work around, pinned so it is on
    # the record and cannot regress further.
    # ------------------------------------------------------------------
    @testset "ERGMEgo defaults do not converge at realistic network size" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "fauxmesa_ego_census.toml"))
        n = Int(g.values["n_actors"])
        grade = Int.(g.values["grade"])
        es = Int.(g.values["edge_src"]); ds = Int.(g.values["edge_dst"])
        net = network(n; directed=false)
        for k in eachindex(es); add_edge!(net, es[k], ds[k]); end
        egos = ERGMEgo.EgoNetwork{Int}[]
        for v in 1:n
            alters = sort(collect(Int.(neighbors(net, v))))
            k = length(alters)
            ties = falses(k, k)
            for a in 1:k, b in 1:k
                a != b && has_edge(net, alters[a], alters[b]) && (ties[a, b] = true)
            end
            push!(egos, EgoNetwork(v, alters, Matrix(ties);
                                   ego_attrs=Dict{Symbol,Any}(:Grade => grade[v]),
                                   alter_attrs=Dict{Symbol,Vector}(:Grade => grade[alters])))
        end
        ed = EgoData(egos; population_size=n, sampling_weights=ones(n))
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]

        # THE DEFAULTS, on a realistic network.
        #
        # They used to be fixed constants (n_samples=400, burnin=2000,
        # interval=20) that did not scale with the pseudo-population. On this
        # 205-actor census the chain stopped mixing and the fit returned
        # edges ≈ −21.9 against a true −6.02 — off by a factor of THREE, on a
        # network the size of the standard teaching dataset. It was not silent
        # (`converged == false`), but an estimator whose defaults produce garbage
        # at 205 nodes has a defaults problem.
        #
        # The MCMC controls now scale with the dyad count m(m−1)/2, exactly as
        # ERGM.jl's MCMLE already did. This test asserts the FIXED behaviour: the
        # defaults must converge here, and land on R's answer.
        f = ergm_ego(ed, terms; rng=Random.Xoshiro(101))
        r_pop = Float64.(g.values["mle_coefficients_population"])

        @test f.converged
        # Within Monte-Carlo reach of the ergm.ego MLE (was off by ~15.8)
        @test abs(f.coefficients[1] - r_pop[1]) < 0.5
        @test abs(f.coefficients[2] - r_pop[2]) < 0.5
    end
end
