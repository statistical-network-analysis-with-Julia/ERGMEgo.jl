using ERGMEgo
using ERGM
using NetworkCore
using DataFrames
using LinearAlgebra: diag, dot, norm
using Random
using Statistics
import Graphs, StatsAPI   # Graphs is a test-only extra (the documented-elsewhere check below); not a package dependency
import Aqua
using Test

# Text files read by the tests are compared line by line; a Windows checkout
# (git's core.autocrlf) gives them CRLF endings, so normalise to LF.
_readtext(path) = replace(read(path, String), "\r\n" => "\n")

# The 205-actor census egodata of R's faux.mesa.high (the bundled teaching
# dataset): every actor is an ego, unit weights, population 205. Built ONCE
# per testset from the loaded dataset, not by hand; the golden testset checks
# the dataset against the fixture's frozen edge list first.
fauxmesa_census(net=load_dataset(:faux_mesa_high)) =
    simulate_ego_sample(net, nv(net); ego_attrs=[:Grade], rng=Random.Xoshiro(0))

# The WEIGHTED sub-design of the second golden fixture: egos `ids` of the
# census (sorted by id, alters and alter ties as observed), case weights `w`,
# population 205. Deterministic — no draw is involved beyond the census's own.
function fauxmesa_weighted(ids, w; net=load_dataset(:faux_mesa_high))
    census = fauxmesa_census(net)
    keep = sort([e for e in census.egos if e.ego in ids]; by=e -> e.ego)
    return ego_design(EgoData(keep; population_size=nv(net)); weights=Float64.(w))
end

# `converged`, the termination report and the recorded diagnostics must
# describe ONE sample: the diagnostics are exactly `ERGM.mcmc_convergence`
# recomputed on `fit.sim_stats`, the termination report counts that sample's
# draws, and `converged` is EXACTLY the rule recomputed from that sample —
# under the confidence rule, R's equivalence test at the recorded step (the
# full Newton step from a sample that passed inside the loop, which also
# needs the Hummel step length 1; zeros for the fresh sample drawn at the
# returned coefficients after `maxiter`), under the Hotelling rule the
# t-ratio + Hotelling test
function assert_one_sample(fit; conv_threshold=0.1, hotelling_alpha=0.05,
                           max_step_norm=5.0)
    c = fit.mcmc_convergence
    t = ERGM.mcmc_convergence(fit.sim_stats, fit.model.targets;
                              conv_threshold, hotelling_alpha)
    @test t.t_ratios == c.t_ratios
    @test t.hotelling_p == c.hotelling_p
    @test t.n_eff == c.n_eff
    term = fit.termination
    @test term.n_samples == size(fit.sim_stats, 1)
    if term.rule === :confidence
        S = fit.sim_stats
        targets = fit.model.targets
        stepped = !iszero(term.step)
        if stepped && !fit.converged
            # the one other path: a singular sampled covariance stopped the
            # loop, and no test was evaluated on the returned sample
            @test any(isnan, fit.std_errors)
        else
            rec = ERGM.Extension.confidence_test(S, [size(S, 1)], targets, term.step;
                                        precision=term.precision,
                                        confidence=term.confidence)
            @test rec.p_value == term.p_value
            if stepped
                diff = targets .- vec(mean(S, dims=1))
                full = cov(S) \ diff
                @test term.step ≈ full .* min(1.0, max_step_norm / norm(full)) rtol = 1e-8
                covers = dot(diff, full) <= ERGMEgo._chisq95(length(targets))
                @test fit.converged == (rec.converged && covers)
                @test c.step_length == 1.0
            else
                @test fit.converged == rec.converged
            end
        end
    else
        @test fit.converged == t.converged
        @test term.p_value == c.hotelling_p
        @test iszero(term.step)
    end
    @test fit.vcov_estimation ≈ inv(cov(fit.sim_stats)) ./ c.n_eff rtol = 1e-6
    return nothing
end

# One Monte-Carlo block of an ergm.ego golden fixture (`ego_terms.toml`,
# `ego_mixing_esp.toml`): R's mean over its seeds against the mean of the
# Julia fits `fitter(rng)` over `seeds`, at the fixture's rule — every band
# is R's own seed spread (see the fixtures' [tolerance] block). Returns the
# fits.
function golden_fit_block(g, key, fitter; seeds=(1, 2))
    v = g.values
    k = g.tolerance["seed_mean_sds"]
    vec_(key) = Float64.(v[key])
    fits = [fitter(Random.Xoshiro(s)) for s in seeds]
    @test all(f -> f.converged, fits)
    names_R = v["$(key)_names"]
    # ergm.ego's labels, three ways: coefnames, the ego terms, the
    # ERGM.jl terms of the simulated model (an offset may follow; a mixing
    # cell's ERGM.jl term is NodeMix, under nodemix's label)
    @test coefnames(fits[1]) == names_R
    @test name.(fits[1].model.ego_terms) == names_R
    @test [startswith(n, "mm[") ? n : name(t) for (t, n) in
           zip(fits[1].model.ergm_terms, names_R)] == names_R
    nJ, nR = length(seeds), Int(v["$(key)_n_fits"])
    n_failed = Int(v["$(key)_n_failed"])
    # the seed-mean band, plus what a failed or timed-out R seed can
    # have moved R's mean had its estimate lain within the band
    band(sd) = k .* sd .* (sqrt(1 / nR + 1 / nJ) + n_failed / (nR + n_failed))
    meanR, sdR = vec_("$(key)_mean"), vec_("$(key)_sd")
    fin = isfinite.(meanR)
    C = reduce(vcat, [coef(f)' for f in fits])
    # a coefficient R fixes at ∓Inf is fixed at the same, exactly
    @test all(C[:, j] == fill(meanR[j], nJ) for j in findall(.!fin))
    # The premise of the band: one Julia fit is no noisier than one R fit
    for a in 1:nJ, b in (a + 1):nJ
        @test all(abs.(C[a, fin] .- C[b, fin]) .<= k * sqrt(2) .* sdR[fin])
    end
    @test all(abs.(vec(mean(C[:, fin]; dims=1)) .- meanR[fin]) .<= band(sdR)[fin])
    # Standard errors: the design component is R's design component;
    # the total lies between R's design component and R's total (a
    # longer final sample, a smaller MCMC-estimation component)
    seR, seR_sd = vec_("$(key)_se_mean"), vec_("$(key)_se_sd")
    semR, semR_sd = vec_("$(key)_se_model_mean"), vec_("$(key)_se_model_sd")
    jse = vec(mean(reduce(vcat, [stderror(f)' for f in fits]); dims=1))
    jsd = vec(mean(reduce(vcat, [sqrt.(diag(f.vcov_design))' for f in fits]); dims=1))
    @test all(abs.(jsd[fin] .- semR[fin]) .<= band(semR_sd)[fin])
    @test all(semR[fin] .- band(semR_sd)[fin] .<= jse[fin] .<= seR[fin] .+ band(seR_sd)[fin])
    @test all(jse[.!fin] .== 0)
    @test all(f -> isapprox(f.netsize_adjustment, v["$(key)_netsize_adj"]; atol=1e-12), fits)
    @test all(f -> f.model.ppopsize == v["$(key)_ppopsize"], fits)
    return fits
end

# The sibling checkouts (each a directory with a Project.toml) present beside
# the package. The workflow testset runs the layout step only when at least
# one is present: with none, this is a lone checkout or a registry install,
# where `[sources]` is not used and there is no layout to check; with some
# but not all, the layout is broken and the step's assertion fails.
present_siblings(parent::AbstractString, siblings) =
    [s for s in siblings if isfile(joinpath(parent, s, "Project.toml"))]

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

        # A duplicated alter (the wide-to-long survey mistake) used to be
        # accepted with ego_degree == 2, doubling the ego's degree and its
        # nodematch count in every target; it is refused naming ego and alter
        errmsg(f) = (try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end)
        msg = errmsg(() -> EgoNetwork(5, [10, 10], zeros(Bool, 2, 2)))
        @test occursin("ego 5", msg) && occursin("alter 10 twice", msg)
        msg = errmsg(() -> EgoNetwork(6, [10, 11, 10, 12], zeros(Bool, 4, 4)))
        @test occursin("ego 6", msg) && occursin("alter 10 twice", msg)
        # ...and a diagonal entry (an alter tied to itself) used to be accepted
        # and dropped by `sum ÷ 2` (1 ÷ 2 == 0) instead of refused
        msg = errmsg(() -> EgoNetwork(4, [10, 11], Bool[1 0; 0 0]))
        @test occursin("self-tie on the diagonal", msg) && occursin("ego 4", msg)
        @test occursin("alter 10", msg)
        @test_throws ArgumentError EgoNetwork(4, [10, 11], Bool[0 0; 0 1])
        # the legal neighbours of both mistakes still construct
        @test n_alter_ties(EgoNetwork(4, [10, 11], Bool[0 1; 1 0])) == 1
        @test ego_degree(EgoNetwork(5, [10, 11], zeros(Bool, 2, 2))) == 2
        @test n_alters(EgoNetwork(7, Int[], Matrix{Bool}(undef, 0, 0))) == 0
    end

    @testset "Sampling weights are validated at construction" begin
        # A zero-sum, NaN or Inf weight vector used to give NaN targets
        # silently (and then the unrelated "density ≥ 1" error from the fit);
        # a negative weight ran the whole MCMC and RETURNED coefficients from a
        # negative-weight Hájek mean. Each is an ArgumentError naming the ego
        # index, its id and the value, from the EgoData constructor — so
        # `ego_design`, `as_egodata` and `EgoData` itself all refuse it before
        # any statistic is computed.
        ed = fixture_egodata()
        errmsg(f) = (try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end)
        msg = errmsg(() -> ego_design(ed; weights=[0.0, 0.0, 0.0]))
        @test occursin("sum to 0.0", msg) && occursin("all 3 weights are zero", msg)
        @test occursin("EgoData", msg)
        msg = errmsg(() -> ego_design(ed; weights=[1.0, NaN, 1.0]))
        @test occursin("ego 2 (id 2)", msg) && occursin("is NaN", msg)
        @test occursin("finite, non-negative", msg)
        msg = errmsg(() -> ego_design(ed; weights=[1.0, 1.0, Inf]))
        @test occursin("ego 3 (id 3)", msg) && occursin("is Inf", msg)
        msg = errmsg(() -> ego_design(ed; weights=[1.0, -1.0, 1.0]))
        @test occursin("ego 2 (id 2)", msg) && occursin("negative (-1.0)", msg)
        @test_throws ArgumentError EgoData(ed.egos; sampling_weights=[1.0, -Inf, 1.0])
        @test_throws ArgumentError EgoData(ed.egos; sampling_weights=[-1.0, 2.0, 3.0])
        # ...so no NaN target and no coefficient can come out of them
        for w in ([0.0, 0.0, 0.0], [1.0, NaN, 1.0], [1.0, -1.0, 1.0], [Inf, 1.0, 1.0])
            @test_throws ArgumentError compute(EgoEdges(), ego_design(ed; weights=w))
            @test_throws ArgumentError fit_ergm_ego(ego_design(ed; weights=w), [EgoEdges()];
                                                    ppopsize=20, rng=Random.Xoshiro(1))
        end
        # A single zero weight is legal (an ego that contributes nothing), as
        # are the empty EgoData and unit weights; the weighted median keeps
        # its own sum guard as a second line of defence
        wz = ego_design(ed; weights=[1.0, 0.0, 1.0])
        @test wz.sampling_weights == [1.0, 0.0, 1.0]
        @test compute(EgoEdges(), wz) ≈ (3 + 4) / 2 / 2
        @test summary_stats(wz).mean_degree ≈ 3.5
        @test length(EgoData(EgoNetwork{Int}[])) == 0
        @test EgoData(ed.egos).sampling_weights == ones(3)
        @test_throws ArgumentError ERGMEgo._weighted_median([1.0, 2.0], [0.0, 0.0])
        # ...and the weight column of as_egodata meets the same guard
        ego_df = DataFrame(ego_id=[1, 2], w=[1.0, -2.0])
        alter_df = DataFrame(ego_id=[1, 2], alter_id=[10, 11])
        msg = errmsg(() -> as_egodata(ego_df, alter_df; weight_col=:w))
        @test occursin("ego 2 (id 2)", msg) && occursin("negative (-2.0)", msg)
        @test_throws ArgumentError as_egodata(DataFrame(ego_id=[1, 2], w=[NaN, 1.0]), alter_df;
                                              weight_col=:w)
    end

    @testset "summary_stats" begin
        ed = fixture_egodata()
        s = summary_stats(ed)
        @test s.n_egos == 3
        @test s.mean_degree ≈ 3.0        # (3 + 2 + 4)/3
        @test s.mean_alter_ties ≈ 1.0    # (1 + 0 + 2)/3
        @test s.total_alters == 9

        @test s.median_degree == median([3.0, 2.0, 4.0]) == 3.0

        # Weighted version: mean AND median honour the same design weights
        wed = ego_design(ed; weights=[2.0, 1.0, 1.0])
        ws = summary_stats(wed)
        @test ws.mean_degree ≈ (2 * 3 + 2 + 4) / 4
        # Degrees 3, 2, 4 with weights 2:1:1 — the weight-estimated degree
        # distribution puts mass 0.25 at 2, 0.5 at 3, 0.25 at 4, whose 0.5
        # quantile is 3
        @test ws.median_degree == 3.0
        # ...and with weights 1:1:4 the mass is 1/6, 1/6, 2/3, so the weighted
        # median is 4 where the unweighted one is 3 — the two estimands differ
        ws4 = summary_stats(ego_design(ed; weights=[1.0, 1.0, 4.0]))
        @test ws4.median_degree == 4.0 != median([3.0, 2.0, 4.0])
        @test ws4.mean_degree ≈ (3 + 2 + 16) / 6
        # The helper reduces to Statistics.median under (scaled) unit weights,
        # midpoint convention included; StatsBase's interpolating weighted
        # median (2.75 for the 2:1:1 case) is deliberately not used
        for n in 1:8, c in (1.0, 1/3)
            x = randn(Random.Xoshiro(n), n)
            @test ERGMEgo._weighted_median(x, fill(c, n)) == median(x)
        end
        @test_throws ArgumentError ERGMEgo._weighted_median([1.0, 2.0], [0.0, 0.0])

        # No egos: an ArgumentError that says so, not `mean`'s "reducing over
        # an empty collection" (an as_egodata whose ego frame filtered to
        # zero rows lands here)
        err = try summary_stats(EgoData(EgoNetwork{Int}[])); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("no egos", err.msg) && occursin("summary_stats", err.msg)
        empty_df = DataFrame(ego_id=Int[], g=String[])
        empty_ed = as_egodata(empty_df, DataFrame(ego_id=Int[], alter_id=Int[]))
        @test length(empty_ed) == 0
        @test_throws ArgumentError summary_stats(empty_ed)
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

        # An ego without the attribute is refused (it used to be skipped, so
        # the matrix silently described a subset of the sample)
        err = try ego_mixing_matrix(ed, :nope); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin(":nope", err.msg) && occursin("ego 1", err.msg)
        bare = EgoNetwork(4, [108], zeros(Bool, 1, 1))
        err = try ego_mixing_matrix(EgoData([ed[1], bare]), :group); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("ego 4", err.msg) && occursin("ego attribute :group", err.msg)
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

        # Every named column is checked up front: the error names the column,
        # the keyword and the frame, and lists the frame's columns
        colerr(f) = (try f(); nothing catch e; e end)
        err = colerr(() -> as_egodata(ego_df, alter_df; weight_col=:missing_col))
        @test err isa ArgumentError
        @test occursin(":missing_col", err.msg) && occursin("ego_df", err.msg)
        err = colerr(() -> as_egodata(ego_df, alter_df; ego_attrs=[:nope]))
        @test err isa ArgumentError
        @test occursin(":nope", err.msg) && occursin("ego_df", err.msg)
        @test occursin("ego_attrs", err.msg) && occursin("ego_id, group, w", err.msg)
        err = colerr(() -> as_egodata(ego_df, alter_df; alter_attrs=[:nope]))
        @test err isa ArgumentError
        @test occursin(":nope", err.msg) && occursin("alter_df", err.msg)
        err = colerr(() -> as_egodata(ego_df, alter_df; aatie_df=aatie_df, source_col=:from))
        @test err isa ArgumentError
        @test occursin(":from", err.msg) && occursin("aatie_df", err.msg) &&
              occursin("source_col", err.msg)
        err = colerr(() -> as_egodata(ego_df, alter_df; aatie_df=aatie_df, target_col=:to))
        @test err isa ArgumentError
        @test occursin(":to", err.msg) && occursin("target_col", err.msg)
        err = colerr(() -> as_egodata(ego_df, alter_df; alter_id=:alter))
        @test err isa ArgumentError
        @test occursin(":alter", err.msg) && occursin("alter_df", err.msg)
        err = colerr(() -> as_egodata(rename(ego_df, :ego_id => :id), alter_df))
        @test err isa ArgumentError
        @test occursin(":ego_id", err.msg) && occursin("ego_df", err.msg)

        # A duplicated (ego, alter) row — the wide-to-long reshape mistake —
        # is named in the frame's own terms (it used to build an EgoNetwork
        # with the alter listed twice, doubling that ego's degree)
        dup_df = DataFrame(ego_id=[1, 1, 1, 2], alter_id=[10, 11, 10, 10],
                           group=["A", "B", "A", "B"])
        err = colerr(() -> as_egodata(ego_df, dup_df; alter_attrs=[:group]))
        @test err isa ArgumentError
        @test occursin("1 duplicate (ego, alter) row", err.msg) && occursin("ego 1", err.msg)
        @test occursin("alter 10", err.msg) && occursin("(:ego_id, :alter_id)", err.msg)
        dup2 = DataFrame(ego_id=[2, 2, 2], alter_id=[10, 10, 10])
        err = colerr(() -> as_egodata(ego_df, dup2))
        @test occursin("2 duplicate (ego, alter) rows", err.msg) && occursin("ego 2", err.msg)
        # ...and an alter–alter tie row from an alter to itself is refused
        # (it used to set the diagonal, which n_alter_ties then dropped)
        self_tie = DataFrame(ego_id=[1], src=[10], dst=[10])
        err = colerr(() -> as_egodata(ego_df, alter_df; aatie_df=self_tie))
        @test err isa ArgumentError
        @test occursin("(1, 10, 10)", err.msg) && occursin("self-tie", err.msg)
    end

    @testset "Population size estimation" begin
        ed = fixture_egodata()

        # Horvitz-Thompson with weights
        wed = ego_design(ed; weights=[100.0, 150.0, 250.0])
        @test estimate_popsize(wed) == 500.0

        # Capture-recapture matches nominated alters against the sampled egos:
        # here no alter (101…107) is an ego (1, 2, 3), so there is nothing to
        # recapture and the method says so (the estimator has its own testset)
        err = try estimate_popsize(ed; method=:capture_recapture); nothing catch e; e end
        @test err isa ArgumentError && occursin("no nominated alter is a sampled ego", err.msg)

        # An unknown method names itself and lists the two valid ones (the
        # one guard that used to say only "Unknown method: bogus")
        err = try estimate_popsize(ed; method=:bogus); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("unknown method :bogus", err.msg)
        @test occursin(":horvitz_thompson", err.msg) && occursin(":capture_recapture", err.msg)
        @test occursin("estimate_popsize", err.msg)
        @test_throws ArgumentError estimate_popsize(ed; method=:lincoln_petersen)
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
                          n_samples=300, rng=rng)

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
                          n_samples=400,
                          rng=Random.Xoshiro(2))

        @test result.converged
        # The sampled nodematch statistics vary; they would be identically
        # zero if the sampler's network copies dropped vertex attributes
        @test std(result.sim_stats[:, 2]) > 0
        # Strong homophily is recovered as a positive nodematch coefficient
        @test result.coefficients[2] > 0
    end

    @testset "Netsize adjustment" begin
        rng = Random.Xoshiro(5)
        n = 40
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.1 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)

        r_same = ergm_ego(ed, [EgoEdges()]; ppopsize=n, popsize=n,
                          n_samples=200, rng=Random.Xoshiro(1))
        r_big = ergm_ego(ed, [EgoEdges()]; ppopsize=n, popsize=4n,
                         n_samples=200, rng=Random.Xoshiro(1))

        # Same pseudo-population fit; the popsize enters only through
        # ergm.ego's offset netsize.adj = −log(ppopsize/popsize) on edges
        @test r_big.netsize_adjustment ≈ -log(n / 4n) ≈ log(4.0)
        @test r_same.netsize_adjustment === 0.0
        @test (r_big.model.popsize, r_big.model.ppopsize) == (4n, n)
        @test r_big.coefficients[1] ≈ r_same.coefficients[1] - log(4.0) atol = 0.35

        # Descriptive terms (the GOF bins) are rejected with a clear error
        @test_throws ArgumentError ergm_ego(ed, [EgoEdges(), ERGMEgo._EgoDegreeAtLeast(2)];
                                            ppopsize=n)
        # Models must include EgoEdges
        @test_throws ArgumentError ergm_ego(ed, [EgoTriangle()]; ppopsize=n)
    end

    @testset "ego_gof" begin
        rng = Random.Xoshiro(9)
        n = 30
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        result = ergm_ego(ed, [EgoEdges()]; ppopsize=n,
                          n_samples=200, rng=rng)

        g = ego_gof(result; n_sim=10, rng=Random.Xoshiro(4))
        @test g.n_sim == 10
        @test 0.0 < g.p_values.mean_degree <= 1.0
        @test isfinite(g.simulated.mean_degree)
        # A well-specified edges-only model should not be wildly rejected
        # on mean degree
        @test g.p_values.mean_degree > 0.01

        # ego_gof is a thin wrapper over gof: the same simulations from the
        # same rng state, and its p-values ARE NetworkCore.mc_pvalue on them (the
        # local `(mean(sim .>= o), ...)` closure that could return exactly 0
        # is gone)
        G = gof(result; n_sim=10, rng=Random.Xoshiro(4))
        @test G.statistics[end].name == "ego summary statistics"
        sim = G.statistics[end].simulated
        obs = G.statistics[end].observed
        @test g.p_values.mean_degree == mc_pvalue(sim[:, 1], obs[1]) == G.statistics[end].p_values[1]
        @test g.p_values.mean_alter_ties == mc_pvalue(sim[:, 2], obs[2])
        @test g.observed.mean_degree == obs[1] == summary_stats(ed).mean_degree
        @test g.simulated.mean_degree == mean(sim[:, 1])
        @test g.simulated.mean_alter_ties == mean(sim[:, 2])
        @test !isdefined(ERGMEgo, :mc_p)

        # gof.ergm.ego's GOF="model": the first statistic is every model
        # statistic per capita — here edges, half the mean degree
        @test G.statistics[1].name == "model statistics"
        @test G.statistics[1].labels == ["edges"]
        @test G.statistics[1].observed == [compute(EgoEdges(), ed)]
        @test G.statistics[1].simulated[:, 1] ≈ sim[:, 1] ./ 2
        # gof.ergm.ego's GOF="degree": the second statistic is the ego degree
        # distribution over R's bins — degree 0 … maxdeg−1 plus a "≥ maxdeg"
        # tail, maxdeg = 2·max(K, 3), K the largest OBSERVED ego degree
        # (`degree(0:(maxdeg-1)) + degrange(maxdeg)`) — the design-weighted
        # proportion of egos in each bin, observed from `ed` via EgoDegree,
        # simulated per ego sample. The mean-degree row of the first statistic
        # is a fitted target (every model has EgoEdges), so this is the row
        # set that can actually detect misfit.
        @test length(G.statistics) == 4
        @test G.statistics[3].name == "edgewise shared partners"
        deg = G.statistics[2]
        @test deg.name == "degree distribution"
        K = Int(summary_stats(ed).max_degree)
        maxdeg = 2 * max(K, 3)
        @test maxdeg < n - 1                    # the tail branch applies here
        @test deg.labels == vcat(["degree $d" for d in 0:(maxdeg - 1)], ["degree ≥ $maxdeg"])
        @test length(deg.labels) == maxdeg + 1
        @test deg.observed[1:maxdeg] == [compute(EgoDegree(d), ed) for d in 0:(maxdeg - 1)]
        @test deg.observed[end] == 0.0          # nothing observed above 2·K, by construction
        @test sum(deg.observed) ≈ 1.0
        @test size(deg.simulated) == (10, maxdeg + 1)
        @test all(0 .<= deg.simulated .<= 1)
        # THE point of the tail bin: every simulated ego lands in exactly one
        # row, so every simulated row is a full distribution. The earlier
        # statistic stopped at K and let up to 5 % of simulated egos vanish
        # from every row (rows summed to 0.95 on a 40-vertex Bernoulli census).
        @test all(sum(deg.simulated; dims=2) .≈ 1)
        @test all(p -> 0 < p <= 1, deg.p_values)
        @test deg.p_values == [mc_pvalue(deg.simulated[:, j], deg.observed[j]) for j in 1:maxdeg + 1]
        @test deg.simulated == gof(result; n_sim=10, rng=Random.Xoshiro(4)).statistics[2].simulated
        @test occursin("degree distribution", sprint(show, G))
        @test occursin("degree ≥ $maxdeg", sprint(show, G))
        # The tail row is where a model that over-produces high degrees shows:
        # simulate at an edges coefficient 3 larger than fitted (mean simulated
        # degree ≈ 22 on this 30-vertex pseudo-population, against the ≥ 24
        # tail for K = 12) and the simulated tail mass is positive where the
        # observed one is 0
        heavy = EgoERGMResult(result.model, result.coefficients .+ 3.0, result.std_errors,
                              result.vcov, result.vcov_design, result.vcov_estimation,
                              result.netsize_adjustment, result.converged,
                              result.mcmc_convergence, result.sim_stats,
                              result.termination, result.se_type, result.boot_replicates)
        gh = gof(heavy; n_sim=10, rng=Random.Xoshiro(4))
        @test gh.statistics[2].labels == deg.labels
        @test maximum(gh.statistics[2].simulated[:, end]) > 0
        @test all(sum(gh.statistics[2].simulated; dims=2) .≈ 1)
        # The binning rule itself, including R's no-tail branch: when maxdeg
        # ≥ m − 1 every attainable degree of an m-vertex pseudo-population is
        # its own bin (`degree(0:(n-1))`)
        terms, labels = ERGMEgo._degree_gof_bins(2, 30)
        @test labels == vcat(["degree $d" for d in 0:5], ["degree ≥ 6"])
        @test terms[1:6] == [EgoDegree(d) for d in 0:5]
        @test terms[end] isa ERGMEgo._EgoDegreeAtLeast && terms[end].d == 6
        @test compute(terms[end], ed) == compute(EgoDegree(6), ed) + sum(compute(EgoDegree(d), ed) for d in 7:K)
        @test ERGMEgo._degree_gof_bins(5, 8)[2] == ["degree $d" for d in 0:7]
        @test ERGMEgo._degree_gof_bins(5, 12)[2] == vcat(["degree $d" for d in 0:9], ["degree ≥ 10"])
        @test_throws ArgumentError ERGMEgo.ergm_term(ERGMEgo._EgoDegreeAtLeast(3))   # descriptive only
    end

    @testset "fit aliases, shared show, and NetworkCore.gof" begin
        # Standardized fit_<model> entry point and its statnet-named alias,
        # bound to the same function; the development-era alias is gone
        @test ergm_ego === fit_ergm_ego
        @test !isdefined(ERGMEgo, :fit_ego_ergm)

        # One gof generic across the ecosystem: the method is added to
        # NetworkCore.gof, not a package-local function
        @test ERGMEgo.gof === NetworkCore.gof

        rng = Random.Xoshiro(21)
        n = 25
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.15 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        result = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n,
                              n_samples=200, rng=rng)

        # show renders through the shared coefficient-table printer
        out = sprint(show, result)
        @test occursin("Egocentric ERGM Results", out)
        @test occursin("Estimate", out)
        @test occursin("Pr(>|z|)", out)
        @test occursin("Signif. codes", out)

        # gof returns the shared GOFResult container
        g = gof(result; n_sim=8, rng=rng)
        @test g isa NetworkCore.GOFResult
        @test NetworkCore.n_simulations(g) == 8
        @test g.statistics[1].labels == ["edges"]
        stat = g.statistics[end]
        @test stat.labels == ["mean degree", "mean alter ties"]
        @test stat.observed[1] ≈ summary_stats(ed).mean_degree
        @test all(p -> 0 < p <= 1, stat.p_values)
        gout = sprint(show, g)
        @test occursin("Goodness-of-fit assessment: Egocentric ERGM", gout)
        @test occursin("MC p-value", gout)

        # Result metadata protocol: the fit says what it actually did
        md = NetworkCore.fit_metadata(result)
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
        @test occursin("strata, clusters", replace(out, r"\s+" => " "))
    end

    @testset "Reproducibility: every draw flows through rng, not the global RNG" begin
        rng = Random.Xoshiro(3)
        n = 30
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)

        # Two fits from the same rng state are bit-identical whatever the global
        # RNG holds: the pseudo-population seeding and the MCMC chain draw
        # from the caller's rng only (there is no bare rand() anywhere)
        Random.seed!(1)
        a = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, rng=Random.Xoshiro(7))
        Random.seed!(987654)
        b = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, rng=Random.Xoshiro(7))
        @test coef(a) == coef(b)
        @test vcov(a) == vcov(b)
        @test a.sim_stats == b.sim_stats
        @test a.mcmc_convergence == b.mcmc_convergence
        # ...and the global RNG is left alone, so a seed set by the caller
        # before the fit still governs whatever the caller draws after it
        Random.seed!(5); x = rand()
        Random.seed!(5)
        fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=50, rng=Random.Xoshiro(1))
        @test rand() == x
    end

    @testset "Shared contracts: name/compute/gof identities, StatsAPI surface" begin
        # ONE statistic protocol and ONE gof generic across the ecosystem
        @test ERGMEgo.name === NetworkCore.name
        @test ERGMEgo.compute === NetworkCore.compute
        @test ERGMEgo.gof === NetworkCore.gof
        @test ERGMEgo.coeftable === NetworkCore.coeftable
        # The z → p helper is NetworkCore's (no private copy left in this package)
        @test !isdefined(ERGMEgo, :_z_pvalues)

        # The documented public surface is declared so: `EgoTerm` (the type in
        # fit_ergm_ego's signature and what a custom term subtypes) is
        # exported as ERGM exports AbstractERGMTerm, and the two hooks a
        # custom term extends are `public`, not exported
        @test :EgoTerm in names(ERGMEgo)
        @test Base.ispublic(ERGMEgo, :EgoTerm)
        for hook in (:ergm_term, :ego_contribution)
            @test Base.ispublic(ERGMEgo, hook) && !Base.isexported(ERGMEgo, hook)
            @test hook in names(ERGMEgo)
        end
        # No underscore name is public: an underscore says private (the
        # budget rule and the GOF tail bin are internal)
        @test isempty([n for n in names(ERGMEgo) if startswith(string(n), "_")])
        @test isdefined(ERGMEgo, :_mcmc_controls) && !Base.ispublic(ERGMEgo, :_mcmc_controls)
        @test !Base.ispublic(ERGMEgo, :_EgoDegreeAtLeast)
        @test !isdefined(ERGMEgo, :_ergm_term) && !isdefined(ERGMEgo, :_ego_contribution)
        @test EgoTerm <: ERGM.AbstractERGMTerm
        # No cross-package `ERGM._name` reach-in, by `import` or written out:
        # ERGM.jl's building blocks come from its extension API,
        # `ERGM.Extension` (the sampler rule, `mcmle`'s own sampler, R ergm's
        # stopping rule and the attainable range ERGMEgo extends for its ego
        # terms)
        src = _readtext(joinpath(dirname(@__DIR__), "src", "ERGMEgo.jl"))
        @test isempty(collect(eachmatch(r"\b(?:ERGM|NetworkCore)\._\w+", src)))
        imported = Symbol[]
        for m in eachmatch(r"import ERGM(?:\.Extension)?:\s*([^\n]+)", src)
            append!(imported, Symbol.(strip.(split(m.captures[1], ","))))
        end
        @test !any(n -> startswith(string(n), "_"), imported)
        for n in (:mcmc_defaults, :mcmle_sampler, :confidence_test, :attainable_range)
            @test n in imported
            @test getfield(ERGMEgo, n) === getfield(ERGM.Extension, n)
        end
        @test Base.ispublic(ERGM, :mcmc_convergence)
        @test Base.ispublic(ERGM, :MCMLEConvergence)
        # Graphs is not a dependency: every graph primitive the package uses
        # (`nv`, `neighbors`, `has_edge`, `vertices`) is NetworkCore.jl's re-export
        @test !isdefined(ERGMEgo, :Graphs)
        @test ERGMEgo.nv === NetworkCore.nv && ERGMEgo.neighbors === NetworkCore.neighbors
        @test ERGMEgo.has_edge === NetworkCore.has_edge
        project = _readtext(joinpath(dirname(@__DIR__), "Project.toml"))
        deps_block = match(r"\[deps\]\n(.*?)\n\n"s, project).captures[1]
        compat_block = match(r"\[compat\]\n(.*?)\n\n"s, project).captures[1]
        @test !occursin("Graphs", deps_block)
        @test occursin("Graphs", compat_block)   # a [compat] bound for the test-only extra (Aqua's deps_compat)
        @test occursin(r"\[extras\][^\[]*Graphs"s, project)

        rng = Random.Xoshiro(21)
        n = 25
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.15 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, rng=rng)

        # The full surface the fit can honestly answer, pinned in one line;
        # loglikelihood/aic/bic are deliberately absent (objective == :moment,
        # no likelihood is evaluated), so they are NOT methods, not NaN-returners
        # `coefnames` (StatsAPI's optional verb) is required too: the labels
        # are ergm.ego's and must equal the coefficient table's rows
        verbs = (:coef, :stderror, :vcov, :confint, :nobs, :dof, :coeftable, :coefnames)
        st = NetworkCore.check_statsapi(fit; required=verbs, strict=true)
        @test all(st[k] for k in verbs)
        @test coefnames(fit) == coeftable(fit).names == ["edges"]
        @test coefnames === ERGMEgo.StatsAPI.coefnames === NetworkCore.coefnames
        cn = coefnames(fit); push!(cn, "x")
        @test coefnames(fit) == ["edges"]           # a fresh copy each call
        SA = ERGMEgo.StatsAPI
        @test !hasmethod(SA.loglikelihood, Tuple{EgoERGMResult})
        @test !hasmethod(SA.aic, Tuple{EgoERGMResult})
        @test !hasmethod(SA.bic, Tuple{EgoERGMResult})
        @test NetworkCore.check_statsapi(fit).loglikelihood == false

        tbl = coeftable(fit)
        @test tbl isa NetworkCore.CoefficientTable
        @test tbl["edges"].estimate == coef(fit)[1]
        @test tbl[1].std_error == stderror(fit)[1]
        @test tbl.p_values == z_pvalues(coef(fit), stderror(fit)).p
        ci = confint(fit)
        @test size(ci) == (1, 2)
        @test ci[1, 1] < coef(fit)[1] < ci[1, 2]
        @test ci[1, 2] - ci[1, 1] ≈ 2 * 1.959963984540054 * stderror(fit)[1]
        ci90 = confint(fit; level=0.9)
        @test ci90[1, 2] - ci90[1, 1] < ci[1, 2] - ci[1, 1]
        @test_throws ArgumentError confint(fit; level=1.5)
        @test nobs(fit) == n                   # egos, not pseudo-population dyads
        @test dof(fit) == 1
        @test objective(fit) == :moment

        # The printed table IS coeftable(fit), and the SE decomposition line is
        # printed under it
        out = sprint(show, fit)
        @test occursin(sprint(show, tbl), out)
        @test occursin("MCMC % of the standard error (100·(se − se_design)/se)", out)
        @test occursin("design component ⊕ MCMC-estimation component", out)
    end

    @testset "Parametric result types and EgoGWDegree(decay >= 0)" begin
        ed = fixture_egodata()
        @test ed isa EgoData{Int}

        # The model and result carry the ego data's ID type, so the
        # data field is concretely typed
        rng = Random.Xoshiro(2)
        n = 20
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.2 && add_edge!(net, i, j)
        end
        sed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(sed, [EgoEdges()]; ppopsize=n, n_samples=100, rng=rng)
        @test fit isa EgoERGMResult{Int}
        @test fit.model isa EgoERGMModel{Int}
        @test isconcretetype(fieldtype(typeof(fit.model), :data))
        @test isconcretetype(fieldtype(typeof(fit), :model))
        @test fit.mcmc_convergence isa ERGM.MCMLEConvergence
        @test fit.netsize_adjustment === 0.0          # not -0.0

        # EgoGWDegree accepts any Real and decay = 0 (statnet's gwdegree(0,
        # fixed=TRUE)); its ERGM counterpart carries R's label
        @test EgoGWDegree(0.0).decay == 0.0
        @test EgoGWDegree(1).decay === 1.0
        @test EgoGWDegree().decay == 0.5
        @test_throws ArgumentError EgoGWDegree(-0.1)
        @test name(ERGMEgo.ergm_term(EgoGWDegree(0.5))) == "gwdeg.fixed.0.5"
        @test name(ERGMEgo.ergm_term(EgoGWDegree(0))) == "gwdeg.fixed.0"
        # At α = 0 the per-ego weight is exactly 1[degree > 0]
        @test compute(EgoGWDegree(0.0), ed) ≈ mean(ego_degree(e) > 0 for e in ed.egos)
        e0 = EgoNetwork(9, Int[], Matrix{Bool}(undef, 0, 0))
        @test compute(EgoGWDegree(0.0), EgoData([ed[1], e0])) ≈ 0.5
    end

    @testset "Missing dyads and the conversion contract (simulate_ego_sample)" begin
        net = load_dataset(:faux_mesa_high)          # undirected; :Grade, :Race, :Sex
        n = nv(net)
        errmsg(f) = (try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end)

        # The declaration half of the missing-data contract
        @test supports_missing(simulate_ego_sample)
        @test missing_policies(simulate_ego_sample) == (:error, :face)
        # ...and the docstring's promise that the positional is a Network
        @test hasmethod(simulate_ego_sample, Tuple{Network, Int})
        @test !hasmethod(simulate_ego_sample, Tuple{Matrix{Bool}, Int})

        # A masked network is refused by default, naming the routine and the
        # written opt-in; a bogus policy is refused outright
        masked = copy(net)
        set_missing_dyad!(masked, 1, 2)
        set_missing_dyad!(masked, 3, 4)
        msg = errmsg(() -> simulate_ego_sample(masked, 30; rng=Random.Xoshiro(1)))
        @test occursin("simulate_ego_sample", msg)
        @test occursin("missing=:face", msg)
        @test occursin("2 masked dyads", msg)
        @test_throws ArgumentError simulate_ego_sample(masked, 30; missing=:bogus)
        # :face reads the stored values and the report says so
        edf, repf = simulate_ego_sample(masked, 30; missing=:face, report=true,
                                        rng=Random.Xoshiro(1))
        @test edf isa EgoData{Int}
        @test repf isa ConversionReport
        @test repf.source == :Network && repf.target == :EgoData
        @test :missing_dyads in dropped_fields(repf)
        @test any(f == :missing_dyads && occursin("2 masked dyads", why)
                  for (f, why) in repf.dropped)
        # ...and is the same draw the unmasked network gives (face values are
        # the stored ones), so the mask changed nothing but the audit trail
        edu = simulate_ego_sample(net, 30; rng=Random.Xoshiro(1))
        @test [e.ego for e in edf.egos] == [e.ego for e in edu.egos]
        @test [e.alters for e in edf.egos] == [e.alters for e in edu.egos]
        # report=false returns the EgoData alone, exactly as before
        @test simulate_ego_sample(masked, 30; missing=:face, rng=Random.Xoshiro(1)) isa EgoData

        # Directed and two-mode networks are refused with the fix in the message
        d = network(10; directed=true)
        add_edge!(d, 1, 2)
        msg = errmsg(() -> simulate_ego_sample(d, 5))
        @test occursin("undirected", msg) && occursin("out-neighbours", msg)
        @test occursin("ymmetris", msg)
        msg = errmsg(() -> simulate_ego_sample(network(10; directed=false, bipartite=4), 5))
        @test occursin("two-mode", msg) && occursin("Project onto one mode", msg)
        msg = errmsg(() -> simulate_ego_sample(BipartiteNetwork(2, 3), 2))
        @test occursin("two-mode", msg) && occursin("BipartiteNetwork", msg)
        # More egos than vertices, and an unknown / partial attribute
        @test_throws ArgumentError simulate_ego_sample(net, n + 1)
        msg = errmsg(() -> simulate_ego_sample(net, 5; ego_attrs=[:nope]))
        @test occursin(":nope", msg) && occursin("Grade, Race, Sex", msg)
        partial = network(5; directed=false)
        set_vertex_attribute!(partial, :g, Dict(1 => "a", 2 => "b"))
        msg = errmsg(() -> simulate_ego_sample(partial, 5; ego_attrs=[:g]))
        @test occursin(":g", msg) && occursin("2 of 5", msg)

        # The conversion report. A census with every vertex attribute
        # requested is lossless (faux.mesa.high has no edge or network
        # attributes and no loops)...
        ed, rep = simulate_ego_sample(net, n; ego_attrs=[:Grade, :Race, :Sex],
                                      rng=Random.Xoshiro(0), report=true)
        @test is_lossless(rep)
        @test length(ed) == n && ed.population_size == n
        @test all(haskey(e.ego_attrs, :Race) && haskey(e.alter_attrs, :Sex) for e in ed.egos)
        @test !any(any(ismissing, e.alter_attrs[:Grade]) for e in ed.egos)
        # ...a 30-of-205 sample reports the unobserved ties between unsampled
        # vertices and each attribute not requested
        ed30, rep30 = simulate_ego_sample(net, 30; ego_attrs=[:Grade],
                                          rng=Random.Xoshiro(0), report=true)
        @test !is_lossless(rep30)
        @test dropped_fields(rep30) == [:edges, :vertex_attrs, :vertex_attrs]
        @test any(f == :edges && occursin("unsampled", why) && occursin("175 of 205", why)
                  for (f, why) in rep30.dropped)
        @test any(f == :vertex_attrs && occursin(":Race", why) for (f, why) in rep30.dropped)
        @test any(f == :vertex_attrs && occursin(":Sex", why) for (f, why) in rep30.dropped)
        @test occursin("2 dropped", sprint(show, rep30)) || occursin("3 dropped", sprint(show, rep30))
        # ...edge attributes, network attributes and a loops flag are named
        rich = network(6; directed=false, loops=true)
        add_edge!(rich, 1, 2); add_edge!(rich, 2, 3); add_edge!(rich, 1, 1)
        set_edge_attribute!(rich, :weight, Dict((1, 2) => 2.0, (2, 3) => 1.0))
        set_network_attribute!(rich, :title, "toy")
        edr, repr_ = simulate_ego_sample(rich, 6; report=true, rng=Random.Xoshiro(1))
        @test sort(dropped_fields(repr_)) == [:edge_attrs, :loops, :network_attrs]
        @test any(f == :edge_attrs && occursin(":weight", why) for (f, why) in repr_.dropped)
        @test any(f == :network_attrs && occursin(":title", why) for (f, why) in repr_.dropped)
        @test any(f == :loops && occursin("1 self-loop", why) for (f, why) in repr_.dropped)
        # the ego's own loop is never an alter
        e1 = only(e for e in edr.egos if e.ego == 1)
        @test e1.alters == [2]
        # Every ego's attribute column has ONE concrete element type — an
        # isolate's empty column included (it used to be a `Vector{Any}` next
        # to its neighbours' `Vector{String}`): the columns are built once per
        # attribute, not per ego
        sparse = network(500; directed=false)
        srng = Random.Xoshiro(4)
        for i in 1:500, j in (i+1):500
            rand(srng) < 2 / 500 && add_edge!(sparse, i, j)
        end
        set_vertex_attribute!(sparse, :g, Dict(v => ("A", "B")[mod1(v, 2)] for v in 1:500))
        set_vertex_attribute!(sparse, :k, Dict(v => v % 3 for v in 1:500))
        eds = simulate_ego_sample(sparse, 500; ego_attrs=[:g, :k], rng=Random.Xoshiro(5))
        isolates = [e for e in eds.egos if n_alters(e) == 0]
        @test !isempty(isolates)
        @test unique(typeof(e.alter_attrs[:g]) for e in eds.egos) == [Vector{String}]
        @test unique(typeof(e.alter_attrs[:k]) for e in eds.egos) == [Vector{Int}]
        @test typeof(first(isolates).alter_attrs[:g]) == Vector{String}
        @test all(e.ego_attrs[:g] isa String for e in eds.egos)
        @test compute(EgoNodeMatch(:g), eds) * 500 ≈ compute(NodeMatch(:g), sparse)
        # The census draw itself is unchanged by the guard: the frozen
        # per-testset egodata is the same object the golden testset uses
        @test [e.ego for e in fauxmesa_census(net).egos] ==
              [e.ego for e in simulate_ego_sample(net, n; ego_attrs=[:Grade], rng=Random.Xoshiro(0)).egos]
    end

    @testset "Actionable errors for missing attributes (no silent zero-fill)" begin
        ed = fixture_egodata()
        errmsg(f) = (try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end)

        # compute(EgoNodeMatch(:nope), ed) used to return 0.0 — the same
        # silent zero-fill as ERGM's NodeCov once had; it now names
        # the attribute, the ego and the side that lacks it
        msg = errmsg(() -> compute(EgoNodeMatch(:nope), ed))
        @test occursin(":nope", msg) && occursin("ego 1", msg) && occursin("ego attribute", msg)
        @test occursin("nodematch.nope", msg)
        # alters lacking it while the ego has it: the other side is named
        lop = EgoNetwork(7, [1, 2], zeros(Bool, 2, 2);
                         ego_attrs=Dict{Symbol,Any}(:group => "A"))
        msg = errmsg(() -> compute(EgoNodeMatch(:group), EgoData([ed[1], lop])))
        @test occursin("ego 7", msg) && occursin("alter attribute :group", msg)
        @test occursin("its alter attributes: none", msg)
        # the good data still evaluates
        @test compute(EgoNodeMatch(:group), ed) ≈ 1.0

        # fit_ergm_ego fails at the targets, before any MCMC: the caller's rng
        # is untouched (the first draw would be the pseudo-population seeding)
        rng = Random.Xoshiro(3)
        before = copy(rng)
        msg = errmsg(() -> fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:nope)];
                                         ppopsize=20, rng=rng))
        @test occursin(":nope", msg) && occursin("ego 1", msg)
        @test rng == before
        @test_throws ArgumentError ego_target_stats([EgoNodeMatch(:nope)], ed, 10)
        @test_throws ArgumentError ERGMEgo._design_cov([EgoEdges(), EgoNodeMatch(:nope)], ed, 10)

        # A carried attribute holding `missing` (an unknown alter attribute
        # in a real survey) used to be a TypeError ("non-boolean (Missing)
        # used in boolean context") from inside `_count_matches`; it is now
        # the same rule as an absent attribute — refused by name, with the
        # ego, the count of alters lacking a value, and the fix
        m1 = EgoNetwork(11, [10, 13, 12], zeros(Bool, 3, 3);
                        ego_attrs=Dict{Symbol,Any}(:group => "A"),
                        alter_attrs=Dict{Symbol,Vector}(:group => Union{String,Missing}["A", missing, missing]))
        msg = errmsg(() -> compute(EgoNodeMatch(:group), EgoData([ed[1], m1])))
        @test occursin("ego 11", msg) && occursin(":group", msg)
        @test occursin("2 alters of 3", msg) && occursin("drop those alters", msg)
        @test occursin("nodematch.group", msg)
        m2 = EgoNetwork(12, [10], zeros(Bool, 1, 1);
                        ego_attrs=Dict{Symbol,Any}(:group => missing),
                        alter_attrs=Dict{Symbol,Vector}(:group => ["A"]))
        msg = errmsg(() -> compute(EgoNodeMatch(:group), EgoData([m2])))
        @test occursin("ego 12", msg) && occursin("the ego's own value", msg)
        @test !occursin("alter of", msg)
        msg = errmsg(() -> ego_mixing_matrix(EgoData([ed[1], m1]), :group))
        @test occursin("ego_mixing_matrix", msg) && occursin("ego 11", msg) && occursin("`missing`", msg)
        # ...the fit fails at the targets, before any draw
        rng = Random.Xoshiro(3); before = copy(rng)
        @test_throws ArgumentError fit_ergm_ego(EgoData([ed[1], m1]), [EgoEdges(), EgoNodeMatch(:group)];
                                                ppopsize=20, rng=rng)
        @test rng == before
        # ...and as_egodata passes such a column through (a DataFrame with a
        # `missing` entry), so the refusal is what a survey user meets
        ego_df = DataFrame(ego_id=[1, 2], g=["A", "B"])
        alter_df = DataFrame(ego_id=[1, 1, 2], alter_id=[10, 11, 10], g=Union{String,Missing}["A", missing, "B"])
        mde = as_egodata(ego_df, alter_df; ego_attrs=[:g], alter_attrs=[:g])
        msg = errmsg(() -> compute(EgoNodeMatch(:g), mde))
        @test occursin("ego 1", msg) && occursin("1 alter of 2", msg)
        @test compute(EgoNodeMatch(:g), EgoData([mde[2]])) == 0.5    # the clean ego still evaluates
    end

    @testset "Informative show methods" begin
        ed = fixture_egodata()
        # EgoNetwork: ego id, alter count, alter-tie count, attribute names
        out = sprint(show, ed[1])
        @test occursin("EgoNetwork{Int64}", out)
        @test occursin("ego 1", out) && occursin("3 alters", out)
        @test occursin("1 alter–alter tie;", out)          # singular
        @test occursin("ego attributes: group", out) && occursin("alter attributes: group", out)
        out2 = sprint(show, ed[2])
        @test occursin("2 alters", out2) && occursin("0 alter–alter ties", out2)
        bare = EgoNetwork(9, [1], zeros(Bool, 1, 1))
        outb = sprint(show, bare)
        @test occursin("1 alter,", outb) && occursin("ego attributes: none", outb)

        # EgoData: n egos, weighted mean degree, population size, weights
        out = sprint(show, ed)
        @test occursin("EgoData{Int64}: 3 egos", out)
        @test occursin("weighted mean degree 3.0", out)
        @test occursin("population size unknown", out)
        @test occursin("unit weights", out)
        wed = ego_design(ed; popsize=500, weights=[100.0, 150.0, 250.0])
        outw = sprint(show, wed)
        @test occursin("population size 500", outw)
        @test occursin("sampling weights summing to 500.0", outw)
        @test occursin("weighted mean degree $(round((300 + 300 + 1000) / 500; digits=3))", outw)
        @test occursin("1 ego,", sprint(show, EgoData([ed[1]])))
        @test occursin("0 egos;", sprint(show, EgoData(EgoNetwork{Int}[])))

        # EgoERGMModel: terms, ppopsize/popsize, targets (ERGMModel's style)
        rng = Random.Xoshiro(2)
        n = 20
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.2 && add_edge!(net, i, j)
        end
        sed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(sed, [EgoEdges()]; ppopsize=n, popsize=4n, n_samples=100, rng=rng)
        outm = sprint(show, fit.model)
        @test occursin("EgoERGMModel{Int64}: 20 egos", outm)
        @test occursin("terms: edges", outm)
        @test occursin("ppopsize 20, popsize 80", outm)
        @test occursin("targets [$(ERGMEgo._fmt3(fit.model.targets[1]))]", outm)
        @test occursin("targets [$(Float64(ne(net)))]", outm)
        # the model line is also usable inside the result's own show
        @test occursin("Egocentric ERGM Results", sprint(show, fit))
    end

    @testset "gof: rng, n_chains, the dyad-scaled budget and thread-count independence" begin
        rng = Random.Xoshiro(9)
        n = 30
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, rng=Random.Xoshiro(1))

        # The failing reproduction: `_gof_simulations`
        # dropped `rng` on the way to `sample_networks`, so two calls with the
        # same seed differed. Now every draw flows through rng.
        Random.seed!(1)
        a = gof(fit; n_sim=10, rng=Random.Xoshiro(5))
        Random.seed!(2)
        b = gof(fit; n_sim=10, rng=Random.Xoshiro(5))
        @test a.statistics[1].simulated == b.statistics[1].simulated
        @test a.statistics[1].p_values == b.statistics[1].p_values
        @test ego_gof(fit; n_sim=10, rng=Random.Xoshiro(5)) ==
              ego_gof(fit; n_sim=10, rng=Random.Xoshiro(5))
        # ...and the global RNG is left alone
        Random.seed!(5); x = rand()
        Random.seed!(5); gof(fit; n_sim=4, rng=Random.Xoshiro(1))
        @test rand() == x
        # a different seed is a different sample (the rng is actually used)
        c = gof(fit; n_sim=10, rng=Random.Xoshiro(6))
        @test c.statistics[1].simulated != a.statistics[1].simulated

        # Keyword vocabulary on both entry points: burnin/interval default to
        # the fit's dyad-scaled rule (`_mcmc_controls`), n_chains to ERGM's
        kw = Base.kwarg_decl(which(gof, Tuple{EgoERGMResult}))
        @test all(k in kw for k in (:n_sim, :rng, :burnin, :interval, :n_chains))
        kwe = Base.kwarg_decl(which(ego_gof, Tuple{EgoERGMResult}))
        @test all(k in kwe for k in (:n_sim, :rng, :burnin, :interval, :n_chains))
        ctl = ERGMEgo._mcmc_controls(n)
        explicit = gof(fit; n_sim=10, rng=Random.Xoshiro(5), burnin=ctl.burnin,
                       interval=ctl.interval)
        @test explicit.statistics[1].simulated == a.statistics[1].simulated
        # the literals 2000/200 are gone: passing them gives a different draw
        legacy = gof(fit; n_sim=10, rng=Random.Xoshiro(5), burnin=2000, interval=200)
        @test legacy.statistics[1].simulated != a.statistics[1].simulated
        @test ctl.burnin == 20 * (n * (n - 1) ÷ 2)
        @test_throws ArgumentError gof(fit; n_sim=0)
        @test_throws ArgumentError gof(fit; n_sim=4, n_chains=0)

        # n_chains=2 is reproducible in-process...
        two = gof(fit; n_sim=8, rng=Random.Xoshiro(5), n_chains=2)
        @test two.statistics[1].simulated ==
              gof(fit; n_sim=8, rng=Random.Xoshiro(5), n_chains=2).statistics[1].simulated
        @test NetworkCore.n_simulations(two) == 8
        # ...and thread-count independent, for real: the same gof in a fresh
        # process with a DIFFERENT thread count is bit-identical (chains are
        # seeded from the caller's rng and concatenated in order, as in
        # ERGM.jl's own test; `n_chains` never defaults to Threads.nthreads())
        other_threads = Threads.nthreads() == 1 ? 4 : 1
        script = """
            using ERGMEgo, NetworkCore, Random
            rng = Xoshiro(9)
            n = 30
            net = network(n; directed=false)
            for i in 1:n, j in (i+1):n
                rand(rng) < 0.12 && add_edge!(net, i, j)
            end
            ed = simulate_ego_sample(net, n; rng=rng)
            fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, rng=Xoshiro(1))
            g = gof(fit; n_sim=8, rng=Xoshiro(5), n_chains=2)
            println(Threads.nthreads())
            println(repr(g.statistics[1].simulated))
            println(repr(g.statistics[1].p_values))
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$other_threads --project=$(dirname(@__DIR__)) -e $script`
        errbuf = IOBuffer()
        lines = split(strip(read(pipeline(ignorestatus(cmd); stderr=errbuf), String)), '\n')
        length(lines) == 3 || println(stderr, "fresh-process gof failed:\n",
                                      String(take!(errbuf)))
        @test length(lines) == 3
        @test lines[1] == string(other_threads)
        @test lines[2] == repr(two.statistics[1].simulated)
        @test lines[3] == repr(two.statistics[1].p_values)
    end

    # ------------------------------------------------------------------
    # Golden fixture: statnet `ergm.ego` on faux.mesa.high under a CENSUS
    # (the direct check of the design variance against ergm.ego's).
    # test/fixtures/r/fauxmesa_ego_census.R regenerates it.
    #
    # The design is a census — every one of the 205 actors is an ego, unit
    # weights, ppopsize = popsize = 205 — chosen precisely because it makes
    # three of the things being compared DETERMINISTIC:
    #
    #   * the target statistics (they become the observed network's own),
    #   * the design variance of those targets, and
    #   * the design standard errors of the coefficients under the EXACT
    #     information (the model is dyad-independent, so a plain ERGM's
    #     MPLE = MLE and its vcov is the exact I⁻¹),
    #
    # so none can be excused as Monte-Carlo noise. Only the fitted
    # coefficients and the MCMC-estimated information remain stochastic, and
    # those get tolerances measured from both packages' seed-to-seed spread.
    # ------------------------------------------------------------------
    @testset "Golden fixture: ergm.ego on faux.mesa.high (census design)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "fauxmesa_ego_census.toml"))
        @test g.provenance["ergm_ego_version"] == "1.1.4"

        # The bundled teaching dataset IS R's faux.mesa.high: same edge list,
        # same grades, so the census egodata can be drawn from it directly
        n = Int(g.values["n_actors"])
        net = load_dataset(:faux_mesa_high)
        @test nv(net) == n
        @test ne(net) == Int(g.values["n_edges"])
        es = Int.(g.values["edge_src"]); ds = Int.(g.values["edge_dst"])
        r_edges = Set((min(es[k], ds[k]), max(es[k], ds[k])) for k in eachindex(es))
        el = as_edgelist(net)
        jl_edges = Set((min(el[k, 1], el[k, 2]), max(el[k, 1], el[k, 2]))
                       for k in axes(el, 1))
        @test jl_edges == r_edges
        @test vertex_attribute_vector(net, :Grade, Int) == Int.(g.values["grade"])

        ed = fauxmesa_census(net)
        @test length(ed) == n
        @test ed.population_size == n
        @test all(==(1.0), ed.sampling_weights)
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]
        m = Int(g.values["ppopsize"])

        # --- (1) TARGETS: deterministic under a census, asserted exactly ------
        # A census must reproduce the observed network's own statistics
        # (edges = 203, nodematch.Grade = 163). It does, exactly.
        targets = ego_target_stats(terms, ed, m)
        @test check_golden(g, "targets", targets) ||
              error(golden_report(g, "targets", targets))

        # --- (2) THE DESIGN VARIANCE — A BUG FOUND BY THE FIXTURE, FIXED ------
        # Σ_design is a function of the 205 per-ego contributions and the
        # weights and NOTHING else, so a disagreement cannot be blamed on MCMC,
        # on an I⁻¹ sandwich, or on a tolerance. The first run of this fixture
        # found ERGMEgo.jl's design variance too small by exactly (n−1)/n
        # (`_design_cov` divided the squared deviations by n where the survey
        # variance of a mean divides by n−1); the Bessel factor is now applied
        # and the agreement is exact, every entry, off-diagonal included.
        Σ_jl = ERGMEgo._design_cov(terms, ed, m)
        Σ_r = reduce(vcat, [Float64.(r)' for r in g.values["design_cov"]])
        @test size(Σ_jl) == size(Σ_r)
        @test Σ_jl ≈ Σ_r atol = 1e-9              # exact agreement, every entry

        jl_se = sqrt.(diag(Σ_jl))
        @test check_golden(g, "design_std_errors", jl_se) ||
              error(golden_report(g, "design_std_errors", jl_se))

        # Pin the Bessel correction against an INDEPENDENT implementation so
        # it cannot be removed silently: under unit weights the design
        # covariance is m² · cov(H)/n with Statistics.cov's own n−1 divisor,
        # H[i, j] the per-ego contribution of term j
        H = [ERGMEgo.ego_contribution(terms[j], ed[i]) for i in 1:n, j in eachindex(terms)]
        @test Σ_jl ≈ m^2 .* cov(H) ./ n atol = 1e-9
        # ...and a hand-computed, non-unit-weight reference: three egos with
        # weights 2:1:1 (normalised 0.5, 0.25, 0.25), EgoEdges contributions
        # h = [1.5, 1.0, 2.0], h̄ = 1.5, deviations [0, −0.5, 0.5], so
        # Σᵢ wᵢ²(hᵢ−h̄)² = 0.0625·0.25 + 0.0625·0.25 = 0.03125, times the
        # with-replacement factor n/(n−1) = 3/2 gives 0.046875, times m² = 100
        wed = ego_design(fixture_egodata(); weights=[2.0, 1.0, 1.0])
        @test ERGMEgo._design_cov([EgoEdges()], wed, 10)[1, 1] ≈ 4.6875 atol = 1e-12

        # --- (3) DESIGN SEs OF THE COEFFICIENTS WITH THE EXACT INFORMATION ----
        # edges + nodematch is dyad-independent, so the plain ERGM's MPLE is
        # the MLE and its vcov is the exact I⁻¹; sandwiching Σ_design with it
        # is deterministic on both sides. ERGM.jl reproduces R's MPLE and
        # its vcov to 1e-6, and the sandwich agrees to the same.
        plain = fit_ergm(net, [Edges(), NodeMatch(:Grade)])
        @test check_golden(g, "plain_ergm_mple", plain.coefficients) ||
              error(golden_report(g, "plain_ergm_mple", plain.coefficients))
        V_plain = vcov(plain)
        @test check_golden(g, "plain_ergm_vcov", vec(permutedims(V_plain))) ||
              error(golden_report(g, "plain_ergm_vcov", vec(permutedims(V_plain))))
        I_exact = inv(V_plain)
        se_exact = sqrt.(diag(inv(I_exact) * Σ_jl * inv(I_exact)))
        @test check_golden(g, "design_se_exact", se_exact) ||
              error(golden_report(g, "design_se_exact", se_exact))
        # R's own DtDe is ONE MCMC estimate of that information, 12 % off in
        # I⁻¹[1,1]; its sandwich is the frozen design component. Pinned so
        # the size of the Monte-Carlo effect on a standard error is on record.
        DtDe = reshape(Float64.(g.values["r_DtDe"]), 2, 2)'
        se_dtde = sqrt.(diag(inv(DtDe) * Σ_jl * inv(DtDe)))
        @test se_dtde ≈ Float64.(g.values["mle_se_design_component"]) atol = 1e-9
        @test abs(inv(DtDe)[1, 1] / inv(I_exact)[1, 1] - 1) > 0.1

        # --- (4) THE FIT ------------------------------------------------------
        # PARAMETERIZATION: the fixture's R fit used ergm.ego's default
        # popsize = 1: a fixed offset netsize.adj = −log(205/1) = −5.3230 plus
        # a free, per-capita `edges` coefficient (−0.6974). The census ego data
        # carries population_size = 205, so the default Julia fit is ergm.ego
        # with popsize = 205 — offset 0, edges −6.02 — and `popsize=1`
        # reproduces R's default output; the two differ by exactly the offset.
        @test Float64(g.values["netsize_adjustment"]) ≈ -log(n) atol = 1e-9

        # The DEFAULTS (dyad-scaled budget, annealed start, confidence rule)
        # converge here in one or two iterations. Five seeds, as in R.
        # (se=:design: ergm.ego's standard errors, which the fixture freezes)
        fits = [ergm_ego(ed, terms; se=:design, rng=Random.Xoshiro(s)) for s in (101, 202, 303, 404, 505)]
        @test all(f.converged for f in fits)
        @test all(f.se_type === :design for f in fits)
        @test all(f.mcmc_convergence.iterations <= 5 for f in fits)
        @test all(f.netsize_adjustment === 0.0 && f.model.popsize == 205 for f in fits)

        # R's default output, per capita: popsize = 1 on the same
        # pseudo-population. netsize.adj is R's, and the free coefficients are
        # R's `mle_coefficients` (same tolerance as the population-scale pin)
        pc = ergm_ego(ed, terms; popsize=1, ppopsize=m, rng=Random.Xoshiro(101))
        @test pc.converged && pc.model.popsize == 1
        @test pc.netsize_adjustment ≈ Float64(g.values["netsize_adjustment"]) atol = 1e-9
        r_free = Float64.(g.values["mle_coefficients"])
        @test maximum(abs.(coef(pc) .- r_free)) < g.tolerance["mle_coefficients_population"]
        @test coef(pc)[1] + pc.netsize_adjustment ≈ coef(fits[1])[1] atol = 0.05
        # An EgoData with no recorded population size gets the same default
        @test fit_ergm_ego(EgoData(ed.egos), [EgoEdges()]; ppopsize=m,
                           rng=Random.Xoshiro(1)).model.popsize == 1
        coefs = mean(f.coefficients for f in fits)
        @test check_golden(g, "mle_coefficients_population", coefs) ||
              error(golden_report(g, "mle_coefficients_population", coefs))

        # ...and a census ego fit must reduce to a plain ERGM fit of the same
        # model on the whole network — the strongest available check that the
        # pseudo-population construction is not distorting anything. The
        # five-seed mean lands within 0.02 of the MPLE (0.05 = 2× the combined
        # Monte-Carlo sd of the two means).
        @test maximum(abs.(coefs .- plain.coefficients)) < 0.05

        # STANDARD ERRORS: ergm.ego's decomposition, per fit. The design
        # component sandwiches Σ_design with each package's MCMC estimate of the
        # information (tolerance justified in the fixture: R's DtDe offset from
        # the exact value + 3× Julia's per-fit sd); the estimation component is
        # I⁻¹/n_eff and is small next to it; there is no I⁻¹ term.
        for f in fits
            se_design = sqrt.(diag(f.vcov_design))
            se_est = sqrt.(diag(f.vcov_estimation))
            @test check_golden(g, "mle_se_design_component", se_design) ||
                  error(golden_report(g, "mle_se_design_component", se_design))
            @test check_golden(g, "mle_std_errors", stderror(f)) ||
                  error(golden_report(g, "mle_std_errors", stderror(f)))
            @test stderror(f) ≈ sqrt.(se_design .^ 2 .+ se_est .^ 2)
            @test vcov(f) == f.vcov_design .+ f.vcov_estimation
            @test all(se_est .< 0.3 .* se_design)
            @test all(se_est .> 0)
            # The design SE scatters around the EXACT sandwich; every fit is
            # within 0.03 of it (R's DtDe is 0.022 above it)
            @test maximum(abs.(se_design .- se_exact)) < 0.03
            # Final-sample budget: the effective sample size behind the
            # estimation term and the Hotelling test is respectable
            @test f.mcmc_convergence.n_eff >= 64
            @test f.mcmc_convergence.n_eff <= size(f.sim_stats, 1)
            # ESS-adaptive: a few hundred stored draws, never above the cap
            @test size(f.sim_stats, 2) == 2 && 256 <= size(f.sim_stats, 1) <= 4 * 3000
        end
        mean_se = mean(stderror(f) for f in fits)
        @test check_golden(g, "mle_std_errors", mean_se) ||
              error(golden_report(g, "mle_std_errors", mean_se))
        # The estimation component is sqrt(diag(I⁻¹)/n_eff) by construction
        # (asserted per fit above through the decomposition), so its ratio to
        # ergm.ego's (0.0083 / 0.0103, i.e. about 360 effective draws) is the
        # square root of the ratio of effective sample sizes. Both samplers
        # are ESS-adaptive — R ergm 4's design, ERGMEgo's default — and stop
        # growing the sample once the equivalence test passes, from a target
        # of 64 effective draws. Held two-sided around R's value: n_eff ≥ 64
        # bounds the component at sqrt(360/64) = 2.4 times R's, and n_eff
        # cannot exceed the 4·3000 cap (sqrt(360/12000) = 0.17 of R's)
        r_est = Float64.(g.values["mle_se_estimation_component"])
        j_est = mean(sqrt.(diag(f.vcov_estimation)) for f in fits)
        @test all(r_est ./ 6 .< j_est .< 2.4 .* r_est)
        for f in fits
            @test f.vcov_estimation ≈ inv(cov(f.sim_stats)) ./ f.mcmc_convergence.n_eff rtol = 1e-6
            @test 64 <= f.mcmc_convergence.n_eff <= 4 * 3000
        end
    end

    # ------------------------------------------------------------------
    # Golden fixture: statnet `ergm.ego` on faux.mesa.high under a WEIGHTED,
    # NON-CENSUS design — the estimator's actual use case, which the census
    # cannot exercise: case-weighted (Hájek) targets, the design variance of
    # a WEIGHTED mean, the weight-proportional pseudo-population, and the two
    # other fittable terms (EgoTriangle with its /3, EgoGWDegree).
    # test/fixtures/r/fauxmesa_ego_weighted.R regenerates it.
    #
    # The design is deterministic (egos 1, 4, …, 205 of the census, weights
    # Grade − 6, ppopsize = popsize = 205), so the targets, their 4×4 design
    # covariance, the pseudo-population's composition and the EXACT
    # information at R's estimate are all asserted at machine precision; the
    # fitted coefficients and the MCMC-estimated standard errors get
    # tolerances measured from both packages' seed-to-seed spread.
    # ------------------------------------------------------------------
    @testset "Golden fixture: ergm.ego on faux.mesa.high (weighted sub-design)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "fauxmesa_ego_weighted.toml"))
        @test g.provenance["ergm_ego_version"] == "1.1.4"
        @test occursin("WEIGHTED", g.provenance["sampling_design"])

        net = load_dataset(:faux_mesa_high)
        n = Int(g.values["n_actors"])
        m = Int(g.values["ppopsize"])
        ids = Int.(g.values["ego_ids"])
        w = Float64.(g.values["weights"])
        n_e = Int(g.values["n_egos"])
        @test ids == collect(1:3:n) && length(ids) == n_e == 69
        @test m == Int(g.values["popsize"]) == n

        # The same EgoData, rebuilt from the bundled dataset with no draw:
        # the census's egos 1, 4, …, 205 in id order, with weights Grade − 6
        wed = fauxmesa_weighted(ids, w; net)
        @test wed isa EgoData{Int}
        @test [e.ego for e in wed.egos] == ids
        @test wed.sampling_weights == w
        @test w == [Float64(e.ego_attrs[:Grade] - 6) for e in wed.egos]
        @test all(1 .<= w .<= 6) && !all(==(w[1]), w)
        @test wed.population_size == n
        @test estimate_popsize(wed) == sum(w)
        terms4 = [EgoEdges(), EgoNodeMatch(:Grade), EgoGWDegree(0.5), EgoTriangle()]
        @test [name(ERGMEgo.ergm_term(t)) for t in terms4] == g.values["term_names"]

        # --- (1) WEIGHTED TARGETS: deterministic, exact ------------------------
        # Hájek estimates m·Σwᵢhᵢ/Σwᵢ of edges, nodematch, gwdegree(0.5) and
        # triangle — the weights enter, and so do the /3 of EgoTriangle and the
        # geometric weights of EgoGWDegree
        targets = ego_target_stats(terms4, wed, m)
        @test check_golden(g, "targets", targets) ||
              error(golden_report(g, "targets", targets))
        @test targets[1] ≈ m * sum(w .* [ego_degree(e) / 2 for e in wed.egos]) / sum(w)
        @test targets[4] ≈ m * sum(w .* [n_alter_ties(e) / 3 for e in wed.egos]) / sum(w)
        # ...and under unit weights they are different numbers (the weights
        # are not a no-op the way they are under the census)
        unit = EgoData(wed.egos; population_size=n)
        @test !(ego_target_stats(terms4, unit, m) ≈ targets)
        @test maximum(abs.(ego_target_stats(terms4, unit, m) .- targets)) > 1.0

        # --- (2) THE DESIGN VARIANCE OF A WEIGHTED MEAN: deterministic, exact --
        # m² · n/(n−1) · Σᵢ wᵢ²(hᵢ−h̄)(hᵢ−h̄)′ with normalised weights, every
        # entry of the 4×4 matrix (off-diagonal included)
        Σ_jl = ERGMEgo._design_cov(terms4, wed, m)
        Σ_r = reduce(vcat, [Float64.(r)' for r in g.values["design_cov"]])
        @test size(Σ_jl) == (4, 4) == size(Σ_r)
        @test Σ_jl ≈ Σ_r atol = 1e-9
        jl_se = sqrt.(diag(Σ_jl))
        @test check_golden(g, "design_std_errors", jl_se) ||
              error(golden_report(g, "design_std_errors", jl_se))
        # ...against the textbook formula written out
        H = [ERGMEgo.ego_contribution(t, e) for e in wed.egos, t in terms4]
        wn = w ./ sum(w)
        h̄ = vec(wn' * H)
        D = H .- h̄'
        @test Σ_jl ≈ m^2 * (n_e / (n_e - 1)) .* ((D .* wn .^ 2)' * D) atol = 1e-9

        # --- (3) THE PSEUDO-POPULATION: weight-proportional, same as R's ------
        # 205 vertices, each ego replicated round(205·wᵢ/Σw) times (ergm.ego's
        # ppop.wt = "round"): the composition by grade is identical and does
        # not depend on the rng
        pp = ERGMEgo._pseudo_population(wed, m, 0.01, Random.Xoshiro(1))
        grades = vertex_attribute_vector(pp, :Grade, Int)
        counts = [count(==(k), grades) for k in 7:12]
        @test counts == Int.(g.values["ppop_grade_counts"])
        @test sum(counts) == m
        @test counts == [count(==(k), vertex_attribute_vector(
            ERGMEgo._pseudo_population(wed, m, 0.01, Random.Xoshiro(99)), :Grade, Int)) for k in 7:12]

        # --- (4) THE EXACT INFORMATION at R's estimate: deterministic ---------
        # edges + nodematch is dyad-independent with two dyad types, so I is a
        # closed-form dyad sum over that composition; sandwiching Σ_design with
        # it gives the design SE of the coefficients with no MCMC anywhere
        θ_r = Float64.(g.values["mle_coefficients_population"])
        n_mat = sum(c * (c - 1) ÷ 2 for c in counts)
        @test n_mat == Int(g.values["n_same_grade_dyads"])
        n_mis = m * (m - 1) ÷ 2 - n_mat
        p0 = 1 / (1 + exp(-θ_r[1]))
        p1 = 1 / (1 + exp(-(θ_r[1] + θ_r[2])))
        I_exact = n_mis * p0 * (1 - p0) .* [1.0 0.0; 0.0 0.0] .+ n_mat * p1 * (1 - p1) .* ones(2, 2)
        @test check_golden(g, "exact_information", vec(permutedims(I_exact))) ||
              error(golden_report(g, "exact_information", vec(permutedims(I_exact))))
        Σ2 = Σ_jl[1:2, 1:2]
        se_exact = sqrt.(diag(inv(I_exact) * Σ2 * inv(I_exact)))
        @test check_golden(g, "design_se_exact", se_exact) ||
              error(golden_report(g, "design_se_exact", se_exact))
        # R's own MCMC estimate of I (DtDe) sandwiches the same Σ to R's frozen
        # design component — the Julia Σ_design is R's, to the last digit
        DtDe = reshape(Float64.(g.values["r_DtDe"]), 2, 2)'
        @test sqrt.(diag(inv(DtDe) * Σ2 * inv(DtDe))) ≈
              Float64.(g.values["mle_se_design_component"]) atol = 1e-9
        @test Float64(g.values["netsize_adjustment"]) == 0.0

        # --- (5) THE FIT, under the defaults ---------------------------------
        # ppopsize defaults to popsize = 205 (known and ≤ 1000); the adjustment
        # is exactly 0, so coef(fit) is on R's scale directly. Three seeds.
        terms = terms4[1:2]
        fits = [ergm_ego(wed, terms; se=:design, rng=Random.Xoshiro(s)) for s in (1, 2, 3)]
        for f in fits
            @test f.converged
            assert_one_sample(f)
            @test f.model.ppopsize == m && f.model.popsize == n
            @test f.netsize_adjustment === 0.0
            @test f.model.targets == targets[1:2]
            @test nobs(f) == n_e
            @test check_golden(g, "mle_coefficients_population", coef(f)) ||
                  error(golden_report(g, "mle_coefficients_population", coef(f)))
            se_design = sqrt.(diag(f.vcov_design))
            se_est = sqrt.(diag(f.vcov_estimation))
            @test check_golden(g, "mle_se_design_component", se_design) ||
                  error(golden_report(g, "mle_se_design_component", se_design))
            @test check_golden(g, "mle_std_errors", stderror(f)) ||
                  error(golden_report(g, "mle_std_errors", stderror(f)))
            # the design SE scatters around the EXACT sandwich (tolerance
            # justified in the fixture: > 3× the per-fit sd)
            @test maximum(abs.(se_design .- se_exact)) < 0.08
            @test stderror(f) ≈ sqrt.(se_design .^ 2 .+ se_est .^ 2)
            @test all(0 .< se_est .< 0.3 .* se_design)
            @test f.mcmc_convergence.n_eff >= 64      # the ESS target, at least
            @test size(f.sim_stats, 2) == 2 && size(f.sim_stats, 1) <= 4 * 3000
            # the SE decomposition is built on THIS design covariance
            I_mc = cov(f.sim_stats)
            @test f.vcov_design ≈ inv(I_mc) * Σ2 * inv(I_mc) rtol = 1e-6
        end
        coefs = mean(coef(f) for f in fits)
        @test check_golden(g, "mcmle_seed_mean", coefs) ||
              error(golden_report(g, "mcmle_seed_mean", coefs))
        # A weighted fit differs from the unit-weight fit of the same egos:
        # the targets differ by more than the Monte-Carlo noise, so the
        # weights demonstrably reach the estimate
        fu = ergm_ego(unit, terms; rng=Random.Xoshiro(1))
        @test fu.model.targets != fits[1].model.targets
        @test fu.converged
        @test maximum(abs.(coef(fu) .- coefs)) > 0.05
    end

    # ------------------------------------------------------------------
    # The DEFAULTS at realistic network size — the defect the July fixture
    # had to work around, pinned in its FIXED state.
    # ------------------------------------------------------------------
    @testset "Defaults converge at realistic network size (205-actor census)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "fauxmesa_ego_census.toml"))
        net = load_dataset(:faux_mesa_high)
        ed = fauxmesa_census(net)
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]
        r_pop = Float64.(g.values["mle_coefficients_population"])

        # The pre-0.2 defaults were fixed constants (n_samples=400, burnin=2000,
        # interval=20) that did not scale with the pseudo-population: on this
        # 205-actor census the chain stopped mixing and the fit returned
        # edges ≈ −21.9 against a true −6.02. The MCMC controls follow
        # ERGM.jl's one dyad-scaled rule (`_mcmc_controls`), the chains start
        # from a pseudo-population annealed to the targets at its MPLE, and the
        # stopping rule is R ergm's confidence test. The defaults must converge
        # here and land on R's answer within the single-fit Monte-Carlo sd.
        f = ergm_ego(ed, terms; rng=Random.Xoshiro(7))
        @test f.converged
        @test abs(f.coefficients[1] - r_pop[1]) < 0.1
        @test abs(f.coefficients[2] - r_pop[2]) < 0.1
        n_dyads = 205 * 204 ÷ 2
        ctl = ERGMEgo._mcmc_controls(205)
        @test ctl == (n_samples=3000, burnin=20 * n_dyads, interval=max(100, n_dyads ÷ 10))
        @test ctl.burnin == ERGM.Extension.mcmc_defaults(n_dyads).burnin
        @test ERGMEgo._mcmc_controls(205; n_samples=500, interval=7) ==
              (n_samples=500, burnin=20 * n_dyads, interval=7)

        # A converged fit's report describes the sample that passed — the
        # final sample IS the passing one (R's ergm design; no fresh draw is
        # taken afterwards), so `converged`, `termination`, `mcmc_convergence`,
        # `sim_stats` and the standard errors cannot disagree
        c = f.mcmc_convergence
        @test f.termination.rule === :confidence
        @test f.termination.p_value < 0.01
        @test (f.termination.precision, f.termination.confidence) == (0.1, 0.99)
        @test 1 <= c.iterations <= 5
        @test c.step_length == 1.0
        @test occursin("Termination: 99% equivalence test p", sprint(show, f))
        assert_one_sample(f)
        for s in (101, 505)
            assert_one_sample(ergm_ego(ed, terms; rng=Random.Xoshiro(s)))
        end

        # The annealed start: the pseudo-population the chains start from has
        # the target statistics (203 edges, 163 same-grade ties), so the MPLE
        # start is the solution of the moment equations for this
        # dyad-independent model and one sample confirms it
        pm = ERGMEgo._annealed_model(ed, fill(1, 205), [Edges(), NodeMatch(:Grade)],
                                     f.model.targets, 2, 1, Random.Xoshiro(1))
        @test compute_all(pm.formula.terms, pm.network) == f.model.targets == [203.0, 163.0]
        @test sort(vertex_attribute_vector(pm.network, :Grade, Int)) ==
              sort([e.ego_attrs[:Grade] for e in ed.egos])

        # --- Non-convergence is LOUD: a chain too short to mix (fixed-size
        # samples of 100 tie/no-tie draws five toggles apart on 20 910 dyads,
        # effective sample size ≈ 3) cannot show that the moment equations
        # are solved; the fit warns with the diagnostics, records
        # converged == false, prints the caveat under `Converged: false`, and
        # lists it in approximations
        short = (maxiter=2, n_samples=100, burnin=1000, interval=5, effective_size=nothing,
                 proposal=:tnt)
        u = @test_logs (:warn, r"did not converge") match_mode=:any ergm_ego(
            ed, terms; short..., rng=Random.Xoshiro(1))
        @test !u.converged
        @test u.mcmc_convergence.iterations == 2
        @test u.mcmc_convergence.n_eff < 20
        @test !(u.termination.p_value < 0.01)
        @test any(occursin("did not converge", a) for a in approximations(u))
        @test any(occursin("did not converge", a) for a in NetworkCore.fit_metadata(u).approximations)
        uout = sprint(show, u)
        @test occursin("Converged: false", uout)
        @test occursin("did not converge", uout)
        @test occursin("equivalence test p", uout) && occursin("max t-ratio", uout)
        @test !occursin("Termination:", uout)
        @test all(isfinite, stderror(u))            # SEs exist, but are flagged unreliable
        # The caveat quotes the sample at the RETURNED coefficients — the one
        # `converged` was decided on — and says so
        @test occursin("on the final sample at the returned coefficients", uout)
        assert_one_sample(u)

        # The pre-0.2 rule is still available: termination=:hotelling stops
        # when every t-ratio is below conv_threshold and the Hotelling test
        # does not reject. With maxiter=1 the one sample (at the start, or — if
        # that one fails and a step is taken — a fresh one at the returned
        # coefficients) decides, whatever the seed: `converged` is exactly the
        # rule applied to the recorded report, and a fit warns if and only if
        # it did not converge.
        rng = Random.Xoshiro(3)
        sn = network(30; directed=false)
        for i in 1:30, j in (i+1):30
            rand(rng) < 0.12 && add_edge!(sn, i, j)
        end
        sed = simulate_ego_sample(sn, 30; rng=rng)
        # (fixed-size tie/no-tie samples: the paths below are the loop's, and
        # each is pinned on a seed of that sampler)
        one_step(s; rule=:hotelling) =
            fit_ergm_ego(sed, [EgoEdges()]; ppopsize=50, maxiter=1, n_samples=200,
                         termination=rule, effective_size=nothing, proposal=:tnt,
                         rng=Random.Xoshiro(s))
        verdicts = Bool[]
        for s in 1:6
            logs, fit = Test.collect_test_logs(() -> one_step(s))
            warned = any(occursin("did not converge", string(l.message)) for l in logs)
            push!(verdicts, fit.converged)
            @test warned == !fit.converged
            @test fit.termination.rule === :hotelling
            @test fit.mcmc_convergence.iterations == 1
            @test fit.converged == (all(fit.mcmc_convergence.t_ratios .< 0.1) &&
                                    fit.mcmc_convergence.hotelling_p > 0.05)
            @test any(occursin("did not converge", a) for a in approximations(fit)) == !fit.converged
            @test occursin("Converged: $(fit.converged)", sprint(show, fit))
            assert_one_sample(fit)
        end
        # Both verdicts occur among the six seeds (4 converge, 2 do not), so
        # the equivalences above were exercised on each side
        @test any(verdicts) && !all(verdicts)

        # Under the default confidence rule with maxiter=1, the three paths a
        # one-iteration fit can take, each pinned on a seed that takes it:
        # (i) the first sample passes inside the loop — converged, and the
        # test was evaluated at the full step taken from it (seed 2);
        ok_loop = one_step(2; rule=:confidence)
        @test ok_loop.converged && !iszero(ok_loop.termination.step)
        assert_one_sample(ok_loop)
        # (ii) the first sample fails, a step is taken, and the fresh sample
        # at the returned coefficients passes — CONVERGED, no warning, and the
        # report is that post-step sample's (seed 11);
        logs, ok_post = Test.collect_test_logs(() -> one_step(11; rule=:confidence))
        @test !any(occursin("did not converge", string(l.message)) for l in logs)
        @test ok_post.converged && iszero(ok_post.termination.step)
        @test ok_post.mcmc_convergence.iterations == 1
        @test !any(occursin("did not converge", a) for a in approximations(ok_post))
        @test occursin("Converged: true", sprint(show, ok_post))
        assert_one_sample(ok_post)
        # (iii) ...and when that post-step sample fails, the fit is not
        # converged and says so, quoting that sample (seed 3)
        bad = @test_logs (:warn, r"on the final sample at the returned coefficients") match_mode=:any one_step(3; rule=:confidence)
        @test !bad.converged && iszero(bad.termination.step)
        assert_one_sample(bad)
        @test_throws ArgumentError fit_ergm_ego(sed, [EgoEdges()]; termination=:relative)
        @test_throws ArgumentError fit_ergm_ego(sed, [EgoEdges()]; conv_confidence=1.0)
        @test_throws ArgumentError fit_ergm_ego(sed, [EgoEdges()]; n_samples=100, max_n_samples=50)

        # --- The pseudo-population guard names the size, its origin and the fix
        errmsg(f) = (try f(); "" catch e; e isa ArgumentError ? e.msg : rethrow() end)
        msg = errmsg(() -> fit_ergm_ego(sed, [EgoEdges()]; ppopsize=1))
        @test occursin("at least 5 vertices", msg) && occursin("ppopsize=1", msg)
        @test occursin("the ppopsize keyword", msg) && occursin("pass ppopsize=<n> ≥ 5", msg)
        tiny = EgoData(sed.egos[1:0])
        msg = errmsg(() -> fit_ergm_ego(tiny, [EgoEdges()]))
        @test occursin("ppopsize=0", msg) && occursin("default rule", msg)
        @test occursin("0 egos", msg) && occursin("popsize=unknown", msg)
        msg = errmsg(() -> fit_ergm_ego(EgoData(sed.egos[1:0]; population_size=3), [EgoEdges()]))
        @test occursin("ppopsize=3", msg) && occursin("popsize=3", msg)

        # --- Keyword vocabulary: maxiter is the name. The development-era
        # spellings `max_iter` and `tol` were removed (never released): they
        # are unknown keywords now, before any fitting
        kw = Base.kwarg_decl(first(methods(fit_ergm_ego)))
        for name in (:maxiter, :n_samples, :rng, :burnin, :interval, :conv_threshold,
                     :hotelling_alpha, :ppopsize, :popsize, :termination, :conv_precision,
                     :conv_confidence, :max_n_samples, :proposal, :se)
            @test name in kw
        end
        @test !(:max_iter in kw) && !(:tol in kw)
        @test_throws MethodError ergm_ego(ed, terms; max_iter=2)
        @test_throws MethodError ergm_ego(ed, terms; tol=0.01)
        @test_throws ArgumentError ergm_ego(ed, terms; maxiter=0)
    end

    @testset "Golden fixture: ergm.ego network-size offset, triangle and gwdegree fits" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ego_netsize.toml"))
        v = g.values
        @test g.provenance["ergm_ego_version"] == "1.1.4"
        @test occursin("transitiveties = -0.333", v["offset_formula"])

        # --- (a) THE OFFSET STATISTIC: deterministic ---------------------------
        # R's transitiveties on an undirected network is gwesp at decay 0 (the
        # ties with at least one shared partner), so the offset statistic
        # edges − transitiveties/3 is Edges − GWESP(0.0)/3
        gnet = network(Int(v["g_n"]); directed=false)
        for (a, b) in zip(v["g_edge_src"], v["g_edge_dst"])
            add_edge!(gnet, Int(a), Int(b))
        end
        fmh = load_dataset(:faux_mesa_high)
        for (net, key) in ((gnet, "g"), (fmh, "fmh"))
            s = summary_stats(net, [Edges(), GWESP(0.0)])
            @test s[2] == v["$(key)_transitiveties"]
            @test s[1] - s[2] / 3 ≈ v["$(key)_offset_statistic"] atol = g.tolerance["offset_statistic"]
        end
        @test summary_stats(gnet, [Edges(), Triangle()]) == (edges = Float64(v["g_edges"]),
                                                             triangle = Float64(v["g_triangles"]))

        # Monte-Carlo comparison of a few seeded Julia fits with R's seed
        # mean: the fixture's rule (see its [tolerance] block)
        function check_block(key, fitter; seeds=(1, 2, 3), adj_atol=1e-12)
            fits = [fitter(Random.Xoshiro(s)) for s in seeds]
            @test all(f -> f.converged, fits)
            nJ, nR = length(seeds), Int(v["$(key)_n_fits"])
            jmean = mean(coef(f) for f in fits)
            sdR = Float64.(v["$(key)_sd"])
            seR = Float64.(v["$(key)_se_mean"])
            tol = max.(g.tolerance["seed_mean_sds"] .* sdR .* sqrt(1 / nR + 1 / nJ),
                       g.tolerance["coef_floor_se"] .* seR)
            @test all(abs.(jmean .- Float64.(v["$(key)_mean"])) .<= tol)
            jse = mean(stderror(f) for f in fits)
            @test all(abs.(jse .- seR) .<= g.tolerance["std_errors_rel"] .* seR)
            # R's netsize.adj row and the constructed pseudo-population size
            @test all(f -> isapprox(f.netsize_adjustment, v["$(key)_netsize_adj"]; atol=adj_atol), fits)
            @test all(f -> f.model.ppopsize == v["$(key)_ppopsize"], fits)
            return fits
        end

        # --- (b) THE TRIANGLE MODEL UNDER THREE POPULATION SIZES ---------------
        gcensus = simulate_ego_sample(gnet, nv(gnet); rng=Random.Xoshiro(0))
        tri = [EgoEdges(), EgoTriangle()]
        @test ego_target_stats(tri, gcensus, 100) ≈ [v["g_edges"], v["g_triangles"]]
        same = check_block("tri_same", r -> fit_ergm_ego(gcensus, tri; popsize=100, ppopsize=100, se=:design, rng=r))
        # N == m: no offset, the plain ERGM's terms
        @test length(same[1].model.ergm_terms) == 2
        @test same[1].netsize_adjustment === 0.0
        double = check_block("tri_double", r -> fit_ergm_ego(gcensus, tri; popsize=200, ppopsize=100, se=:design, rng=r))
        # N ≠ m with a triangle term: the transitive-ties part of the offset
        # is a fixed-coefficient statistic of the simulated model
        off = double[1].model.ergm_terms[end]
        @test off isa ERGM.Offset && off.term == GWESP(0.0)
        @test off.coef ≈ [-log(2) / 3]
        @test occursin("edges − transitiveties/3", sprint(show, double[1]))
        @test any(occursin("edges − transitiveties/3", a) for a in approximations(double[1]))
        # Unknown population size: R's default popsize = 1, per-capita coefficients
        unknown = EgoData(gcensus.egos)
        @test unknown.population_size === nothing
        # ...at the DEFAULT pseudo-population size, which is ergm.ego's: the
        # number of egos (100) when the population size is unknown
        percap = check_block("tri_percap", r -> fit_ergm_ego(unknown, tri; rng=r))
        @test percap[1].model.popsize == 1
        @test percap[1].netsize_adjustment ≈ -log(100)
        @test percap[1].model.ergm_terms[end].coef ≈ [log(100) / 3]
        out = sprint(show, percap[1])
        @test occursin("population: unknown (popsize = 1)", out)
        @test occursin("per capita", out)
        # The three are different models: shifting only the edges coefficient
        # (the pre-0.2 code) would leave the triangle coefficient where the
        # N == m fit has it
        tri_coef(fits) = mean(coef(f)[2] for f in fits)
        @test tri_coef(double) - tri_coef(same) > 0.4
        @test tri_coef(percap) - tri_coef(same) < -3.5

        # --- (c) gwdegree UNDER A WEIGHTED DESIGN ------------------------------
        # (Before 0.2 about a third of the seeds of this fit ran off to
        # gwdegree ≈ −5.5; R itself fails on one of its six seeds.)
        ids = Int.(v["ego_ids"])
        w = Float64.(v["weights"])
        wed = fauxmesa_weighted(ids, w; net=fmh)
        gw_terms = [EgoEdges(), EgoNodeMatch(:Grade), EgoGWDegree(0.5)]
        @test [name(ERGMEgo.ergm_term(t)) for t in gw_terms] == v["gwdeg_names"]
        gw = check_block("gwdeg", r -> fit_ergm_ego(wed, gw_terms; popsize=205, ppopsize=205, se=:design, rng=r);
                         seeds=(1, 2, 3, 4, 5))
        @test all(f -> f.mcmc_convergence.iterations <= 10, gw)
        @test all(f -> all(isfinite, stderror(f)), gw)

        # --- (d) PER-CAPITA COEFFICIENTS ARE PSEUDO-POPULATION-SIZE INVARIANT --
        sub = EgoData(wed.egos)                   # unit weights, population unknown
        dyadic = [EgoEdges(), EgoNodeMatch(:Grade)]
        pc207 = check_block("percap207", r -> fit_ergm_ego(sub, dyadic; ppopsize=207, rng=r))
        pc414 = check_block("percap414", r -> fit_ergm_ego(sub, dyadic; ppopsize=414, rng=r);
                            seeds=(1, 2))
        @test abs(mean(coef(f)[1] for f in pc207) - mean(coef(f)[1] for f in pc414)) < 0.03
        # ...where the pre-0.2 default (popsize := ppopsize) moved the edges
        # coefficient by log(414/207) = 0.69
        @test abs((coef(pc207[1])[1] + pc207[1].netsize_adjustment) -
                  (coef(pc414[1])[1] + pc414[1].netsize_adjustment)) > 0.6

        # --- (e) popsize ≠ ppopsize ON A WEIGHTED DESIGN -----------------------
        # ppopsize = 500 requested; rounding the replication counts gives 508
        # vertices, as in R, and the offset uses the realised size
        big = @test_logs (:info, r"508 vertices, not the requested 500") match_mode=:any check_block(
            "big", r -> fit_ergm_ego(wed, dyadic; popsize=2050, ppopsize=500, se=:design, rng=r); seeds=(1,))
        @test big[1].netsize_adjustment ≈ -log(508 / 2050)
        @test sum(big[1].model.ppop_counts) == 508
    end

    @testset "Attribute and degree terms: nodefactor, nodecov, absdiff, degree" begin
        # --- Under a census the per-capita targets scaled to the network size
        # ARE the network's own statistics: every new term against ERGM.jl's
        # statistic on the network (brute force, no R), on faux.mesa.high and
        # on a random network with a non-integer numeric attribute
        net = load_dataset(:faux_mesa_high)
        census = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Race, :Sex],
                                     rng=Random.Xoshiro(0))
        terms = EgoTerm[EgoEdges(), EgoNodeFactor(:Race), EgoNodeFactor(:Sex),
                        EgoNodeFactor(:Grade; base=0), EgoNodeFactor(:Grade; levels=[12, 9]),
                        EgoNodeCov(:Grade), EgoAbsDiff(:Grade), EgoAbsDiff(:Grade; pow=0.5),
                        EgoDegree.(0:4)...]
        expanded = ERGMEgo._expand_ego_terms(terms, census)
        @test all(t -> !(t isa EgoNodeFactor) || t.level !== nothing, expanded)
        @test ego_target_stats(terms, census, 205) ≈
              [compute(ERGMEgo.ergm_term(t), net) for t in expanded] atol = 1e-9
        rng = Random.Xoshiro(11)
        g = network(40; directed=false)
        x = Dict(v => round(10 * rand(rng); digits=2) for v in 1:40)
        c = Dict(v => rand(rng, ["p", "q", "r"]) for v in 1:40)
        set_vertex_attribute!(g, :x, x); set_vertex_attribute!(g, :c, c)
        for i in 1:40, j in (i+1):40
            rand(rng) < 0.1 && add_edge!(g, i, j)
        end
        gc = simulate_ego_sample(g, 40; ego_attrs=[:x, :c], rng=Random.Xoshiro(1))
        gterms = EgoTerm[EgoNodeFactor(:c), EgoNodeCov(:x), EgoAbsDiff(:x; pow=2), EgoDegree(3)]
        @test ego_target_stats(gterms, gc, 40) ≈
              [compute(ERGMEgo.ergm_term(t), g) for t in ERGMEgo._expand_ego_terms(gterms, gc)] rtol = 1e-12

        # --- The per-ego contributions by hand (fixture_egodata: groups A/B;
        # ego 1 (A) has alters A, B, A; ego 2 (B) has B, B; ego 3 (A) has
        # A, A, B, B)
        ed = fixture_egodata()
        @test [ERGMEgo.ego_contribution(EgoNodeFactor(:group; level="B"), e) for e in ed.egos] ==
              [0.5, 2.0, 1.0]           # (0·3 + 1)/2, (1·2 + 2)/2, (0·4 + 2)/2
        @test [ERGMEgo.ego_contribution(EgoNodeFactor(:group; level="A"), e) for e in ed.egos] ==
              [2.5, 0.0, 3.0]           # (3 + 2)/2, 0, (4 + 2)/2
        @test compute(EgoNodeFactor(:group), ed) == compute(EgoNodeFactor(:group; level="B"), ed) ≈ 3.5 / 3
        @test compute(EgoNodeFactor(:group; base=0), ed) ≈ (3.5 + 5.5) / 3   # = 2·edges per ego
        @test compute(EgoNodeFactor(:group; base=0), ed) ≈ 2 * compute(EgoEdges(), ed)
        e = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3);
                       ego_attrs=Dict{Symbol,Any}(:x => 2.0),
                       alter_attrs=Dict{Symbol,Vector}(:x => [1.0, 4.0, 2.5]))
        @test ERGMEgo.ego_contribution(EgoNodeCov(:x), e) == (2.0 * 3 + 7.5) / 2
        @test ERGMEgo.ego_contribution(EgoAbsDiff(:x), e) == (1.0 + 2.0 + 0.5) / 2
        @test ERGMEgo.ego_contribution(EgoAbsDiff(:x; pow=2), e) == (1.0 + 4.0 + 0.25) / 2
        @test ERGMEgo.ego_contribution(EgoDegree(3), e) == 1.0

        # --- Levels are the EGOS' sorted values, first dropped (R's default)
        @test name.(ERGMEgo._expand_ego_terms([EgoNodeFactor(:Race)], census)) ==
              ["nodefactor.Race.$l" for l in ("Hisp", "NatAm", "Other", "White")]
        @test name.(ERGMEgo._expand_ego_terms([EgoNodeFactor(:Grade; base=[1, 2])], census)) ==
              ["nodefactor.Grade.$l" for l in 9:12]
        @test name(ERGMEgo.ergm_term(EgoNodeFactor(:Sex; level="M"))) == "nodefactor.Sex.M"
        @test name(ERGMEgo.ergm_term(EgoAbsDiff(:Grade; pow=2))) == "absdiff2.Grade"
        @test name(EgoAbsDiff(:Grade; pow=2)) == "absdiff2.Grade"
        @test name(ERGMEgo.ergm_term(EgoNodeCov(:Grade))) == "nodecov.Grade"
        @test name(ERGMEgo.ergm_term(EgoDegree(2))) == "degree2"
        # an alter's level that no ego has counts for nothing (R's match(…, 0))
        odd = EgoNetwork(1, [10], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:g => "a"),
                         alter_attrs=Dict{Symbol,Vector}(:g => ["z"]))
        odd2 = EgoNetwork(2, [11], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:g => "b"),
                          alter_attrs=Dict{Symbol,Vector}(:g => ["z"]))
        @test name.(ERGMEgo._expand_ego_terms([EgoNodeFactor(:g; base=0)], EgoData([odd, odd2]))) ==
              ["nodefactor.g.a", "nodefactor.g.b"]

        # --- Refusals, in words
        errmsg(f) = (try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end)
        @test occursin("not a value of any ego", errmsg(() -> compute(EgoNodeFactor(:Race; levels=["Martian"]), census)))
        @test occursin("no levels remain", errmsg(() -> ego_target_stats([EgoNodeFactor(:Sex; base=[1, 2])], census, 10)))
        @test_throws ArgumentError EgoNodeFactor(:Sex; level="M", levels=["M"])
        @test_throws ArgumentError EgoNodeFactor(:Sex; base=-1)
        @test_throws ArgumentError EgoDegree(-1)
        @test occursin("multi-level", errmsg(() -> ERGMEgo.ego_contribution(EgoNodeFactor(:Sex), census.egos[1])))
        @test occursin("multi-level", errmsg(() -> ERGMEgo.ergm_term(EgoNodeFactor(:Sex))))
        noalt = EgoNetwork(1, [10], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:x => 1))
        @test occursin("no alter attribute :x", errmsg(() -> compute(EgoNodeCov(:x), EgoData([noalt]))))
        @test occursin("no ego attribute :y", errmsg(() -> compute(EgoNodeFactor(:y), EgoData([noalt]))))
        txt = EgoNetwork(1, [10], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:x => 1),
                         alter_attrs=Dict{Symbol,Vector}(:x => Any["high"]))
        msg = errmsg(() -> compute(EgoAbsDiff(:x), EgoData([txt])))
        @test occursin("\"high\"", msg) && occursin("real number", msg)
        unk = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2); ego_attrs=Dict{Symbol,Any}(:x => 1),
                         alter_attrs=Dict{Symbol,Vector}(:x => [1, missing]))
        @test occursin("`missing`", errmsg(() -> compute(EgoNodeCov(:x), EgoData([unk]))))

        # --- Fitting expands once, against the data: the model holds the
        # one-level terms, `coef` has a row per level, and the simulated
        # model's terms are ERGM.jl's with R's labels
        small = simulate_ego_sample(net, 60; ego_attrs=[:Sex, :Grade], rng=Random.Xoshiro(2))
        f = fit_ergm_ego(small, [EgoEdges(), EgoNodeFactor(:Sex), EgoAbsDiff(:Grade), EgoDegree(0)];
                         n_samples=400, rng=Random.Xoshiro(1))
        @test name.(f.model.ego_terms) == ["edges", "nodefactor.Sex.M", "absdiff.Grade", "degree0"]
        @test name.(f.model.ergm_terms) == ["edges", "nodefactor.Sex.M", "absdiff.Grade", "degree0"]
        @test length(coef(f)) == 4 && f.converged
        @test f.model.targets ≈ ego_target_stats(f.model.ego_terms, small, f.model.ppopsize)
    end

    @testset "Golden fixture: ergm.ego's nodefactor, nodecov, absdiff, degree and gwesp" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ego_terms.toml"))
        v = g.values
        net = load_dataset(:faux_mesa_high)
        @test nv(net) == v["n_actors"]
        census = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Race, :Sex],
                                     rng=Random.Xoshiro(0))
        # R's formula, term for term
        terms = EgoTerm[EgoEdges(), EgoNodeFactor(:Race), EgoNodeFactor(:Sex), EgoNodeCov(:Grade),
                        EgoAbsDiff(:Grade), EgoAbsDiff(:Grade; pow=2), EgoDegree.(0:3)...,
                        EgoNodeFactor(:Grade; base=[1, 2]), EgoNodeFactor(:Grade; base=0),
                        EgoGWESP(0.0), EgoGWESP(0.5)]
        expanded = ERGMEgo._expand_ego_terms(terms, census)
        # R's labels: the ego terms' own labels, and those of the ERGM.jl
        # terms they estimate
        @test name.(expanded) == v["term_names"]
        @test [name(ERGMEgo.ergm_term(t)) for t in expanded] == v["term_names"]
        @test ego_target_stats(terms, census, 205) ≈ Float64.(v["census_targets"]) atol = g.tolerance["targets"]

        # The weighted sub-design: Hájek targets and their design covariance
        ids = Int.(v["ego_ids"])
        keep = sort([e for e in census.egos if e.ego in ids]; by=e -> e.ego)
        wed = ego_design(EgoData(keep; population_size=205); weights=Float64.(v["weights"]))
        @test name.(ERGMEgo._expand_ego_terms(terms, wed)) == name.(expanded)   # same levels on the sub-design
        @test ego_target_stats(terms, wed, 205) ≈ Float64.(v["weighted_targets"]) atol = g.tolerance["targets"]
        Σ_r = reduce(vcat, [Float64.(r)' for r in v["weighted_design_cov"]])
        Σ_jl = ERGMEgo._design_cov(expanded, wed, 205)
        @test size(Σ_jl) == size(Σ_r)
        # entries up to 7·10⁴ (nodecov): 1e-9 relative to the largest
        @test maximum(abs.(Σ_jl .- Σ_r)) <= g.tolerance["design_cov_rel"] * maximum(abs.(Σ_r))

        # Monte-Carlo: R's mean over its seeds, at the fixture's rule — every
        # band is R's own seed spread (see the fixture's [tolerance] block)
        check_block(key, fitter; seeds=(1, 2)) = golden_fit_block(g, key, fitter; seeds)
        # (b) the ergm.ego example model (without gwesp) on the census
        example = EgoTerm[EgoEdges(), EgoDegree.(0:3)..., EgoNodeFactor(:Race), EgoNodeMatch(:Race),
                          EgoNodeFactor(:Sex), EgoNodeMatch(:Sex), EgoAbsDiff(:Grade)]
        check_block("example", r -> fit_ergm_ego(census, example; se=:design, rng=r))
        # (c) popsize unknown: the DEFAULT pseudo-population is ergm.ego's, the
        # number of egos, and the per-capita coefficients are R's
        unweighted = EgoData(keep)
        pc = check_block("default_ppop", r -> fit_ergm_ego(unweighted,
                         [EgoEdges(), EgoNodeFactor(:Sex), EgoAbsDiff(:Grade)]; rng=r))
        @test pc[1].model.ppopsize == length(keep) == 69 && pc[1].model.popsize == 1
        # (d) gwesp(0, fixed=TRUE), the help page's gwesp term, on the census
        check_block("gwesp_census", r -> fit_ergm_ego(census,
                    [EgoEdges(), EgoNodeMatch(:Grade), EgoGWESP(0.0)]; se=:design, rng=r))
        # (e) gwesp per capita: an order-3 statistic, so the offset is on
        # edges − transitiveties/3 (a trailing GWESP(0) offset); a port that
        # left the offset on edges alone misses R's gwesp coefficient
        gp = check_block("gwesp_percap", r -> fit_ergm_ego(unweighted,
                         [EgoEdges(), EgoGWESP(0.5)]; rng=r))
        @test gp[1].model.ergm_terms[end] isa ERGM.Offset &&
              gp[1].model.ergm_terms[end].term == GWESP(0.0)
        # (f) a target at its bound: no ego of the census without isolates
        # has degree 0, so degree0 is fixed at -Inf as ergm.ego fixes it (the
        # code before the drop returned a finite, "converged" −5.6)
        noniso = sort([e for e in census.egos if e.ego in Int.(v["noniso_ego_ids"])]; by=e -> e.ego)
        @test all(e -> ego_degree(e) > 0, noniso) && length(noniso) == 148
        bed = EgoData(noniso; population_size=148)
        bfits = @test_logs (:warn, r"degree0 are at their smallest attainable values") match_mode=:any check_block(
            "boundary", r -> fit_ergm_ego(bed, [EgoEdges(), EgoDegree(0), EgoNodeMatch(:Grade)];
                                          se=:design, rng=r))
        @test coef(bfits[1])[2] == -Inf && dof(bfits[1]) == 2
    end

    @testset "Golden fixture: ergm.ego's esp, mm, concurrent, the help-page model and the top of a range" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ego_mixing_esp.toml"))
        v = g.values
        net = load_dataset(:faux_mesa_high)
        @test nv(net) == v["n_actors"]
        census = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Race, :Sex],
                                     rng=Random.Xoshiro(0))
        # (a) R's formula, term for term: targets and labels on the census
        terms = EgoTerm[EgoESP.(0:3)..., EgoMM(:Race), EgoMM(:Sex), EgoConcurrent()]
        expanded = ERGMEgo._expand_ego_terms(terms, census)
        @test name.(expanded) == v["term_names"]
        @test ego_target_stats(terms, census, 205) ≈ Float64.(v["census_targets"]) atol = g.tolerance["targets"]
        # ... on the weighted sub-design, with the design covariance (esp and
        # concurrent: ergm.ego's mm stops with an error on a weighted design,
        # which the fixture records; EgoMM weights its per-ego values as every
        # other term does)
        ids = Int.(v["ego_ids"])
        keep = sort([e for e in census.egos if e.ego in ids]; by=e -> e.ego)
        wed = ego_design(EgoData(keep; population_size=205); weights=Float64.(v["weights"]))
        wterms = EgoTerm[EgoESP.(0:3)..., EgoConcurrent()]
        @test name.(wterms) == v["weighted_names"]
        @test ego_target_stats(wterms, wed, 205) ≈ Float64.(v["weighted_targets"]) atol = g.tolerance["targets"]
        Σ_r = reduce(vcat, [Float64.(r)' for r in v["weighted_design_cov"]])
        Σ_jl = ERGMEgo._design_cov(wterms, wed, 205)
        @test size(Σ_jl) == size(Σ_r)
        @test maximum(abs.(Σ_jl .- Σ_r)) <= g.tolerance["design_cov_rel"] * maximum(abs.(Σ_r))
        @test !isempty(v["r_error_mm_weighted"])
        @test all(isfinite, ego_target_stats([EgoMM(:Race)], wed, 205))
        # ... and mm's levels, the values at either end of a reported tie:
        # the unweighted sub-design (its one "Other" ego reports no alter: no
        # Other cell), the egos who are not Black (Black only among the
        # alters: its cells are there), and the egos without the one friend of
        # an "Other" student (that "Other" ego reports an alter that is not
        # "Other": its cells are there)
        sub = EgoData(keep; population_size=205)
        mmterms = EgoTerm[EgoMM(:Race), EgoMM(:Sex)]
        @test name.(ERGMEgo._expand_ego_terms(mmterms, sub)) == v["sub_mm_names"]
        @test ego_target_stats(mmterms, sub, 205) ≈ Float64.(v["sub_mm_targets"]) atol = g.tolerance["targets"]
        for (key, idkey) in (("nonblack", "nonblack_ego_ids"), ("nofriend", "nofriend_ego_ids"))
            ids_k = Int.(v[idkey])
            edk = EgoData(sort([e for e in census.egos if e.ego in ids_k]; by=e -> e.ego);
                          population_size=205)
            @test name.(ERGMEgo._expand_ego_terms([EgoMM(:Race)], edk)) == v["$(key)_mm_names"]
            @test ego_target_stats([EgoMM(:Race)], edk, 205) ≈ Float64.(v["$(key)_mm_targets"]) atol = g.tolerance["targets"]
        end

        # (b) ergm.ego's help-page model, gwesp(0, fixed=TRUE) included, at the
        # DEFAULT settings (R ergm 4's sampler design: SPDyad, ESS-adaptive
        # sampling on a continued chain). The fixed-size tie/no-tie sampler
        # did not converge on it in 60 iterations
        helppage = EgoTerm[EgoEdges(), EgoDegree.(0:3)..., EgoNodeFactor(:Race), EgoNodeMatch(:Race),
                           EgoNodeFactor(:Sex), EgoNodeMatch(:Sex), EgoAbsDiff(:Grade), EgoGWESP(0.0)]
        hp = golden_fit_block(g, "helppage", r -> fit_ergm_ego(census, helppage; se=:design, rng=r))
        for f in hp
            @test f.mcmc_convergence.iterations < 60 && f.mcmc_convergence.n_eff >= 64
            assert_one_sample(f)
        end
        # (c)-(e) mm, concurrent and esp on the census
        golden_fit_block(g, "mm", r -> fit_ergm_ego(census, [EgoEdges(), EgoMM(:Sex)]; se=:design, rng=r))
        golden_fit_block(g, "concurrent", r -> fit_ergm_ego(census,
                         [EgoEdges(), EgoNodeMatch(:Grade), EgoConcurrent()]; se=:design, rng=r))
        golden_fit_block(g, "esp", r -> fit_ergm_ego(census,
                         [EgoEdges(), EgoNodeMatch(:Grade), EgoESP(1)]; se=:design, rng=r))
        # (f) a target at the TOP of its range: every ego of the census without
        # isolates has an alter, so gwdeg.fixed.0 is at its largest value and
        # is fixed at +Inf, as ergm.ego fixes it
        noniso = sort([e for e in census.egos if e.ego in Int.(v["noniso_ego_ids"])]; by=e -> e.ego)
        @test length(noniso) == 148 && all(e -> ego_degree(e) > 0, noniso)
        ted = EgoData(noniso; population_size=148)
        tfits = @test_logs (:warn, r"gwdeg\.fixed\.0 are at their largest attainable values") match_mode=:any golden_fit_block(
            g, "top", r -> fit_ergm_ego(ted, [EgoEdges(), EgoNodeMatch(:Grade), EgoGWDegree(0.0)];
                                        se=:design, rng=r))
        @test coef(tfits[1])[3] == Inf && dof(tfits[1]) == 2
        # (g) esp per capita: an order-3 statistic, so the offset is on
        # edges − transitiveties/3 (ergm.ego's offset formula, recorded)
        @test occursin("transitiveties = -0.333", v["esp_percap_offset_term"])
        ep = golden_fit_block(g, "esp_percap", r -> fit_ergm_ego(EgoData(keep), [EgoEdges(), EgoESP(1)]; rng=r))
        @test ep[1].model.ergm_terms[end] isa ERGM.Offset &&
              ep[1].model.ergm_terms[end].term == GWESP(0.0)
    end

    @testset "EgoGWESP estimates ERGM.jl's GWESP (ergm.ego's gwesp, fixed=TRUE)" begin
        # Under a census the per-ego values, scaled, are the network's own
        # statistic: every tie's shared partners are seen from both ends.
        # Random networks, several decays, decay 0 included
        rng = Random.Xoshiro(17)
        for trial in 1:12
            n = rand(rng, 8:30)
            net = network(n; directed=false)
            for i in 1:n, j in (i + 1):n
                rand(rng) < 0.3 && add_edge!(net, i, j)
            end
            census = simulate_ego_sample(net, n; rng=rng)
            for α in (0.0, 0.25, 0.5, 1.0, 2.5)
                @test n * compute(EgoGWESP(α), census) ≈ compute(GWESP(α), net) rtol = 1e-12 atol = 1e-12
            end
        end
        # Ego by ego, on a partial sample: half the weighted sum, over the
        # ego's ties, of the shared partners counted in the whole network
        rng = Random.Xoshiro(29)
        for trial in 1:8
            n = rand(rng, 5:25)
            net = network(n; directed=false)
            for i in 1:n, j in (i + 1):n
                rand(rng) < 0.4 && add_edge!(net, i, j)
            end
            sample = simulate_ego_sample(net, rand(rng, 1:n); rng=rng)
            for α in (0.0, 0.5, 2.5), e in sample.egos
                Ni = Set(neighbors(net, e.ego))
                ref = sum((exp(α) * (1 - (1 - exp(-α))^length(intersect(Ni, Set(neighbors(net, j)))))
                           for j in Ni); init=0.0) / 2
                @test ERGMEgo.ego_contribution(EgoGWESP(α), e) ≈ ref rtol = 1e-12 atol = 1e-12
            end
        end
        # No shared-partner cutoff: on the complete graph on 33 vertices every
        # tie has 31 shared partners. ergm.ego's gwesp (cutoff = 30) returns 0
        # there; ergm's gwesp(0.5, fixed=TRUE) returns 870.5248 (R 4.6.1,
        # ergm 4.12.0), and so does EgoGWESP under a census
        k33 = network(33; directed=false)
        for i in 1:33, j in (i + 1):33
            add_edge!(k33, i, j)
        end
        c33 = simulate_ego_sample(k33, 33; rng=Random.Xoshiro(1))
        @test 33 * compute(EgoGWESP(0.5), c33) ≈ 528 * exp(0.5) * (1 - (1 - exp(-0.5))^31) rtol = 1e-12
        @test 33 * compute(EgoGWESP(0.5), c33) ≈ 870.5248 atol = 1e-4
        @test name(EgoGWESP(0.0)) == "gwesp.fixed.0" && name(EgoGWESP(0.5)) == "gwesp.fixed.0.5"
        @test ERGMEgo.ergm_term(EgoGWESP(0.5)) == GWESP(0.5)
        @test ERGMEgo._is_triadic(EgoGWESP(0.5)) && ERGMEgo._needs_alter_ties(EgoGWESP(0.5))
        @test_throws ArgumentError EgoGWESP(-0.1)
        # Data whose alter–alter ties were not collected is refused, not
        # counted as having no shared partners
        e = EgoNetwork(1, [10, 11], Bool[0 1; 1 0])
        nocol = EgoData([e]; design=Dict{Symbol,Any}(:alter_ties_observed => false))
        @test_throws ArgumentError compute(EgoGWESP(0.5), nocol)
    end

    @testset "Moment matching samples as R ergm 4 does (SPDyad, ESS-adaptive, continued chain)" begin
        # The fast proxy of the help-page pin in the golden testset: the
        # default sampler is ERGM.mcmle's (`ERGM.Extension.mcmle_sampler`)
        # with R ergm 4's defaults — a default fit IS the explicit one, draw
        # for draw — and the fixed-size tie/no-tie sampler, under which
        # ergm.ego's help-page model did not converge in 60 iterations, is one
        # keyword pair away
        rng = Random.Xoshiro(5)
        n = 40
        net = network(n; directed=false)
        for i in 1:n, j in (i + 1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        census = simulate_ego_sample(net, n; rng=rng)
        terms = [EgoEdges(), EgoGWESP(0.5)]
        d = fit_ergm_ego(census, terms; se=:design, rng=Random.Xoshiro(1))
        e = fit_ergm_ego(census, terms; se=:design, proposal=:spdyad, effective_size=64,
                         rng=Random.Xoshiro(1))
        @test coef(d) == coef(e) && d.sim_stats == e.sim_stats
        t = fit_ergm_ego(census, terms; se=:design, proposal=:tnt, effective_size=nothing,
                         rng=Random.Xoshiro(1))
        @test coef(t) != coef(d)
        # the fixed-size design stores n_samples draws (max(400, 20m) = 800);
        # the adaptive one stores what its effective-size target needs
        @test size(t.sim_stats, 1) == 800
        @test d.converged && d.mcmc_convergence.n_eff >= 64
        @test 256 <= size(d.sim_stats, 1) <= 4 * 800
        assert_one_sample(d)
        # both estimate the same model
        @test t.converged && maximum(abs.(coef(t) .- coef(d))) < 0.3
        # the adaptive target is validated before any MCMC
        @test_throws ArgumentError fit_ergm_ego(census, terms; effective_size=4)
        @test_throws ArgumentError fit_ergm_ego(census, terms; proposal=:gibbs)
    end

    @testset "EgoESP, EgoMM and EgoConcurrent estimate ERGM.jl's ESP, NodeMix and Concurrent" begin
        # Under a census the per-ego values, scaled, are the network's own
        # statistics, for random networks with a three-level attribute
        rng = Random.Xoshiro(41)
        for trial in 1:10
            n = rand(rng, 8:30)
            net = network(n; directed=false)
            for i in 1:n, j in (i + 1):n
                rand(rng) < 0.3 && add_edge!(net, i, j)
            end
            set_vertex_attribute!(net, :g, Dict(v => rand(rng, ("a", "b", "c")) for v in 1:n))
            census = simulate_ego_sample(net, n; ego_attrs=[:g], rng=rng)
            for k in 0:4
                @test n * compute(EgoESP(k), census) ≈ compute(ESP(k), net) atol = 1e-12
            end
            @test n * compute(EgoConcurrent(), census) == compute(Concurrent(), net)
            for (l1, l2) in (("a", "a"), ("a", "b"), ("b", "a"), ("b", "c"), ("c", "c"))
                @test n * compute(EgoMM(:g, l1, l2), census) ≈ compute(NodeMix(:g, l1, l2), net) atol = 1e-12
            end
            # the specification: R's cells in R's order, the first dropped,
            # each the network's own NodeMix cell
            cells = ERGMEgo._expand_ego_terms([EgoMM(:g)], census)
            @test all(t -> n * compute(t, census) ≈ compute(ERGMEgo.ergm_term(t), net), cells)
            @test n * compute(EgoMM(:g), census) ≈ sum(compute(ERGMEgo.ergm_term(t), net) for t in cells)
        end
        # Ego by ego on partial samples: half the number of the ego's ties
        # whose ends share exactly k partners in the whole network
        rng = Random.Xoshiro(43)
        for trial in 1:6
            n = rand(rng, 6:25)
            net = network(n; directed=false)
            for i in 1:n, j in (i + 1):n
                rand(rng) < 0.4 && add_edge!(net, i, j)
            end
            sample = simulate_ego_sample(net, rand(rng, 1:n); rng=rng)
            for k in 0:3, e in sample.egos
                Ni = Set(neighbors(net, e.ego))
                ref = count(j -> length(intersect(Ni, Set(neighbors(net, j)))) == k, Ni) / 2
                @test ERGMEgo.ego_contribution(EgoESP(k), e) == ref
            end
        end
        # EgoGWESP is the geometric weighting of the EgoESP distribution
        census = fauxmesa_census()
        w(α, k) = exp(α) * (1 - (1 - exp(-α))^k)
        @test compute(EgoGWESP(0.5), census) ≈ sum(w(0.5, k) * compute(EgoESP(k), census) for k in 0:12)

        # Labels, R's; the ERGM.jl term each estimates; the offset flag
        @test name.(EgoESP.(0:1)) == ["esp0", "esp1"] && name(EgoConcurrent()) == "concurrent"
        @test name(EgoMM(:Race, "Black", "Hisp")) == "mm[Race=Black,Race=Hisp]"
        @test ERGMEgo.ergm_term(EgoESP(2)) == ESP(2)
        @test ERGMEgo.ergm_term(EgoConcurrent()) == Concurrent()
        @test ERGMEgo.ergm_term(EgoMM(:g, "a", "b")) == NodeMix(:g, "a", "b")
        @test ERGMEgo._is_triadic(EgoESP(1)) && ERGMEgo._needs_alter_ties(EgoESP(1))
        @test !ERGMEgo._is_triadic(EgoMM(:g, "a", "b")) && !ERGMEgo._is_triadic(EgoConcurrent())
        for m in (5, 30), t in EgoTerm[EgoESP(1), EgoConcurrent(), EgoMM(:g, "a", "b")]
            netm = network(m; directed=false)
            @test ERGM.Extension.attainable_range(t, netm) ==
                  ERGM.Extension.attainable_range(ERGMEgo.ergm_term(t), netm)
        end

        # mm's levels: those at either end of a reported tie. "z" is carried
        # only by an alter (included); "y" only by an ego with no alters
        # (excluded); the first cell (a, a) is the reference
        ea = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2); ego_attrs=Dict{Symbol,Any}(:g => "a"),
                        alter_attrs=Dict{Symbol,Vector}(:g => ["a", "z"]))
        ey = EgoNetwork(2, Int[], zeros(Bool, 0, 0); ego_attrs=Dict{Symbol,Any}(:g => "y"),
                        alter_attrs=Dict{Symbol,Vector}(:g => String[]))
        lv = EgoData([ea, ey])
        @test name.(ERGMEgo._expand_ego_terms([EgoMM(:g)], lv)) == ["mm[g=a,g=z]", "mm[g=z,g=z]"]
        @test ego_target_stats([EgoMM(:g)], lv, 10) == [2.5, 0.0]
        # a weighted design weights the per-ego values like every other term
        wlv = ego_design(lv; weights=[3.0, 1.0])
        @test compute(EgoMM(:g, "a", "z"), wlv) ≈ 3 * 0.5 / 4
        # ...and a cell whose level no ego of the pseudo-population carries
        # cannot be matched: a positive target is refused before any MCMC
        ez = EgoNetwork(3, [12], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:g => "a"),
                        alter_attrs=Dict{Symbol,Vector}(:g => ["a"]))
        msg = try
            fit_ergm_ego(EgoData([ea, ez, ea]), [EgoEdges(), EgoMM(:g, "a", "z")];
                         ppopsize=30, rng=Random.Xoshiro(1))
            ""
        catch err
            err isa ArgumentError ? err.msg : rethrow()
        end
        @test occursin("no vertex of the pseudo-population has level(s) \"z\"", msg)

        # Refusals: R's variants that are not implemented, and specifications
        # evaluated as one statistic
        @test_throws ArgumentError EgoESP(-1)
        @test_throws ArgumentError EgoMM(:Race, :Sex)              # R's mm(Race ~ Sex)
        @test_throws ArgumentError EgoMM(:Race; levels2=0)
        @test_throws ArgumentError EgoMM(:Race; levels=["Hisp"])
        @test_throws ArgumentError EgoConcurrent(by=:Sex)
        @test_throws ArgumentError EgoGWESP(0.5; fixed=false)      # curved gwesp
        @test_throws ArgumentError ERGMEgo.ergm_term(EgoMM(:Race))
        @test_throws ArgumentError ERGMEgo.ego_contribution(EgoMM(:Race), census.egos[1])
        @test_throws ArgumentError compute(EgoMM(:nope), census)
        # alter–alter ties not collected: esp is refused, not counted as zero
        e = EgoNetwork(1, [10, 11], Bool[0 1; 1 0])
        nocol = EgoData([e]; design=Dict{Symbol,Any}(:alter_ties_observed => false))
        @test_throws ArgumentError compute(EgoESP(1), nocol)
        @test compute(EgoConcurrent(), nocol) == 1.0
    end

    @testset "A target at a bound of its statistic is fixed at ∓Inf (ergm.ego's drop)" begin
        # The attainable ranges are ERGM.jl's for the term each ego term
        # estimates (R's minval/maxval), declared through the extension API's
        # generic rather than restated: no range table of its own is left
        @test !isdefined(ERGMEgo, :_ego_attainable_range)
        for m in (5, 30)
            netm = network(m; directed=false)
            @test ERGM.Extension.attainable_range(EgoEdges(), netm) == (0.0, m * (m - 1) / 2)
            @test ERGM.Extension.attainable_range(EgoDegree(2), netm) == (0.0, Float64(m))
            @test ERGM.Extension.attainable_range(EgoNodeCov(:x), netm) == (-Inf, Inf)
            for t in EgoTerm[EgoEdges(), EgoNodeMatch(:g), EgoNodeFactor(:g; level="A"),
                             EgoTriangle(), EgoGWESP(0.5), EgoDegree(2), EgoGWDegree(0.5),
                             EgoNodeCov(:x), EgoAbsDiff(:x)]
                @test ERGM.Extension.attainable_range(t, netm) ==
                      ERGM.Extension.attainable_range(ERGMEgo.ergm_term(t), netm)
            end
        end
        # a descriptive-only term has no ERGM counterpart and no range
        @test_throws ArgumentError ERGM.Extension.attainable_range(
            ERGMEgo._EgoDegreeAtLeast(3), network(5; directed=false))
        @test ERGMEgo._target_boundary(EgoTerm[EgoEdges(), EgoDegree(0), EgoDegree(1)],
                                       [10.0, 0.0, 30.0], 30) == [(2, :min), (3, :max)]
        @test isempty(ERGMEgo._target_boundary(EgoTerm[EgoNodeCov(:x)], [0.0], 30))

        # A 24-vertex ring of 4-cycles: no triangle, so no alter–alter tie is
        # reported and the triangle target is at its smallest value
        n = 24
        net = network(n; directed=false)
        for i in 1:n
            add_edge!(net, i, mod1(i + 1, n))
            iseven(i) && add_edge!(net, i, mod1(i + 5, n))
        end
        @test compute(Triangle(), net) == 0
        ed = simulate_ego_sample(net, n; rng=Random.Xoshiro(3))
        quick = (n_samples=300, se=:design)
        fit = @test_logs (:warn, r"triangle are at their smallest attainable values.*drop=false"s) match_mode=:any fit_ergm_ego(
            ed, [EgoEdges(), EgoTriangle()]; quick..., rng=Random.Xoshiro(1))
        @test coef(fit)[2] == -Inf && isfinite(coef(fit)[1]) && fit.converged
        @test stderror(fit)[2] == 0 && all(iszero, vcov(fit)[2, :]) && all(iszero, vcov(fit)[:, 2])
        @test dof(fit) == 1
        tbl = coeftable(fit)
        @test tbl["triangle"].p_value == 0 && tbl["triangle"].z_value == -Inf
        @test isfinite(tbl["edges"].p_value) && tbl["edges"].p_value > 0
        @test confint(fit)[2, :] == [-Inf, -Inf]
        flat = replace(sprint(show, fit), r"\s+" => " ")
        @test occursin("coefficient(s) triangle fixed at -Inf (target at its smallest attainable value)", flat)
        @test any(occursin("triangle fixed at -Inf", a) for a in approximations(fit))
        @test size(fit.sim_stats, 2) == 1 && length(fit.termination.step) == 1
        # Simulating the fitted model never makes a triangle
        g = gof(fit; n_sim=5, rng=Random.Xoshiro(4))
        @test g.statistics[1].labels == ["edges", "triangle"]
        @test all(iszero, g.statistics[1].simulated[:, 2])
        # edges alone on the same data: the same network with no triangle
        # term is a different model, so only the sign is pinned
        @test coef(fit)[1] < 0
        # The strict mode and the bootstrap refuse, before any MCMC
        @test_throws ArgumentError fit_ergm_ego(ed, [EgoEdges(), EgoTriangle()]; quick...,
                                                drop=false, rng=Random.Xoshiro(1))
        err = try
            fit_ergm_ego(ed, [EgoEdges(), EgoTriangle()]; n_samples=300, se=:bootstrap,
                         rng=Random.Xoshiro(1))
        catch e
            e
        end
        @test err isa ArgumentError && occursin("finite point estimates", err.msg)
        # nothing at a bound: no warning, nothing fixed
        logs, f0 = Test.collect_test_logs(() -> fit_ergm_ego(ed, [EgoEdges()]; quick...,
                                                             rng=Random.Xoshiro(1)))
        @test !any(occursin("attainable", string(l.message)) for l in logs)
        @test all(isfinite, coef(f0))
        # no ego reports an alter: refused
        empty = EgoData([EgoNetwork(i, Int[], zeros(Bool, 0, 0)) for i in 1:6])
        @test_throws ArgumentError fit_ergm_ego(empty, [EgoEdges()]; quick..., rng=Random.Xoshiro(1))

        # --- The TOP of the range: every vertex of the ring of 4-cycles has
        # a tie, so the gwdegree(0) target — the number of non-isolates — is
        # the network size, its largest attainable value. As ergm.ego does
        # (ego_mixing_esp.toml block (f)), the coefficient is fixed at +Inf
        # with ergm's warning, the sampler holds every vertex tied, and edges
        # is estimated given that
        @test 24 * compute(EgoGWDegree(0.0), ed) == 24
        @test ERGMEgo._target_boundary(EgoTerm[EgoEdges(), EgoGWDegree(0.0)],
                                       [30.0, 24.0], 24) == [(2, :max)]
        top = @test_logs (:warn, r"gwdeg\.fixed\.0 are at their largest attainable values \(coefficient \+Inf\)") match_mode=:any fit_ergm_ego(
            ed, [EgoEdges(), EgoGWDegree(0.0)]; quick..., rng=Random.Xoshiro(1))
        @test coef(top)[2] == Inf && isfinite(coef(top)[1]) && top.converged
        @test stderror(top)[2] == 0 && all(iszero, vcov(top)[2, :]) && dof(top) == 1
        ttop = coeftable(top)
        @test ttop["gwdeg.fixed.0"].z_value == Inf && ttop["gwdeg.fixed.0"].p_value == 0
        @test confint(top)[2, :] == [Inf, Inf]
        flat_top = replace(sprint(show, top), r"\s+" => " ")
        @test occursin("gwdeg.fixed.0 fixed at +Inf (target at its largest attainable value)", flat_top)
        @test any(occursin("gwdeg.fixed.0 fixed at +Inf", a) for a in approximations(top))
        @test size(top.sim_stats, 2) == 1
        # the simulated model never leaves a vertex isolated
        gt = gof(top; n_sim=5, rng=Random.Xoshiro(4))
        @test gt.statistics[1].observed[2] == 1.0      # per capita: every ego has a tie
        @test all(==(1.0), gt.statistics[1].simulated[:, 2])
        # strict mode refuses the top as it refuses the bottom
        @test_throws ArgumentError fit_ergm_ego(ed, [EgoEdges(), EgoGWDegree(0.0)]; quick...,
                                                drop=false, rng=Random.Xoshiro(1))
    end

    @testset "The default pseudo-population size is ergm.ego's" begin
        rng = Random.Xoshiro(8)
        net = network(30; directed=false)
        for i in 1:30, j in (i+1):30
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = EgoData(simulate_ego_sample(net, 30; rng=rng).egos)       # popsize unknown
        quick = (n_samples=200,)
        # unknown population size: the number of egos
        u = fit_ergm_ego(ed, [EgoEdges()]; quick..., rng=Random.Xoshiro(1))
        @test u.model.ppopsize == 30 && u.netsize_adjustment ≈ -log(30)
        @test ERGMEgo._default_ppopsize(1, 30) == 30
        # known and at most 1000: the population size
        k = fit_ergm_ego(ed, [EgoEdges()]; popsize=90, quick..., rng=Random.Xoshiro(1))
        @test k.model.ppopsize == 90 && ERGMEgo._default_ppopsize(1000, 30) == 1000
        @test ERGMEgo._ppopsize_note(k.model) === nothing
        # above 1000: ten times the egos, where ergm.ego uses the population
        # itself — disclosed in show and approximations
        @test ERGMEgo._default_ppopsize(1001, 30) == 300
        b = fit_ergm_ego(ed, [EgoEdges()]; popsize=5000, quick..., rng=Random.Xoshiro(1))
        @test b.model.ppopsize == 300
        flatb = replace(sprint(show, b), r"\s+" => " ")
        @test occursin("ergm.ego's default pseudo-population is the population itself", flatb)
        @test occursin("pass ppopsize=5000", flatb)
        @test any(occursin("pass ppopsize=5000", a) for a in approximations(b))
        # ergm.ego's two warnings about the requested size
        @test_logs (:warn, r"smaller pseudo-population size \(20\) than sample size") match_mode=:any fit_ergm_ego(
            ed, [EgoEdges()]; ppopsize=20, quick..., rng=Random.Xoshiro(1))
        wed = ego_design(ed; weights=[isodd(i) ? 1.0 : 3.0 for i in 1:30])
        @test_logs (:warn, r"equal to the sample size \(30\) under weighted sampling") match_mode=:any fit_ergm_ego(
            wed, [EgoEdges()]; quick..., rng=Random.Xoshiro(1))
        # unit weights at the default: no warning about the size
        logs, _ = Test.collect_test_logs(() -> fit_ergm_ego(ed, [EgoEdges()]; quick...,
                                                            rng=Random.Xoshiro(1)))
        @test !any(occursin("pseudo-population size", string(l.message)) for l in logs)
    end

    @testset "The pseudo-population does not depend on the order of the egos" begin
        net = load_dataset(:faux_mesa_high)
        census = fauxmesa_census(net)
        keep = sort([e for e in census.egos if e.ego in 1:3:205]; by=e -> e.ego_attrs[:Grade])
        grades(pp) = sort(vertex_attribute_vector(pp, :Grade, Int))
        # 69 unit-weight egos, ppopsize = 100. Largest-remainder rounding gave
        # the 31 extra copies to whichever egos came first (grade-7 share 0.36
        # sorted by grade, 0.18 reversed); ergm.ego's rounding replicates every
        # ego round(100/69) = 1 time. Realised sizes, by hand: 100/69 → 1,
        # 150/69 → 2, 207/69 = 3, 500/69 → 7 copies of each ego
        for (m, realised) in ((100, 69), (150, 138), (207, 207), (500, 483))
            a = EgoData(keep)
            b = EgoData(reverse(keep))
            ca = ERGMEgo._ppop_counts(a.sampling_weights, m)
            @test ca == reverse(ERGMEgo._ppop_counts(b.sampling_weights, m))
            pa = ERGMEgo._pseudo_population(a, m, 0.02, Random.Xoshiro(1))
            pb = ERGMEgo._pseudo_population(b, m, 0.02, Random.Xoshiro(1))
            @test grades(pa) == grades(pb)
            @test nv(pa) == sum(ca) == realised
        end
        # Unequal weights, random permutations: the count of an ego depends
        # on its own weight only
        rng = Random.Xoshiro(4)
        w = [Float64(e.ego_attrs[:Grade] - 6) + rand(rng) for e in keep]
        base = ERGMEgo._ppop_counts(w, 300)
        # R's round() on m·w/Σw, by hand on small cases: [0.7, 1.4, 2.1, 2.8]
        # → [1, 1, 2, 3] (realised 7); 4/3 each → 1 each (realised 3, where
        # largest remainder would give [2, 1, 1]); halves go to the even
        # neighbour, as R's round does: [0.5, 0.5, 1] → [0, 0, 1] and
        # [1.5, 1.5, 3] → [2, 2, 3]
        @test ERGMEgo._ppop_counts([1.0, 2.0, 3.0, 4.0], 7) == [1, 1, 2, 3]
        @test ERGMEgo._ppop_counts([1.0, 1.0, 1.0], 4) == [1, 1, 1]
        @test ERGMEgo._ppop_counts([1.0, 1.0, 2.0], 2) == [0, 0, 1]
        @test ERGMEgo._ppop_counts([1.0, 1.0, 2.0], 6) == [2, 2, 3]
        for _ in 1:20
            perm = Random.randperm(rng, length(keep))
            ed = EgoData(keep[perm]; sampling_weights=w[perm])
            @test ERGMEgo._ppop_counts(ed.sampling_weights, 300) == base[perm]
            @test grades(ERGMEgo._pseudo_population(ed, 300, 0.02, rng)) ==
                  grades(ERGMEgo._pseudo_population(EgoData(keep; sampling_weights=w), 300, 0.02, rng))
        end
        @test ERGMEgo._ppop_ego_index([2, 0, 1]) == [1, 1, 3]
        # ...and so do the fits: the two orders used to differ by 0.46 in
        # nodematch (2.54 vs 3.00, 1.5 standard errors)
        fit(ed) = @test_logs (:info, r"69 vertices, not the requested 100") match_mode=:any fit_ergm_ego(
            ed, [EgoEdges(), EgoNodeMatch(:Grade)]; ppopsize=100, popsize=205, rng=Random.Xoshiro(1))
        fa, fb = fit(EgoData(keep)), fit(EgoData(reverse(keep)))
        @test fa.model.ppopsize == fb.model.ppopsize == 69
        @test fa.model.targets ≈ fb.model.targets
        @test fa.converged && fb.converged
        @test maximum(abs.(coef(fa) .- coef(fb))) < 0.05
        @test fa.netsize_adjustment ≈ -log(69 / 205)
        # A rounding that leaves fewer than 5 vertices is refused in words
        @test_throws ArgumentError fit_ergm_ego(EgoData(keep), [EgoEdges()]; ppopsize=30)
    end

    @testset "Standard errors: ergm.ego's under-cover, se=:superpopulation corrects (simulation)" begin
        # 100 censuses of a 40-vertex edges + nodematch population. The design
        # variance treats the egos as independent, but every tie is reported
        # by both of its ends: at a census the reported standard error is
        # 1/√2 of the sampling sd of the estimate, and nominal 95 % intervals
        # cover the model parameter about 81 % of the time. Multiplying the
        # design component by 1 + n/N restores it (measured 0.92 / 0.92).
        θ = [-2.5, 1.0]
        N, R = 40, 100
        est = zeros(R, 2); se_d = zeros(R, 2); se_s = zeros(R, 2)
        for r in 1:R
            rng = Random.Xoshiro(r)
            net = network(N; directed=false)
            grp = [isodd(v) ? "a" : "b" for v in 1:N]
            set_vertex_attribute!(net, :g, Dict(v => grp[v] for v in 1:N))
            for i in 1:N, j in (i+1):N
                rand(rng) < 1 / (1 + exp(-(θ[1] + θ[2] * (grp[i] == grp[j])))) && add_edge!(net, i, j)
            end
            ed = simulate_ego_sample(net, N; ego_attrs=[:g], rng=rng)
            fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:g)]; n_samples=400, rng=rng)
            @assert fit.se_type === :superpopulation      # the default: popsize is known
            est[r, :] = coef(fit)
            se_s[r, :] = stderror(fit)
            se_d[r, :] = sqrt.(diag(fit.vcov_design ./ 2 .+ fit.vcov_estimation))   # ergm.ego's
        end
        cover(se) = [mean(abs.(est[:, k] .- θ[k]) .<= 1.96 .* se[:, k]) for k in 1:2]
        sd_est = vec(std(est; dims=1))
        @test all(cover(se_d) .< 0.88)                       # measured 0.81, 0.81
        @test all(cover(se_s) .>= 0.88)                      # measured 0.92, 0.92
        @test all(0.55 .< vec(mean(se_d; dims=1)) ./ sd_est .< 0.82)   # ≈ 1/√2
        @test all(0.8 .< vec(mean(se_s; dims=1)) ./ sd_est .< 1.2)

        # A sample, not a census: 50 of 200 actors (f = 0.25), 100 replicates.
        # The default still covers (measured 0.93 / 0.91 over 300 replicates;
        # ergm.ego's 0.91 / 0.87)
        N, n, R = 200, 50, 100
        θ2 = [-3.2, 1.0]
        hits = zeros(Int, 2)
        for r in 1:R
            rng = Random.Xoshiro(r)
            net = network(N; directed=false)
            grp = [isodd(v) ? "a" : "b" for v in 1:N]
            set_vertex_attribute!(net, :g, Dict(v => grp[v] for v in 1:N))
            for i in 1:N, j in (i+1):N
                rand(rng) < 1 / (1 + exp(-(θ2[1] + θ2[2] * (grp[i] == grp[j])))) && add_edge!(net, i, j)
            end
            ed = simulate_ego_sample(net, n; ego_attrs=[:g], rng=rng)
            fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:g)]; n_samples=400, rng=rng)
            for k in 1:2
                hits[k] += abs(coef(fit)[k] - θ2[k]) <= 1.96 * stderror(fit)[k]
            end
        end
        @test all(hits ./ R .>= 0.87)

        # The two methods differ by exactly that inflation of the design
        # component; the default is :superpopulation when the population size
        # is known and ergm.ego's :design when it is not
        rng = Random.Xoshiro(7)
        net = load_dataset(:faux_mesa_high)
        ed = simulate_ego_sample(net, 82; ego_attrs=[:Grade], rng=rng)
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]
        d = fit_ergm_ego(ed, terms; ppopsize=205, se=:design, rng=Random.Xoshiro(1))
        s = fit_ergm_ego(ed, terms; ppopsize=205, rng=Random.Xoshiro(1))
        @test d.se_type === :design && s.se_type === :superpopulation
        @test fit_ergm_ego(ed, terms; ppopsize=205, se=:superpopulation,
                           rng=Random.Xoshiro(1)).vcov == s.vcov
        u = fit_ergm_ego(EgoData(ed.egos), terms; ppopsize=164, rng=Random.Xoshiro(1))
        @test u.se_type === :design && u.model.popsize == 1
        @test occursin("the default when the population size is unknown",
                       replace(sprint(show, u), r"\s+" => " "))
        @test coef(d) == coef(s)
        @test s.vcov_design ≈ (1 + 82 / 205) .* d.vcov_design
        @test s.vcov_estimation == d.vcov_estimation
        @test se_method(s) === :sandwich
        # Both say what they are, in show and in the metadata
        flat(x) = replace(sprint(show, x), r"\s+" => " ")       # the note is wrapped
        @test occursin("cover about 91 % at a 10 % sampling fraction", flat(d))
        @test occursin("ergm.ego's standard errors (se=:design)", flat(d))
        @test occursin("(se=:superpopulation)", flat(s)) && occursin("= 1.4 ", flat(s))
        @test occursin("attribute composition is itself estimated", flat(s))
        @test all(length(l) <= 100 for l in split(sprint(show, d), '\n')[end-8:end])
        @test any(occursin("reported by both", a) for a in approximations(d))
        @test any(occursin("1 + n_egos/popsize", a) for a in approximations(s))
        # It needs the population size
        err = try fit_ergm_ego(EgoData(ed.egos), terms; ppopsize=205, se=:superpopulation); nothing catch e; e end
        @test err isa ArgumentError && occursin("needs the population size", err.msg)
        err = try fit_ergm_ego(ed, terms; se=:hessian); nothing catch e; e end
        @test err isa ArgumentError && occursin("(:design, :superpopulation, :bootstrap)", err.msg)
        @test Base.kwarg_decl(first(methods(fit_ergm_ego))) ⊇ [:se]
    end

    @testset "se=:bootstrap: a bootstrap over egos through NetworkCore.bootstrap_cov" begin
        ego_df = DataFrame(ego_id=1:6, group=["A", "B", "A", "B", "A", "B"],
                           w=[1.0, 2.0, 1.0, 2.0, 1.0, 3.0])
        alter_df = DataFrame(ego_id=[1, 1, 2, 3, 3, 4, 5, 6, 6],
                             alter_id=[2, 3, 1, 1, 6, 9, 9, 3, 8],
                             group=["B", "A", "A", "A", "B", "A", "A", "A", "B"])
        ed = as_egodata(ego_df, alter_df; ego_attrs=[:group], alter_attrs=[:group], weight_col=:w)
        terms = [EgoEdges(), EgoNodeMatch(:group)]
        boot(d=ed; kw...) = fit_ergm_ego(d, terms; ppopsize=40, se=:bootstrap, n_boot=30, kw...)
        finite_rows(B) = [b for b in axes(B, 1) if all(isfinite, view(B, b, :))]

        # Reproducible from rng alone; the global RNG is neither read nor moved
        Random.seed!(1); a = boot(; rng=Random.Xoshiro(1))
        Random.seed!(2); b = boot(; rng=Random.Xoshiro(1))
        @test isequal(a.boot_replicates, b.boot_replicates) && isequal(vcov(a), vcov(b))
        Random.seed!(5); x = rand()
        Random.seed!(5); boot(; rng=Random.Xoshiro(3))
        @test rand() == x
        @test a.se_type === :bootstrap && se_method(a) === :bootstrap
        @test NetworkCore.fit_metadata(a).se_method === :bootstrap
        @test size(a.boot_replicates) == (30, 2)
        # The point estimate is the fit's: only the covariance is replaced
        d = fit_ergm_ego(ed, terms; ppopsize=40, se=:design, rng=Random.Xoshiro(1))
        @test coef(a) == coef(d)
        @test size(d.boot_replicates) == (0, 2) && se_method(d) === :sandwich
        # Population size unknown: the covariance IS the covariance of the
        # refits (the one shared loop's replicates), nothing added
        ok = finite_rows(a.boot_replicates)
        @test length(ok) >= 25
        # ...with a robust scale: the replicates' correlations, and their
        # normalised interquartile ranges as standard deviations
        Bok = a.boot_replicates[ok, :]
        @test vcov(a) ≈ ERGMEgo._robust_cov(Bok)
        iqr(x) = (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
        @test stderror(a) ≈ [iqr(Bok[:, 1]), iqr(Bok[:, 2])]
        @test vcov(a)[1, 2] / prod(stderror(a)) ≈ cor(Bok[:, 1], Bok[:, 2])
        z = randn(Random.Xoshiro(1), 20_000, 2)
        @test ERGMEgo._robust_cov(z) ≈ cov(z) rtol = 0.05     # the sd, for normal draws
        @test ERGMEgo._robust_cov(hcat(z[:, 1], zeros(20_000)))[2, 2] == 0.0
        @test vcov(a) == a.vcov_design .+ a.vcov_estimation
        @test stderror(a) ≈ sqrt.(diag(vcov(a)))
        @test NetworkCore.check_statsapi(a; required=(:coef, :stderror, :vcov, :confint, :nobs,
                                                   :dof, :coeftable, :coefnames),
                                         strict=true) !== nothing
        # Population size known: n/N times the design sandwich is added for
        # the ties two sampled egos both report
        edN = ego_design(ed; popsize=60)
        k = boot(edN; rng=Random.Xoshiro(1))
        dN = fit_ergm_ego(edN, terms; ppopsize=40, se=:design, rng=Random.Xoshiro(1))
        okk = finite_rows(k.boot_replicates)
        @test vcov(k) ≈ ERGMEgo._robust_cov(k.boot_replicates[okk, :]) .+ (6 / 60) .* dN.vcov_design
        flat(x) = replace(sprint(show, x), r"\s+" => " ")
        @test occursin("bootstrap over egos (se=:bootstrap)", flat(k))
        @test occursin("n_egos/popsize = 0.1 times the design sandwich is added", flat(k))
        @test occursin("the population size is unknown, so nothing is added", flat(a))
        @test any(occursin("bootstrap over egos", x) for x in approximations(a))

        # Each replicate is the refit of the egos drawn with replacement, each
        # with ITS sampling weight, from a rebuilt pseudo-population and
        # targets: replicate 1 by hand, from the draws `simulate` makes first
        controls = ERGMEgo._FitControls(40, nothing, nothing, nothing, nothing, 60, :confidence,
                                        0.1, 0.99, nothing, 0.1, 0.05, 0.1, 5.0, :spdyad,
                                        64.0, :bootstrap, 4, true)
        raw = ERGMEgo._ego_bootstrap(ed, collect(EgoTerm, terms), controls, 40, nothing, 4,
                                     Random.Xoshiro(9))
        r = Random.Xoshiro(9)
        idx = rand(r, 1:6, 6); seed = rand(r, UInt64)
        resampled = EgoData(ed.egos[idx]; sampling_weights=ed.sampling_weights[idx],
                            design=ed.design)
        @test resampled.sampling_weights == ed.sampling_weights[idx]
        by_hand = fit_ergm_ego(resampled, terms; ppopsize=40, se=:design,
                               rng=Random.Xoshiro(seed))
        @test by_hand.converged
        @test raw.replicates[1, :] == coef(by_hand)
        @test by_hand.model.targets != d.model.targets       # targets rebuilt

        # A refit that does not converge is a NaN row: excluded, warned about
        # once, listed. (A fixed-size chain of 40 tie/no-tie draws one toggle
        # apart cannot converge.)
        short = (n_samples=40, burnin=10, interval=1, maxiter=1, effective_size=nothing,
                 proposal=:tnt)
        logs, u = Test.collect_test_logs(() -> boot(; short..., rng=Random.Xoshiro(2)))
        n_bad = 30 - length(finite_rows(u.boot_replicates))
        @test n_bad > 0
        @test count(l -> occursin("bootstrap refits did not converge", string(l.message)), logs) == 1
        @test any(occursin("$n_bad refits did not converge and are excluded", x)
                  for x in approximations(u))
        # ...and every disclosure says what the exclusion does, in the
        # sentence the whole ERGM family uses
        bias = "The standard errors are conditional on a finite refit: the excluded " *
               "replicates are the extreme ones, so the standard errors are biased downward."
        @test ERGMEgo._BOOT_EXCLUSION_BIAS == bias == ERGM._BOOT_EXCLUSION_BIAS
        @test count(l -> occursin(bias, string(l.message)), logs) == 1
        @test any(occursin(bias, x) for x in approximations(u))
        @test occursin(bias, flat(u))
        # a bootstrap with no exclusion does not claim one
        n_bad_a = 30 - length(finite_rows(a.boot_replicates))
        @test occursin(bias, flat(a)) == (n_bad_a > 0)
        oku = finite_rows(u.boot_replicates)
        @test length(oku) < 2 ? all(isnan, vcov(u)) :
                                vcov(u) ≈ ERGMEgo._robust_cov(u.boot_replicates[oku, :])

        @test_throws ArgumentError boot(; n_boot=1)
        @test :n_boot in Base.kwarg_decl(first(methods(fit_ergm_ego)))

        # Thread-count independent: the same fit in a fresh process with a
        # different thread count (every resample and refit seed is drawn from
        # rng before the threaded refits run)
        other_threads = Threads.nthreads() == 1 ? 4 : 1
        script = """
            using ERGMEgo, DataFrames, Random
            ego_df = DataFrame(ego_id=1:6, group=["A", "B", "A", "B", "A", "B"],
                               w=[1.0, 2.0, 1.0, 2.0, 1.0, 3.0])
            alter_df = DataFrame(ego_id=[1, 1, 2, 3, 3, 4, 5, 6, 6],
                                 alter_id=[2, 3, 1, 1, 6, 9, 9, 3, 8],
                                 group=["B", "A", "A", "A", "B", "A", "A", "A", "B"])
            ed = as_egodata(ego_df, alter_df; ego_attrs=[:group], alter_attrs=[:group], weight_col=:w)
            fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:group)]; ppopsize=40,
                               se=:bootstrap, n_boot=30, rng=Xoshiro(1))
            println(Threads.nthreads())
            println(repr(fit.boot_replicates))
            println(repr(vcov(fit)))
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$other_threads --project=$(dirname(@__DIR__)) -e $script`
        errbuf = IOBuffer()
        lines = split(strip(read(pipeline(ignorestatus(cmd); stderr=errbuf), String)), '\n')
        # the subprocess's own error output is shown when it fails
        length(lines) == 3 || println(stderr, "fresh-process bootstrap failed:\n",
                                      String(take!(errbuf)))
        @test length(lines) == 3
        @test lines[1] == string(other_threads)
        @test lines[2] == repr(a.boot_replicates)
        @test lines[3] == repr(vcov(a))

        # COVERAGE at the hard case — few egos with many ties each: 10 of 40
        # actors, mean degree ≈ 12, 30 replicates, 24 bootstrap refits each
        # (conv_precision=0.3 keeps a fit to one sample). The sandwich misses
        # the sampling error of the pseudo-population's composition and
        # under-covers (measured 0.73 / 0.67 here; 0.84 / 0.77 for 20 of 200
        # egos over 200 replicates); the bootstrap covers (0.93 / 0.93 here;
        # 20 of 200: see the fit_ergm_ego docstring).
        θ = [-1.5, 1.0]
        N, n, R = 40, 10, 30
        est = zeros(R, 2); se_b = zeros(R, 2); se_s = zeros(R, 2)
        for r in 1:R
            rng = Random.Xoshiro(r)
            pop = network(N; directed=false)
            grp = [isodd(v) ? "a" : "b" for v in 1:N]
            set_vertex_attribute!(pop, :g, Dict(v => grp[v] for v in 1:N))
            for i in 1:N, j in (i+1):N
                rand(rng) < 1 / (1 + exp(-(θ[1] + θ[2] * (grp[i] == grp[j])))) && add_edge!(pop, i, j)
            end
            smp = simulate_ego_sample(pop, n; ego_attrs=[:g], rng=rng)
            kw = (n_samples=800, conv_precision=0.3)
            Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                fb = fit_ergm_ego(smp, [EgoEdges(), EgoNodeMatch(:g)]; kw..., se=:bootstrap,
                                  n_boot=24, rng=rng)
                fs = fit_ergm_ego(smp, [EgoEdges(), EgoNodeMatch(:g)]; kw..., rng=Random.Xoshiro(r))
                est[r, :] = coef(fb); se_b[r, :] = stderror(fb); se_s[r, :] = stderror(fs)
            end
        end
        cover(se) = [mean(abs.(est[:, k] .- θ[k]) .<= 1.96 .* se[:, k]) for k in 1:2]
        @test all(cover(se_b) .>= 0.87)
        @test all(cover(se_s) .< 0.85)
        @test all(cover(se_b) .> cover(se_s))

        # With few egos the default's note points at the bootstrap
        net = load_dataset(:faux_mesa_high)
        few = simulate_ego_sample(net, 30; ego_attrs=[:Grade], rng=Random.Xoshiro(1))
        f30 = fit_ergm_ego(few, [EgoEdges()]; ppopsize=60, n_samples=300, rng=Random.Xoshiro(1))
        @test f30.se_type === :superpopulation
        @test occursin("With 30 egos, prefer `se=:bootstrap`", flat(f30))
    end

    @testset "estimate_popsize(:capture_recapture) is unbiased for the population size (simulation)" begin
        # Sampled egos are the first capture (equal probability, isolates
        # included), nominations the second: N̂ = 1 + (n−1)(R+2)/(M+2). The
        # pre-0.2 estimator (alter sets of two halves) averaged 153 on
        # faux.mesa.high (N = 205; 148 non-isolates).
        net = load_dataset(:faux_mesa_high)
        est = [estimate_popsize(simulate_ego_sample(net, 80; rng=Random.Xoshiro(r));
                                method=:capture_recapture) for r in 1:300]
        @test abs(mean(est) - 205) < 0.04 * 205              # measured 206.8
        @test 185 < median(est) < 225
        # The formula, by hand: egos 1..4; nominations 2→{1,3}, 1→{2,9}, 3→{2}, 4→{}
        mk(id, alters) = EgoNetwork(id, alters, zeros(Bool, length(alters), length(alters)))
        ed = EgoData([mk(1, [2, 9]), mk(2, [1, 3]), mk(3, [2]), mk(4, Int[])])
        @test estimate_popsize(ed; method=:capture_recapture) == 1 + 3 * (5 + 2) / (4 + 2)
        # It does not depend on the order of the egos (the old split did)
        @test estimate_popsize(EgoData(reverse(ed.egos)); method=:capture_recapture) ==
              estimate_popsize(ed; method=:capture_recapture)
        # Refusals: unequal weights; nothing recaptured
        err = try estimate_popsize(ego_design(ed; weights=[1.0, 2.0, 1.0, 1.0]); method=:capture_recapture); nothing catch e; e end
        @test err isa ArgumentError && occursin("equal-probability", err.msg)
        err = try estimate_popsize(EgoData([mk(1, [7]), mk(2, [8])]); method=:capture_recapture); nothing catch e; e end
        @test err isa ArgumentError && occursin("no nominated alter is a sampled ego", err.msg)
    end

    @testset "gof: every model statistic, shared partners, and the observed design" begin
        net = load_dataset(:faux_mesa_high)
        census = fauxmesa_census(net)
        # The shared-partner statistic on a census is the network's own ESP
        # distribution: m · compute(EgoESP(k)) ties have k shared partners
        A = as_matrix(net)
        esp = zeros(Int, 30)
        for i in 1:205, j in (i+1):205
            A[i, j] != 0 && (esp[round(Int, sum(A[i, :] .* A[j, :])) + 1] += 1)
        end
        @test [205 * compute(EgoESP(k), census) for k in 0:8] ≈ esp[1:9]
        @test sum(esp[2:end]) == 120                       # the transitive ties

        fx = load_golden(joinpath(@__DIR__, "fixtures", "ego_netsize.toml")).values
        wed = fauxmesa_weighted(Int.(fx["ego_ids"]), Float64.(fx["weights"]); net)
        @test wed.sampling_weights == [Float64(e.ego_attrs[:Grade] - 6) for e in wed.egos]
        terms = [EgoEdges(), EgoNodeMatch(:Grade)]
        fit = fit_ergm_ego(wed, terms; rng=Random.Xoshiro(1))
        G = gof(fit; n_sim=12, rng=Random.Xoshiro(2))
        @test [s.name for s in G.statistics] == ["model statistics", "degree distribution",
                                                 "edgewise shared partners", "ego summary statistics"]
        # gof.ergm.ego's GOF="model": every model statistic, per capita
        model_stat = G.statistics[1]
        @test model_stat.labels == ["edges", "nodematch.Grade"]
        @test model_stat.observed ≈ [compute(t, wed) for t in terms]
        @test model_stat.observed ≈ fit.model.targets ./ fit.model.ppopsize
        # a converged fit reproduces its own targets
        @test all(model_stat.p_values .> 0.05)
        # GOF="espartners": esp 0 … 2·(max(K, 3) − 1), observed from the alter ties
        K = Int(summary_stats(wed).max_degree)
        espstat = G.statistics[3]
        @test espstat.labels == ["esp $k" for k in 0:(2 * (max(K, 3) - 1))]
        @test espstat.observed == [compute(EgoESP(k), wed) for k in 0:(2 * (max(K, 3) - 1))]
        @test sum(espstat.observed) ≈ compute(EgoEdges(), wed)     # every tie has some k
        # an edges + nodematch model has no clustering: it misses the ties
        # with one shared partner
        @test espstat.p_values[2] < 0.2 && espstat.observed[2] > maximum(espstat.simulated[:, 2])
        @test G.statistics[4].labels == ["mean degree", "mean alter ties"]

        # The simulated ego samples have the observed design: one replicate
        # per observed ego, with that ego's weight
        counts = fit.model.ppop_counts
        members = [Int[] for _ in 1:length(wed)]
        for (vtx, i) in enumerate(ERGMEgo._ppop_ego_index(counts))
            push!(members[i], vtx)
        end
        present = findall(!isempty, members)
        pp = ERGMEgo._pseudo_population(wed, counts, 0.01, Random.Xoshiro(3))
        columns = Dict{Symbol, Vector}(:Grade => vertex_attribute_vector(pp, :Grade, Int))
        smp = ERGMEgo._design_sample(Random.Xoshiro(4), pp, members, present,
                                     wed.sampling_weights[present], columns)
        @test length(smp) == length(present) == 69
        @test smp.sampling_weights == wed.sampling_weights
        @test [e.ego_attrs[:Grade] for e in smp.egos] == [e.ego_attrs[:Grade] for e in wed.egos]
        @test all(e.ego in members[i] for (e, i) in zip(smp.egos, present))
        # ...so the simulated column is the same weighted estimator as the
        # observed one: its composition by grade is the observed sample's,
        # not the pseudo-population's
        @test count(e -> e.ego_attrs[:Grade] == 7, smp.egos) / 69 ≈
              count(e -> e.ego_attrs[:Grade] == 7, wed.egos) / 69
        @test count(==(7), columns[:Grade]) / nv(pp) < 0.6 * count(e -> e.ego_attrs[:Grade] == 7, wed.egos) / 69
    end

    @testset "Alter–alter ties that were not collected are unobserved, not absent" begin
        ego_df = DataFrame(ego_id=1:6, group=["A", "B", "A", "B", "A", "B"])
        alter_df = DataFrame(ego_id=[1, 1, 2, 3, 3, 4, 5, 6, 6],
                             alter_id=[2, 3, 1, 1, 6, 9, 9, 3, 8],
                             group=["B", "A", "A", "A", "B", "A", "A", "A", "B"])
        ed = as_egodata(ego_df, alter_df; ego_attrs=[:group], alter_attrs=[:group])
        @test ed.design[:alter_ties_observed] === false
        @test isnan(summary_stats(ed).mean_alter_ties)
        @test compute(EgoEdges(), ed) == 0.75
        err = try compute(EgoTriangle(), ed); nothing catch e; e end
        @test err isa ArgumentError && occursin("not collected", err.msg)
        @test_throws ArgumentError ego_target_stats([EgoEdges(), EgoTriangle()], ed, 20)
        @test_throws ArgumentError fit_ergm_ego(ed, [EgoEdges(), EgoTriangle()]; ppopsize=24)
        # ego_design keeps the flag; an empty aatie_df says "collected, none"
        @test ego_design(ed; popsize=50).design[:alter_ties_observed] === false
        none = as_egodata(ego_df, alter_df; aatie_df=DataFrame(ego_id=Int[], src=Int[], dst=Int[]))
        @test !haskey(none.design, :alter_ties_observed)
        @test compute(EgoTriangle(), none) == 0.0
        # gof drops the panels that need the ties instead of showing zeros
        fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=24, rng=Random.Xoshiro(1))
        G = gof(fit; n_sim=5, rng=Random.Xoshiro(2))
        @test [s.name for s in G.statistics] == ["model statistics", "degree distribution",
                                                 "ego summary statistics"]
        @test G.statistics[end].labels == ["mean degree"]
        @test isnan(ego_gof(fit; n_sim=5, rng=Random.Xoshiro(2)).observed.mean_alter_ties)
    end

    @testset "EgoNetwork input checks and ego_design(popsize=)" begin
        # An ego listed as its own alter, and an alter attribute of the wrong length
        err = try EgoNetwork(5, [5, 11], zeros(Bool, 2, 2)); nothing catch e; e end
        @test err isa ArgumentError && occursin("ego 5 is listed as its own alter", err.msg)
        err = try EgoNetwork(1, [10, 11], zeros(Bool, 2, 2);
                             alter_attrs=Dict{Symbol,Vector}(:g => ["a"])); nothing catch e; e end
        @test err isa ArgumentError && occursin("1 values for 2 alters", err.msg)
        @test_throws ArgumentError as_egodata(DataFrame(ego_id=[1]), DataFrame(ego_id=[1], alter_id=[1]))

        # `popsize` is the population size; the development-era spelling
        # `ppopsize` (which set it too) is gone, so it cannot be confused with
        # `fit_ergm_ego`'s pseudo-population size
        ed = fixture_egodata()
        @test ego_design(ed; popsize=500).population_size == 500
        @test_throws MethodError ego_design(ed; ppopsize=500)
        @test !(:ppopsize in Base.kwarg_decl(first(methods(ego_design))))
        @test_throws ArgumentError ego_design(ed; popsize=0)
        @test_throws ArgumentError fit_ergm_ego(ed, [EgoEdges()]; popsize=0, ppopsize=20)
        @test :popsize in Base.kwarg_decl(first(methods(ego_design)))
    end

    @testset "README \"Not implemented\" and CHANGELOG \"Known limitations\" list the same items" begin
        pkgdir = dirname(@__DIR__)
        section(text, header, stop) = (a = findfirst(header, text); b = findnext(stop, text, last(a));
                                       text[last(a)+1:first(b)-1])
        leads(sec) = [replace(m.captures[1], r"\s+" => " ")
                      for m in eachmatch(r"^- \*\*(.+?)\*\*"ms, sec)]
        readme = section(_readtext(joinpath(pkgdir, "README.md")), "## Not implemented", "\n## ")
        changelog = section(_readtext(joinpath(pkgdir, "CHANGELOG.md")),
                            "### Known limitations", "\n## ")
        @test length(leads(readme)) == 9
        @test leads(readme) == leads(changelog)
        # the README's examples install what they load, and no dead links
        full = _readtext(joinpath(pkgdir, "README.md"))
        @test occursin("Pkg.add(\"DataFrames\")", full)
        @test !occursin("issues/1", full) && !occursin("docs-stable", full)
        @test occursin("statistical-network-analysis-with-julia.github.io/citing/", full)
    end

    @testset "Workflows reconstruct the ecosystem layout from [sources]" begin
        pkgdir = dirname(@__DIR__)
        PKG = "ERGMEgo"
        EXPECTED = Set(["ERGMEgo.jl", "ERGM.jl", "NetworkCore.jl"])
        siblings = sort!(collect(setdiff(EXPECTED, ["$PKG.jl"])))
        # the skip rule, on a scratch directory: none present skips, some or
        # all present runs (so a partial layout fails rather than skips)
        mktempdir() do dir
            @test isempty(present_siblings(dir, siblings))
            mkdir(joinpath(dir, "ERGM.jl")); touch(joinpath(dir, "ERGM.jl", "Project.toml"))
            mkdir(joinpath(dir, "NetworkCore.jl"))           # a directory without a project
            @test present_siblings(dir, siblings) == ["ERGM.jl"]
        end
        in_layout = !isempty(present_siblings(dirname(pkgdir), siblings))
        in_layout || @info "The workflows' layout step was not run: none of the sibling " *
                           "checkouts $(join(siblings, ", ")) is beside $pkgdir (a lone " *
                           "checkout or a registry install, where [sources] is not used)."
        for wf in ("CI.yml", "Documentation.yml")
            yml = _readtext(joinpath(pkgdir, ".github", "workflows", wf))
            @test !occursin(r"for pkg in", yml)              # no hand-kept clone list
            @test !occursin("checkout_sources.jl", yml)
            @test occursin("path: $PKG.jl\n", yml)
            step = match(r"\n      - name: Reconstruct the ecosystem layout from \[sources\]\n        shell: julia[^\n]*\n        run: \|\n((?:          [^\n]*\n)+)", yml)
            @test step !== nothing
            step === nothing && continue
            @test first(findfirst("setup-julia", yml)) < step.offset
            if !in_layout
                @test_skip :layout_step_needs_the_sibling_checkouts
                continue
            end
            # Run the workflow's own step without cloning: in the layout this
            # suite runs in, it must find exactly the siblings [sources] names.
            script = replace(step.captures[1], r"^          "m => "")
            out = mktemp() do path, io
                write(io, script); close(io)
                withenv("GITHUB_WORKSPACE" => dirname(pkgdir),
                        "GITHUB_REPOSITORY" => "statistical-network-analysis-with-Julia/$PKG.jl",
                        "LAYOUT_CHECK_ONLY" => "true", "GITHUB_STEP_SUMMARY" => nothing,
                        # the step runs in a plain Julia in CI, not in the
                        # test sandbox's load path (which has no TOML)
                        "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing) do
                    read(`$(Base.julia_cmd()) --startup-file=no $path`, String)
                end
            end
            @test Set(m.captures[1] for m in eachmatch(r"^\| (\S+\.jl) \|"m, out)) == EXPECTED
        end
        ci = _readtext(joinpath(pkgdir, ".github", "workflows", "CI.yml"))
        @test occursin("JULIA_NUM_THREADS", ci)             # the thread-independence cell
        @test occursin("benchmark/regression_tests.jl", ci) # the allocation gates
    end

    @testset "Aqua.jl quality assurance" begin
        # Ambiguities are checked on the package's own methods (the
        # dependencies' are theirs to answer for)
        Aqua.test_all(ERGMEgo; ambiguities=false)
        @test isempty(Test.detect_ambiguities(ERGMEgo))
    end

    @testset "Hot paths are allocation-free" begin
        # The per-ego contribution — the innermost loop of every target
        # statistic and of the design covariance — is allocation-free for the
        # fittable terms: EgoNodeMatch and EgoNodeFactor count through a
        # function barrier over the abstractly typed attribute column, and the
        # numeric terms (EgoNodeCov, EgoAbsDiff) branch on the Int/Float64
        # column layouts so their kernel is a static call. `_design_cov` allocates its H matrix, the normalised weights, h̄ and
        # Σ and nothing else per ego. The same pins live in
        # benchmark/regression_tests.jl (the standalone runner the site's
        # tools/run_benchmarks.jl consumes); here so that `Pkg.test()` alone
        # guards them.
        n = 500
        rng = Random.Xoshiro(1)
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 6 / n && add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :g, Dict(v => ("A", "B", "C")[mod1(v, 3)] for v in 1:n))
        set_vertex_attribute!(net, :x, Dict(v => 0.5 * mod(v, 7) for v in 1:n))
        ed = simulate_ego_sample(net, n; ego_attrs=[:g, :x], rng=Random.Xoshiro(2))
        terms = [EgoEdges(), EgoNodeMatch(:g), EgoTriangle(), EgoGWDegree(0.5),
                 EgoNodeFactor(:g; level="B"), EgoDegree(3), EgoGWESP(0.5), EgoESP(1),
                 EgoMM(:g, "A", "B"), EgoMM(:g, "C", "C"), EgoConcurrent()]
        numeric = [EgoNodeCov(:x), EgoAbsDiff(:x), EgoAbsDiff(:x; pow=2)]
        function worst_alloc(term, egos)
            worst = 0
            for e in egos
                ERGMEgo.ego_contribution(term, e)
                worst = max(worst, @allocated ERGMEgo.ego_contribution(term, e))
            end
            return worst
        end
        for term in terms
            @test worst_alloc(term, ed.egos[1:100]) == 0
        end
        for term in numeric
            @test worst_alloc(term, ed.egos[1:100]) == 0
        end
        function design_cov_bytes(ts)
            ERGMEgo._design_cov(ts, ed, n)
            return @allocated ERGMEgo._design_cov(ts, ed, n)
        end
        for ts in (terms, [EgoEdges()], [EgoEdges(), EgoNodeMatch(:g)], [terms; numeric])
            @test design_cov_bytes(ts) <= 4 * n * length(ts) * 8 + 4096
        end
        # ...and the lean loop is the same number as the textbook formula
        H = hcat([[ERGMEgo.ego_contribution(t, e) for e in ed.egos] for t in terms]...)
        @test ERGMEgo._design_cov(terms, ed, n) ≈ n^2 .* Statistics.cov(H) ./ n
    end

    @testset "Co-loading leaves every shared verb defined (fresh process)" begin
        # `using ERGM, ERGMEgo` — the statnet workflow — must leave the shared
        # verbs defined (two packages each exporting their own `compute` would
        # leave it *undefined*); they are NetworkCore.jl's generics in both.
        pkgdir = dirname(@__DIR__)
        script = """
            using ERGM, ERGMEgo
            import NetworkCore   # the module name is deliberately not re-exported
            for s in (:compute, :name, :gof, :coef, :stderror, :vcov, :coeftable,
                      :summary_stats, :Network)
                @assert isdefined(Main, s) string(s, " undefined after `using ERGM, ERGMEgo`")
            end
            @assert compute === NetworkCore.compute
            @assert gof === NetworkCore.gof
            @assert summary_stats === ERGM.summary_stats
            @assert Network(5) isa Network
            println("COLOAD_OK")
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$pkgdir -e $script`
        @test strip(read(cmd, String)) == "COLOAD_OK"

        # With the monorepo workspace beside us (the root Project.toml that
        # devs every package; absent on CI, which clones only the [sources]
        # siblings), co-load the whole model family the way the site's
        # capability generator does and check the same verbs.
        root = dirname(pkgdir)
        root_project = joinpath(root, "Project.toml")
        family = ["ERGM", "ERGMEgo", "ERGMCount", "ERGMRank", "ERGMMulti", "TERGM",
                  "SNA", "Siena", "REM"]
        if isfile(root_project) && isfile(joinpath(root, "Manifest.toml")) &&
           all(occursin("$pkg = ", _readtext(root_project)) for pkg in family)
            script = """
                using $(join(family, ", "))
                import NetworkCore
                for s in (:compute, :name, :gof, :coef, :coeftable, :Network)
                    @assert isdefined(Main, s) string(s, " undefined after co-loading the family")
                end
                @assert compute === NetworkCore.compute && gof === NetworkCore.gof && name === NetworkCore.name
                println("FAMILY_OK")
                """
            cmd = `$(Base.julia_cmd()) --startup-file=no --project=$root -e $script`
            out = IOBuffer(); err = IOBuffer()
            ok = success(pipeline(cmd; stdout=out, stderr=err))
            errtxt = String(take!(err))
            if ok
                @test strip(String(take!(out))) == "FAMILY_OK"
            elseif (reason = match(r"[^\n]*(?:does not have \S+ in its dependencies|Unsatisfiable requirements|not found in current path|Package \S+ is required but)[^\n]*", errtxt)) !== nothing
                # The workspace Manifest is behind one of the sibling checkouts
                # (a dependency added in another repo, not yet re-resolved at the
                # root): an environment problem, not a co-loading defect — say
                # so rather than fail a test about export conflicts
                @info "Skipping the nine-package co-load: the workspace environment at $root does not resolve" reason.match
            else
                println(stderr, errtxt)
                @test ok
            end
        else
            @info "Skipping the nine-package co-load: no monorepo workspace at $root"
        end
    end

    @testset "Every exported docstring carries a runnable example" begin
        # Every export has a docstring with a runnable
        # example. A docs build with checkdocs=:exports checks presence, not
        # content, so walk the docsystem: every ERGMEgo-owned docstring of an
        # exported binding — including the ones ERGMEgo attaches to the shared
        # NetworkCore/ERGM/StatsAPI generics (`compute`, `gof`, `summary_stats`,
        # `coef`, ...) — must contain a fenced ```julia block, and every such
        # block must RUN in a fresh module that has done nothing but
        # `using ERGMEgo` (so an example that needs `NetworkCore`, `Random` or
        # `DataFrames` says so itself). Names ERGMEgo merely re-exports
        # (`coef`/`stderror`/`vcov` carry ERGMEgo docstrings too, but a name
        # documented only in NetworkCore/ERGM/Graphs/StatsAPI is accepted as
        # documented there). The module docstring is walked as well.
        # Mirrors ERGM.jl's testset of the same name.
        meta = Base.Docs.meta(ERGMEgo)
        documented_elsewhere(b) = any(haskey(Base.Docs.meta(m), b)
                                      for m in (NetworkCore, ERGM, Graphs, StatsAPI))
        undocumented = String[]
        missing_example = String[]
        blocks = Tuple{String,String}[]
        function collect_blocks!(nm, multidoc)
            has_example = false
            for (_, ds) in multidoc.docs
                txt = ds.text isa AbstractString ? ds.text : join(string.(ds.text), "\n")
                for m in eachmatch(r"```julia\n(.*?)```"s, txt)
                    has_example = true
                    push!(blocks, (string(nm), String(m.captures[1])))
                end
                occursin("```jldoctest", txt) && (has_example = true)
            end
            return has_example
        end
        for nm in names(ERGMEgo)
            if nm === :ERGMEgo
                # the module docstring is keyed by the module's own binding
                mods = [md for (b, md) in meta if b.var === :ERGMEgo]
                @test length(mods) == 1
                collect_blocks!(nm, only(mods)) || push!(missing_example, "ERGMEgo (module)")
                continue
            end
            b = Base.Docs.Binding(ERGMEgo, nm)
            if !haskey(meta, b)
                documented_elsewhere(b) || push!(undocumented, string(nm))
                continue
            end
            collect_blocks!(nm, meta[b]) || push!(missing_example, string(nm))
        end
        @test isempty(undocumented)
        @test isempty(missing_example)
        # ERGMEgo-owned docstrings sit on the foreign generics it extends
        for nm in (:coef, :stderror, :vcov, :confint, :coeftable, :nobs, :dof,
                   :gof, :compute, :summary_stats)
            @test haskey(meta, Base.Docs.Binding(ERGMEgo, nm))
        end
        @test length(blocks) >= 30
        for (nm, code) in blocks
            m = Module(Symbol("DocExample_", nm))
            ok = try
                Core.eval(m, :(using ERGMEgo))
                Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    Core.eval(m, Meta.parseall(code; filename="docstring:$nm"))
                end
                true
            catch err
                println(stderr, "docstring example of $nm failed: ", sprint(showerror, err))
                false
            end
            @test ok
        end
    end
end
