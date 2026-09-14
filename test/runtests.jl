using ERGMEgo
using ERGM
using Networks
using DataFrames
using LinearAlgebra: diag
using Random
using Statistics
import Graphs, StatsAPI   # Graphs is a test-only extra (the documented-elsewhere check below); not a package dependency
using Test

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

# `converged` and the recorded report must describe ONE sample: the flag is
# exactly the documented rule applied to `fit.mcmc_convergence`, and the
# report is exactly `ERGM.mcmc_convergence` recomputed on `fit.sim_stats`
function assert_one_sample(fit; conv_threshold=0.1, hotelling_alpha=0.05)
    c = fit.mcmc_convergence
    @test fit.converged == (all(c.t_ratios .< conv_threshold) && c.hotelling_p > hotelling_alpha)
    t = ERGM.mcmc_convergence(fit.sim_stats, fit.model.targets;
                              conv_threshold, hotelling_alpha)
    @test t.t_ratios == c.t_ratios
    @test t.hotelling_p == c.hotelling_p
    @test t.n_eff == c.n_eff
    @test t.converged == fit.converged
    return nothing
end

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

        # Capture-recapture: halves {e1} and {e2, e3};
        # s1 = {101,102,103} (3), s2 = {102,104,101,105,106,107} (6),
        # overlap = {101, 102} (2) → N̂ = 9
        @test estimate_popsize(ed; method=:capture_recapture) ≈ 3 * 6 / 2

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
        rng = Random.Xoshiro(9)
        n = 30
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.12 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        result = ergm_ego(ed, [EgoEdges()]; ppopsize=n,
                          n_samples=200, burnin=1000, interval=10, rng=rng)

        g = ego_gof(result; n_sim=10, rng=Random.Xoshiro(4))
        @test g.n_sim == 10
        @test 0.0 < g.p_values.mean_degree <= 1.0
        @test isfinite(g.simulated.mean_degree)
        # A well-specified edges-only model should not be wildly rejected
        # on mean degree
        @test g.p_values.mean_degree > 0.01

        # ego_gof is a thin wrapper over gof: the same simulations from the
        # same rng state, and its p-values ARE Networks.mc_pvalue on them (the
        # local `(mean(sim .>= o), ...)` closure that could return exactly 0
        # is gone)
        G = gof(result; n_sim=10, rng=Random.Xoshiro(4))
        sim = G.statistics[1].simulated
        obs = G.statistics[1].observed
        @test g.p_values.mean_degree == mc_pvalue(sim[:, 1], obs[1]) == G.statistics[1].p_values[1]
        @test g.p_values.mean_alter_ties == mc_pvalue(sim[:, 2], obs[2])
        @test g.observed.mean_degree == obs[1] == summary_stats(ed).mean_degree
        @test g.simulated.mean_degree == mean(sim[:, 1])
        @test g.simulated.mean_alter_ties == mean(sim[:, 2])
        @test !isdefined(ERGMEgo, :mc_p)

        # gof.ergm.ego's GOF="degree": the second statistic is the ego degree
        # distribution over R's bins — degree 0 … maxdeg−1 plus a "≥ maxdeg"
        # tail, maxdeg = 2·max(K, 3), K the largest OBSERVED ego degree
        # (`degree(0:(maxdeg-1)) + degrange(maxdeg)`) — the design-weighted
        # proportion of egos in each bin, observed from `ed` via EgoDegree,
        # simulated per ego sample. The mean-degree row of the first statistic
        # is a fitted target (every model has EgoEdges), so this is the row
        # set that can actually detect misfit.
        @test length(G.statistics) == 2
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
        # row, so every simulated row is a full distribution. The round-2
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
                              result.mcmc_convergence, result.sim_stats)
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
        @test_throws ArgumentError ERGMEgo._ergm_term(ERGMEgo._EgoDegreeAtLeast(3))   # descriptive only
    end

    @testset "fit aliases, shared show, and Networks.gof" begin
        # Standardized fit_<model> entry point with R-faithful and legacy
        # aliases bound to the same function
        @test ergm_ego === fit_ergm_ego
        @test fit_ego_ergm === fit_ergm_ego

        # One gof generic across the ecosystem: the method is added to
        # Networks.gof, not a package-local function
        @test ERGMEgo.gof === Networks.gof

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
        a = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, burnin=1000,
                         interval=10, rng=Random.Xoshiro(7))
        Random.seed!(987654)
        b = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, burnin=1000,
                         interval=10, rng=Random.Xoshiro(7))
        @test coef(a) == coef(b)
        @test vcov(a) == vcov(b)
        @test a.sim_stats == b.sim_stats
        @test a.mcmc_convergence == b.mcmc_convergence
        # ...and the global RNG is left alone, so a seed set by the caller
        # before the fit still governs whatever the caller draws after it
        Random.seed!(5); x = rand()
        Random.seed!(5)
        fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=50, burnin=100,
                     interval=5, rng=Random.Xoshiro(1))
        @test rand() == x
    end

    @testset "Shared contracts: name/compute/gof identities, StatsAPI surface" begin
        # ONE statistic protocol and ONE gof generic across the ecosystem
        @test ERGMEgo.name === Networks.name
        @test ERGMEgo.compute === Networks.compute
        @test ERGMEgo.gof === Networks.gof
        @test ERGMEgo.coeftable === Networks.coeftable
        # The z → p helper is Networks' (no private copy left in this package)
        @test !isdefined(ERGMEgo, :_z_pvalues)

        # The documented public surface is declared so: `EgoTerm` (the type in
        # fit_ergm_ego's signature and what a custom term subtypes) is
        # exported as ERGM exports AbstractERGMTerm, `_mcmc_controls` (THE
        # budget rule, in the API reference) is `public`
        @test :EgoTerm in names(ERGMEgo)
        @test Base.ispublic(ERGMEgo, :EgoTerm)
        @test Base.ispublic(ERGMEgo, :_mcmc_controls)
        @test !Base.isexported(ERGMEgo, :_mcmc_controls)   # public, not exported (names() lists both)
        # ...and so is the ego-term → ERGM-term hook a custom fittable term
        # extends (the EgoTerm docstring names it; it is not a private reach-in)
        @test Base.ispublic(ERGMEgo, :_ergm_term)
        @test !Base.isexported(ERGMEgo, :_ergm_term)
        @test :_ergm_term in names(ERGMEgo)
        @test !Base.ispublic(ERGMEgo, :_EgoDegreeAtLeast)   # the GOF tail bin stays internal
        @test EgoTerm <: ERGM.AbstractERGMTerm
        # Every cross-package `ERGM._name` reach-in of the source — imported by
        # name or written out — targets a binding ERGM declares public (panel
        # 2026-09, item 13), and so do the convergence machinery it reuses
        src = read(joinpath(dirname(@__DIR__), "src", "ERGMEgo.jl"), String)
        reachins = Set{Symbol}()
        for m in eachmatch(r"ERGM\._(\w+)", src)
            push!(reachins, Symbol("_" * m.captures[1]))
        end
        for m in eachmatch(r"import ERGM:\s*([^\n]+)", src)
            for nm in split(m.captures[1], ",")
                s = strip(nm)
                startswith(s, "_") && push!(reachins, Symbol(s))
            end
        end
        @test :_mcmc_defaults in reachins
        for s in reachins
            @test Base.ispublic(ERGM, s)
        end
        @test Base.ispublic(ERGM, :_mcmc_defaults)
        @test Base.ispublic(ERGM, :mcmc_convergence)
        @test Base.ispublic(ERGM, :MCMLEConvergence)
        # Graphs is not a dependency: every graph primitive the package uses
        # (`nv`, `neighbors`, `has_edge`, `vertices`) is Networks.jl's re-export
        @test !isdefined(ERGMEgo, :Graphs)
        @test ERGMEgo.nv === Networks.nv && ERGMEgo.neighbors === Networks.neighbors
        @test ERGMEgo.has_edge === Networks.has_edge
        project = read(joinpath(dirname(@__DIR__), "Project.toml"), String)
        deps_block = match(r"\[deps\]\n(.*?)\n\n"s, project).captures[1]
        compat_block = match(r"\[compat\]\n(.*?)\n\n"s, project).captures[1]
        @test !occursin("Graphs", deps_block)
        @test !occursin("Graphs", compat_block)
        @test occursin(r"\[extras\][^\[]*Graphs"s, project)

        rng = Random.Xoshiro(21)
        n = 25
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.15 && add_edge!(net, i, j)
        end
        ed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, burnin=1000,
                           interval=10, rng=rng)

        # The full surface the fit can honestly answer, pinned in one line;
        # loglikelihood/aic/bic are deliberately absent (objective == :moment,
        # no likelihood is evaluated), so they are NOT methods, not NaN-returners
        @test Networks.check_statsapi(fit; required=(:coef, :stderror, :vcov, :confint,
                                                     :nobs, :dof, :coeftable),
                                      strict=true) !== nothing
        SA = ERGMEgo.StatsAPI
        @test !hasmethod(SA.loglikelihood, Tuple{EgoERGMResult})
        @test !hasmethod(SA.aic, Tuple{EgoERGMResult})
        @test !hasmethod(SA.bic, Tuple{EgoERGMResult})
        @test Networks.check_statsapi(fit).loglikelihood == false

        tbl = coeftable(fit)
        @test tbl isa Networks.CoefficientTable
        @test tbl["ego.edges"].estimate == coef(fit)[1]
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

        # Item 19: the model and result carry the ego data's ID type, so the
        # data field is concretely typed
        rng = Random.Xoshiro(2)
        n = 20
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.2 && add_edge!(net, i, j)
        end
        sed = simulate_ego_sample(net, n; rng=rng)
        fit = fit_ergm_ego(sed, [EgoEdges()]; ppopsize=n, n_samples=100, burnin=500,
                           interval=5, rng=rng)
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
        @test name(ERGMEgo._ergm_term(EgoGWDegree(0.5))) == "gwdeg.fixed.0.5"
        @test name(ERGMEgo._ergm_term(EgoGWDegree(0))) == "gwdeg.fixed.0"
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
        # silent zero-fill class as ERGM's NodeCov (panel P1-11); it now names
        # the attribute, the ego and the side that lacks it
        msg = errmsg(() -> compute(EgoNodeMatch(:nope), ed))
        @test occursin(":nope", msg) && occursin("ego 1", msg) && occursin("ego attribute", msg)
        @test occursin("ego.nodematch.nope", msg)
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
        m1 = EgoNetwork(11, [10, 11, 12], zeros(Bool, 3, 3);
                        ego_attrs=Dict{Symbol,Any}(:group => "A"),
                        alter_attrs=Dict{Symbol,Vector}(:group => Union{String,Missing}["A", missing, missing]))
        msg = errmsg(() -> compute(EgoNodeMatch(:group), EgoData([ed[1], m1])))
        @test occursin("ego 11", msg) && occursin(":group", msg)
        @test occursin("2 alters of 3", msg) && occursin("drop those alters", msg)
        @test occursin("ego.nodematch.group", msg)
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
        wed = ego_design(ed; ppopsize=500, weights=[100.0, 150.0, 250.0])
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
        fit = fit_ergm_ego(sed, [EgoEdges()]; ppopsize=n, popsize=4n, n_samples=100,
                           burnin=500, interval=5, rng=rng)
        outm = sprint(show, fit.model)
        @test occursin("EgoERGMModel{Int64}: 20 egos", outm)
        @test occursin("terms: ego.edges", outm)
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
        fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, burnin=1000,
                           interval=10, rng=Random.Xoshiro(1))

        # The panel's failing repro (2026-09, item 6): `_gof_simulations`
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
        @test Networks.n_simulations(two) == 8
        # ...and thread-count independent, for real: the same gof in a fresh
        # process with a DIFFERENT thread count is bit-identical (chains are
        # seeded from the caller's rng and concatenated in order, as in
        # ERGM.jl's own test; `n_chains` never defaults to Threads.nthreads())
        other_threads = Threads.nthreads() == 1 ? 4 : 1
        script = """
            using ERGMEgo, Networks, Random
            rng = Xoshiro(9)
            n = 30
            net = network(n; directed=false)
            for i in 1:n, j in (i+1):n
                rand(rng) < 0.12 && add_edge!(net, i, j)
            end
            ed = simulate_ego_sample(net, n; rng=rng)
            fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=200, burnin=1000,
                               interval=10, rng=Xoshiro(1))
            g = gof(fit; n_sim=8, rng=Xoshiro(5), n_chains=2)
            println(Threads.nthreads())
            println(repr(g.statistics[1].simulated))
            println(repr(g.statistics[1].p_values))
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$other_threads --project=$(dirname(@__DIR__)) -e $script`
        lines = split(strip(read(pipeline(cmd; stderr=devnull), String)), '\n')
        @test length(lines) == 3
        @test lines[1] == string(other_threads)
        @test lines[2] == repr(two.statistics[1].simulated)
        @test lines[3] == repr(two.statistics[1].p_values)
    end

    # ------------------------------------------------------------------
    # Golden fixture: statnet `ergm.ego` on faux.mesa.high under a CENSUS
    # (issue #8, and the direct answer to issue ERGMEgo#1).
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

        # --- (2) THE DESIGN VARIANCE — ISSUE ERGMEgo#1, FOUND AND FIXED -------
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
        H = [ERGMEgo._ego_contribution(terms[j], ed[i]) for i in 1:n, j in eachindex(terms)]
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
        # PARAMETERIZATION: ergm.ego splits the population edges parameter into a
        # fixed offset netsize.adj = −log(popsize) = −5.3230 plus a free `edges`
        # coefficient (−0.6974); ERGMEgo.jl reports it as ONE number on the
        # pseudo-population scale. R's −0.697 and Julia's −6.03 are the same
        # parameter in different clothes. The comparable quantity is the sum,
        # which the fixture freezes as `mle_coefficients_population`.
        @test Float64(g.values["netsize_adjustment"]) ≈ -log(n) atol = 1e-9

        # The DEFAULTS (dyad-scaled budget, t-ratio + Hotelling convergence)
        # converge here in 5-25 iterations at ~3 s per fit. Five seeds, as in R.
        fits = [ergm_ego(ed, terms; rng=Random.Xoshiro(s)) for s in (101, 202, 303, 404, 505)]
        @test all(f.converged for f in fits)
        @test all(f.mcmc_convergence.iterations <= 80 for f in fits)
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
            @test f.mcmc_convergence.n_eff >= 100
            @test f.mcmc_convergence.n_eff <= size(f.sim_stats, 1)
            @test size(f.sim_stats) == (3000, 2)
        end
        mean_se = mean(stderror(f) for f in fits)
        @test check_golden(g, "mle_std_errors", mean_se) ||
              error(golden_report(g, "mle_std_errors", mean_se))
        # The estimation component is the same order as ergm.ego's (0.0083 /
        # 0.0103; ERGMEgo.jl's chain has fewer effective draws, so it is larger)
        r_est = Float64.(g.values["mle_se_estimation_component"])
        @test all(r_est .< mean(sqrt.(diag(f.vcov_estimation)) for f in fits) .< 3 .* r_est)
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
        @test [name(ERGMEgo._ergm_term(t)) for t in terms4] == g.values["term_names"]

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
        H = [ERGMEgo._ego_contribution(t, e) for e in wed.egos, t in terms4]
        wn = w ./ sum(w)
        h̄ = vec(wn' * H)
        D = H .- h̄'
        @test Σ_jl ≈ m^2 * (n_e / (n_e - 1)) .* ((D .* wn .^ 2)' * D) atol = 1e-9

        # --- (3) THE PSEUDO-POPULATION: weight-proportional, same as R's ------
        # 205 vertices, egos replicated in proportion to their weights
        # (largest-remainder rounding here, ppop.wt = "round" there): the
        # composition by grade is identical and does not depend on the rng
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
        fits = [ergm_ego(wed, terms; rng=Random.Xoshiro(s)) for s in (1, 2, 3)]
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
            @test f.mcmc_convergence.n_eff >= 100
            @test size(f.sim_stats) == (3000, 2)
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

        # The defaults used to be fixed constants (n_samples=400, burnin=2000,
        # interval=20) that did not scale with the pseudo-population. On this
        # 205-actor census the chain stopped mixing and the fit returned
        # edges ≈ −21.9 against a true −6.02 — off by a factor of THREE, on a
        # network the size of the standard teaching dataset. The MCMC controls
        # now follow ERGM.jl's one dyad-scaled rule (`_mcmc_controls`), and the
        # 1 % relative-change rule that stopped at the noise level is replaced
        # by ERGM.jl's t-ratio + Hotelling tests. The defaults must converge
        # here and land on R's answer within the single-fit Monte-Carlo sd
        # (0.026; 0.1 is ~4 sd).
        f = ergm_ego(ed, terms; rng=Random.Xoshiro(7))
        @test f.converged
        @test abs(f.coefficients[1] - r_pop[1]) < 0.1
        @test abs(f.coefficients[2] - r_pop[2]) < 0.1
        n_dyads = 205 * 204 ÷ 2
        ctl = ERGMEgo._mcmc_controls(205)
        @test ctl == (n_samples=3000, burnin=20 * n_dyads, interval=max(100, n_dyads ÷ 10))
        @test ctl.burnin == ERGM._mcmc_defaults(n_dyads).burnin
        @test ERGMEgo._mcmc_controls(205; n_samples=500, interval=7) ==
              (n_samples=500, burnin=20 * n_dyads, interval=7)

        # A converged fit's report describes the sample that passed — the
        # final sample IS the passing one (R's ergm design; no fresh draw is
        # taken at θ̂ afterwards), so the recorded t-ratios and Hotelling p
        # satisfy the rule by construction, for every seed, and
        # `converged`/`mcmc_convergence`/`sim_stats` cannot disagree. (The
        # round-1 code redrew a sample for the report: one fit in four then
        # printed `Converged: true` next to a Hotelling p of 0.0005.)
        c = f.mcmc_convergence
        @test c.hotelling_p > 0.05
        @test all(c.t_ratios .< 0.1)
        @test 0 < c.hotelling_p <= 1
        @test c.iterations >= 1
        @test c.step_length == 1.0
        assert_one_sample(f)
        for s in (101, 505)
            assert_one_sample(ergm_ego(ed, terms; rng=Random.Xoshiro(s)))
        end
        # ...and the standard errors come from that same sample
        @test f.vcov_estimation ≈ inv(cov(f.sim_stats)) ./ c.n_eff rtol = 1e-6

        # --- Non-convergence is LOUD (item 24): maxiter=1 cannot solve the
        # moment equations from the initial values; the fit warns with the
        # diagnostics, records converged == false, prints the caveat under
        # `Converged: false`, and lists it in approximations
        u = @test_logs (:warn, r"did not converge") match_mode=:any ergm_ego(
            ed, terms; maxiter=1, rng=Random.Xoshiro(1))
        @test !u.converged
        @test u.mcmc_convergence.iterations == 1
        @test maximum(u.mcmc_convergence.t_ratios) > 0.1
        @test any(occursin("did not converge", a) for a in approximations(u))
        @test any(occursin("did not converge", a) for a in Networks.fit_metadata(u).approximations)
        uout = sprint(show, u)
        @test occursin("Converged: false", uout)
        @test occursin("did not converge", uout)
        @test occursin("max t-ratio", uout)
        @test all(isfinite, stderror(u))            # SEs exist, but are flagged unreliable
        # The caveat quotes the sample at the RETURNED coefficients — the one
        # `converged` was decided on — and says so; its numbers therefore
        # always FAIL the documented rule (the round-1 message quoted a fresh
        # draw that could pass it while the verdict said "did not converge")
        @test occursin("on the final sample at the returned coefficients", uout)
        assert_one_sample(u)
        @test !(all(u.mcmc_convergence.t_ratios .< 0.1) && u.mcmc_convergence.hotelling_p > 0.05)

        # When maxiter is exhausted after a Newton step, a fresh sample at the
        # returned coefficients decides: a small edges-only fit whose first
        # sample (at the initial values) fails but whose post-step sample
        # passes is CONVERGED — no warning, and the report is that sample's
        rng = Random.Xoshiro(3)
        sn = network(30; directed=false)
        for i in 1:30, j in (i+1):30
            rand(rng) < 0.12 && add_edge!(sn, i, j)
        end
        sed = simulate_ego_sample(sn, 30; rng=rng)
        one_step(s) = fit_ergm_ego(sed, [EgoEdges()]; ppopsize=50, maxiter=1, n_samples=200,
                                   burnin=1000, interval=10, rng=Random.Xoshiro(s))
        logs, ok = Test.collect_test_logs(() -> one_step(2))
        @test !any(occursin("did not converge", string(l.message)) for l in logs)
        @test ok.converged
        @test ok.mcmc_convergence.iterations == 1
        @test !any(occursin("did not converge", a) for a in approximations(ok))
        @test occursin("Converged: true", sprint(show, ok))
        assert_one_sample(ok)
        # ...while a seed whose post-step sample fails is not, and says so
        bad = @test_logs (:warn, r"on the final sample at the returned coefficients") match_mode=:any one_step(1)
        @test !bad.converged
        assert_one_sample(bad)

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

        # --- Keyword vocabulary (item 16): maxiter is the name; max_iter is a
        # deprecated, honoured shim; tol is deprecated and ignored
        kw = Base.kwarg_decl(first(methods(fit_ergm_ego)))
        for name in (:maxiter, :n_samples, :rng, :burnin, :interval, :conv_threshold,
                     :hotelling_alpha, :ppopsize, :popsize)
            @test name in kw
        end
        # (two iterations cannot converge here, so each of the three fits warns)
        f_new = @test_logs (:warn, r"did not converge") match_mode=:any ergm_ego(
            ed, terms; maxiter=2, rng=Random.Xoshiro(3))
        f_old = @test_logs (:warn, r"max_iter.*deprecated") match_mode=:any ergm_ego(
            ed, terms; max_iter=2, rng=Random.Xoshiro(3))
        @test f_old.mcmc_convergence.iterations <= 2
        @test f_old.coefficients == f_new.coefficients   # honoured, identical fit
        @test f_old.sim_stats == f_new.sim_stats
        f_tol = @test_logs (:warn, r"tol.*deprecated") match_mode=:any ergm_ego(
            ed, terms; maxiter=2, tol=0.01, rng=Random.Xoshiro(3))
        @test f_tol.coefficients == f_new.coefficients   # ignored: changes nothing
        @test f_tol.sim_stats == f_new.sim_stats
        @test_throws ArgumentError ergm_ego(ed, terms; maxiter=0)
    end

    @testset "Hot paths are allocation-free" begin
        # The per-ego contribution — the innermost loop of every target
        # statistic and of the design covariance — is allocation-free for the
        # four fittable terms (EgoNodeMatch through a function barrier over
        # the abstractly typed attribute column), and `_design_cov` allocates
        # its H matrix, the normalised weights, h̄ and Σ and nothing per ego.
        # The same pins live in benchmark/regression_tests.jl (the standalone
        # runner the site's tools/run_benchmarks.jl consumes); here so that
        # `Pkg.test()` alone guards them (panel 2026-09, item 7).
        n = 500
        rng = Random.Xoshiro(1)
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 6 / n && add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :g, Dict(v => ("A", "B", "C")[mod1(v, 3)] for v in 1:n))
        ed = simulate_ego_sample(net, n; ego_attrs=[:g], rng=Random.Xoshiro(2))
        terms = [EgoEdges(), EgoNodeMatch(:g), EgoTriangle(), EgoGWDegree(0.5)]
        function worst_alloc(term, egos)
            worst = 0
            for e in egos
                ERGMEgo._ego_contribution(term, e)
                worst = max(worst, @allocated ERGMEgo._ego_contribution(term, e))
            end
            return worst
        end
        for term in terms
            @test worst_alloc(term, ed.egos[1:100]) == 0
        end
        function design_cov_bytes(ts)
            ERGMEgo._design_cov(ts, ed, n)
            return @allocated ERGMEgo._design_cov(ts, ed, n)
        end
        for ts in (terms, [EgoEdges()], [EgoEdges(), EgoNodeMatch(:g)])
            @test design_cov_bytes(ts) <= 4 * n * length(ts) * 8 + 4096
        end
        # ...and the lean loop is the same number as the textbook formula
        H = hcat([[ERGMEgo._ego_contribution(t, e) for e in ed.egos] for t in terms]...)
        @test ERGMEgo._design_cov(terms, ed, n) ≈ n^2 .* Statistics.cov(H) ./ n
    end

    @testset "Co-loading leaves every shared verb defined (fresh process)" begin
        # `using ERGM, ERGMEgo` — the statnet workflow — must leave the shared
        # verbs defined (two packages each exporting their own `compute` would
        # leave it *undefined*); they are Networks.jl's generics in both.
        pkgdir = dirname(@__DIR__)
        script = """
            using ERGM, ERGMEgo
            import Networks   # the module name is deliberately not re-exported
            for s in (:compute, :name, :gof, :coef, :stderror, :vcov, :coeftable,
                      :summary_stats, :Network)
                @assert isdefined(Main, s) string(s, " undefined after `using ERGM, ERGMEgo`")
            end
            @assert compute === Networks.compute
            @assert gof === Networks.gof
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
           all(occursin("$pkg = ", read(root_project, String)) for pkg in family)
            script = """
                using $(join(family, ", "))
                import Networks
                for s in (:compute, :name, :gof, :coef, :coeftable, :Network)
                    @assert isdefined(Main, s) string(s, " undefined after co-loading the family")
                end
                @assert compute === Networks.compute && gof === Networks.gof && name === Networks.name
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

    @testset "Every exported docstring carries a runnable example (criterion 5)" begin
        # Grade-A criterion 5: every export has a docstring with a runnable
        # example. A docs build with checkdocs=:exports checks presence, not
        # content, so walk the docsystem: every ERGMEgo-owned docstring of an
        # exported binding — including the ones ERGMEgo attaches to the shared
        # Networks/ERGM/StatsAPI generics (`compute`, `gof`, `summary_stats`,
        # `coef`, ...) — must contain a fenced ```julia block, and every such
        # block must RUN in a fresh module that has done nothing but
        # `using ERGMEgo` (so an example that needs `Networks`, `Random` or
        # `DataFrames` says so itself). Names ERGMEgo merely re-exports
        # (`coef`/`stderror`/`vcov` carry ERGMEgo docstrings too, but a name
        # documented only in Networks/ERGM/Graphs/StatsAPI is accepted as
        # documented there). The module docstring is walked as well.
        # Mirrors ERGM.jl's testset of the same name.
        meta = Base.Docs.meta(ERGMEgo)
        documented_elsewhere(b) = any(haskey(Base.Docs.meta(m), b)
                                      for m in (Networks, ERGM, Graphs, StatsAPI))
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
