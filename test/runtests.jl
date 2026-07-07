using ERGMEgo
using ERGM
using Network
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

        g = ego_gof(result; n_sim=10, rng=rng)
        @test g.n_sim == 10
        @test 0.0 <= g.p_values.mean_degree <= 1.0
        @test isfinite(g.simulated.mean_degree)
        # A well-specified edges-only model should not be wildly rejected
        # on mean degree
        @test g.p_values.mean_degree > 0.01
    end
end
