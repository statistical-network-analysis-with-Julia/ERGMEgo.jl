#!/usr/bin/env julia
# benchmark/regression_tests.jl — allocation-regression assertions for the
# ERGMEgo.jl hot paths. Standalone; run with
#     julia --project=benchmark benchmark/regression_tests.jl
#
# The per-ego contribution `ego_contribution(term, ego)` is the innermost
# loop of every target statistic and of the design covariance, and it is
# allocation-free for the fittable terms (`EgoNodeMatch` and `EgoNodeFactor`
# through a function barrier over the abstractly typed attribute column; the
# numeric `EgoNodeCov` and `EgoAbsDiff` branch on the Int/Float64 column
# layouts, so their kernel is a static call). `_design_cov`
# allocates its H matrix, the normalised weights, h̄ and Σ and nothing else
# per ego: ≤ 4·n·p·8 bytes plus a constant (measured 1.3× at p = 4, 2.0× at
# p = 1). Any allocation appearing here is a performance regression — these
# tests assert the loops STAY that way rather than tracking a noisy budget.
# The same pins live in test/runtests.jl ("Hot paths are allocation-free")
# so `Pkg.test()` alone guards them; this file is the standalone runner the
# site's tools/run_benchmarks.jl consumes.

using ERGMEgo
using NetworkCore
using Random
using Test

function er_network(rng::AbstractRNG, n::Int, mean_degree::Int)
    net = network(n; directed=false)
    p = mean_degree / n
    for i in 1:n, j in (i + 1):n
        rand(rng) < p && add_edge!(net, i, j)
    end
    set_vertex_attribute!(net, :g, Dict(v => ("A", "B", "C")[mod1(v, 3)] for v in 1:n))
    set_vertex_attribute!(net, :x, Dict(v => 0.5 * mod(v, 7) for v in 1:n))
    return net
end

"Bytes allocated by `ego_contribution` on a pre-warmed call, worst over `egos`."
function max_allocs_contribution(term, egos)
    worst = 0
    for e in egos
        ERGMEgo.ego_contribution(term, e)                # warm up / compile
        worst = max(worst, @allocated ERGMEgo.ego_contribution(term, e))
    end
    return worst
end

"Bytes allocated by `_design_cov` on a pre-warmed call."
function allocs_design_cov(terms, ed, m)
    ERGMEgo._design_cov(terms, ed, m)
    return @allocated ERGMEgo._design_cov(terms, ed, m)
end

@testset "ERGMEgo allocation regressions" begin
    n = 2000
    net = er_network(Random.Xoshiro(1), n, 6)
    ed = simulate_ego_sample(net, n; ego_attrs=[:g, :x], rng=Random.Xoshiro(2))
    terms = [EgoEdges(), EgoNodeMatch(:g), EgoTriangle(), EgoGWDegree(0.5),
             EgoNodeFactor(:g; level="B"), EgoDegree(3), EgoGWESP(0.5), EgoESP(1),
             EgoMM(:g, "A", "B"), EgoMM(:g, "C", "C"), EgoConcurrent()]
    numeric = [EgoNodeCov(:x), EgoAbsDiff(:x), EgoAbsDiff(:x; pow=2)]

    @testset "ego_contribution is allocation-free (count terms)" begin
        for term in terms
            @test max_allocs_contribution(term, ed.egos[1:100]) == 0
        end
    end

    @testset "ego_contribution is allocation-free (numeric terms)" begin
        for term in numeric
            @test max_allocs_contribution(term, ed.egos[1:100]) == 0
        end
    end

    @testset "_design_cov allocates O(n·p) bytes and nothing per ego" begin
        for ts in (terms, [EgoEdges()], [EgoEdges(), EgoNodeMatch(:g)], [terms; numeric])
            p = length(ts)
            @test allocs_design_cov(ts, ed, n) <= 4 * n * p * 8 + 4096
        end
    end
end
