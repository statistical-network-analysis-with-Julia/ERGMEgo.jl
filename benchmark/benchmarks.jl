#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for ERGMEgo.jl's hot paths.
#
# What is timed, and why:
#   * `compute(term, ed)` for the four fittable ego terms on 2000 simulated
#     egos — the per-ego contribution loop behind every target statistic;
#   * `ERGMEgo._design_cov` on the same egos — the survey-design covariance
#     of the targets, the one O(n·p²) pass of a fit;
#   * `simulate_ego_sample` as a census of a 500-vertex network — the
#     `Network → EgoData` adapter (O(Σ degree²) alter–alter lookups);
#   * one small `fit_ergm_ego` (20-vertex census, a short chain) — the whole
#     moment-matching loop end to end, so a regression in the sampler call
#     or the Newton step shows up in one number.
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite and prints one tab-separated `BENCHJL` line
# per benchmark (consumed by the site repo's tools/run_benchmarks.jl). The
# allocation pins live in benchmark/regression_tests.jl.

using BenchmarkTools
using ERGMEgo
using Networks
using Random
using Logging: NullLogger, with_logger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

const N_EGOS = 2000            # egos behind the term / design-covariance rows
const N_CENSUS = 500           # vertices of the network the adapter samples
const MEAN_DEGREE = 6

"Sparse undirected Erdős–Rényi network with a three-level vertex attribute `:g`."
function er_network(rng::AbstractRNG, n::Int)
    net = network(n; directed=false)
    p = MEAN_DEGREE / n
    for i in 1:n, j in (i + 1):n
        rand(rng) < p && add_edge!(net, i, j)
    end
    set_vertex_attribute!(net, :g, Dict(v => ("A", "B", "C")[mod1(v, 3)] for v in 1:n))
    return net
end

const NET_EGOS = er_network(Random.Xoshiro(1), N_EGOS)
const ED = simulate_ego_sample(NET_EGOS, N_EGOS; ego_attrs=[:g], rng=Random.Xoshiro(2))
const NET_CENSUS = er_network(Random.Xoshiro(3), N_CENSUS)

const TERMS = [("edges", EgoEdges()),
               ("nodematch", EgoNodeMatch(:g)),
               ("triangle", EgoTriangle()),
               ("gwdegree", EgoGWDegree(0.5))]
const ALL_TERMS = [t for (_, t) in TERMS]

"A 20-vertex census fit with an explicit short chain (deterministic in `rng`)."
function small_fit()
    n = 20
    rng = Random.Xoshiro(21)
    net = network(n; directed=false)
    for i in 1:n, j in (i + 1):n
        rand(rng) < 0.15 && add_edge!(net, i, j)
    end
    ed = simulate_ego_sample(net, n; rng=rng)
    # A short chain and a small iteration cap: a benchmark, not an estimate,
    # so the non-convergence warning it may emit is silenced
    return with_logger(NullLogger()) do
        fit_ergm_ego(ed, [EgoEdges()]; ppopsize=n, n_samples=100, burnin=200,
                     interval=5, maxiter=5, rng=rng)
    end
end

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "compute")
    for (label, term) in TERMS
        g["$(label)_n$(N_EGOS)"] = @benchmarkable compute($term, $ED)
    end
end

let g = addgroup!(SUITE, "design_cov")
    g["four_terms_n$(N_EGOS)"] = @benchmarkable ERGMEgo._design_cov($ALL_TERMS, $ED, $N_EGOS)
end

let g = addgroup!(SUITE, "simulate_ego_sample")
    g["census_n$(N_CENSUS)"] =
        @benchmarkable simulate_ego_sample($NET_CENSUS, $N_CENSUS; ego_attrs=[:g],
                                           rng=Random.Xoshiro(4))
end

let g = addgroup!(SUITE, "fit_ergm_ego")
    g["census_n20_short_chain"] = @benchmarkable small_fit()
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
