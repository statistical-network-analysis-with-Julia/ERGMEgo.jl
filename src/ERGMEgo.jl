"""
    ERGMEgo.jl - ERGMs for Ego-Centric Network Data

Fits ERGMs to egocentrically sampled network data (a sample of "egos" with
their local networks: alters and ties among alters), enabling inference
about complete-network properties from ego samples.

The methodology follows R `ergm.ego` (Krivitsky & Morris 2017): ego
statistics are design-weighted and scaled to **target statistics** for a
pseudo-population network of size `ppopsize`, an ERGM is fit to those
targets by method-of-moments (MCMC moment matching), and the edges
coefficient receives the network-size adjustment `−log(popsize/ppopsize)`
to put it on the population scale. Standard errors combine the model-based
(inverse-information) and survey-design variance components.

Port of the R ergm.ego package from the StatNet collection.
"""
module ERGMEgo

using DataFrames
using Distributions
using ERGM
using Graphs
using LinearAlgebra
using Network
using Random
using Statistics
using StatsBase

import ERGM: name, compute, summary_stats
# Shared presentation infrastructure (Network.jl): the ONE `gof` generic all
# model packages extend, plus the common coefficient-table printer and
# GOF containers
import Network: gof, print_coeftable, GOFStatistic, GOFResult
import StatsAPI
import StatsAPI: coef, stderror, vcov

# Data structures
export EgoData, EgoNetwork
export n_alters, ego_degree, alter_degree, n_alter_ties

# Data preparation
export as_egodata, ego_design

# Ego-specific terms and statistics
export EgoEdges, EgoNodeMatch, EgoDegree, EgoGWDegree, EgoTriangle
export ego_mixing_matrix, ego_target_stats
export summary_stats

# Estimation
export fit_ergm_ego, ergm_ego, fit_ego_ergm, EgoERGMModel, EgoERGMResult

# Population size estimation
export estimate_popsize

# Simulation
export simulate_ego_sample

# Diagnostics: gof is a method of the shared Network.jl generic; ego_gof is
# the legacy NamedTuple-returning form
export gof, ego_gof

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGMEgo`)
export coef, stderror, vcov

# =============================================================================
# Ego Network Data Structures
# =============================================================================

"""
    EgoNetwork{T}

An ego-centric network observation.

# Fields
- `ego::T`: Ego ID
- `alters::Vector{T}`: Alter IDs (original IDs are preserved so that
  cross-ego alter overlap remains meaningful)
- `alter_ties::Matrix{Bool}`: Symmetric adjacency among alters (ego not
  included)
- `ego_attrs::Dict{Symbol, Any}`: Ego attributes
- `alter_attrs::Dict{Symbol, Vector}`: Alter attributes (column per attribute)
"""
struct EgoNetwork{T}
    ego::T
    alters::Vector{T}
    alter_ties::Matrix{Bool}
    ego_attrs::Dict{Symbol, Any}
    alter_attrs::Dict{Symbol, Vector}

    function EgoNetwork{T}(ego::T, alters::Vector{T}, alter_ties::Matrix{Bool};
                           ego_attrs::Dict{Symbol,Any}=Dict{Symbol,Any}(),
                           alter_attrs::Dict{Symbol,Vector}=Dict{Symbol,Vector}()) where T
        n_alters = length(alters)
        size(alter_ties) == (n_alters, n_alters) ||
            throw(ArgumentError("alter_ties must be $(n_alters)×$(n_alters)"))
        alter_ties == transpose(alter_ties) ||
            throw(ArgumentError("alter_ties must be symmetric (undirected)"))
        new{T}(ego, alters, alter_ties, ego_attrs, alter_attrs)
    end
end

EgoNetwork(ego::T, alters::Vector{T}, alter_ties::Matrix{Bool}; kwargs...) where T =
    EgoNetwork{T}(ego, alters, alter_ties; kwargs...)

"""
    n_alters(ego_net::EgoNetwork) -> Int

Number of alters in an ego network.
"""
n_alters(ego_net::EgoNetwork) = length(ego_net.alters)

"""
    ego_degree(ego_net::EgoNetwork) -> Int

The ego's degree (number of alters).
"""
ego_degree(ego_net::EgoNetwork) = n_alters(ego_net)

"""
    alter_degree(ego_net::EgoNetwork) -> Vector{Int}

Degree of each alter within the ego network (not counting ego).
"""
alter_degree(ego_net::EgoNetwork) = vec(sum(ego_net.alter_ties, dims=2))

"""
    n_alter_ties(ego_net::EgoNetwork) -> Int

Number of (undirected) ties among alters.
"""
n_alter_ties(ego_net::EgoNetwork) = sum(ego_net.alter_ties) ÷ 2

"""
    EgoData

Collection of ego networks with sampling information.

# Fields
- `egos::Vector{EgoNetwork}`: Individual ego network observations
- `population_size::Union{Int, Nothing}`: Known or estimated population size
- `sampling_weights::Vector{Float64}`: Per-ego sampling weights (design
  weights, ideally inverse inclusion probabilities)
- `design::Dict{Symbol, Any}`: Survey design information
"""
struct EgoData{T}
    egos::Vector{EgoNetwork{T}}
    population_size::Union{Int, Nothing}
    sampling_weights::Vector{Float64}
    design::Dict{Symbol, Any}

    function EgoData(egos::Vector{EgoNetwork{T}};
                     population_size::Union{Int, Nothing}=nothing,
                     sampling_weights::Vector{Float64}=Float64[],
                     design::Dict{Symbol, Any}=Dict{Symbol, Any}()) where T
        n_egos = length(egos)
        sw = isempty(sampling_weights) ? ones(n_egos) : sampling_weights
        length(sw) == n_egos ||
            throw(ArgumentError("need one sampling weight per ego"))
        new{T}(egos, population_size, sw, design)
    end
end

Base.length(ed::EgoData) = length(ed.egos)
Base.iterate(ed::EgoData, state=1) = state > length(ed) ? nothing : (ed.egos[state], state + 1)
Base.getindex(ed::EgoData, i) = ed.egos[i]

"""
    summary_stats(ed::EgoData) -> NamedTuple

Design-weighted summary statistics for ego data.
"""
function summary_stats(ed::EgoData)
    w = Weights(ed.sampling_weights)
    degrees = [Float64(ego_degree(e)) for e in ed.egos]
    alter_ties = [Float64(n_alter_ties(e)) for e in ed.egos]

    return (
        n_egos = length(ed.egos),
        mean_degree = mean(degrees, w),
        median_degree = median(degrees),
        min_degree = minimum(degrees),
        max_degree = maximum(degrees),
        mean_alter_ties = mean(alter_ties, w),
        total_alters = sum(degrees),
        population_size = ed.population_size
    )
end

# =============================================================================
# Data Preparation
# =============================================================================

"""
    as_egodata(ego_df::DataFrame, alter_df::DataFrame;
               aatie_df=nothing, kwargs...) -> EgoData

Create `EgoData` from `ergm.ego`-style data frames:

- `ego_df`: one row per ego (`ego_id` column plus ego attributes)
- `alter_df`: one row per ego–alter pair (`ego_id`, `alter_id` plus alter
  attributes)
- `aatie_df`: optional alter–alter ties, one row per tie with columns
  `ego_id`, `source_col`, `target_col` (alter IDs)

# Keyword Arguments
- `ego_id::Symbol=:ego_id`, `alter_id::Symbol=:alter_id`
- `ego_attrs::Vector{Symbol}=Symbol[]`: Ego attribute columns from `ego_df`
- `alter_attrs::Vector{Symbol}=Symbol[]`: Alter attribute columns from `alter_df`
- `weight_col::Union{Symbol,Nothing}=nothing`: Ego sampling-weight column
  in `ego_df`
- `source_col::Symbol=:src`, `target_col::Symbol=:dst`: Alter-tie columns
- `population_size::Union{Int,Nothing}=nothing`

Alter IDs are preserved (not relabeled), so cross-ego alter overlap
remains available to `estimate_popsize`.
"""
function as_egodata(ego_df::DataFrame, alter_df::DataFrame;
                    aatie_df::Union{DataFrame, Nothing}=nothing,
                    ego_id::Symbol=:ego_id,
                    alter_id::Symbol=:alter_id,
                    ego_attrs::Vector{Symbol}=Symbol[],
                    alter_attrs::Vector{Symbol}=Symbol[],
                    weight_col::Union{Symbol, Nothing}=nothing,
                    source_col::Symbol=:src,
                    target_col::Symbol=:dst,
                    population_size::Union{Int, Nothing}=nothing)
    egos = EgoNetwork{Int}[]
    weights = Float64[]

    if !isnothing(weight_col) && !(string(weight_col) in names(ego_df))
        throw(ArgumentError("weight column :$weight_col not found in ego_df"))
    end

    for row in eachrow(ego_df)
        eid = Int(row[ego_id])
        a_rows = alter_df[alter_df[!, ego_id] .== eid, :]
        alters = Int.(a_rows[!, alter_id])
        n_a = length(alters)
        alter_index = Dict(a => k for (k, a) in enumerate(alters))

        # Alter-alter ties
        ties = zeros(Bool, n_a, n_a)
        if !isnothing(aatie_df)
            t_rows = aatie_df[aatie_df[!, ego_id] .== eid, :]
            for t in eachrow(t_rows)
                a, b = Int(t[source_col]), Int(t[target_col])
                (haskey(alter_index, a) && haskey(alter_index, b)) ||
                    throw(ArgumentError("alter tie ($a, $b) of ego $eid references unknown alters"))
                ties[alter_index[a], alter_index[b]] = true
                ties[alter_index[b], alter_index[a]] = true
            end
        end

        e_attrs = Dict{Symbol, Any}(attr => row[attr] for attr in ego_attrs)
        a_attrs = Dict{Symbol, Vector}(attr => collect(a_rows[!, attr])
                                       for attr in alter_attrs)

        push!(egos, EgoNetwork(eid, alters, ties;
                               ego_attrs=e_attrs, alter_attrs=a_attrs))
        push!(weights, isnothing(weight_col) ? 1.0 : Float64(row[weight_col]))
    end

    return EgoData(egos; sampling_weights=weights,
                   population_size=population_size)
end

"""
    ego_design(ed::EgoData; ppopsize=nothing, weights=nothing) -> EgoData

Attach survey-design information (population size and/or per-ego weights)
to ego data.
"""
function ego_design(ed::EgoData{T};
                    ppopsize::Union{Int, Nothing}=nothing,
                    weights::Union{Vector{Float64}, Nothing}=nothing) where T
    new_weights = isnothing(weights) ? ed.sampling_weights : weights

    return EgoData(ed.egos;
                   population_size=something(ppopsize, ed.population_size, Some(nothing)),
                   sampling_weights=new_weights,
                   design=copy(ed.design))
end

# =============================================================================
# Ego-Specific Terms
# =============================================================================
#
# Each ego term is a per-capita statistic: compute(term, ed) returns the
# design-weighted mean per-ego contribution h̄. The population target for
# a network of size m is m·h̄, and each term maps to the ERGM.jl term whose
# sufficient statistic it estimates:
#
#   term            per-ego contribution h_i        ERGM term
#   EgoEdges        degree_i / 2                    Edges()
#   EgoNodeMatch    (matching alters)_i / 2         NodeMatch(attr)
#   EgoTriangle     (alter-alter ties)_i / 3        Triangle()
#   EgoGWDegree     e^α(1−(1−e^−α)^degree_i)        GWDegree(α)
#
# EgoDegree(d) is a descriptive statistic (proportion of egos with degree
# d); ERGM.jl has no degree-count term, so it cannot be used in ergm_ego.

abstract type EgoTerm <: AbstractERGMTerm end

_wmean(values, weights) = sum(values .* weights) / sum(weights)

"""
    EgoEdges <: EgoTerm

Per-capita edge statistic: the design-weighted mean of `degree/2` over
egos. Scaled by the pseudo-population size this estimates the `edges`
sufficient statistic.
"""
struct EgoEdges <: EgoTerm end

name(::EgoEdges) = "ego.edges"

_ego_contribution(::EgoEdges, e::EgoNetwork) = ego_degree(e) / 2

"""
    EgoNodeMatch(attr) <: EgoTerm

Per-capita homophily statistic: the design-weighted mean of half the
number of alters whose `attr` matches the ego's. Estimates the
`nodematch(attr)` sufficient statistic.
"""
struct EgoNodeMatch <: EgoTerm
    attr::Symbol
end

name(t::EgoNodeMatch) = "ego.nodematch.$(t.attr)"

function _ego_contribution(t::EgoNodeMatch, e::EgoNetwork)
    haskey(e.ego_attrs, t.attr) || return 0.0
    haskey(e.alter_attrs, t.attr) || return 0.0
    ego_val = e.ego_attrs[t.attr]
    return count(==(ego_val), e.alter_attrs[t.attr]) / 2
end

"""
    EgoTriangle <: EgoTerm

Per-capita triangle statistic: the design-weighted mean of
`(alter–alter ties)/3` (each population triangle appears in the local view
of each of its three vertices). Estimates the `triangle` sufficient
statistic.
"""
struct EgoTriangle <: EgoTerm end

name(::EgoTriangle) = "ego.triangle"

_ego_contribution(::EgoTriangle, e::EgoNetwork) = n_alter_ties(e) / 3

"""
    EgoGWDegree(decay) <: EgoTerm

Per-capita geometrically weighted degree: the design-weighted mean of
`e^α(1 − (1 − e^{−α})^degree)`. Estimates the `gwdegree(decay, fixed=TRUE)`
sufficient statistic.
"""
struct EgoGWDegree <: EgoTerm
    decay::Float64

    function EgoGWDegree(decay::Float64=0.5)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        new(decay)
    end
end

name(t::EgoGWDegree) = "ego.gwdegree.$(t.decay)"

function _ego_contribution(t::EgoGWDegree, e::EgoNetwork)
    d = ego_degree(e)
    α = t.decay
    return d > 0 ? exp(α) * (1 - (1 - exp(-α))^d) : 0.0
end

"""
    EgoDegree(d) <: EgoTerm

Descriptive statistic: the design-weighted proportion of egos with degree
exactly `d`. Not usable in `ergm_ego` (ERGM.jl has no degree-count term).
"""
struct EgoDegree <: EgoTerm
    d::Int
end

name(t::EgoDegree) = "ego.degree.$(t.d)"

_ego_contribution(t::EgoDegree, e::EgoNetwork) = Float64(ego_degree(e) == t.d)

"""
    compute(term::EgoTerm, ed::EgoData) -> Float64

The design-weighted mean per-ego contribution of the term (a per-capita
statistic; multiply by a network size to get a target sufficient
statistic — see [`ego_target_stats`](@ref)).
"""
function compute(term::EgoTerm, ed::EgoData)
    h = [_ego_contribution(term, e) for e in ed.egos]
    return _wmean(h, ed.sampling_weights)
end

"""
    ego_mixing_matrix(ed::EgoData, attr::Symbol) -> (levels, matrix)

Design-weighted ego–alter mixing counts for a categorical attribute:
`matrix[a, b]` is the weighted number of ego–alter pairs with ego level
`levels[a]` and alter level `levels[b]`.
"""
function ego_mixing_matrix(ed::EgoData, attr::Symbol)
    levels = Any[]
    for e in ed.egos
        haskey(e.ego_attrs, attr) && push!(levels, e.ego_attrs[attr])
        haskey(e.alter_attrs, attr) && append!(levels, e.alter_attrs[attr])
    end
    levels = sort(unique(levels))
    index = Dict(l => k for (k, l) in enumerate(levels))

    mix = zeros(length(levels), length(levels))
    for (e, w) in zip(ed.egos, ed.sampling_weights)
        (haskey(e.ego_attrs, attr) && haskey(e.alter_attrs, attr)) || continue
        a = index[e.ego_attrs[attr]]
        for v in e.alter_attrs[attr]
            mix[a, index[v]] += w
        end
    end

    return (levels=levels, matrix=mix)
end

# Mapping from ego terms to the ERGM terms whose sufficient statistics
# they estimate
_ergm_term(::EgoEdges) = Edges()
_ergm_term(t::EgoNodeMatch) = NodeMatch(t.attr)
_ergm_term(::EgoTriangle) = Triangle()
_ergm_term(t::EgoGWDegree) = GWDegree(t.decay)
_ergm_term(t::EgoTerm) =
    throw(ArgumentError("$(name(t)) is a descriptive statistic with no " *
                        "ERGM.jl counterpart; it cannot be used in ergm_ego"))

"""
    ego_target_stats(terms, ed::EgoData, m::Int) -> Vector{Float64}

Target sufficient statistics for a network of size `m`: `m` times the
design-weighted per-capita ego statistics.
"""
ego_target_stats(terms, ed::EgoData, m::Int) =
    [m * compute(t, ed) for t in terms]

# =============================================================================
# Model and Estimation
# =============================================================================

"""
    EgoERGMModel

Specification of an egocentric ERGM fit: ego terms, their ERGM
counterparts, the pseudo-population network, and target statistics.
"""
struct EgoERGMModel
    ego_terms::Vector{EgoTerm}
    ergm_terms::Vector{AbstractERGMTerm}
    data::EgoData
    ppopsize::Int
    popsize::Int
    targets::Vector{Float64}
end

"""
    EgoERGMResult

Results from [`fit_ergm_ego`](@ref).

# Fields
- `coefficients`: Population-scale coefficients (the edges coefficient
  includes the network-size adjustment `−log(popsize/ppopsize)`)
- `std_errors`: Standard errors combining the model-based and
  survey-design variance components
- `vcov`: Estimated covariance matrix of the coefficients,
  `I⁻¹ + I⁻¹ Σ_design I⁻¹`
- `netsize_adjustment`: The `−log(popsize/ppopsize)` adjustment applied to
  the edges coefficient
- `converged`: Whether moment matching converged
- `sim_stats`: Statistics sampled at the fitted coefficients (pseudo-
  population scale)
"""
struct EgoERGMResult
    model::EgoERGMModel
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    netsize_adjustment::Float64
    converged::Bool
    sim_stats::Matrix{Float64}
end

# Two-sided normal p-values via the complementary CDF (the naive
# 2(1 − cdf) form underflows to exactly 0 beyond |z| ≈ 8.3); NaN standard
# errors give NaN p-values, which the shared printer renders as "NaN"
_z_pvalues(z::AbstractVector{Float64}) = 2 .* ccdf.(Normal(), abs.(z))

function Base.show(io::IO, result::EgoERGMResult)
    println(io, "Egocentric ERGM Results")
    println(io, "=======================")
    println(io, "Egos: $(length(result.model.data)); pseudo-population: " *
                "$(result.model.ppopsize); population: $(result.model.popsize)")
    println(io, "Netsize adjustment (edges): $(round(result.netsize_adjustment, digits=4))")
    println(io, "Converged: $(result.converged)")
    println(io)
    println(io, "Coefficients (population scale):")
    z = result.coefficients ./ result.std_errors
    print_coeftable(io, [name(term) for term in result.model.ego_terms],
                    result.coefficients, result.std_errors, _z_pvalues(z);
                    z_values=z)
end

# StatsAPI interface: methods on the shared statistics generics (mirroring
# ERGM.jl), so `coef(fit)` etc. work on egocentric fits too
StatsAPI.coef(result::EgoERGMResult) = result.coefficients
StatsAPI.stderror(result::EgoERGMResult) = result.std_errors
StatsAPI.vcov(result::EgoERGMResult) = result.vcov

# Build the pseudo-population network: m vertices whose attributes are
# ego attributes replicated proportionally to the sampling weights, with
# edges seeded at the target density
function _pseudo_population(ed::EgoData, m::Int, target_density::Float64,
                            rng::Random.AbstractRNG)
    net = network(m; directed=false)

    # Replicate egos proportionally to weight (largest-remainder rounding)
    n_egos = length(ed.egos)
    w = ed.sampling_weights ./ sum(ed.sampling_weights)
    counts = floor.(Int, w .* m)
    remainder = m - sum(counts)
    order = sortperm(w .* m .- counts; rev=true)
    for k in 1:remainder
        counts[order[k]] += 1
    end

    # Assign ego attributes to pseudo-population vertices
    attrs = Dict{Symbol, Dict{Int, Any}}()
    v = 0
    for (i, e) in enumerate(ed.egos)
        for _ in 1:counts[i]
            v += 1
            for (attr, val) in e.ego_attrs
                get!(attrs, attr, Dict{Int, Any}())[v] = val
            end
        end
    end
    for (attr, vals) in attrs
        set_vertex_attribute!(net, attr, vals)
    end

    # Seed edges at approximately the target density
    p = clamp(target_density, 1e-4, 0.5)
    for i in 1:m, j in (i+1):m
        rand(rng) < p && add_edge!(net, i, j)
    end

    return net
end

"""
    fit_ergm_ego(ed::EgoData, terms::Vector{<:EgoTerm}; kwargs...) -> EgoERGMResult

Fit an ERGM to egocentrically sampled data, following `ergm.ego`:

1. Compute design-weighted **target statistics** scaled to a
   pseudo-population of size `ppopsize`.
2. Build a pseudo-population network with ego attributes replicated
   proportionally to the sampling weights.
3. Fit coefficients by **MCMC moment matching** (Newton iterations on
   `targets − E_θ[g]`, the method-of-moments estimator that `ergm` uses
   for `target.stats`).
4. Apply the **network-size adjustment** `−log(popsize/ppopsize)` to the
   edges coefficient so it is on the population scale.

Standard errors combine the inverse-information (model) component with the
survey-design variance of the targets:
`V(θ̂) = I⁻¹ + I⁻¹ Σ_design I⁻¹`.

[`ergm_ego`](@ref) is the R-faithful alias (matching the `ergm.ego`
package); `fit_ego_ergm` is a legacy alias.

# Keyword Arguments
- `ppopsize::Int`: Pseudo-population size (default: `popsize` if known and
  ≤ 1000, otherwise `10 ×` the number of egos)
- `popsize::Union{Int,Nothing}`: Population size for the offset (default:
  `ed.population_size`, falling back to `ppopsize`, i.e. no adjustment)
- `n_samples, burnin, interval, max_iter, tol`: MCMC moment-matching controls
- `rng`: Random number generator
"""
function fit_ergm_ego(ed::EgoData, terms::Vector{<:EgoTerm};
                      ppopsize::Union{Int, Nothing}=nothing,
                      popsize::Union{Int, Nothing}=nothing,
                      n_samples::Int=400,
                      burnin::Int=2000,
                      interval::Int=20,
                      max_iter::Int=25,
                      tol::Float64=0.05,
                      rng::Random.AbstractRNG=Random.default_rng())
    isempty(terms) && throw(ArgumentError("need at least one term"))
    any(t -> t isa EgoEdges, terms) ||
        throw(ArgumentError("the model must include EgoEdges() (as ergm.ego models include edges)"))

    N = something(popsize, ed.population_size, Some(nothing))
    m = if !isnothing(ppopsize)
        ppopsize
    elseif !isnothing(N) && N <= 1000
        N
    else
        10 * length(ed.egos)
    end
    N = something(N, m)
    m >= 5 || throw(ArgumentError("pseudo-population size too small"))

    ego_terms = collect(EgoTerm, terms)
    ergm_terms = AbstractERGMTerm[_ergm_term(t) for t in ego_terms]
    p = length(ego_terms)

    # Target statistics on the pseudo-population scale
    targets = ego_target_stats(ego_terms, ed, m)

    edges_idx = findfirst(t -> t isa EgoEdges, ego_terms)
    n_dyads = m * (m - 1) / 2
    target_density = targets[edges_idx] / n_dyads
    target_density < 1 ||
        throw(ArgumentError("target mean degree implies density ≥ 1; increase ppopsize"))

    net = _pseudo_population(ed, m, target_density, rng)
    model = ERGMModel(ERGMFormula(ergm_terms), net)

    # Initialize: edges at logit of target density, others at 0
    θ = zeros(p)
    θ[edges_idx] = log(target_density / (1 - target_density))

    # MCMC moment matching (Newton on targets − E[g])
    converged = false
    samples = Matrix{Float64}(undef, 0, p)
    for iter in 1:max_iter
        samples = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                            interval=interval, rng=rng).stats
        mean_stats = vec(mean(samples, dims=1))
        diff = targets .- mean_stats

        if maximum(abs.(diff) ./ max.(abs.(targets), 1.0)) < tol
            converged = true
            break
        end

        cov_stats = cov(samples)
        step = try
            cov_stats \ diff
        catch
            break
        end
        # Damp large steps for stability
        maxstep = maximum(abs.(step))
        maxstep > 1.0 && (step .*= 1.0 / maxstep)
        θ .+= step
    end

    # Final sample at the fitted coefficients
    samples = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                        interval=interval, rng=rng).stats

    # Variance: model component I⁻¹ plus design component I⁻¹ Σ_t I⁻¹,
    # where Σ_t is the survey variance of the target statistics
    I_mat = cov(samples)
    Σ_t = _design_cov(ego_terms, ed, m)
    vcov_θ = try
        Iinv = inv(Symmetric(I_mat))
        Matrix(Iinv) .+ Matrix(Iinv) * Σ_t * Matrix(Iinv)
    catch
        fill(NaN, p, p)
    end
    se = sqrt.(abs.(diag(vcov_θ)))

    # Network-size adjustment: put the edges coefficient on the
    # population (size N) scale
    adjustment = -log(N / m)
    coefficients = copy(θ)
    coefficients[edges_idx] += adjustment

    ego_model = EgoERGMModel(ego_terms, ergm_terms, ed, m, N, targets)
    return EgoERGMResult(ego_model, coefficients, se, vcov_θ, adjustment, converged, samples)
end

"""
    ergm_ego(ed::EgoData, terms; kwargs...)

R-faithful alias for [`fit_ergm_ego`](@ref) (the same function), matching
the R `ergm.ego` package name.
"""
const ergm_ego = fit_ergm_ego

"""
    fit_ego_ergm(ed::EgoData, terms; kwargs...)

Alias for [`fit_ergm_ego`](@ref), kept for backward compatibility.
"""
const fit_ego_ergm = fit_ergm_ego

# Survey-design covariance of the target statistics: targets are
# m·(weighted mean of per-ego contributions), so
# V(target) = m² · V_w(h̄) with the standard weighted-mean variance
function _design_cov(terms, ed::EgoData, m::Int)
    n = length(ed.egos)
    p = length(terms)
    w = ed.sampling_weights ./ sum(ed.sampling_weights)

    H = Matrix{Float64}(undef, n, p)
    for (j, t) in enumerate(terms), (i, e) in enumerate(ed.egos)
        H[i, j] = _ego_contribution(t, e)
    end
    h̄ = vec(sum(H .* w, dims=1))

    Σ = zeros(p, p)
    for i in 1:n
        d = H[i, :] .- h̄
        Σ .+= (w[i]^2) .* (d * d')
    end
    return (m^2) .* Σ
end

# =============================================================================
# Population Size Estimation
# =============================================================================

"""
    estimate_popsize(ed::EgoData; method=:horvitz_thompson) -> Float64

Estimate the population size from an ego sample.

- `:horvitz_thompson`: The sum of the sampling weights. Only meaningful
  when the weights are inverse inclusion probabilities; with unit weights
  this is just the number of egos.
- `:capture_recapture`: Two-sample Lincoln–Petersen using alter overlap —
  the egos are split in half; with `n₁`/`n₂` the distinct alters named in
  each half and `m` the overlap, `N̂ = n₁·n₂/m`. Requires globally
  meaningful alter IDs (preserved by `as_egodata`).
"""
function estimate_popsize(ed::EgoData; method::Symbol=:horvitz_thompson)
    if method == :horvitz_thompson
        return sum(ed.sampling_weights)
    elseif method == :capture_recapture
        n = length(ed.egos)
        n >= 2 || throw(ArgumentError("need at least two egos"))
        half = n ÷ 2
        s1 = Set{Int}()
        s2 = Set{Int}()
        for (i, e) in enumerate(ed.egos)
            target = i <= half ? s1 : s2
            for a in e.alters
                push!(target, Int(a))
            end
        end
        (isempty(s1) || isempty(s2)) &&
            throw(ArgumentError("both halves must contain alters"))
        overlap = length(intersect(s1, s2))
        overlap > 0 ||
            throw(ArgumentError("no alter overlap between sample halves; " *
                                "capture-recapture requires shared alter IDs"))
        return length(s1) * length(s2) / overlap
    else
        throw(ArgumentError("Unknown method: $method"))
    end
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_ego_sample(net::Network, n_egos::Int;
                        ego_attrs=Symbol[], rng=Random.default_rng()) -> EgoData

Draw an egocentric sample from a complete (undirected) network: sample
`n_egos` egos uniformly without replacement and record each ego's alters,
the ties among those alters, and the requested vertex attributes for ego
and alters. Alter IDs are the network's vertex IDs.
"""
function simulate_ego_sample(net, n_egos::Int;
                             ego_attrs::Vector{Symbol}=Symbol[],
                             rng::Random.AbstractRNG=Random.default_rng())
    n = Int(nv(net))
    n_egos <= n || throw(ArgumentError("cannot sample more egos than vertices"))

    ego_ids = sample(rng, 1:n, n_egos; replace=false)
    egos = EgoNetwork{Int}[]

    for eid in ego_ids
        alters = sort(collect(neighbors(net, eid)))
        n_a = length(alters)

        ties = zeros(Bool, n_a, n_a)
        for a in 1:n_a, b in (a+1):n_a
            if has_edge(net, alters[a], alters[b])
                ties[a, b] = true
                ties[b, a] = true
            end
        end

        e_attrs = Dict{Symbol, Any}()
        a_attrs = Dict{Symbol, Vector}()
        for attr in ego_attrs
            vals = get_vertex_attribute(net, attr)
            haskey(vals, eid) && (e_attrs[attr] = vals[eid])
            a_attrs[attr] = [get(vals, a, missing) for a in alters]
        end

        push!(egos, EgoNetwork(Int(eid), Int.(alters), ties;
                               ego_attrs=e_attrs, alter_attrs=a_attrs))
    end

    return EgoData(egos; population_size=n)
end

# =============================================================================
# Diagnostics
# =============================================================================

"""
    ego_gof(result::EgoERGMResult; n_sim=50, rng=Random.default_rng()) -> NamedTuple

Goodness of fit for an egocentric ERGM: simulate pseudo-population
networks at the fitted (pseudo-population scale) coefficients, take ego
samples of the observed size from each, and compare the observed
design-weighted mean degree and mean alter-tie count against their
simulated distributions (two-sided Monte Carlo p-values).
"""
function ego_gof(result::EgoERGMResult; n_sim::Int=50,
                 rng::Random.AbstractRNG=Random.default_rng())
    model = result.model
    ed = model.data
    m = model.ppopsize
    n_egos = length(ed)

    # Coefficients on the pseudo-population scale (undo the adjustment)
    θ = copy(result.coefficients)
    edges_idx = findfirst(t -> t isa EgoEdges, model.ego_terms)
    θ[edges_idx] -= result.netsize_adjustment

    net = _pseudo_population(ed, m, model.targets[edges_idx] / (m * (m - 1) / 2), rng)
    ergm_model = ERGMModel(ERGMFormula(model.ergm_terms), net)
    sims = sample_networks(ergm_model, θ; n_sim=n_sim, burnin=2000, interval=200)

    obs = summary_stats(ed)
    sim_mean_degree = Float64[]
    sim_mean_aaties = Float64[]
    for s in sims
        sample_ed = simulate_ego_sample(s, min(n_egos, Int(nv(s))); rng=rng)
        ss = summary_stats(sample_ed)
        push!(sim_mean_degree, ss.mean_degree)
        push!(sim_mean_aaties, ss.mean_alter_ties)
    end

    mc_p = (sim, o) -> min(1.0, 2.0 * min(mean(sim .>= o), mean(sim .<= o)))

    return (
        observed = (mean_degree = obs.mean_degree,
                    mean_alter_ties = obs.mean_alter_ties),
        simulated = (mean_degree = mean(sim_mean_degree),
                     mean_alter_ties = mean(sim_mean_aaties)),
        p_values = (mean_degree = mc_p(sim_mean_degree, obs.mean_degree),
                    mean_alter_ties = mc_p(sim_mean_aaties, obs.mean_alter_ties)),
        n_sim = n_sim
    )
end

end # module
