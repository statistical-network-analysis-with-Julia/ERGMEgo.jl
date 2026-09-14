"""
    ERGMEgo.jl - ERGMs for Ego-Centric Network Data

Fits ERGMs to egocentrically sampled network data (a sample of "egos" with
their local networks: alters and ties among alters), enabling inference
about complete-network properties from ego samples.

The methodology follows R `ergm.ego` (Krivitsky & Morris 2017): ego
statistics are design-weighted and scaled to **target statistics** for a
pseudo-population network of size `ppopsize`, an ERGM is fit to those
targets by method-of-moments (MCMC moment matching, with ERGM.jl's t-ratio
and Hotelling T² convergence tests), and the edges coefficient receives the
network-size adjustment `−log(popsize/ppopsize)` to put it on the population
scale. Standard errors are `ergm.ego`'s decomposition: the survey-design
variance of the targets sandwiched by the inverse information, plus the
Monte-Carlo estimation term of the moment equations.

Port of the R ergm.ego package from the StatNet collection.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)                       # 205 students, 203 ties
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))   # a census
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                     # true
coef(fit)                         # ≈ [-6.02, 2.82], ergm.ego's population-scale estimates
gof(fit; n_sim=20, rng=Xoshiro(2)) # a Networks.GOFResult on mean degree / mean alter ties
```
"""
module ERGMEgo

using DataFrames
using ERGM
using LinearAlgebra
using Networks          # graph primitives (nv, neighbors, has_edge, vertices) are its re-exports
using Random
using SpecialFunctions: erfcinv
using Statistics
using StatsBase

# The shared statistic protocol (Networks.jl `src/statistics.jl`): `name` and
# `compute` are the ONE pair of generics every model package extends
# (`ERGMEgo.name === Networks.name`); `summary_stats` is ERGM.jl's
import Networks: name, compute
import ERGM: summary_stats
# ERGM.jl's public MCMLE convergence machinery (t-ratios, Hotelling T²,
# Geyer effective sample size) and its one dyad-scaled sampler rule, reused
# by the moment-matching loop instead of an ad-hoc relative-change rule
import ERGM: mcmc_convergence, MCMLEConvergence, _mcmc_defaults
# Shared presentation infrastructure (Networks.jl): the ONE `gof` generic all
# model packages extend, the common coefficient table and printer, the ONE
# z → p helper, and the GOF containers
import Networks: gof, print_coeftable, CoefficientTable, z_pvalues,
                 GOFStatistic, GOFResult

# The shared result-metadata protocol (Networks.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMEgo adds methods for `EgoERGMResult`; `fit_metadata(fit)` collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 approximations
import StatsAPI
import StatsAPI: coef, stderror, vcov, confint, nobs, dof, coeftable

# Data structures
export EgoData, EgoNetwork
export n_alters, ego_degree, alter_degree, n_alter_ties

# Data preparation
export as_egodata, ego_design

# Ego-specific terms and statistics. `compute` and `name` are the shared
# Networks.jl statistic generics (re-exported, as ERGM.jl re-exports them),
# so `compute(EgoEdges(), ed)` works with `using ERGMEgo` alone. `EgoTerm`
# is exported as ERGM.jl exports `AbstractERGMTerm`: it is the type in
# `fit_ergm_ego`'s signature and what a custom ego term subtypes
export EgoTerm, EgoEdges, EgoNodeMatch, EgoDegree, EgoGWDegree, EgoTriangle
export ego_mixing_matrix, ego_target_stats
export summary_stats, compute, name

# Documented internals: THE MCMC budget rule (API reference, "Estimation")
# and the ego-term → ERGM-term hook a custom fittable term extends (API
# reference, "Terms"), public in the sense ERGM.jl declares `_mcmc_defaults`
# public — stable names a user may call or extend and a test may pin, not
# exports
public _mcmc_controls, _ergm_term

# Estimation
export fit_ergm_ego, ergm_ego, fit_ego_ergm, EgoERGMModel, EgoERGMResult

# Population size estimation
export estimate_popsize

# Simulation
export simulate_ego_sample

# Diagnostics: gof is a method of the shared Networks.jl generic; ego_gof is
# the legacy NamedTuple-returning form
export gof, ego_gof

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using
# ERGMEgo`). Deliberately absent: `loglikelihood`, `aic`, `bic` — the fit is
# moment matching (`objective(fit) == :moment`); no likelihood is evaluated.
export coef, stderror, vcov, confint, coeftable, nobs, dof

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

`alter_ties` must be square (`n_alters × n_alters`) and symmetric (ego
networks are undirected); an asymmetric matrix is an `ArgumentError`. So is
an alter listed twice (a duplicated ego–alter row in a survey's wide-to-long
reshape, which would double the ego's degree and its homophily count in
every target) and a `true` on the diagonal of `alter_ties` (an alter tied
to itself, which `n_alter_ties` would silently drop by integer division):
each is refused naming the ego and the alter. The constructor
`EgoNetwork(ego, alters, alter_ties; ego_attrs, alter_attrs)` infers `T`
from `ego`.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12], Bool[0 1 0; 1 0 0; 0 0 0];
               ego_attrs=Dict{Symbol,Any}(:group => "A"),
               alter_attrs=Dict{Symbol,Vector}(:group => ["A", "B", "A"]))
n_alters(e), n_alter_ties(e)      # (3, 1)
e                                 # EgoNetwork{Int64}: ego 1, 3 alters, 1 alter–alter tie; ...
try EgoNetwork(2, [10, 11], Bool[0 1; 0 0]) catch e; e isa ArgumentError end   # true — ArgumentError: alter_ties must be symmetric
try EgoNetwork(3, [10, 10], zeros(Bool, 2, 2)) catch e; e isa ArgumentError end   # true — ArgumentError: ego 3 lists alter 10 twice
try EgoNetwork(4, [10, 11], Bool[1 0; 0 0]) catch e; e isa ArgumentError end   # true — ArgumentError: self-tie on the diagonal
```
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
        # A duplicated alter (the wide-to-long survey mistake) would double
        # the ego's degree and its nodematch contribution in every target
        # and in the design variance; refused by name rather than counted
        if !allunique(alters)
            dup = first(a for a in alters if count(==(a), alters) > 1)
            throw(ArgumentError("EgoNetwork: ego $ego lists alter $dup twice " *
                                "(alters must be distinct; a duplicated ego–alter row " *
                                "would double the ego's degree in every statistic). " *
                                "Deduplicate the alter list."))
        end
        # A diagonal entry is an alter tied to itself: `n_alter_ties` would
        # drop it by integer division instead of counting it, so it is refused
        for i in 1:n_alters
            alter_ties[i, i] && throw(ArgumentError(
                "EgoNetwork: alter_ties has a self-tie on the diagonal for ego $ego " *
                "(alter $(alters[i]) tied to itself); alter–alter ties are between " *
                "distinct alters. Clear the diagonal."))
        end
        new{T}(ego, alters, alter_ties, ego_attrs, alter_attrs)
    end
end

EgoNetwork(ego::T, alters::Vector{T}, alter_ties::Matrix{Bool}; kwargs...) where T =
    EgoNetwork{T}(ego, alters, alter_ties; kwargs...)

"""
    n_alters(ego_net::EgoNetwork) -> Int

Number of alters in an ego network.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3))
n_alters(e)          # 3
```
"""
n_alters(ego_net::EgoNetwork) = length(ego_net.alters)

"""
    ego_degree(ego_net::EgoNetwork) -> Int

The ego's degree (number of alters) — the same number as [`n_alters`](@ref),
under the name the ERGM literature uses.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
ego_degree(e)        # 2
ego_degree(EgoNetwork(2, Int[], Matrix{Bool}(undef, 0, 0)))   # 0 — an isolate ego
```
"""
ego_degree(ego_net::EgoNetwork) = n_alters(ego_net)

"""
    alter_degree(ego_net::EgoNetwork) -> Vector{Int}

Degree of each alter within the ego network (not counting ego): the row
sums of `alter_ties`, in the order of `alters`.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12, 13], Bool[0 1 1 0; 1 0 0 0; 1 0 0 0; 0 0 0 0])
alter_degree(e)      # [2, 1, 1, 0] — alter 10 is tied to 11 and 12
```
"""
alter_degree(ego_net::EgoNetwork) = vec(sum(ego_net.alter_ties, dims=2))

"""
    n_alter_ties(ego_net::EgoNetwork) -> Int

Number of (undirected) ties among alters — half the number of `true`
entries of the symmetric `alter_ties` matrix.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12], Bool[0 1 1; 1 0 0; 1 0 0])
n_alter_ties(e)      # 2 — the ties 10–11 and 10–12
```
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

`EgoData(egos; population_size, sampling_weights, design)` defaults to unit
weights (one per ego; a weight vector of another length is an
`ArgumentError`). Every weight must be a finite, non-negative number and
the weights must sum to a positive number: a `NaN`, `Inf` or negative case
weight (an `NA` inclusion probability, a `1/0`), or an all-zero vector, is
an `ArgumentError` naming the ego and the value — as `ergm.ego`/`survey`
refuse — never a `NaN` target or a coefficient computed from a
negative-weight Hájek mean. An `EgoData` iterates and indexes as its vector
of egos.
The design the weights describe is **independent egos with case weights** —
there is no field for strata, clusters, a finite-population correction or
replicate weights, and the design variance of a fit encodes none (see
[`fit_ergm_ego`](@ref)).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [11], zeros(Bool, 1, 1))
ed = EgoData([e1, e2]; population_size=50, sampling_weights=[2.0, 1.0])
length(ed), ed[2].ego                 # (2, 2)
[ego_degree(e) for e in ed]           # [2, 1]
summary_stats(ed).mean_degree         # 1.6667 — the weighted mean (2·2 + 1·1)/3
try EgoData([e1, e2]; sampling_weights=[1.0, -1.0]) catch e; e isa ArgumentError end   # true — ArgumentError naming ego 2 and the weight
```
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
        _check_sampling_weights(egos, sw)
        new{T}(egos, population_size, sw, design)
    end
end

# The one validation of the case weights, at construction — so no routine
# downstream (`compute`, `ego_target_stats`, `_design_cov`, the
# pseudo-population) can see a NaN/Inf/negative weight and return a number
# computed around it (a NaN target used to surface as the unrelated
# "density ≥ 1" error; a negative weight ran the whole MCMC and RETURNED
# coefficients from a negative-weight Hájek mean). Zero weights are legal
# individually (an ego that contributes nothing), not all together.
function _check_sampling_weights(egos::Vector, sw::AbstractVector{Float64})
    for (i, w) in enumerate(sw)
        if !isfinite(w) || w < 0
            what = isnan(w) ? "NaN" : !isfinite(w) ? "$w" : "negative ($w)"
            throw(ArgumentError(
                "EgoData: the sampling weight of ego $i (id $(egos[i].ego)) is $what; " *
                "every sampling weight must be a finite, non-negative number " *
                "(an inverse inclusion probability or a case weight). Check the " *
                "weight column for NA/Inf/negative entries before building the data."))
        end
    end
    if !isempty(sw) && sum(sw) <= 0
        throw(ArgumentError(
            "EgoData: the sampling weights sum to $(sum(sw)) (all " *
            "$(_plural(length(sw), "weight")) are zero); they must sum to a positive " *
            "number — no design-weighted statistic is defined otherwise. Supply " *
            "positive weights, or omit them for unit weights."))
    end
    return nothing
end

Base.length(ed::EgoData) = length(ed.egos)
Base.iterate(ed::EgoData, state=1) = state > length(ed) ? nothing : (ed.egos[state], state + 1)
Base.getindex(ed::EgoData, i) = ed.egos[i]

# The design-weighted median: the 0.5 quantile of the weight-estimated
# (Horvitz–Thompson) distribution of `x`, with the midpoint convention at an
# exact half so that unit weights reproduce `Statistics.median`. Not
# StatsBase's `median(x, Weights(w))`, which interpolates between values
# (2.75 for degrees [3, 2, 4] under weights [2, 1, 1]) — a number no ego has.
function _weighted_median(x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    isempty(x) && return NaN
    order = sortperm(x)
    total = sum(w)
    total > 0 || throw(ArgumentError("summary_stats: the sampling weights must sum to a positive number"))
    # A relative tolerance on "exactly half", so scaled unit weights (1/n
    # each) hit the midpoint convention the way integer ones do
    half = total / 2
    tol = sqrt(eps(Float64)) * total
    cum = 0.0
    lower = upper = NaN
    for k in order
        cum += w[k]
        if isnan(lower) && cum >= half - tol
            lower = Float64(x[k])
        end
        if cum > half + tol
            upper = Float64(x[k])
            break
        end
    end
    return (lower + upper) / 2
end

"""
    summary_stats(ed::EgoData) -> NamedTuple

Design-weighted summary statistics for ego data: `n_egos`, the weighted
`mean_degree` and weighted `median_degree` (the 0.5 quantile of the
weight-estimated degree distribution with the midpoint convention, so it
reduces to `Statistics.median` under unit weights and both centre statistics
honour the same sampling weights), `min_degree`, `max_degree`, the weighted
`mean_alter_ties`, the unweighted `total_alters` (the number of ego–alter
rows) and `population_size`. An `EgoData` with no egos (an `as_egodata`
whose ego frame filtered down to zero rows, say) is an `ArgumentError` that
says so, not a bare "reducing over an empty collection".

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], Bool[0 1 0; 1 0 0; 0 0 0])
e2 = EgoNetwork(2, [10, 13], Bool[0 0; 0 0])
e3 = EgoNetwork(3, [11, 14, 15, 16], zeros(Bool, 4, 4))
ed = ego_design(EgoData([e1, e2, e3]); weights=[1.0, 1.0, 4.0])
summary_stats(ed).mean_degree      # 3.5 — (3 + 2 + 4·4)/6
summary_stats(ed).median_degree    # 4.0 — the design-weighted median; the unweighted one is 3.0
```
"""
function summary_stats(ed::EgoData)
    isempty(ed.egos) && throw(ArgumentError(
        "summary_stats: the EgoData has no egos (an ego_df with zero rows, or a " *
        "filter that removed every ego?); there is no degree distribution to summarise"))
    w = Weights(ed.sampling_weights)
    degrees = [Float64(ego_degree(e)) for e in ed.egos]
    alter_ties = [Float64(n_alter_ties(e)) for e in ed.egos]

    return (
        n_egos = length(ed.egos),
        mean_degree = mean(degrees, w),
        # Design-weighted like the mean: a descriptive that ignored the
        # weights beside one that honoured them was two estimands under one
        # heading
        median_degree = _weighted_median(degrees, ed.sampling_weights),
        min_degree = minimum(degrees),
        max_degree = maximum(degrees),
        mean_alter_ties = mean(alter_ties, w),
        total_alters = sum(degrees),
        population_size = ed.population_size
    )
end

# One-line `show` methods in the style of ERGM.jl's `ERGMModel` (`Type{T}:
# counts; details`), so an ego network, a data set and a model specification
# each say what they hold instead of dumping their fields
_plural(n::Integer, noun::AbstractString) = "$n $noun$(n == 1 ? "" : "s")"

function Base.show(io::IO, e::EgoNetwork{T}) where T
    print(io, "EgoNetwork{$T}: ego $(e.ego), $(_plural(n_alters(e), "alter")), ",
          _plural(n_alter_ties(e), "alter–alter tie"))
    ego_attrs = sort!(collect(keys(e.ego_attrs)))
    alter_attrs = sort!(collect(keys(e.alter_attrs)))
    print(io, "; ego attributes: ", isempty(ego_attrs) ? "none" : join(ego_attrs, ", "),
          "; alter attributes: ", isempty(alter_attrs) ? "none" : join(alter_attrs, ", "))
    return nothing
end

function Base.show(io::IO, ed::EgoData{T}) where T
    n = length(ed.egos)
    print(io, "EgoData{$T}: ", _plural(n, "ego"))
    if n > 0
        degrees = [Float64(ego_degree(e)) for e in ed.egos]
        print(io, ", weighted mean degree ",
              round(mean(degrees, Weights(ed.sampling_weights)); digits=3))
    end
    print(io, "; population size ",
          isnothing(ed.population_size) ? "unknown" : string(ed.population_size))
    if all(==(1.0), ed.sampling_weights)
        print(io, "; unit weights")
    else
        print(io, "; sampling weights summing to ", round(sum(ed.sampling_weights); digits=3))
    end
    return nothing
end

# =============================================================================
# Data Preparation
# =============================================================================

# The one column check of `as_egodata`: an ArgumentError naming the column,
# the keyword that named it, the frame it was looked up in, and the columns
# that frame actually has
function _require_column(df::DataFrame, col::Symbol, frame::AbstractString,
                         keyword::AbstractString)
    string(col) in names(df) && return nothing
    throw(ArgumentError("as_egodata: column :$col ($keyword) not found in $frame; " *
                        "its columns are $(join(names(df), ", "))"))
end

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

Every column the keywords name must exist in its frame: a missing
`ego_id`/`alter_id`, ego or alter attribute, weight, or alter-tie column is an
`ArgumentError` naming the column, the frame (`ego_df`, `alter_df`,
`aatie_df`) and the columns that frame does have, instead of a `KeyError`
from deep inside the row loop. A duplicated `(ego, alter)` row in `alter_df`
(the wide-to-long reshape mistake, which would double that ego's degree in
every statistic) and an `aatie_df` row tying an alter to itself are
likewise `ArgumentError`s naming the ego and the alter; so is a weight
column holding a `NaN`, `Inf` or negative entry (see [`EgoData`](@ref)).

# Example
```julia
using ERGMEgo, DataFrames
ego_df   = DataFrame(ego_id=[1, 2], group=["A", "B"], w=[2.0, 1.0])
alter_df = DataFrame(ego_id=[1, 1, 2], alter_id=[10, 11, 10], group=["A", "B", "B"])
aatie_df = DataFrame(ego_id=[1], src=[10], dst=[11])
ed = as_egodata(ego_df, alter_df; aatie_df=aatie_df, ego_attrs=[:group],
                alter_attrs=[:group], weight_col=:w)
ed.sampling_weights            # [2.0, 1.0]
n_alter_ties(ed[1])            # 1
try as_egodata(ego_df, alter_df; ego_attrs=[:nope]) catch e; e isa ArgumentError end   # true — ArgumentError: column :nope not found in ego_df
```
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

    # Every named column is checked up front, so a typo in a keyword is one
    # ArgumentError naming the column and the frame, not a KeyError from the
    # middle of the row loop
    _require_column(ego_df, ego_id, "ego_df", "ego_id")
    _require_column(alter_df, ego_id, "alter_df", "ego_id")
    _require_column(alter_df, alter_id, "alter_df", "alter_id")
    for attr in ego_attrs
        _require_column(ego_df, attr, "ego_df", "ego_attrs")
    end
    for attr in alter_attrs
        _require_column(alter_df, attr, "alter_df", "alter_attrs")
    end
    isnothing(weight_col) || _require_column(ego_df, weight_col, "ego_df", "weight_col")
    if !isnothing(aatie_df)
        _require_column(aatie_df, ego_id, "aatie_df", "ego_id")
        _require_column(aatie_df, source_col, "aatie_df", "source_col")
        _require_column(aatie_df, target_col, "aatie_df", "target_col")
    end

    for row in eachrow(ego_df)
        eid = Int(row[ego_id])
        a_rows = alter_df[alter_df[!, ego_id] .== eid, :]
        alters = Int.(a_rows[!, alter_id])
        n_a = length(alters)
        # A duplicated (ego, alter) row — the wide-to-long reshape mistake —
        # is named here in the frame's own terms, before the EgoNetwork
        # constructor would refuse it (it would double the ego's degree)
        if !allunique(alters)
            n_dup = n_a - length(unique(alters))
            dup = first(a for a in alters if count(==(a), alters) > 1)
            throw(ArgumentError(
                "as_egodata: alter_df has $(_plural(n_dup, "duplicate (ego, alter) row")) " *
                "for ego $eid (alter $dup appears more than once); each ego–alter pair " *
                "must be one row, or the ego's degree is over-counted in every " *
                "statistic. Deduplicate alter_df on (:$ego_id, :$alter_id)."))
        end
        alter_index = Dict(a => k for (k, a) in enumerate(alters))

        # Alter-alter ties
        ties = zeros(Bool, n_a, n_a)
        if !isnothing(aatie_df)
            t_rows = aatie_df[aatie_df[!, ego_id] .== eid, :]
            for t in eachrow(t_rows)
                a, b = Int(t[source_col]), Int(t[target_col])
                (haskey(alter_index, a) && haskey(alter_index, b)) ||
                    throw(ArgumentError("alter tie ($a, $b) of ego $eid references unknown alters"))
                a == b && throw(ArgumentError(
                    "as_egodata: aatie_df row ($eid, $a, $b) is a self-tie (an alter tied " *
                    "to itself); alter–alter ties are between distinct alters. Drop the row."))
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
to ego data, returning a new `EgoData` (the egos are shared, the weights
replaced when given, the population size replaced when given). This is the
whole of the design an `EgoData` can express: independent egos with case
weights — there is no strata/cluster/finite-population-correction input.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [11], zeros(Bool, 1, 1))
ed = ego_design(EgoData([e1, e2]); ppopsize=1000, weights=[3.0, 1.0])
ed.population_size, ed.sampling_weights    # (1000, [3.0, 1.0])
estimate_popsize(ed)                       # 4.0 — Horvitz–Thompson: the weight sum
```
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

"""
    EgoTerm <: AbstractERGMTerm

The abstract type of every ego statistic — the `terms::Vector{<:EgoTerm}`
of [`fit_ergm_ego`](@ref) and the type a custom ego term subtypes (exported
as ERGM.jl exports `AbstractERGMTerm`). A subtype provides

- `name(t)::String` — its label (a method of the shared `Networks.name`);
- `ERGMEgo._ego_contribution(t, e::EgoNetwork)::Float64` — the per-ego
  contribution, whose design-weighted mean is `compute(t, ed)` (the shared
  `Networks.compute`; the generic method over `EgoTerm` does the weighting);
- and, to be fittable, [`ERGMEgo._ergm_term`](@ref)`(t)` — the ERGM.jl
  term whose sufficient statistic `m · compute(t, ed)` estimates (a `public`
  hook, like `_mcmc_controls`). Without it the term is descriptive only,
  like [`EgoDegree`](@ref), and `fit_ergm_ego` refuses it with an
  `ArgumentError`. The four built-in fittable terms are the whole fittable
  vocabulary today: `ergm.ego`'s `nodefactor`, `nodecov`, `absdiff`,
  `gwesp`, `mm` and `degree`/`concurrent` have no ego counterpart yet.

# Example
```julia
using ERGMEgo
struct EgoIsolate <: EgoTerm end                     # proportion of egos with no alters
ERGMEgo.name(::EgoIsolate) = "ego.isolate"
ERGMEgo._ego_contribution(::EgoIsolate, e::EgoNetwork) = Float64(n_alters(e) == 0)
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, Int[], Matrix{Bool}(undef, 0, 0))
compute(EgoIsolate(), EgoData([e1, e2]))             # 0.5
EgoIsolate() isa EgoTerm                             # true
```
"""
abstract type EgoTerm <: AbstractERGMTerm end

_wmean(values, weights) = sum(values .* weights) / sum(weights)

"""
    EgoEdges <: EgoTerm

Per-capita edge statistic: the design-weighted mean of `degree/2` over
egos. Scaled by the pseudo-population size this estimates the `edges`
sufficient statistic. Every model passed to [`fit_ergm_ego`](@ref) must
include it (as every `ergm.ego` model includes `edges`).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3))
e2 = EgoNetwork(2, [10], zeros(Bool, 1, 1))
ed = EgoData([e1, e2])
compute(EgoEdges(), ed)              # 1.0 — mean degree 2, halved
ego_target_stats([EgoEdges()], ed, 100)   # [100.0] — the edges target for 100 vertices
```
"""
struct EgoEdges <: EgoTerm end

"""
    name(term::EgoTerm) -> String

The label of an ego term, `ego.<statistic>[.<parameter>]` — `ego.edges`,
`ego.nodematch.<attr>`, `ego.triangle`, `ego.gwdegree.<decay>`,
`ego.degree.<d>` — used for the rows of `coeftable(fit)` and by `show`. A
method of the shared `Networks.name` generic (`ERGMEgo.name === Networks.name`).

# Example
```julia
using ERGMEgo
name(EgoEdges()), name(EgoNodeMatch(:Grade)), name(EgoGWDegree(0.5))
# ("ego.edges", "ego.nodematch.Grade", "ego.gwdegree.0.5")
```
"""
name(::EgoEdges) = "ego.edges"

_ego_contribution(::EgoEdges, e::EgoNetwork) = ego_degree(e) / 2

"""
    EgoNodeMatch(attr) <: EgoTerm

Per-capita homophily statistic: the design-weighted mean of half the
number of alters whose `attr` matches the ego's. Estimates the
`nodematch(attr)` sufficient statistic.

Every ego must carry `attr` in `ego_attrs` and in `alter_attrs`: an ego that
lacks it is an `ArgumentError` naming the attribute, the ego and which side
(ego or alters) is missing it, raised by `compute`/`fit_ergm_ego` before any
MCMC runs. (The pre-0.2 code scored such an ego as 0 — a silent zero-fill
that biased the target toward "no homophily".) A carried attribute whose
value is `missing` — for the ego, or for some of its alters (an unknown
alter attribute in a real survey; `as_egodata` passes such a column through)
— is likewise an `ArgumentError` naming the attribute, the ego and how many
alters lack a value, instead of a `TypeError` from the matching loop.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3);
               ego_attrs=Dict{Symbol,Any}(:group => "A"),
               alter_attrs=Dict{Symbol,Vector}(:group => ["A", "B", "A"]))
compute(EgoNodeMatch(:group), EgoData([e]))   # 1.0 — two matching alters / 2
try compute(EgoNodeMatch(:nope), EgoData([e])) catch e; e isa ArgumentError end   # true — ArgumentError naming :nope and ego 1
```
"""
struct EgoNodeMatch <: EgoTerm
    attr::Symbol
end

name(t::EgoNodeMatch) = "ego.nodematch.$(t.attr)"

# The one "attribute not carried" error of the ego terms and descriptives:
# names the attribute, the ego, the side that lacks it and what that side has
function _missing_attribute_error(what::AbstractString, attr::Symbol, e::EgoNetwork,
                                  side::AbstractString, have)
    have = sort!(collect(keys(have)))
    return ArgumentError("$what: ego $(e.ego) has no $side attribute :$attr " *
                         "(its $side attributes: " *
                         (isempty(have) ? "none" : join(have, ", ")) *
                         "). Every ego needs the attribute on both the ego and " *
                         "its alters — build the data with `ego_attrs`/" *
                         "`alter_attrs` naming :$attr, or drop the term.")
end

# Function barrier: `alter_attrs` is a `Dict{Symbol,Vector}` and `ego_attrs` a
# `Dict{Symbol,Any}`, so at the call site both the alter column and the ego's
# value are abstractly typed. Dispatching once on their concrete types makes
# the element comparison static; the `::Int` assertion on the result lets
# the caller divide without a second dynamic call (which boxed the Float64).
# The whole contribution is allocation-free (the interned small-`Int` box
# costs nothing), pinned by the "Hot paths are allocation-free" testset and
# benchmark/regression_tests.jl.
_count_matches(alters::AbstractVector, ego_val) = count(a -> a == ego_val, alters)

# The same barrier for the `missing`-VALUE check: on a `Vector{String}` (the
# column `as_egodata`/`simulate_ego_sample` build) `ismissing` folds to
# `false` at compile time, so the check costs nothing on clean data
_n_missing(alters::AbstractVector) = count(ismissing, alters)

# An attribute that is carried but holds `missing` for the ego or for some
# alters: refused by name (a real survey's unknown alter attribute), never
# scored around — `a == missing` is `missing`, and `count` would throw a
# TypeError from inside the loop
function _missing_value_error(what::AbstractString, attr::Symbol, e::EgoNetwork,
                              ego_missing::Bool, n_alters_missing::Int)
    where_ = String[]
    ego_missing && push!(where_, "the ego's own value")
    n_alters_missing > 0 && push!(where_, "$(_plural(n_alters_missing, "alter")) of " *
                                          "$(n_alters(e))")
    return ArgumentError("$what: ego $(e.ego) has `missing` for attribute :$attr " *
                         "($(join(where_, " and "))). A missing value cannot be " *
                         "matched: drop those alters (or the ego), recode the " *
                         "value, or drop the term.")
end

# `what` is a String or the term itself; its label is built only on the
# error path, so the check stays allocation-free on clean data
function _check_attribute_values(what, attr::Symbol, e::EgoNetwork)
    ego_missing = ismissing(e.ego_attrs[attr])
    n_alters_missing = _n_missing(e.alter_attrs[attr])::Int
    (ego_missing || n_alters_missing > 0) &&
        throw(_missing_value_error(what isa AbstractString ? what : name(what),
                                   attr, e, ego_missing, n_alters_missing))
    return nothing
end

function _ego_contribution(t::EgoNodeMatch, e::EgoNetwork)
    haskey(e.ego_attrs, t.attr) ||
        throw(_missing_attribute_error(name(t), t.attr, e, "ego", e.ego_attrs))
    haskey(e.alter_attrs, t.attr) ||
        throw(_missing_attribute_error(name(t), t.attr, e, "alter", e.alter_attrs))
    _check_attribute_values(t, t.attr, e)
    n_match = _count_matches(e.alter_attrs[t.attr], e.ego_attrs[t.attr])::Int
    return n_match / 2
end

"""
    EgoTriangle <: EgoTerm

Per-capita triangle statistic: the design-weighted mean of
`(alter–alter ties)/3` (each population triangle appears in the local view
of each of its three vertices). Estimates the `triangle` sufficient
statistic.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], Bool[0 1 1; 1 0 0; 1 0 0])   # two alter–alter ties
e2 = EgoNetwork(2, [10, 11], Bool[0 1; 1 0])                  # one
e3 = EgoNetwork(3, [12], zeros(Bool, 1, 1))                   # none
ed = EgoData([e1, e2, e3])
compute(EgoTriangle(), ed)          # 0.3333 — (2 + 1 + 0)/3 alter ties per ego, over 3
```
"""
struct EgoTriangle <: EgoTerm end

name(::EgoTriangle) = "ego.triangle"

_ego_contribution(::EgoTriangle, e::EgoNetwork) = n_alter_ties(e) / 3

"""
    EgoGWDegree(decay::Real=0.5) <: EgoTerm

Per-capita geometrically weighted degree: the design-weighted mean of
`e^α(1 − (1 − e^{−α})^degree)` with `α = decay`. Estimates the
`gwdegree(decay, fixed=TRUE)` sufficient statistic, i.e. ERGM.jl's
`GWDegree(decay)` (labelled `gwdeg.fixed.<decay>`, R's name). `decay` must be
non-negative (`ArgumentError` otherwise); at `decay = 0` every ego with at
least one alter contributes exactly 1, so the statistic is the proportion of
non-isolate egos.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], Bool[0 0; 0 0])
e2 = EgoNetwork(2, Int[], Matrix{Bool}(undef, 0, 0))
ed = EgoData([e1, e2])
compute(EgoGWDegree(0.0), ed)      # 0.5 — one of two egos has alters
EgoGWDegree(1).decay               # 1.0 — any Real is accepted
```
"""
struct EgoGWDegree <: EgoTerm
    decay::Float64

    function EgoGWDegree(decay::Real=0.5)
        decay >= 0 || throw(ArgumentError("decay must be non-negative (got $decay)"))
        new(Float64(decay))
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
exactly `d`. Not usable in `ergm_ego` (ERGM.jl has no degree-count term):
[`fit_ergm_ego`](@ref) refuses it with an `ArgumentError` naming the term.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [10], zeros(Bool, 1, 1))
e3 = EgoNetwork(3, [11, 12], zeros(Bool, 2, 2))
ed = EgoData([e1, e2, e3])
compute(EgoDegree(2), ed)           # 0.6667 — two of three egos have degree 2
compute(EgoDegree(0), ed)           # 0.0
try fit_ergm_ego(ed, [EgoEdges(), EgoDegree(2)]; ppopsize=20) catch e; e isa ArgumentError end   # true — ArgumentError: descriptive statistic
```
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
statistic — see [`ego_target_stats`](@ref)). A method of the shared
`Networks.compute` statistic protocol (`ERGMEgo.compute === Networks.compute`).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3))
e2 = EgoNetwork(2, [10], zeros(Bool, 1, 1))
ed = ego_design(EgoData([e1, e2]); weights=[1.0, 3.0])
compute(EgoEdges(), ed)             # 0.75 — weighted mean degree (3 + 3·1)/4 = 1.5, halved
compute(EgoEdges(), EgoData([e1, e2]))   # 1.0 under unit weights
```
"""
function compute(term::EgoTerm, ed::EgoData)
    h = [_ego_contribution(term, e) for e in ed.egos]
    return _wmean(h, ed.sampling_weights)
end

"""
    ego_mixing_matrix(ed::EgoData, attr::Symbol) -> (levels, matrix)

Design-weighted ego–alter mixing counts for a categorical attribute:
`matrix[a, b]` is the weighted number of ego–alter pairs with ego level
`levels[a]` and alter level `levels[b]`. Every ego must carry `attr` on both
the ego and its alters; one that does not is an `ArgumentError` naming the
attribute and the ego (it used to be skipped silently, so the matrix quietly
described a subset of the sample), and so is a `missing` value on either side
(the same rule as [`EgoNodeMatch`](@ref)).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2); ego_attrs=Dict{Symbol,Any}(:g => "A"),
                alter_attrs=Dict{Symbol,Vector}(:g => ["A", "B"]))
e2 = EgoNetwork(2, [12], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:g => "B"),
                alter_attrs=Dict{Symbol,Vector}(:g => ["B"]))
mm = ego_mixing_matrix(EgoData([e1, e2]), :g)
mm.levels            # ["A", "B"]
mm.matrix            # [1.0 1.0; 0.0 1.0]
```
"""
function ego_mixing_matrix(ed::EgoData, attr::Symbol)
    levels = Any[]
    for e in ed.egos
        haskey(e.ego_attrs, attr) ||
            throw(_missing_attribute_error("ego_mixing_matrix", attr, e, "ego", e.ego_attrs))
        haskey(e.alter_attrs, attr) ||
            throw(_missing_attribute_error("ego_mixing_matrix", attr, e, "alter", e.alter_attrs))
        _check_attribute_values("ego_mixing_matrix", attr, e)
        push!(levels, e.ego_attrs[attr])
        append!(levels, e.alter_attrs[attr])
    end
    levels = sort(unique(levels))
    index = Dict(l => k for (k, l) in enumerate(levels))

    mix = zeros(length(levels), length(levels))
    for (e, w) in zip(ed.egos, ed.sampling_weights)
        a = index[e.ego_attrs[attr]]
        for v in e.alter_attrs[attr]
            mix[a, index[v]] += w
        end
    end

    return (levels=levels, matrix=mix)
end

"""
    ERGMEgo._ergm_term(term::EgoTerm) -> AbstractERGMTerm

The ERGM.jl term whose sufficient statistic `m · compute(term, ed)`
estimates — the hook that makes an ego term **fittable** by
[`fit_ergm_ego`](@ref): `EgoEdges → Edges()`, `EgoNodeMatch(a) →
NodeMatch(a)`, `EgoTriangle → Triangle()`, `EgoGWDegree(α) → GWDegree(α)`.
Those four are the whole fittable vocabulary; a term without a method
(`EgoDegree`, or a custom [`EgoTerm`](@ref) that has not added one) is
descriptive only, and `fit_ergm_ego` refuses it with an `ArgumentError`
naming the term. A `public` (not exported) name, as `_mcmc_controls` is:
add a method to it (with the matching `_ego_contribution`) to fit a custom
term, provided the per-ego contribution really is an unbiased per-capita
estimate of the ERGM term's statistic.

# Example
```julia
using ERGMEgo, ERGM
ERGMEgo._ergm_term(EgoEdges())                       # Edges()
name(ERGMEgo._ergm_term(EgoGWDegree(0.5)))           # "gwdeg.fixed.0.5" — R's label
try ERGMEgo._ergm_term(EgoDegree(2)) catch e; e isa ArgumentError end   # true — descriptive only
struct EgoTwoStar <: EgoTerm end                     # per-capita 2-stars: C(degree, 2) — each 2-star has ONE centre, no divisor
ERGMEgo.name(::EgoTwoStar) = "ego.kstar.2"
ERGMEgo._ego_contribution(::EgoTwoStar, e::EgoNetwork) = Float64(binomial(ego_degree(e), 2))
ERGMEgo._ergm_term(::EgoTwoStar) = Kstar(2)          # now fittable
Base.ispublic(ERGMEgo, :_ergm_term)                  # true
```
"""
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
design-weighted per-capita ego statistics — what `ergm.ego` passes to
`ergm` as `target.stats`. Under a census (every vertex an ego, unit
weights, `m = nv`) they reduce exactly to the network's own statistics.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))   # a census
ego_target_stats([EgoEdges(), EgoNodeMatch(:Grade)], ed, 205)   # [203.0, 163.0] — the network's own
ego_target_stats([EgoEdges()], ed, 1000)                         # [990.24] — scaled to 1000 vertices
```
"""
ego_target_stats(terms, ed::EgoData, m::Int) =
    [m * compute(t, ed) for t in terms]

# =============================================================================
# Model and Estimation
# =============================================================================

"""
    EgoERGMModel{T}

Specification of an egocentric ERGM fit: ego terms, their ERGM
counterparts, the ego data, the pseudo-population and population sizes, and
the target statistics. The type parameter `T` is the vertex-ID type of the
ego data, so `data::EgoData{T}` is concretely typed (`fit.model.data` is an
`EgoData{Int}` for data built by [`as_egodata`](@ref) or
[`simulate_ego_sample`](@ref)).

# Fields
- `ego_terms::Vector{EgoTerm}`: the ego terms of the model, in order
- `ergm_terms::Vector{AbstractERGMTerm}`: the ERGM.jl term each one estimates
- `data::EgoData{T}`: the ego data the fit was made on
- `ppopsize::Int`, `popsize::Int`: pseudo-population and population sizes
- `targets::Vector{Float64}`: target statistics on the pseudo-population
  scale ([`ego_target_stats`](@ref))

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.model isa EgoERGMModel{Int}     # true
fit.model.targets                   # [203.0, 163.0] — the census reproduces the network's own
```
"""
struct EgoERGMModel{T}
    ego_terms::Vector{EgoTerm}
    ergm_terms::Vector{AbstractERGMTerm}
    data::EgoData{T}
    ppopsize::Int
    popsize::Int
    targets::Vector{Float64}
end

function Base.show(io::IO, m::EgoERGMModel{T}) where T
    print(io, "EgoERGMModel{$T}: ", _plural(length(m.data), "ego"),
          "; terms: ", join((name(t) for t in m.ego_terms), " + "),
          "; ppopsize $(m.ppopsize), popsize $(m.popsize); targets [",
          join((_fmt3(x) for x in m.targets), ", "), "]")
    return nothing
end

"""
    EgoERGMResult{T}

Results from [`fit_ergm_ego`](@ref); `T` is the vertex-ID type of the ego
data (`EgoERGMModel{T}`).

# Fields
- `model::EgoERGMModel{T}`: the fitted specification
- `coefficients`: population-scale coefficients (the edges coefficient
  includes the network-size adjustment `−log(popsize/ppopsize)`)
- `std_errors`: standard errors, `sqrt.(diag(vcov))`
- `vcov`: the covariance of the coefficients, `vcov_design + vcov_estimation`
  — `ergm.ego`'s decomposition (`vcov(fit, sources="all")`)
- `vcov_design`: the survey-design component `I⁻¹ Σ_design I⁻¹`, where
  `Σ_design` is the design variance of the target statistics and `I` the
  information (covariance of the statistics on the final MCMC sample) —
  `ergm.ego`'s `sources="model"`
- `vcov_estimation`: the Monte-Carlo estimation component `I⁻¹ / n_eff`
  (`= I⁻¹ Σ_mc I⁻¹` with `Σ_mc = I / n_eff` the variance of the sampled
  mean, `n_eff` the Geyer effective sample size of the final sample) —
  `ergm.ego`'s `sources="estimation"`. There is deliberately **no**
  standalone `I⁻¹` term: the estimand is a population parameter estimated
  from a sample of egos, and the population network is not modelled as a
  draw from the ERGM (Krivitsky & Morris 2017, §4).
- `netsize_adjustment`: the `−log(popsize/ppopsize)` adjustment applied to
  the edges coefficient
- `converged`: whether the sample at the returned coefficients passes
  ERGM.jl's per-statistic t-ratio and Hotelling T² tests
  (`ERGM.mcmc_convergence`)
- `mcmc_convergence::ERGM.MCMLEConvergence`: the convergence report of that
  same sample — `(iterations, step_length, t_ratios, hotelling_p, n_eff)`,
  where `iterations` is the number of Newton iterations run and
  `step_length` the damping factor applied to the last Newton step (1.0 when
  it was taken in full). `converged` is exactly `all(t_ratios .<
  conv_threshold) && hotelling_p > hotelling_alpha` of this report: the two
  describe ONE sample and cannot disagree
- `sim_stats`: that sample — the statistics drawn at the returned
  coefficients (pseudo-population scale) behind `converged`,
  `mcmc_convergence` and `vcov`. It is the sample that passed the tests
  (R's `ergm` design: no fresh draw is taken after convergence), or, when
  `maxiter` was exhausted after a Newton step, a fresh sample at the
  returned coefficients that decided `converged`

# StatsAPI
`coef`, `stderror`, `vcov`, `confint`, `coeftable`, `nobs` (the number of
egos — the independent units the design variance divides by; R's `nobs()` on
the underlying `ergm` object counts pseudo-population dyads instead) and `dof`
(the number of finite coefficients) are defined. `loglikelihood`, `aic` and
`bic` are deliberately **not**: the fit is moment matching
(`objective(fit) == :moment`) and no likelihood is ever evaluated, so there
is no number to return.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                                  # true
fit.mcmc_convergence.n_eff > 100               # true
coeftable(fit)["ego.edges"].estimate == coef(fit)[1]   # true
nobs(fit)                                      # 205 egos
```
"""
struct EgoERGMResult{T}
    model::EgoERGMModel{T}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    vcov::Matrix{Float64}
    vcov_design::Matrix{Float64}
    vcov_estimation::Matrix{Float64}
    netsize_adjustment::Float64
    converged::Bool
    mcmc_convergence::MCMLEConvergence
    sim_stats::Matrix{Float64}
end

_fmt3(x::Real) = isfinite(x) ? string(round(x; sigdigits=3)) : string(x)

# THE non-convergence sentence: printed by `show` under `Converged: false`,
# listed by `approximations` and quoted by the `@warn` in `fit_ergm_ego`,
# from the same numbers, so the three cannot disagree
function _nonconvergence_caveat(c::MCMLEConvergence, maxiter::Union{Nothing,Int}=nothing)
    cap = maxiter === nothing ? "" : " in maxiter=$maxiter iterations"
    # The numbers quoted are those of the sample at the RETURNED coefficients
    # — the one `converged` was decided on — so they always fail the rule
    return "moment matching did not converge$cap (on the final sample at the " *
           "returned coefficients: max t-ratio $(_fmt3(maximum(c.t_ratios))), " *
           "Hotelling p $(_fmt3(c.hotelling_p)), step length " *
           "$(_fmt3(c.step_length)) after $(c.iterations) " *
           "iteration$(c.iterations == 1 ? "" : "s")): the estimates do not solve " *
           "the moment equations and the standard errors are unreliable — " *
           "increase maxiter/n_samples, or burnin/interval for a better-mixing chain"
end
_nonconvergence_caveat(result::EgoERGMResult) = _nonconvergence_caveat(result.mcmc_convergence)

# R's "MCMC %" for an egocentric fit: `100 · (se − se_design)/se`, the share
# of the TOTAL standard error that the Monte-Carlo estimation term adds over
# the design part, rounded to an integer (NaN when the SE is NaN)
function _mcmc_percent(result::EgoERGMResult)
    se_design = sqrt.(max.(diag(result.vcov_design), 0.0))
    return [isfinite(se) && isfinite(d) && se > 0 ? round(Int, 100 * (se - d) / se) : NaN
            for (d, se) in zip(se_design, result.std_errors)]
end

function Base.show(io::IO, result::EgoERGMResult)
    c = result.mcmc_convergence
    println(io, "Egocentric ERGM Results")
    println(io, "=======================")
    println(io, "Egos: $(length(result.model.data)); pseudo-population: " *
                "$(result.model.ppopsize); population: $(result.model.popsize)")
    println(io, "Netsize adjustment (edges): $(round(result.netsize_adjustment, digits=4))")
    println(io, "Converged: $(result.converged)")
    if !result.converged
        # The caveat sits right under the verdict: an unconverged fit must
        # never look like a fit with a footnote
        println(io, "  ", _nonconvergence_caveat(result))
    end
    println(io)
    println(io, "Coefficients (population scale):")
    # Shared ecosystem presentation layer: the printed table IS
    # `coeftable(result)` (a Networks.CoefficientTable rendered through
    # `print_coeftable`), so what is shown and what is inspected agree
    show(io, coeftable(result))

    # ergm.ego's standard-error decomposition, in R's "MCMC %" convention
    names = [name(t) for t in result.model.ego_terms]
    shares = _mcmc_percent(result)
    println(io)
    println(io, "Std.Error = design component ⊕ MCMC-estimation component " *
                "(n_eff = $(round(c.n_eff, digits=1)) on the final sample)")
    println(io, "MCMC % of the standard error (100·(se − se_design)/se): ",
            join(("$(names[k]) $(shares[k])" for k in eachindex(names)), ", "))

    # Honest-uncertainty caveat, the prose twin of `approximations(result)`:
    # the "survey-design" variance component assumes independent egos carrying
    # the given case weights and nothing more (issue #1).
    println(io)
    println(io, "Note: the design variance component assumes independently sampled egos")
    println(io, "with the given case weights. It encodes no strata, clusters, finite-")
    println(io, "population correction, replicate weights, without-replacement inclusion")
    println(io, "probabilities, or alter dependence, so the standard errors are narrower")
    println(io, "than \"survey-design variance\" implies for any richer design.")
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these accessors. The caveats below are the
# machine-readable twin of the note `show` prints, so the two cannot disagree.

estimand(::EgoERGMResult) = :ergm_ego

"""
    objective(::EgoERGMResult) -> Symbol

`:moment` — the fit is MCMC **moment matching** (Newton iterations on
`targets − E_θ[g]`), not a likelihood maximization: no likelihood, exact or
approximate, is ever evaluated (which is why `loglikelihood`/`aic`/`bic`
have no methods for an egocentric fit).
"""
objective(::EgoERGMResult) = :moment

"""
    is_exact(::EgoERGMResult) -> Bool

Always `false`. The estimator matches design-weighted target statistics against
Monte-Carlo means simulated on a *pseudo-population* network of size `ppopsize`
— two distinct approximations to the population likelihood, neither of which
collapses to it for any formula.
"""
is_exact(::EgoERGMResult) = false

"""
    se_method(::EgoERGMResult) -> Symbol

`:sandwich`: `V(θ̂) = I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff` — the survey-design
covariance of the target statistics sandwiched by the inverse MCMC information,
plus the Monte-Carlo estimation term of the moment equations (`ergm.ego`'s
`sources="model"` and `sources="estimation"`). See [`approximations`](@ref) for
what that design covariance does and does not encode.
"""
se_method(::EgoERGMResult) = :sandwich

# Egocentric data is a sample of egos and their reported alters, not a
# sociomatrix with a dyad mask: the missing-dyad concept does not arise.
missing_method(::EgoERGMResult) = :none

"""
    approximations(result::EgoERGMResult) -> Vector{String}

What the reported numbers do and do not account for, as one sentence per
item (the shared `Networks.approximations` protocol, collected by
`fit_metadata`): the Monte-Carlo nature of the moment-matching fit with the
final sample's size, effective size, max t-ratio and Hotelling p; the
pseudo-population construction and the size adjustment; the independent-egos
assumption behind the design variance (no strata, clusters, finite-population
correction, replicate weights, without-replacement inclusion probabilities, or
alter dependence); and the design + MCMC-estimation composition of the
standard errors. A fit that did not converge lists the non-convergence caveat
first — the same sentence `show` prints and the `@warn` quoted.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
length(approximations(fit))                                   # 4
any(occursin("no strata, clusters", a) for a in approximations(fit))   # true
```
"""
function approximations(result::EgoERGMResult)
    c = result.mcmc_convergence
    out = [
        "method-of-moments fit by MCMC: the targets are matched against " *
        "simulated means, so the estimates carry Monte-Carlo error (final " *
        "sample: $(size(result.sim_stats, 1)) draws, effective sample size " *
        "$(round(c.n_eff, digits=1)), max convergence t-ratio " *
        "$(_fmt3(maximum(c.t_ratios))), Hotelling p $(_fmt3(c.hotelling_p)))",
        "the model is simulated on a pseudo-population network of size " *
        "$(result.model.ppopsize), not the population of size " *
        "$(result.model.popsize); the edges coefficient is put on the " *
        "population scale by the size adjustment " *
        "$(round(result.netsize_adjustment, digits=4))",
        # Issue ERGMEgo#1: the design variance is narrower than advertised.
        "the survey-design variance component is the weighted-mean variance of " *
        "the target statistics under INDEPENDENT egos with the given case " *
        "weights: it encodes no strata, clusters, finite-population correction, " *
        "replicate weights, without-replacement inclusion probabilities, or " *
        "alter dependence, so the standard errors are narrower than " *
        "\"survey-design variance\" implies for any richer sampling design",
        "the standard errors are the design sandwich I⁻¹ Σ_design I⁻¹ plus the " *
        "Monte-Carlo estimation term I⁻¹/n_eff of the moment equations " *
        "(ergm.ego's decomposition); the information I is itself the covariance " *
        "of the statistics on a finite MCMC sample, so both components carry " *
        "Monte-Carlo noise — increase n_samples or interval to reduce it",
    ]
    result.converged || pushfirst!(out, _nonconvergence_caveat(result))
    return out
end

# StatsAPI interface: methods on the shared statistics generics (mirroring
# ERGM.jl), so `coef(fit)` etc. work on egocentric fits too. `loglikelihood`,
# `aic` and `bic` are deliberately absent (see `EgoERGMResult`).

"""
    coef(result::EgoERGMResult) -> Vector{Float64}

The population-scale coefficients, in the order of the model's ego terms
(the edges coefficient includes the network-size adjustment) — a method of
`StatsAPI.coef`.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
coef(fit) == fit.coefficients        # true
coef(fit)[1] - fit.netsize_adjustment    # the edges coefficient on the pseudo-population scale
```
"""
StatsAPI.coef(result::EgoERGMResult) = result.coefficients

"""
    stderror(result::EgoERGMResult) -> Vector{Float64}

The standard errors `sqrt.(diag(vcov(result)))`: `ergm.ego`'s design +
MCMC-estimation decomposition (see [`EgoERGMResult`](@ref)) — a method of
`StatsAPI.stderror`. `NaN` when the information matrix was singular.

# Example
```julia
using ERGMEgo, Networks, Random, LinearAlgebra
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
stderror(fit) ≈ sqrt.(diag(vcov(fit)))                    # true
all(stderror(fit) .> sqrt.(diag(fit.vcov_design)))        # true — the estimation term adds to the design part
```
"""
StatsAPI.stderror(result::EgoERGMResult) = result.std_errors

"""
    vcov(result::EgoERGMResult) -> Matrix{Float64}

The covariance of the coefficients, `vcov_design + vcov_estimation`
(`ergm.ego`'s `vcov(fit, sources="all")`) — a method of `StatsAPI.vcov`.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
vcov(fit) == fit.vcov_design .+ fit.vcov_estimation       # true
size(vcov(fit))                                           # (2, 2)
```
"""
StatsAPI.vcov(result::EgoERGMResult) = result.vcov

"""
    coeftable(result::EgoERGMResult) -> Networks.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `Networks.CoefficientTable` — exactly the table
`show(result)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`); p-values come from the shared `Networks.z_pvalues`.
Rows can be read by index or by ego-term name.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
tbl = coeftable(fit)
tbl["ego.edges"].estimate == coef(fit)[1]          # true
tbl[2].p_value == z_pvalues(coef(fit), stderror(fit)).p[2]   # true
```
"""
StatsAPI.coeftable(result::EgoERGMResult) =
    CoefficientTable([name(t) for t in result.model.ego_terms],
                     result.coefficients, result.std_errors)

"""
    confint(result::EgoERGMResult; level=0.95) -> Matrix{Float64}

Normal-theory confidence intervals `θ̂ ± z_{1−α/2} · se`, one row per
coefficient (lower, upper), on the population scale — a method of
`StatsAPI.confint`. The standard errors are the design + MCMC-estimation
decomposition of [`EgoERGMResult`](@ref).

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
ci = confint(fit)                            # 2×2
all(ci[:, 1] .< coef(fit) .< ci[:, 2])       # true
confint(fit; level=0.9)                      # narrower
```
"""
function StatsAPI.confint(result::EgoERGMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must be in (0, 1) (got $level)"))
    q = sqrt(2.0) * erfcinv(1 - level)       # z_{1−α/2}, α = 1 − level
    θ, se = result.coefficients, result.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    nobs(result::EgoERGMResult) -> Int

The number of egos — the independent sampling units the design variance
divides by. Note that R's `nobs()` on the `ergm` object underlying an
`ergm.ego` fit counts the pseudo-population's dyads, not egos; this method
reports the egocentric sample size.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 30; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges()]; ppopsize=100, popsize=205, rng=Xoshiro(1))
nobs(fit)                            # 30 — egos, not the 4950 pseudo-population dyads
```
"""
StatsAPI.nobs(result::EgoERGMResult) = length(result.model.data)

"""
    dof(result::EgoERGMResult) -> Int

The number of estimated (finite) coefficients — a method of `StatsAPI.dof`.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
dof(fit)                             # 2
```
"""
StatsAPI.dof(result::EgoERGMResult) = count(isfinite, result.coefficients)

# Build the pseudo-population network: m vertices whose attributes are
# ego attributes replicated proportionally to the sampling weights, with
# edges seeded at the target density
function _pseudo_population(ed::EgoData, m::Int, target_density::Float64,
                            rng::Random.AbstractRNG)
    net = network(m; directed=false)

    # Replicate egos proportionally to weight (largest-remainder rounding)
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
    _mcmc_controls(m::Int; n_samples=nothing, burnin=nothing, interval=nothing)
        -> (n_samples, burnin, interval)

THE MCMC budget rule of the package, used by every sampler call on a
pseudo-population of size `m` (the Newton iterations and final sample of
[`fit_ergm_ego`](@ref), and the GOF simulations): `nothing` selects the
default, an `Int` overrides it. The controls scale with the size of the
PSEUDO-POPULATION, not with the ego sample — the chain has to mix over
`m(m−1)/2` dyads, and fixed defaults (the pre-0.2 `400/2000/20`) silently
stopped mixing as `m` grew (edges ≈ −21.9 against a true −6.02 on the
205-actor faux.mesa census).

- `n_samples = max(400, min(3000, 20m))` draws per sample;
- `burnin`/`interval` from ERGM.jl's one dyad-scaled rule,
  `ERGM._mcmc_defaults(n_dyads)`: `20·n_dyads` and `max(100, n_dyads ÷ 10)`
  toggles. On the 205-actor census (20 910 dyads) this is 3000 draws at
  interval 2091 after 418 200 burn-in toggles — an effective sample size of
  ≈ 150 per sample at ~0.3 s, where the pre-panel interval of `n_dyads ÷ 70`
  gave n_eff ≈ 23.

A `public` (not exported) name: `ERGMEgo._mcmc_controls`.

# Example
```julia
using ERGMEgo
ERGMEgo._mcmc_controls(205)                    # (n_samples = 3000, burnin = 418200, interval = 2091)
ERGMEgo._mcmc_controls(30)                     # (n_samples = 600, burnin = 8700, interval = 100)
ERGMEgo._mcmc_controls(205; n_samples = 500)   # one control overridden, the others scaled
```
"""
function _mcmc_controls(m::Int;
                        n_samples::Union{Int, Nothing}=nothing,
                        burnin::Union{Int, Nothing}=nothing,
                        interval::Union{Int, Nothing}=nothing)
    n_dyads = m * (m - 1) ÷ 2
    d = _mcmc_defaults(n_dyads)
    return (n_samples = something(n_samples, max(400, min(3000, 20 * m))),
            burnin    = something(burnin, d.burnin),
            interval  = something(interval, d.interval))
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
   for `target.stats`). Convergence is ERGM.jl's MCMLE rule
   (`ERGM.mcmc_convergence`): every per-statistic t-ratio
   `|target − mean| / sd` below `conv_threshold` AND a Hotelling T² test of
   `mean == targets` not rejected at `hotelling_alpha`, evaluated on the
   sample drawn at every iteration. The sample that passes is the final
   sample — it is what `converged`, `mcmc_convergence`, `sim_stats` and the
   standard errors describe, so they cannot disagree; only when `maxiter` is
   exhausted after a Newton step is one more sample drawn at the returned
   coefficients, and then that sample decides `converged`.
4. Apply the **network-size adjustment** `−log(popsize/ppopsize)` to the
   edges coefficient so it is on the population scale.

Standard errors are `ergm.ego`'s decomposition,
`V(θ̂) = I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff`: the survey-design variance of the
targets sandwiched by the inverse information, plus the Monte-Carlo
estimation term (see [`EgoERGMResult`](@ref) for `vcov_design`,
`vcov_estimation` and why there is no standalone `I⁻¹` term).

A fit that does not converge is **loud**: a warning quoting the max t-ratio,
Hotelling p-value and iteration count is emitted, `converged` is `false`,
`show` prints the caveat under `Converged: false`, and
`approximations(fit)` lists it. A singular covariance of the sampled
statistics (collinear terms, degenerate model) stops the Newton iterations
with a warning and leaves the standard errors `NaN`.

[`ergm_ego`](@ref) is the R-faithful alias (matching the `ergm.ego`
package); `fit_ego_ergm` is a legacy alias.

# Keyword Arguments
- `ppopsize::Int`: Pseudo-population size (default: `popsize` if known and
  ≤ 1000, otherwise `10 ×` the number of egos)
- `popsize::Union{Int,Nothing}`: Population size for the offset (default:
  `ed.population_size`, falling back to `ppopsize`, i.e. no adjustment)
- `n_samples`, `burnin`, `interval`: MCMC controls per sample; `nothing`
  (the default) selects the dyad-scaled rule of [`_mcmc_controls`](@ref)
- `maxiter::Int=80`: maximum number of Newton iterations
- `conv_threshold::Float64=0.1`, `hotelling_alpha::Float64=0.05`: the
  convergence tests, as in `ERGM.mcmle`
- `rng::AbstractRNG=Random.default_rng()`: source of every random draw
  (pseudo-population seeding and the MCMC chain); the same `rng` state gives
  a bit-identical fit whatever the global RNG holds
- `max_iter`: deprecated spelling of `maxiter` (honoured, warns once);
  `tol`: deprecated and ignored (warns once) — convergence is no longer a
  relative-change rule

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)                 # 205 students, 203 ties
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))   # a census
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                     # true
coef(fit)                         # ≈ [-6.02, 2.82] — ergm.ego's netsize.adj + edges, nodematch.Grade
stderror(fit)                     # ≈ [0.16, 0.19] (ergm.ego: 0.178, 0.196; the gap is Monte-Carlo noise in I)
fit                               # prints the coefficient table and the SE decomposition
```
"""
function fit_ergm_ego(ed::EgoData{T}, terms::Vector{<:EgoTerm};
                      ppopsize::Union{Int, Nothing}=nothing,
                      popsize::Union{Int, Nothing}=nothing,
                      n_samples::Union{Int, Nothing}=nothing,
                      burnin::Union{Int, Nothing}=nothing,
                      interval::Union{Int, Nothing}=nothing,
                      maxiter::Int=80,
                      conv_threshold::Float64=0.1,
                      hotelling_alpha::Float64=0.05,
                      rng::Random.AbstractRNG=Random.default_rng(),
                      max_iter::Union{Nothing, Int}=nothing,
                      tol::Union{Nothing, Real}=nothing) where T
    if !isnothing(tol)
        @warn "The `tol` keyword to `fit_ergm_ego` is deprecated and ignored: " *
              "convergence is now assessed with per-statistic t-ratios " *
              "(`conv_threshold`) and a Hotelling T² test (`hotelling_alpha`), as " *
              "in `ERGM.mcmle`." maxlog=1
    end
    if !isnothing(max_iter)
        # Deprecated spelling of the ecosystem-wide `maxiter` keyword (panel
        # 2026-09, item 16): honoured, with a one-time warning.
        @warn "The `max_iter` keyword to `fit_ergm_ego` is deprecated; use " *
              "`maxiter` (the ecosystem-wide spelling). `max_iter=$max_iter` is " *
              "honoured for now." maxlog=1
        maxiter = max_iter
    end
    maxiter >= 1 || throw(ArgumentError("fit_ergm_ego: maxiter must be ≥ 1 (got $maxiter)"))
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
    m >= 5 || throw(ArgumentError(
        "fit_ergm_ego: the pseudo-population must have at least 5 vertices (got " *
        "ppopsize=$m, from " *
        (isnothing(ppopsize) ? "the default rule — popsize when it is known and " *
                               "≤ 1000, else 10·n_egos — with popsize=" *
                               "$(something(popsize, ed.population_size, Some("unknown"))) " *
                               "and $(_plural(length(ed.egos), "ego"))" :
                               "the ppopsize keyword") *
        "); pass ppopsize=<n> ≥ 5"))

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

    # The one MCMC budget rule (dyad-scaled; see `_mcmc_controls`)
    n_samples, burnin, interval = _mcmc_controls(m; n_samples, burnin, interval)
    n_samples >= 2 || throw(ArgumentError("fit_ergm_ego: n_samples must be ≥ 2 (got $n_samples)"))

    net = _pseudo_population(ed, m, target_density, rng)
    model = ERGMModel(ERGMFormula(ergm_terms), net)
    term_names = [name(t) for t in ego_terms]

    # Initialize: edges at logit of target density, others at 0
    θ = zeros(p)
    θ[edges_idx] = log(target_density / (1 - target_density))

    # MCMC moment matching: Newton on targets − E_θ[g], with ERGM.jl's
    # MCMLE convergence tests (t-ratios + Hotelling T² on the Geyer effective
    # sample size) in place of the pre-panel 1 % relative-change rule, which
    # never looked at the Monte-Carlo sd and so declared convergence at the
    # noise level (panel 2026-09, item 24)
    #
    # ONE sample describes the returned fit (R's `ergm` design): the sample
    # that passes the tests IS the final sample — the basis of `converged`,
    # `mcmc_convergence`, `sim_stats` and the standard errors — so the four
    # cannot disagree. (Redrawing a fresh sample at θ̂ for the report, as the
    # 2026-09 round-1 code did, made `converged == true` sit next to a
    # recorded Hotelling p of 0.0005 in about one fit in four: the two
    # samples were different draws.) A fresh sample at the returned θ is
    # drawn only when the loop exhausts `maxiter` — θ moved after the last
    # sample — and then THAT sample decides `converged`, so the caveat, when
    # printed, always quotes numbers that fail the rule.
    converged = false
    iterations = 0
    step_length = 1.0
    samples = Matrix{Float64}(undef, 0, p)
    tests = nothing
    for iter in 1:maxiter
        iterations = iter
        samples = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                            interval=interval, rng=rng).stats
        tests = mcmc_convergence(samples, targets;
                                 conv_threshold=conv_threshold,
                                 hotelling_alpha=hotelling_alpha,
                                 chain_lengths=[n_samples])
        if tests.converged
            converged = true
            break
        end

        cov_stats = cov(samples)
        F = cholesky(Symmetric(cov_stats); check=false)
        if !issuccess(F)
            source = iter == 1 ? "the initial values" :
                                 "the iteration-$(iter - 1) update"
            @warn "The covariance matrix of the sampled statistics is singular at " *
                  "iteration $iter (collinear statistics, a degenerate model, or a " *
                  "collapsed sampler; terms $(join(term_names, ", "))). Moment " *
                  "matching cannot take further Newton steps; the returned " *
                  "coefficients are $source, unrefined, and standard errors will " *
                  "be NaN. Check the model for degeneracy or redundant terms."
            # θ did not move: `samples`/`tests` already describe the returned θ
            break
        end

        diff = targets .- vec(mean(samples, dims=1))
        step = F \ diff
        # Damp large steps for stability: no coefficient moves by more than
        # 1 per iteration; `step_length` is the factor actually applied
        maxstep = maximum(abs.(step))
        step_length = maxstep > 1.0 ? 1.0 / maxstep : 1.0
        θ .+= step_length .* step

        if iter == maxiter
            # maxiter exhausted after a Newton step: the returned θ has no
            # sample yet. Draw one and let it decide — "evaluated at every
            # iteration and once more on the final sample"
            samples = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                                interval=interval, rng=rng).stats
            tests = mcmc_convergence(samples, targets;
                                     conv_threshold=conv_threshold,
                                     hotelling_alpha=hotelling_alpha,
                                     chain_lengths=[n_samples])
            converged = tests.converged
        end
    end
    convergence = MCMLEConvergence((iterations, step_length, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))

    # Variance — ergm.ego's decomposition (vcov.ergm.ego, sources="model" /
    # "estimation"): the design component I⁻¹ Σ_t I⁻¹, where Σ_t is the
    # survey variance of the target statistics and I the information
    # (covariance of the sampled statistics), plus the Monte-Carlo estimation
    # component I⁻¹ Σ_mc I⁻¹ with Σ_mc = I/n_eff the variance of the sampled
    # mean. No standalone I⁻¹: the population network is not a draw from the
    # model, the egos are the sample.
    I_mat = cov(samples)
    Σ_t = _design_cov(ego_terms, ed, m)
    F = cholesky(Symmetric(I_mat); check=false)
    vcov_design, vcov_estimation = if issuccess(F)
        Iinv = Matrix(inv(F))
        Vd = Iinv * Σ_t * Iinv
        ((Vd .+ Vd') ./ 2, Iinv ./ tests.n_eff)
    else
        @warn "The covariance matrix of the statistics sampled at the fitted " *
              "coefficients is singular (collinear statistics or a collapsed " *
              "sampler; terms $(join(term_names, ", "))): no standard errors are " *
              "available — `stderror`, `vcov`, z-values and p-values are NaN. " *
              "The point estimates are unaffected."
        (fill(NaN, p, p), fill(NaN, p, p))
    end
    vcov_θ = vcov_design .+ vcov_estimation
    se = sqrt.(max.(diag(vcov_θ), 0.0))

    # Non-convergence is loud: warned here with the diagnostics of the
    # returned estimate, recorded in `converged`/`mcmc_convergence` (hence in
    # `approximations(fit)` and `show`) so a reader and a machine both see it.
    converged || @warn _nonconvergence_caveat(convergence, maxiter)

    # Network-size adjustment: put the edges coefficient on the
    # population (size N) scale
    adjustment = log(m / N)          # = −log(N/m); exactly 0.0 (not −0.0) when N == m
    coefficients = copy(θ)
    coefficients[edges_idx] += adjustment

    ego_model = EgoERGMModel(ego_terms, ergm_terms, ed, m, N, targets)
    return EgoERGMResult(ego_model, coefficients, se, vcov_θ, vcov_design,
                         vcov_estimation, adjustment, converged, convergence, samples)
end

"""
    ergm_ego(ed::EgoData, terms; kwargs...)

R-faithful alias for [`fit_ergm_ego`](@ref) (the same function, so
`ergm_ego === fit_ergm_ego`), matching the R `ergm.ego` package name — the
ecosystem convention of one `fit_<model>` name and one statnet-style name.

# Example
```julia
using ERGMEgo, Networks, Random
ergm_ego === fit_ergm_ego            # true
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))   # R: ergm.ego(egor ~ edges + nodematch("Grade"))
fit.converged                        # true
```
"""
const ergm_ego = fit_ergm_ego

"""
    fit_ego_ergm(ed::EgoData, terms; kwargs...)

Alias for [`fit_ergm_ego`](@ref), kept for backward compatibility
(`fit_ego_ergm === fit_ergm_ego`); new code should spell it `fit_ergm_ego`
or `ergm_ego`.

# Example
```julia
using ERGMEgo
fit_ego_ergm === fit_ergm_ego        # true
```
"""
const fit_ego_ergm = fit_ergm_ego

# The per-ego contributions of ONE (concretely typed) term into column `j`
function _contribution_column!(H::Matrix{Float64}, j::Int, t::EgoTerm, egos::Vector)
    @inbounds for (i, e) in enumerate(egos)
        H[i, j] = _ego_contribution(t, e)
    end
    return H
end

# Survey-design covariance of the target statistics: targets are
# m·(weighted mean of per-ego contributions), so
# V(target) = m² · V_w(h̄) with the standard weighted-mean variance
function _design_cov(terms, ed::EgoData, m::Int)
    n = length(ed.egos)
    p = length(terms)
    n >= 2 || throw(ArgumentError(
        "the design variance of the ego targets needs at least 2 egos (got $n)"))
    w = ed.sampling_weights ./ sum(ed.sampling_weights)

    # One column per term through a function barrier (`terms` is a
    # `Vector{EgoTerm}`, so `t` is abstractly typed here; inside
    # `_contribution_column!` the term is concrete and the per-ego loop is
    # static): the whole routine allocates H, w, h̄ and Σ and nothing per
    # ego — pinned at ≤ 4·n·p·8 bytes plus a constant by the allocation gates.
    H = Matrix{Float64}(undef, n, p)
    for (j, t) in enumerate(terms)
        _contribution_column!(H, j, t, ed.egos)
    end
    h̄ = zeros(p)
    @inbounds for j in 1:p, i in 1:n
        h̄[j] += w[i] * H[i, j]
    end

    Σ = zeros(p, p)
    @inbounds for i in 1:n
        w2 = w[i]^2
        for k in 1:p
            dk = H[i, k] - h̄[k]
            for j in 1:p
                Σ[j, k] += w2 * (H[i, j] - h̄[j]) * dk
            end
        end
    end

    # Bessel/finite-sample correction. The weighted sum above divides the
    # squared deviations by n (each wᵢ = 1/n under a census, so Σwᵢ² = 1/n);
    # the survey (Horvitz–Thompson / SRS) variance of a weighted mean — which
    # is what `ergm.ego` computes — divides by n − 1. Without this factor the
    # design variance, and hence every standard error, is too small by exactly
    # (n − 1)/n. That was measurably wrong against `ergm.ego`
    # (test/fixtures/fauxmesa_ego_census.toml) and it matters: the design
    # component of V(θ̂) is an order of magnitude larger than the estimation
    # component, so an ego SE essentially *is* its design variance.
    Σ .*= n / (n - 1)

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
  meaningful alter IDs (preserved by `as_egodata`); no overlap at all is an
  `ArgumentError`.

An unknown `method` is an `ArgumentError` naming it and listing the two
methods above (`:horvitz_thompson`, `:capture_recapture`).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3))
e2 = EgoNetwork(2, [11, 13], zeros(Bool, 2, 2))
e3 = EgoNetwork(3, [10, 14], zeros(Bool, 2, 2))
e4 = EgoNetwork(4, [12, 15], zeros(Bool, 2, 2))
ed = ego_design(EgoData([e1, e2, e3, e4]); weights=[10.0, 10.0, 5.0, 5.0])
estimate_popsize(ed)                                  # 30.0 — the weight sum
estimate_popsize(ed; method=:capture_recapture)       # 8.0 — halves name {10,11,12,13} and {10,12,14,15}, overlap 2: 4·4/2
try estimate_popsize(ed; method=:lincoln_petersen) catch e; e isa ArgumentError end   # true — ArgumentError: unknown method :lincoln_petersen; expected :horvitz_thompson ... or :capture_recapture ...
```
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
        throw(ArgumentError(
            "estimate_popsize: unknown method :$method; expected :horvitz_thompson " *
            "(the weight sum) or :capture_recapture (two-sample Lincoln–Petersen on " *
            "alter overlap)"))
    end
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate_ego_sample(net::Network, n_egos::Int;
                        ego_attrs=Symbol[], missing=:error, report=false,
                        rng=Random.default_rng()) -> EgoData
    simulate_ego_sample(net, n_egos; report=true) -> (EgoData, ConversionReport)

Draw an egocentric sample from a complete **undirected, one-mode** network:
sample `n_egos` egos uniformly without replacement (`rng`) and record each
ego's alters, the ties among those alters, and the vertex attributes named in
`ego_attrs` for ego and alters. Alter IDs are the network's vertex IDs, so
cross-ego overlap is preserved; `population_size` is `nv(net)`.

This is the ecosystem's `Network → EgoData` conversion adapter and follows
the conversion contract (`Networks.ConversionReport`) and the missing-data
contract (`Networks.require_observed`):

- **Refused** (`ArgumentError` naming the fix): a network with masked
  (unobserved) dyads under the default `missing=:error` — a masked dyad is
  unobserved, not absent, and the sample would read its face value as an
  observed ego–alter or alter–alter tie; a **directed** network (ego
  networks are undirected; `neighbors` would be read as out-neighbours only
  — symmetrise first, e.g. `A = as_matrix(net); network_from_matrix(A .| A';
  directed=false)`); a **two-mode** network;
  an entry of `ego_attrs` the network does not carry, or carries for only
  some vertices (the pre-0.2 code silently filled `missing`, which a
  homophily term then miscounted).
- **`missing=:face`** reads the stored face value of every masked dyad — an
  explicit, auditable opt-in, recorded in the report as `:missing_dyads`.
  `supports_missing(simulate_ego_sample) == true`,
  `missing_policies(simulate_ego_sample) == (:error, :face)`.
- **Reported** with `report=true`, which returns `(ed, report)`: `:edges`
  (ties between unsampled vertices, whenever `n_egos < nv(net)`),
  `:vertex_attrs` (each attribute not in `ego_attrs`), `:edge_attrs`,
  `:network_attrs`, `:loops` (a network allowing self-loops: an ego's own
  loop is never read and alter self-loops are not recorded) and
  `:missing_dyads` (under `missing=:face`). A census of an undirected
  network with every vertex attribute requested and no edge/network
  attributes is lossless (`is_lossless(report)`).

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)                       # undirected, :Grade/:Race/:Sex
ed, rep = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Race, :Sex],
                              rng=Xoshiro(1), report=true)
is_lossless(rep)                                          # true — a census, every attribute kept
ed30, rep30 = simulate_ego_sample(net, 30; ego_attrs=[:Grade], rng=Xoshiro(1), report=true)
dropped_fields(rep30)                                     # [:edges, :vertex_attrs, :vertex_attrs]
try simulate_ego_sample(net, 30; ego_attrs=[:nope]) catch e; e isa ArgumentError end   # true — ArgumentError naming :nope and the attributes it has

masked = copy(net)
set_missing_dyad!(masked, 1, 2)
try simulate_ego_sample(masked, 30) catch e; e isa ArgumentError end   # true — ArgumentError: ... pass `missing=:face` ...
edf, repf = simulate_ego_sample(masked, 30; missing=:face, report=true)
:missing_dyads in dropped_fields(repf)                    # true
```
"""
function simulate_ego_sample(net::Network, n_egos::Int;
                             ego_attrs::Vector{Symbol}=Symbol[],
                             missing::Symbol=:error,
                             report::Bool=false,
                             rng::Random.AbstractRNG=Random.default_rng())
    # The missing-data contract first: a masked dyad is unobserved, not
    # absent, and `:face` is the written opt-in (as for Network ↔ Matrix)
    require_observed(net, missing; context="simulate_ego_sample")
    is_directed(net) && throw(ArgumentError(
        "simulate_ego_sample: ergm.ego models are undirected, but the network is " *
        "directed; simulate_ego_sample would read `neighbors` as out-neighbours " *
        "only and record alter–alter ties from one arc direction. Symmetrise " *
        "first, e.g. `A = as_matrix(net); network_from_matrix(A .| A'; " *
        "directed=false)` (weak rule), or rebuild with `network(n; directed=false)`."))
    is_two_mode(net) && throw(ArgumentError(
        "simulate_ego_sample: the network is two-mode (bipartite = $(net.bipartite)); " *
        "ego networks have one vertex set and ergm.ego has no two-mode " *
        "counterpart. Project onto one mode first."))
    n = Int(nv(net))
    0 <= n_egos <= n || throw(ArgumentError(
        "simulate_ego_sample: cannot sample $n_egos egos from a network of $n vertices"))

    # Every requested attribute must exist and cover every vertex — a
    # vertex without a value used to become `missing` in `alter_attrs`, which
    # `EgoNodeMatch` then miscounted (silent zero-fill class, panel P1-11)
    available = sort!(list_vertex_attributes(net))
    for attr in ego_attrs
        attr in available || throw(ArgumentError(
            "simulate_ego_sample: the network has no vertex attribute :$attr " *
            "(ego_attrs); its vertex attributes are " *
            (isempty(available) ? "none" : join(available, ", ")) *
            ". Set it with `set_vertex_attribute!` or drop it from `ego_attrs`."))
        vals = get_vertex_attribute(net, attr)
        n_set = count(v -> haskey(vals, v), vertices(net))
        n_set == n || throw(ArgumentError(
            "simulate_ego_sample: vertex attribute :$attr is set for $n_set of $n " *
            "vertices; an ego sample cannot carry a partial attribute (an alter " *
            "without a value has no level to match). Set it for every vertex " *
            "or drop it from `ego_attrs`."))
    end

    # One concretely typed column per attribute, built once (a `Vector{String}`
    # for a string attribute, the comprehension's widened eltype otherwise),
    # so every ego's `alter_attrs[attr]` — an isolate's empty one included —
    # has the same element type as `as_egodata` would give it. Indexing the
    # per-ego dictionary inside the loop gave a runtime-widened `Vector{String}`
    # for egos with alters and a `Vector{Any}` for isolates.
    columns = Dict{Symbol, Vector}()
    for attr in ego_attrs
        vals = get_vertex_attribute(net, attr)
        columns[attr] = [vals[v] for v in 1:n]
    end

    ego_ids = sample(rng, 1:n, n_egos; replace=false)
    egos = EgoNetwork{Int}[]

    for eid in ego_ids
        # An ego's own self-loop is never an alter
        alters = sort!(filter(!=(eid), collect(neighbors(net, eid))))
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
            col = columns[attr]
            e_attrs[attr] = col[eid]
            a_attrs[attr] = col[alters]      # same eltype as `col`, empty or not
        end

        push!(egos, EgoNetwork(Int(eid), Int.(alters), ties;
                               ego_attrs=e_attrs, alter_attrs=a_attrs))
    end

    ed = EgoData(egos; population_size=n)
    report || return ed

    # The conversion contract: what an EgoData cannot hold, named
    rep = ConversionReport(:Network, :EgoData)
    n_egos < n && record_drop!(rep, :edges,
        "ties between unsampled vertices are not observed ($(n - n_egos) of $n " *
        "vertices are not egos; only ego–alter and alter–alter ties per ego are kept)")
    for attr in available
        attr in ego_attrs || record_drop!(rep, :vertex_attrs,
            "vertex attribute :$attr is not in ego_attrs")
    end
    for attr in sort!(list_edge_attributes(net))
        record_drop!(rep, :edge_attrs,
            "edge attribute :$attr: an ego network records tie presence only")
    end
    for attr in sort!(list_network_attributes(net))
        record_drop!(rep, :network_attrs,
            "network attribute :$attr: an ego sample has no network-level attributes")
    end
    if net.loops
        n_loops = count(v -> has_edge(net, v, v), vertices(net))
        record_drop!(rep, :loops,
            "the network allows self-loops ($(_plural(n_loops, "self-loop")) present): " *
            "an ego's own loop is never read and alter self-loops are not recorded")
    end
    n_masked = n_missing_dyads(net)
    n_masked > 0 && record_drop!(rep, :missing_dyads,
        "$(_plural(n_masked, "masked dyad")) read at face value (missing=:face): " *
        "an unobserved tie is recorded as its stored value")
    return (ed, rep)
end

# A two-mode network wrapper is refused with the same actionable message as
# a `Network` carrying two-mode metadata
simulate_ego_sample(net::BipartiteNetwork, n_egos::Int; kwargs...) =
    throw(ArgumentError(
        "simulate_ego_sample: the network is two-mode (BipartiteNetwork); ego " *
        "networks have one vertex set and ergm.ego has no two-mode counterpart. " *
        "Project onto one mode first."))

# The missing-data contract's declaration half: the adapter guards with
# `require_observed` and offers `:face` as its one written opt-in
Networks.supports_missing(::typeof(simulate_ego_sample)) = true
Networks.missing_policies(::typeof(simulate_ego_sample)) = (:error, :face)

# =============================================================================
# Diagnostics
# =============================================================================

# The ONE simulation engine behind `gof` (and hence `ego_gof`): simulate
# pseudo-population networks at the fitted (pseudo-population scale)
# coefficients, take ego samples of the observed size from each, and
# summarize each sample. Returns the observed summary statistics plus the
# simulated mean-degree and mean-alter-tie vectors.
#
# Every draw flows through `rng`: the pseudo-population seeding, the
# `sample_networks` chains (seeded per chain from `rng`, concatenated in
# order, so the result is thread-count independent) and each ego sample.
# The MCMC budget is the package's one dyad-scaled rule (`_mcmc_controls`),
# not the pre-panel literals `burnin=2000, interval=200`, which on the
# 205-actor census were 0.1 % of the fit's burn-in.
function _gof_simulations(result::EgoERGMResult, n_sim::Int,
                          rng::Random.AbstractRNG;
                          burnin::Union{Int, Nothing}=nothing,
                          interval::Union{Int, Nothing}=nothing,
                          n_chains::Int=min(n_sim, 4))
    n_sim >= 1 || throw(ArgumentError("gof: n_sim must be ≥ 1 (got $n_sim)"))
    n_chains >= 1 || throw(ArgumentError("gof: n_chains must be ≥ 1 (got $n_chains)"))
    model = result.model
    ed = model.data
    m = model.ppopsize
    n_egos = length(ed)

    # Coefficients on the pseudo-population scale (undo the adjustment)
    θ = copy(result.coefficients)
    edges_idx = findfirst(t -> t isa EgoEdges, model.ego_terms)
    θ[edges_idx] -= result.netsize_adjustment

    ctl = _mcmc_controls(m; burnin=burnin, interval=interval)
    net = _pseudo_population(ed, m, model.targets[edges_idx] / (m * (m - 1) / 2), rng)
    ergm_model = ERGMModel(ERGMFormula(model.ergm_terms), net)
    sims = sample_networks(ergm_model, θ; n_sim=n_sim, burnin=ctl.burnin,
                           interval=ctl.interval, rng=rng, n_chains=n_chains)

    obs = summary_stats(ed)
    # The ego degree distribution (gof.ergm.ego's GOF="degree"): the
    # design-weighted proportion of egos at each degree, observed and per
    # simulated sample, over R's bins — `degree(0:(maxdeg−1)) +
    # degrange(maxdeg)` with maxdeg = 2·max(K, 3), K the largest OBSERVED
    # ego degree, so the bins reach twice the observed maximum and close
    # with a "≥ maxdeg" tail. Without the tail (the round-2 code stopped at
    # K) a model that over-produces high degrees put up to 5 % of its
    # simulated egos in no row at all: the rows did not sum to 1 and the
    # failure mode this diagnostic exists for was invisible. R's one
    # exception is kept too: when maxdeg ≥ m − 1 every degree a network of m
    # vertices can have is already a bin, so there is no tail.
    degree_terms, degree_labels = _degree_gof_bins(Int(obs.max_degree), m)
    obs_degree = [compute(t, ed) for t in degree_terms]
    sim_mean_degree = Float64[]
    sim_mean_aaties = Float64[]
    sim_degree = Matrix{Float64}(undef, length(sims), length(degree_terms))
    for (k, s) in enumerate(sims)
        sample_ed = simulate_ego_sample(s, min(n_egos, Int(nv(s))); rng=rng)
        ss = summary_stats(sample_ed)
        push!(sim_mean_degree, ss.mean_degree)
        push!(sim_mean_aaties, ss.mean_alter_ties)
        for (j, t) in enumerate(degree_terms)
            sim_degree[k, j] = compute(t, sample_ed)
        end
    end

    return obs, sim_mean_degree, sim_mean_aaties, obs_degree, sim_degree, degree_labels
end

# The upper-tail bin of the degree GOF: the design-weighted proportion of
# egos with degree ≥ d (R's `degrange(d)` on an egor). Descriptive only, like
# EgoDegree — never fittable — and internal: it exists so every simulated
# ego lands in exactly one row of the "degree distribution" statistic.
struct _EgoDegreeAtLeast <: EgoTerm
    d::Int
end
name(t::_EgoDegreeAtLeast) = "ego.degree.$(t.d)+"
_ego_contribution(t::_EgoDegreeAtLeast, e::EgoNetwork) = Float64(ego_degree(e) >= t.d)

# gof.ergm.ego's GOF="degree" bins for an observed maximum ego degree K on a
# pseudo-population of m vertices: `maxdeg = 2·max(K, 3)`; degrees
# `0:maxdeg−1` plus a `≥ maxdeg` tail, or `0:m−1` with no tail when
# `maxdeg ≥ m − 1` (every attainable degree is then its own bin)
function _degree_gof_bins(K::Int, m::Int)
    maxdeg = 2 * max(K, 3)
    if maxdeg >= m - 1
        terms = EgoTerm[EgoDegree(d) for d in 0:(m - 1)]
        labels = ["degree $d" for d in 0:(m - 1)]
    else
        terms = EgoTerm[EgoDegree(d) for d in 0:(maxdeg - 1)]
        push!(terms, _EgoDegreeAtLeast(maxdeg))
        labels = ["degree $d" for d in 0:(maxdeg - 1)]
        push!(labels, "degree ≥ $maxdeg")
    end
    return terms, labels
end

"""
    gof(result::EgoERGMResult; n_sim=50, rng=Random.default_rng(),
        burnin=nothing, interval=nothing, n_chains=min(n_sim, 4)) -> GOFResult

Goodness-of-fit assessment of a fitted egocentric ERGM: pseudo-population
networks are simulated at the fitted (pseudo-population scale)
coefficients, ego samples of the observed size are drawn from each, and
two sets of observed design-weighted statistics are compared against their
simulated distributions — `gof.ergm.ego`'s two diagnostics:

1. `"ego summary statistics"` — the mean degree and the mean alter–alter tie
   count (`GOF = "model"`). Note that every model includes `EgoEdges()`, so
   the mean degree is a **fitted target**: its p-value says whether the
   chain reproduced the target, not whether the model fits (a large
   p-value there is expected of any converged fit). Mean alter ties is the
   informative row of the pair — a model without a triangle term misses it.
2. `"degree distribution"` — the proportion of egos at each degree, over
   `gof.ergm.ego`'s bins (`GOF = "degree"`): `degree 0 … degree maxdeg−1`
   plus an upper tail `degree ≥ maxdeg`, with `maxdeg = 2·max(K, 3)` and
   `K` the largest **observed** ego degree — R's `degree(0:(maxdeg−1)) +
   degrange(maxdeg)` — so the bins reach twice the observed maximum and
   every simulated ego falls in exactly one row (observed and every
   simulated row sum to 1; the observed tail is 0 by construction, and a
   model that over-produces high degrees shows up there). When `maxdeg ≥
   ppopsize − 1` every attainable degree is its own bin and there is no
   tail, as in R. [`EgoDegree`](@ref) is evaluated on each simulated
   sample. An edges-only model matches the mean degree by construction but
   not the distribution around it, which is what this statistic detects.

This is a method of the shared `Networks.gof` generic; it returns the
shared `Networks.GOFResult` (observed value, simulation envelope, and
two-sided Monte-Carlo p-value per level from `Networks.mc_pvalue`, the
`(1 + k)/(N + 1)` estimator, so it is never exactly zero).

Every random draw — the pseudo-population, the MCMC chains and the ego
samples — flows through `rng`, so two calls from the same `rng` state give
identical simulated statistics, and the multi-chain sampler is seeded per
chain from `rng` so the result does not depend on the thread count.

[`ego_gof`](@ref) is the legacy NamedTuple-returning form of the same
simulations.

# Keyword Arguments
- `n_sim::Int=50`: number of simulated networks
- `rng::AbstractRNG=Random.default_rng()`: source of every random draw
- `burnin`, `interval`: MCMC controls of the network sampler; `nothing`
  (the default) selects the dyad-scaled rule the fit uses
  ([`_mcmc_controls`](@ref): `20·n_dyads` and `max(100, n_dyads ÷ 10)`)
- `n_chains::Int=min(n_sim, 4)`: chains the `n_sim` networks are split over
  (`ERGM.sample_networks`, one seed per chain drawn from `rng`); for a given
  `n_chains` and `rng` state the result is bit-identical whatever the thread
  count (changing `n_chains` changes the chain seeding, hence the draws)

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
g = gof(fit; n_sim=20, rng=Xoshiro(5))
g.statistics[1].labels                          # ["mean degree", "mean alter ties"]
g.statistics[2].labels[1:3]                     # ["degree 0", "degree 1", "degree 2"]
g.statistics[2].labels[end]                     # "degree ≥ 26" — the tail bin, maxdeg = 2·max(13, 3); K = 13 observed
all(sum(g.statistics[2].simulated; dims=2) .≈ 1) # true — every simulated ego lands in one row
g.statistics[1].simulated == gof(fit; n_sim=20, rng=Xoshiro(5)).statistics[1].simulated   # true
g.statistics[1].p_values[1] == mc_pvalue(g.statistics[1].simulated[:, 1], g.statistics[1].observed[1])   # true
```
"""
function gof(result::EgoERGMResult; n_sim::Int=50,
             rng::Random.AbstractRNG=Random.default_rng(),
             burnin::Union{Int, Nothing}=nothing,
             interval::Union{Int, Nothing}=nothing,
             n_chains::Int=min(n_sim, 4))
    obs, sim_mean_degree, sim_mean_aaties, obs_degree, sim_degree, degree_labels =
        _gof_simulations(result, n_sim, rng; burnin, interval, n_chains)
    # p-values are the shared `Networks.mc_pvalue` (GOFStatistic's default)
    stat = GOFStatistic("ego summary statistics",
                        ["mean degree", "mean alter ties"],
                        [obs.mean_degree, obs.mean_alter_ties],
                        hcat(sim_mean_degree, sim_mean_aaties))
    degree = GOFStatistic("degree distribution", degree_labels, obs_degree, sim_degree)
    return GOFResult([stat, degree]; model="Egocentric ERGM")
end

"""
    ego_gof(result::EgoERGMResult; n_sim=50, rng=Random.default_rng(),
            burnin=nothing, interval=nothing, n_chains=min(n_sim, 4)) -> NamedTuple

Goodness of fit for an egocentric ERGM as a NamedTuple
`(observed, simulated, p_values, n_sim)` keyed by `mean_degree` /
`mean_alter_ties`: `observed` the design-weighted sample statistics,
`simulated` the means over the simulated ego samples, `p_values` the
two-sided Monte-Carlo p-values (`Networks.mc_pvalue`).

A thin wrapper over [`gof`](@ref) — the same simulations, the same
keywords, the same p-value estimator, read out of the `GOFResult`'s first
statistic — so the two cannot disagree. Prefer `gof`, which returns the
shared `Networks.GOFResult` and also carries the degree distribution
(`gof.ergm.ego`'s `GOF = "degree"`), which this NamedTuple does not.

# Example
```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
e = ego_gof(fit; n_sim=20, rng=Xoshiro(5))
g = gof(fit; n_sim=20, rng=Xoshiro(5))
e.p_values.mean_degree == g.statistics[1].p_values[1]   # true
e.observed.mean_degree == summary_stats(ed).mean_degree # true
```
"""
function ego_gof(result::EgoERGMResult; n_sim::Int=50,
                 rng::Random.AbstractRNG=Random.default_rng(),
                 burnin::Union{Int, Nothing}=nothing,
                 interval::Union{Int, Nothing}=nothing,
                 n_chains::Int=min(n_sim, 4))
    g = gof(result; n_sim, rng, burnin, interval, n_chains)
    stat = g.statistics[1]
    sim = stat.simulated
    return (
        observed = (mean_degree = stat.observed[1],
                    mean_alter_ties = stat.observed[2]),
        simulated = (mean_degree = mean(view(sim, :, 1)),
                     mean_alter_ties = mean(view(sim, :, 2))),
        p_values = (mean_degree = stat.p_values[1],
                    mean_alter_ties = stat.p_values[2]),
        n_sim = n_sim
    )
end

end # module
