"""
    ERGMEgo.jl - ERGMs for Ego-Centric Network Data

Fits ERGMs to egocentrically sampled network data (a sample of "egos" with
their local networks: alters and ties among alters), enabling inference
about complete-network properties from ego samples.

The methodology follows R `ergm.ego` (Krivitsky & Morris 2017): ego
statistics are design-weighted and scaled to **target statistics** for a
pseudo-population network of size `ppopsize`, an ERGM is fit to those
targets by method-of-moments (MCMC moment matching with ERGM.jl's MCMLE
step and stopping rule), with `ergm.ego`'s network-size offset
`netsize.adj = −log(ppopsize/popsize)`, so the coefficients are those of a
population of `popsize` members — per capita when the population size is
unknown. Standard errors are `ergm.ego`'s decomposition: the survey-design
variance of the targets sandwiched by the inverse information, plus the
Monte-Carlo estimation term of the moment equations.

Port of the R ergm.ego package from the StatNet collection.

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)                       # 205 students, 203 ties
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))   # a census
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                     # true
coef(fit)                         # ≈ [-6.03, 2.83], ergm.ego's estimates for popsize = 205
gof(fit; n_sim=20, rng=Xoshiro(2)) # a NetworkCore.GOFResult: model statistics, degrees, shared partners
```
"""
module ERGMEgo

using DataFrames
using ERGM
using LinearAlgebra
using NetworkCore          # graph primitives (nv, neighbors, has_edge, vertices) are its re-exports
using Random
using SpecialFunctions: erfcinv, gamma_inc_inv
using Statistics
using StatsBase

# The shared statistic protocol (NetworkCore.jl `src/statistics.jl`): `name` and
# `compute` are the ONE pair of generics every model package extends
# (`ERGMEgo.name === NetworkCore.name`); `summary_stats` is ERGM.jl's
import NetworkCore: name, compute
import ERGM: summary_stats
# ERGM.jl's public MCMLE convergence machinery (t-ratios, Hotelling T²,
# Geyer effective sample size), reused by the moment-matching loop instead of
# an ad-hoc relative-change rule, and from its extension API
# (`ERGM.Extension`, the semver-covered surface for packages built on
# ERGM.jl) the one dyad-scaled sampler rule, `mcmle`'s own sampler (R ergm
# 4's SPDyad proposal and ESS-adaptive, continued chains), R ergm's
# confidence stopping rule and the attainable range, which ERGMEgo extends
# for its ego terms
import ERGM: mcmc_convergence, MCMLEConvergence
import ERGM.Extension: mcmc_defaults, mcmle_sampler, confidence_test, attainable_range
# Shared presentation infrastructure (NetworkCore.jl): the ONE `gof` generic all
# model packages extend, the common coefficient table and printer, the ONE
# z → p helper, and the GOF containers
import NetworkCore: gof, print_coeftable, CoefficientTable, z_pvalues,
                 GOFStatistic, GOFResult, check_se, bootstrap_cov
using Logging: with_logger, NullLogger

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): the
# generic accessors that say what a fit actually did. Imported by name because
# ERGMEgo adds methods for `EgoERGMResult`; `fit_metadata(fit)` collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 approximations
import StatsAPI
import StatsAPI: coef, coefnames, stderror, vcov, confint, nobs, dof, coeftable

# Data structures
export EgoData, EgoNetwork
export n_alters, ego_degree, alter_degree, n_alter_ties

# Data preparation
export as_egodata, ego_design

# Ego-specific terms and statistics. `compute` and `name` are the shared
# NetworkCore.jl statistic generics (re-exported, as ERGM.jl re-exports them),
# so `compute(EgoEdges(), ed)` works with `using ERGMEgo` alone. `EgoTerm`
# is exported as ERGM.jl exports `AbstractERGMTerm`: it is the type in
# `fit_ergm_ego`'s signature and what a custom ego term subtypes
export EgoTerm, EgoEdges, EgoNodeMatch, EgoDegree, EgoGWDegree, EgoTriangle
export EgoNodeFactor, EgoNodeCov, EgoAbsDiff, EgoGWESP, EgoESP, EgoMM, EgoConcurrent
export ego_mixing_matrix, ego_target_stats
export summary_stats, compute, name

# The two hooks a custom ego term extends (API reference, "Terms"): its
# per-ego contribution and, to be fittable, the ERGM.jl term it estimates.
# Stable names a user extends and a test may pin, not exports
public ergm_term, ego_contribution

# Estimation
export fit_ergm_ego, ergm_ego, EgoERGMModel, EgoERGMResult

# Population size estimation
export estimate_popsize

# Simulation
export simulate_ego_sample

# Diagnostics: gof is a method of the shared NetworkCore.jl generic; ego_gof is
# the legacy NamedTuple-returning form
export gof, ego_gof

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using
# ERGMEgo`). Deliberately absent: `loglikelihood`, `aic`, `bic` — the fit is
# moment matching (`objective(fit) == :moment`); no likelihood is evaluated.
export coef, coefnames, stderror, vcov, confint, coeftable, nobs, dof

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
each is refused naming the ego and the alter. So are an ego listed as its
own alter and an alter attribute whose length is not the number of alters.
The constructor
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
try EgoNetwork(5, [5, 11], zeros(Bool, 2, 2)) catch e; e isa ArgumentError end   # true — ArgumentError: ego 5 is listed as its own alter
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
        # The ego listed as its own alter would raise its degree (and every
        # statistic built on it) by one
        ego in alters && throw(ArgumentError(
            "EgoNetwork: ego $ego is listed as its own alter; an ego is never " *
            "its own alter (it would add one to the ego's degree in every " *
            "statistic). Drop that row."))
        # One attribute value per alter, in the order of `alters`
        for (attr, vals) in alter_attrs
            length(vals) == n_alters || throw(ArgumentError(
                "EgoNetwork: alter attribute :$attr of ego $ego has " *
                "$(length(vals)) values for $(n_alters) alters; give one value " *
                "per alter, in the order of `alters`."))
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

`design[:alter_ties_observed] == false` records that the ties among alters
were **not collected** ([`as_egodata`](@ref) sets it when no `aatie_df` is
given). An uncollected tie is unobserved, not absent: every statistic that
reads alter–alter ties ([`EgoTriangle`](@ref), the shared-partner panel of
[`gof`](@ref)) then refuses with an `ArgumentError` instead of counting
zero triangles, and `summary_stats(ed).mean_alter_ties` is `NaN`.
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

# Whether the ties among alters were collected (the default), or are
# unobserved (`as_egodata` without `aatie_df`)
_alter_ties_observed(ed::EgoData) = get(ed.design, :alter_ties_observed, true) === true

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
`mean_alter_ties` (`NaN` when the alter–alter ties were not collected), the
unweighted `total_alters` (the number of ego–alter rows) and
`population_size`. An `EgoData` with no egos (an `as_egodata`
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
        mean_alter_ties = _alter_ties_observed(ed) ? mean(alter_ties, w) : NaN,
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
  `ego_id`, `source_col`, `target_col` (alter IDs). Without it the ties
  among alters are recorded as **not collected**
  (`ed.design[:alter_ties_observed] == false`): [`EgoTriangle`](@ref) and
  the shared-partner goodness-of-fit panel refuse such data rather than
  read "no triangles" from it. Pass an empty frame (with the three columns)
  when the ties were collected and there are none.

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

    design = Dict{Symbol, Any}()
    isnothing(aatie_df) && (design[:alter_ties_observed] = false)
    return EgoData(egos; sampling_weights=weights,
                   population_size=population_size, design=design)
end

"""
    ego_design(ed::EgoData; popsize=nothing, weights=nothing) -> EgoData

Attach survey-design information (population size and/or per-ego weights)
to ego data, returning a new `EgoData` (the egos are shared, the weights
replaced when given, the population size replaced when given). This is the
whole of the design an `EgoData` can express: independent egos with case
weights — there is no strata/cluster/finite-population-correction input.

`popsize` is the size of the **population** the egos were sampled from
(`ergm.ego`'s `popsize`), stored as `ed.population_size`. It is not the
pseudo-population size, which is the `ppopsize` keyword of
[`fit_ergm_ego`](@ref).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [11], zeros(Bool, 1, 1))
ed = ego_design(EgoData([e1, e2]); popsize=1000, weights=[3.0, 1.0])
ed.population_size, ed.sampling_weights    # (1000, [3.0, 1.0])
estimate_popsize(ed)                       # 4.0 — Horvitz–Thompson: the weight sum
```
"""
function ego_design(ed::EgoData{T};
                    popsize::Union{Int, Nothing}=nothing,
                    weights::Union{Vector{Float64}, Nothing}=nothing) where T
    isnothing(popsize) || popsize >= 1 || throw(ArgumentError(
        "ego_design: popsize must be a positive number of population members (got $popsize)"))
    new_weights = isnothing(weights) ? ed.sampling_weights : weights

    return EgoData(ed.egos;
                   population_size=something(popsize, ed.population_size, Some(nothing)),
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
#   EgoDegree       1[degree_i = d]                 Degree(d)
#   EgoNodeFactor   (1[x_i = l]·degree_i + #alters with x = l)/2   NodeFactor(attr; level=l)
#   EgoNodeCov      (x_i·degree_i + Σ_alters x)/2   NodeCov(attr)
#   EgoAbsDiff      Σ_alters |x_i − x|^pow / 2      AbsDiff(attr; pow)
#
# Each is derived from the term's definition seen from one ego, and equals
# ergm.ego's per-ego value (its `nodefactor`, `nodecov` and `absdiff` with
# alter attributes observed), as the fixtures check. A multi-level
# EgoNodeFactor expands into one-level terms against the ego data
# (`_expand_ego_terms`), as ERGM.jl's NodeFactor expands against a network.

"""
    EgoTerm <: AbstractERGMTerm

The abstract type of every ego statistic — the `terms::Vector{<:EgoTerm}`
of [`fit_ergm_ego`](@ref) and the type a custom ego term subtypes (exported
as ERGM.jl exports `AbstractERGMTerm`). A subtype provides

- `name(t)::String` — its label (a method of the shared `NetworkCore.name`);
- `ERGMEgo.ego_contribution(t, e::EgoNetwork)::Float64` — the per-ego
  contribution, whose design-weighted mean is `compute(t, ed)` (the shared
  `NetworkCore.compute`; the generic method over `EgoTerm` does the weighting);
- and, to be fittable, [`ERGMEgo.ergm_term`](@ref)`(t)` — the ERGM.jl
  term whose sufficient statistic `m · compute(t, ed)` estimates (a `public`
  hook, like `ego_contribution`). Without it the term is descriptive only
  (`compute` works) and `fit_ergm_ego` refuses it with an `ArgumentError`. The built-in fittable terms are `EgoEdges`,
  `EgoNodeMatch`, `EgoNodeFactor`, `EgoNodeCov`, `EgoAbsDiff`, `EgoDegree`,
  `EgoTriangle`, `EgoGWDegree`, `EgoGWESP`, `EgoESP`, `EgoMM` and
  `EgoConcurrent`; `ergm.ego`'s other terms have no ego counterpart yet.

# Example
```julia
using ERGMEgo
struct EgoIsolate <: EgoTerm end                     # proportion of egos with no alters
ERGMEgo.name(::EgoIsolate) = "isolates"
ERGMEgo.ego_contribution(::EgoIsolate, e::EgoNetwork) = Float64(n_alters(e) == 0)
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, Int[], Matrix{Bool}(undef, 0, 0))
compute(EgoIsolate(), EgoData([e1, e2]))             # 0.5
EgoIsolate() isa EgoTerm                             # true
```
"""
abstract type EgoTerm <: AbstractERGMTerm end

"""
    ERGMEgo.ego_contribution(term::EgoTerm, e::EgoNetwork) -> Float64

The contribution of one ego's local network `e` to the per-capita statistic
of `term`: [`compute`](@ref)`(term, ed)` is its design-weighted mean over
the egos of `ed`. Every built-in [`EgoTerm`](@ref) is a method of it (`EgoEdges`
contributes `degree/2`, `EgoTriangle` the ties among the alters over 3), and
a custom ego term adds one. A `public` (not exported) name: the hook a custom
term extends, with [`ERGMEgo.ergm_term`](@ref) to make it fittable.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], zeros(Bool, 3, 3))
ERGMEgo.ego_contribution(EgoEdges(), e1)        # 1.5 — degree 3, halved
struct EgoIsolated <: EgoTerm end
ERGMEgo.name(::EgoIsolated) = "isolated"
ERGMEgo.ego_contribution(::EgoIsolated, e::EgoNetwork) = Float64(n_alters(e) == 0)
ERGMEgo.ego_contribution(EgoIsolated(), e1)     # 0.0
Base.ispublic(ERGMEgo, :ego_contribution)       # true
```
"""
function ego_contribution end

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

The label of an ego term: `ergm.ego`'s label for the statistic it
estimates, which is R ergm's label of the same term — `edges`,
`nodematch.<attr>`, `nodefactor.<attr>.<level>`, `nodecov.<attr>`,
`absdiff.<attr>`, `degree<d>`, `triangle`, `gwdeg.fixed.<decay>`,
`gwesp.fixed.<decay>`, `esp<k>`, `mm[<attr>=<l1>,<attr>=<l2>]`,
`concurrent` — and so the label of its row in `coeftable(fit)`
and of its coefficient in [`coefnames`](@ref)`(fit)`, so a coefficient is
found by its R name (`coeftable(fit)["edges"]`). For a fittable term it is
`name(ERGMEgo.ergm_term(term))`, except `EgoMM`, whose ERGM.jl counterpart
`NodeMix` carries R's `nodemix` label. A method of the shared `NetworkCore.name`
generic (`ERGMEgo.name === NetworkCore.name`).

# Example
```julia
using ERGMEgo
name(EgoEdges()), name(EgoNodeMatch(:Grade)), name(EgoGWDegree(0.5))
# ("edges", "nodematch.Grade", "gwdeg.fixed.0.5") — ergm.ego's labels
name(EgoDegree(0)), name(EgoGWESP(0.0))   # ("degree0", "gwesp.fixed.0")
```
"""
name(::EgoEdges) = "edges"

ego_contribution(::EgoEdges, e::EgoNetwork) = ego_degree(e) / 2

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

name(t::EgoNodeMatch) = "nodematch.$(t.attr)"

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

function ego_contribution(t::EgoNodeMatch, e::EgoNetwork)
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
statistic. Data whose alter–alter ties were not collected
(`ed.design[:alter_ties_observed] == false`) is refused with an
`ArgumentError`.

In a fit whose pseudo-population size differs from the population size, a
triangle term changes the network-size offset as in `ergm.ego`: the offset
statistic is `edges − transitiveties/3` instead of `edges` (see
[`fit_ergm_ego`](@ref)).

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

name(::EgoTriangle) = "triangle"

ego_contribution(::EgoTriangle, e::EgoNetwork) = n_alter_ties(e) / 3

# Whether a term reads the ties among alters (and so cannot be computed on
# data whose alter–alter ties were not collected), and whether it is a
# triadic (order-3) statistic, for which ergm.ego's network-size offset
# carries the `transitiveties = -1/3` adjustment
_needs_alter_ties(::EgoTerm) = false
_needs_alter_ties(::EgoTriangle) = true
_is_triadic(::EgoTerm) = false
_is_triadic(::EgoTriangle) = true

function _require_alter_ties(t::EgoTerm, ed::EgoData)
    (_needs_alter_ties(t) && !_alter_ties_observed(ed)) && throw(ArgumentError(
        "$(name(t)): the ties among alters were not collected for this EgoData " *
        "(`design[:alter_ties_observed] == false`, e.g. `as_egodata` without " *
        "`aatie_df`); an uncollected tie is unobserved, not absent, so the " *
        "statistic is not computed from it. Supply the alter–alter ties, or " *
        "drop the term."))
    return nothing
end

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

name(t::EgoGWDegree) = name(GWDegree(t.decay))

function ego_contribution(t::EgoGWDegree, e::EgoNetwork)
    d = ego_degree(e)
    α = t.decay
    return d > 0 ? exp(α) * (1 - (1 - exp(-α))^d) : 0.0
end

"""
    EgoGWESP(decay::Real=0.5) <: EgoTerm

Per-capita geometrically weighted edgewise shared partners (`ergm.ego`'s
`gwesp(decay, fixed=TRUE)`). Seen from an ego, the shared partners of its
tie to alter `a` are the ego's other alters tied to `a` — `a`'s degree
`s` among the alter–alter ties — so the per-ego value is
`Σ_alters w(s)/2` with `w(s) = e^α(1 − (1 − e^{−α})^s)`, `α = decay`
(each tie is seen from both of its ends, hence the half). Scaled by the
pseudo-population size it estimates ERGM.jl's `GWESP(decay)`, labelled
`gwesp.fixed.<decay>` (R's name). At `decay = 0` it is the number of ties
with at least one shared partner (`ergm.ego`'s help-page model uses
`gwesp(0, fixed=TRUE)`). `decay` must be non-negative.

It reads the alter–alter ties, so data whose alter ties were not collected
is refused, as for [`EgoTriangle`](@ref); and, like `EgoTriangle`, it is a
triadic statistic: in a fit whose pseudo-population size differs from the
population size the network-size offset is `edges − transitiveties/3`, as
in `ergm.ego`.

The decay is fixed: `EgoGWESP(decay; fixed=false)` (`ergm.ego`'s curved
`gwesp(decay, fixed=FALSE)`) is refused with an `ArgumentError`.

There is no shared-partner cutoff. `ergm.ego`'s `gwesp` takes `cutoff`
(default 30) and leaves out of its statistic every tie with more than
`cutoff` shared partners: on a census of the complete graph on 33 vertices
(31 shared partners per tie) `ergm.ego` returns 0 where `ergm`'s own
`gwesp(0.5, fixed=TRUE)` returns 870.52. `EgoGWESP` weights every tie by its
full count, as `ergm`'s `gwesp` and ERGM.jl's `GWESP` (the statistic it
estimates) do, so it differs from `ergm.ego` only on data in which an ego and
one of its alters share more than 30 alters.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], Bool[0 1 1; 1 0 0; 1 0 0])   # alter 10 shares 11 and 12
e2 = EgoNetwork(2, [10, 11], Bool[0 1; 1 0])
ed = EgoData([e1, e2])
compute(EgoGWESP(0.0), ed)          # 1.25 — (3/2 + 2/2)/2: every tie has a shared partner
name(EgoGWESP(0.5))                 # "gwesp.fixed.0.5"
```
"""
struct EgoGWESP <: EgoTerm
    decay::Float64

    function EgoGWESP(decay::Real=0.5; fixed::Bool=true)
        fixed || throw(ArgumentError(
            "EgoGWESP: the curved gwesp(decay, fixed=FALSE), whose decay is " *
            "estimated, is not implemented for egocentric data (ERGM.jl's curved " *
            "MCMLE fits a whole observed network, not target statistics); fix the " *
            "decay — EgoGWESP(decay) is gwesp(decay, fixed=TRUE)"))
        decay >= 0 || throw(ArgumentError("EgoGWESP: decay must be non-negative (got $decay)"))
        new(Float64(decay))
    end
end

name(t::EgoGWESP) = name(GWESP(t.decay))
_needs_alter_ties(::EgoGWESP) = true
_is_triadic(::EgoGWESP) = true

# Provenance: implemented from the published definition only. The statistic
# is Hunter's (2007) geometrically weighted edgewise shared partner count
# (ergm's `gwesp` term documentation; ERGM.jl's `GWESP`), and the per-ego
# reduction below is derived here from that definition, within the
# egocentric framework of Krivitsky & Morris (2017) that ergm.ego's term
# documentation describes and that the package's other ego terms follow.
# No code of ergm.ego or ergm was consulted. It is validated against
# ergm.ego's outputs, as a black box, by the golden fixture `ego_terms.toml`,
# and against ERGM.jl's `GWESP` on whole networks by brute force.
#
# Derivation. GWESP_α = Σ over ties {i,j} of e^α(1 − q^{sp(i,j)}), with
# q = 1 − e^{−α} and sp(i,j) the number of vertices tied to both i and j.
# For a tie between ego i and its alter a, the vertices tied to i are i's
# alters, so sp(i,a) is the number of i's alters tied to a: a's column count
# in `alter_ties`. Summing over i's alters visits every population tie once
# from each of its two ends, so the ego's share is half of that sum (as
# `EgoEdges` is half the degree). At α = 0, q = 0 and q^0 = 1, so a tie
# counts 1 exactly when it has a shared partner. No allocation.
function ego_contribution(t::EgoGWESP, e::EgoNetwork)
    A = e.alter_ties
    k = size(A, 1)
    q = 1 - exp(-t.decay)
    acc = 0.0                      # Σ over alters of (1 − q^{shared partners})
    @inbounds for a in 1:k
        sp = 0
        for b in 1:k
            sp += A[b, a]
        end
        acc += 1 - q^sp
    end
    return exp(t.decay) * acc / 2
end

"""
    EgoESP(k::Integer) <: EgoTerm

Per-capita edgewise shared partners (`ergm.ego`'s `esp(k)`): the
design-weighted mean of half the number of the ego's alters that have
exactly `k` shared partners with the ego — an alter's shared partners with
the ego are the ego's other alters tied to it, so its count is its degree
among the alter–alter ties. Scaled by the pseudo-population size it
estimates ERGM.jl's `ESP(k)`, the number of ties with exactly `k` shared
partners (R's label `esp<k>`). R's `esp(0:3)` is `EgoESP.(0:3)` (one term
per count); `k` must be non-negative.

It reads the alter–alter ties, so data whose alter ties were not collected
is refused, as for [`EgoTriangle`](@ref); and it is a triadic statistic: in
a fit whose pseudo-population size differs from the population size the
network-size offset is `edges − transitiveties/3`, as in `ergm.ego`.
`EgoGWESP(α)` is `Σₖ eᵅ(1 − (1 − e⁻ᵅ)ᵏ)·EgoESP(k)`.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11, 12], Bool[0 1 1; 1 0 0; 1 0 0])   # alter 10 shares 11 and 12
e2 = EgoNetwork(2, [10, 11], Bool[0 1; 1 0])
ed = EgoData([e1, e2])
compute(EgoESP(1), ed)              # 1.0 — (2/2 + 2/2)/2: alters 11, 12 of ego 1; both of ego 2
compute(EgoESP(2), ed)              # 0.25 — alter 10 of ego 1, halved, averaged
name.(EgoESP.(0:1))                 # ["esp0", "esp1"] — R's esp(0:1)
```
"""
struct EgoESP <: EgoTerm
    k::Int

    function EgoESP(k::Integer)
        k >= 0 || throw(ArgumentError(
            "EgoESP: the shared-partner count must be non-negative (got $k)"))
        return new(Int(k))
    end
end

name(t::EgoESP) = "esp$(t.k)"
_needs_alter_ties(::EgoESP) = true
_is_triadic(::EgoESP) = true

# Provenance: implemented from the published definition only — ergm's `esp`
# term documentation (the number of ties with exactly k shared partners), in
# the egocentric framework of Krivitsky & Morris (2017) that ergm.ego's term
# documentation describes. No code of ergm.ego or ergm was consulted. It is
# validated against ergm.ego's outputs, as a black box, by the golden fixture
# `ego_mixing_esp.toml`, and against ERGM.jl's `ESP` on whole networks by
# brute force. Derivation: as for `EgoGWESP`, the shared partners of the tie
# between ego i and its alter a are i's other alters tied to a (a's column
# count in `alter_ties`), and each population tie is seen from both of its
# ends, so the ego's share is half the number of its alters whose count is
# k. No allocation.
function ego_contribution(t::EgoESP, e::EgoNetwork)
    A = e.alter_ties
    k = size(A, 1)
    n_k = 0
    @inbounds for a in 1:k
        sp = 0
        for b in 1:k
            sp += A[b, a]
        end
        n_k += sp == t.k
    end
    return n_k / 2
end

"""
    EgoDegree(d::Integer) <: EgoTerm

Per-capita degree count: the design-weighted proportion of egos with degree
exactly `d` (`d ≥ 0`). Estimates `ergm.ego`'s `degree(d)` — the number of
vertices of degree `d` — i.e. ERGM.jl's `Degree(d)` (R's label `degree<d>`).
R's `degree(1:3)` is `EgoDegree.(1:3)` (one term per degree). The `by=` and
`homophily=` forms of R's term are not ported.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [10], zeros(Bool, 1, 1))
e3 = EgoNetwork(3, [11, 12], zeros(Bool, 2, 2))
ed = EgoData([e1, e2, e3])
compute(EgoDegree(2), ed)           # 0.6667 — two of three egos have degree 2
compute(EgoDegree(0), ed)           # 0.0
name.(EgoDegree.(1:2))              # ["degree1", "degree2"] — R's degree(1:2)
```
"""
struct EgoDegree <: EgoTerm
    d::Int

    function EgoDegree(d::Integer)
        d >= 0 || throw(ArgumentError("EgoDegree: the degree must be non-negative (got $d)"))
        return new(Int(d))
    end
end

name(t::EgoDegree) = "degree$(t.d)"

ego_contribution(t::EgoDegree, e::EgoNetwork) = Float64(ego_degree(e) == t.d)

"""
    EgoConcurrent() <: EgoTerm

Per-capita concurrency (`ergm.ego`'s `concurrent`): the design-weighted
proportion of egos with two or more alters. Scaled by the pseudo-population
size it estimates ERGM.jl's `Concurrent()`, the number of vertices of degree
two or more (R's label `concurrent`). Like [`EgoDegree`](@ref) it is a
property of the ego, not of a tie, so it is not halved. R's `by=` form
(concurrency by an attribute) is not implemented and is refused.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2))
e2 = EgoNetwork(2, [10], zeros(Bool, 1, 1))
e3 = EgoNetwork(3, [11, 12, 13], zeros(Bool, 3, 3))
compute(EgoConcurrent(), EgoData([e1, e2, e3]))   # 0.6667 — two of three egos have ≥ 2 alters
name(EgoConcurrent())                              # "concurrent"
```
"""
struct EgoConcurrent <: EgoTerm
    function EgoConcurrent(; by=nothing)
        by === nothing || throw(ArgumentError(
            "EgoConcurrent: R's concurrent(by=) — concurrency counted separately " *
            "for each level of an attribute — is not implemented; use " *
            "EgoConcurrent() for the whole population"))
        return new()
    end
end

name(::EgoConcurrent) = "concurrent"

ego_contribution(::EgoConcurrent, e::EgoNetwork) = Float64(ego_degree(e) >= 2)

# The attribute checks every attribute term runs on one ego: carried on both
# sides, and no `missing` value (the same errors as `EgoNodeMatch`)
function _require_attribute(t::EgoTerm, attr::Symbol, e::EgoNetwork)
    haskey(e.ego_attrs, attr) ||
        throw(_missing_attribute_error(name(t), attr, e, "ego", e.ego_attrs))
    haskey(e.alter_attrs, attr) ||
        throw(_missing_attribute_error(name(t), attr, e, "alter", e.alter_attrs))
    _check_attribute_values(t, attr, e)
    return nothing
end

"""
    EgoNodeFactor(attr; levels=nothing, base=1) <: EgoTerm
    EgoNodeFactor(attr; level=x) <: EgoTerm

Per-capita main effect of a categorical attribute (`ergm.ego`'s
`nodefactor(attr)`): for a level `x`, the design-weighted mean of
`(1[ego is x]·degree + #alters that are x) / 2` — each tie of the
population has two endpoints and is reported by both of them, hence the
halving. Scaled to a network of `m` vertices it estimates the number of
times a vertex of level `x` is an endpoint of a tie, ERGM.jl's
`NodeFactor(attr; level=x)` (R's label `nodefactor.<attr>.<x>`).

The levels follow `ergm.ego`: the sorted distinct values **of the egos**,
with the first dropped as the reference category (`base=1`, R's default
`levels = -1`). `base` gives the indices into the sorted levels to drop
(`base=0` keeps every level, R's `levels = TRUE`); `levels` names the
included values explicitly. A multi-level term expands into one statistic
per included level when the model is fitted (and in
[`ego_target_stats`](@ref)), so `coef(fit)` has one row per level;
`compute` of the unexpanded term is the sum over its levels, as for
ERGM.jl's `NodeFactor`. `level=x` builds a single-level term directly.

Every ego must carry `attr` for itself and for its alters (an
`ArgumentError` names the ego otherwise; `ergm.ego`'s fallback for data
without alter attributes, which counts the ego's endpoint only, is not
ported).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2);
                ego_attrs=Dict{Symbol,Any}(:sex => "F"),
                alter_attrs=Dict{Symbol,Vector}(:sex => ["F", "M"]))
e2 = EgoNetwork(2, [12], zeros(Bool, 1, 1);
                ego_attrs=Dict{Symbol,Any}(:sex => "M"),
                alter_attrs=Dict{Symbol,Vector}(:sex => ["M"]))
ed = EgoData([e1, e2])
compute(EgoNodeFactor(:sex; level="M"), ed)    # 0.75 — ((0 + 1) + (1 + 1))/2 per ego, averaged
ego_target_stats([EgoNodeFactor(:sex)], ed, 10)                   # [7.5] — level "M" only: "F" is the reference
ego_target_stats([EgoNodeFactor(:sex; base=0)], ed, 10)            # [7.5, 7.5]
```
"""
struct EgoNodeFactor <: EgoTerm
    attr::Symbol
    level::Any
    levels::Union{Nothing, Vector{Any}}
    base::Vector{Int}

    function EgoNodeFactor(attr::Symbol; level=nothing, levels=nothing, base=1)
        (level !== nothing && levels !== nothing) && throw(ArgumentError(
            "EgoNodeFactor: give either `level` (a single-level term) or `levels`, not both"))
        base_vec = base isa Integer ? (base == 0 ? Int[] : Int[base]) : Int[b for b in base]
        all(>=(1), base_vec) || throw(ArgumentError(
            "EgoNodeFactor: base indices must be positive (use base=0 to keep all levels)"))
        return new(attr, level, levels === nothing ? nothing : collect(Any, levels), base_vec)
    end
end

name(t::EgoNodeFactor) =
    t.level === nothing ? "nodefactor.$(t.attr)" : "nodefactor.$(t.attr).$(t.level)"

# The included levels against the ego data: R's `ergm.ego_attr_levels` over
# the EGOS' values, sorted, minus `base` (or the explicit `levels`)
function _nodefactor_levels(t::EgoNodeFactor, ed::EgoData)
    vals = Any[]
    for e in ed.egos
        haskey(e.ego_attrs, t.attr) ||
            throw(_missing_attribute_error(name(t), t.attr, e, "ego", e.ego_attrs))
        v = e.ego_attrs[t.attr]
        ismissing(v) && throw(_missing_value_error(name(t), t.attr, e, true, 0))
        push!(vals, v)
    end
    lv = sort!(unique(vals))
    if t.levels !== nothing
        unknown = [l for l in t.levels if !(l in lv)]
        isempty(unknown) || throw(ArgumentError(
            "$(name(t)): levels $(unknown) are not a value of any ego (the egos' " *
            "levels: $(lv))"))
        return copy(t.levels)
    end
    keep = Any[l for (k, l) in enumerate(lv) if !(k in t.base)]
    isempty(keep) && throw(ArgumentError(
        "$(name(t)): no levels remain after dropping base level(s) $(t.base) of " *
        "$(lv); pass base=0 or levels=[...]"))
    return keep
end

# The one-level ego terms a term stands for: a multi-level `EgoNodeFactor`
# expands (ERGM.jl's NodeFactor expansion, against the ego data), every
# other term is itself
_expand_ego_term(t::EgoTerm, ::EgoData) = EgoTerm[t]
_expand_ego_term(t::EgoNodeFactor, ed::EgoData) =
    t.level === nothing ?
    EgoTerm[EgoNodeFactor(t.attr; level=l) for l in _nodefactor_levels(t, ed)] : EgoTerm[t]
_expand_ego_terms(terms, ed::EgoData) =
    reduce(vcat, (_expand_ego_term(t, ed) for t in terms); init=EgoTerm[])

# Function barrier (see `_count_matches`): whether the ego has the level
_is_level(v, level) = isequal(v, level)

function ego_contribution(t::EgoNodeFactor, e::EgoNetwork)
    t.level === nothing && throw(ArgumentError(
        "$(name(t)): a multi-level term has one statistic per level; it is " *
        "expanded against the ego data (fit_ergm_ego, ego_target_stats, compute), " *
        "or build one level with EgoNodeFactor(:$(t.attr); level=x)"))
    _require_attribute(t, t.attr, e)
    ego_in = _is_level(e.ego_attrs[t.attr], t.level)::Bool
    n_in = _count_matches(e.alter_attrs[t.attr], t.level)::Int
    return (ego_in * ego_degree(e) + n_in) / 2
end

function compute(t::EgoNodeFactor, ed::EgoData)
    t.level === nothing || return _wmean([ego_contribution(t, e) for e in ed.egos],
                                         ed.sampling_weights)
    return sum(compute(s, ed) for s in _expand_ego_term(t, ed))
end

# The numeric attribute terms read the ego's value and the alters' column,
# both abstractly typed at the call site. `_numeric_call` branches on the two
# common concrete layouts (Int and Float64 values with a column of the same
# type, which `simulate_ego_sample` and `as_egodata` build), so the kernel is
# a static call there and allocates nothing; any other layout goes through a
# dynamic call (which boxes the Float64 it returns). A value that is not a
# real number is an `ArgumentError`, never a conversion error from inside the
# loop.
function _numeric_error(t::EgoTerm, attr::Symbol, e::EgoNetwork, v)
    return ArgumentError("$(name(t)): attribute :$attr of ego $(e.ego) (or of one of " *
                         "its alters) is $(repr(v)) of type $(typeof(v)); the term needs " *
                         "a real number for the ego and every alter")
end

function _nodecov_sum(t, e, xe, alters::AbstractVector)
    xe isa Real || throw(_numeric_error(t, t.attr, e, xe))
    s = Float64(xe) * length(alters)
    for a in alters
        a isa Real || throw(_numeric_error(t, t.attr, e, a))
        s += Float64(a)
    end
    return s / 2
end

function _absdiff_sum(t, e, xe, alters::AbstractVector)
    xe isa Real || throw(_numeric_error(t, t.attr, e, xe))
    s = 0.0
    for a in alters
        a isa Real || throw(_numeric_error(t, t.attr, e, a))
        s += abs(Float64(xe) - Float64(a))^t.pow
    end
    return s / 2
end

"""
    EgoNodeCov(attr) <: EgoTerm

Per-capita main effect of a numeric attribute (`ergm.ego`'s
`nodecov(attr)`): the design-weighted mean of `(x_ego·degree + Σ x_alter)/2`
(each tie reported by both endpoints, hence the halving). Scaled to a
network of `m` vertices it estimates the sum of the attribute over the
endpoints of every tie, ERGM.jl's `NodeCov(attr)` (R's label
`nodecov.<attr>`). Every ego must carry a real-valued `attr` for itself and
its alters.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2);
                ego_attrs=Dict{Symbol,Any}(:age => 20),
                alter_attrs=Dict{Symbol,Vector}(:age => [30, 40]))
e2 = EgoNetwork(2, Int[], zeros(Bool, 0, 0);
                ego_attrs=Dict{Symbol,Any}(:age => 50),
                alter_attrs=Dict{Symbol,Vector}(:age => Int[]))
compute(EgoNodeCov(:age), EgoData([e1, e2]))   # 27.5 — (20·2 + 30 + 40)/2 = 55 and 0, averaged
name(EgoNodeCov(:age))                         # "nodecov.age"
```
"""
struct EgoNodeCov <: EgoTerm
    attr::Symbol
end

name(t::EgoNodeCov) = "nodecov.$(t.attr)"

@inline function _numeric_call(kernel::F, t, e, xe, col) where {F}
    if xe isa Float64 && col isa Vector{Float64}
        return kernel(t, e, xe, col)
    elseif xe isa Int && col isa Vector{Int}
        return kernel(t, e, xe, col)
    else
        return kernel(t, e, xe, col)::Float64
    end
end

function ego_contribution(t::EgoNodeCov, e::EgoNetwork)
    _require_attribute(t, t.attr, e)
    return _numeric_call(_nodecov_sum, t, e, e.ego_attrs[t.attr], e.alter_attrs[t.attr])
end

"""
    EgoAbsDiff(attr; pow=1) <: EgoTerm

Per-capita absolute difference of a numeric attribute across ties
(`ergm.ego`'s `absdiff(attr, pow)`): the design-weighted mean of
`Σ_alters |x_ego − x_alter|^pow / 2`. Scaled to a network of `m` vertices
it estimates `Σ_ties |xᵢ − xⱼ|^pow`, ERGM.jl's `AbsDiff(attr; pow)` (R's
labels `absdiff.<attr>`, and `absdiff<pow>.<attr>` for `pow ≠ 1`). Every
ego must carry a real-valued `attr` for itself and its alters.

# Example
```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2);
               ego_attrs=Dict{Symbol,Any}(:grade => 9),
               alter_attrs=Dict{Symbol,Vector}(:grade => [9, 12]))
compute(EgoAbsDiff(:grade), EgoData([e]))            # 1.5 — (0 + 3)/2
compute(EgoAbsDiff(:grade; pow=2), EgoData([e]))     # 4.5 — (0 + 9)/2
name(EgoAbsDiff(:grade; pow=2))                      # "absdiff2.grade"
```
"""
struct EgoAbsDiff <: EgoTerm
    attr::Symbol
    pow::Float64

    EgoAbsDiff(attr::Symbol; pow::Real=1) = new(attr, Float64(pow))
end

name(t::EgoAbsDiff) = name(AbsDiff(t.attr; pow=t.pow))

function ego_contribution(t::EgoAbsDiff, e::EgoNetwork)
    _require_attribute(t, t.attr, e)
    return _numeric_call(_absdiff_sum, t, e, e.ego_attrs[t.attr], e.alter_attrs[t.attr])
end

"""
    EgoMM(attr) <: EgoTerm
    EgoMM(attr, l1, l2) <: EgoTerm

Per-capita mixing-matrix cells of a categorical attribute (`ergm.ego`'s
`mm(attr)`, the default form `mm(~attr, levels2 = -1)`): for the cell of
levels `(l1, l2)`, the design-weighted mean of half the number of the ego's
alters such that the ego and the alter carry `l1` and `l2`, in either order.
Scaled by the pseudo-population size it estimates the number of ties
between a vertex of level `l1` and one of level `l2`, ERGM.jl's
`NodeMix(attr, l1, l2)`; R's label is `mm[<attr>=<l1>,<attr>=<l2>]`.

`EgoMM(attr)` is a specification that expands into one statistic per cell
when the model is fitted (and in [`ego_target_stats`](@ref)), as `ergm.ego`
does: the levels are the sorted values found at either end of a reported
ego–alter tie (an alter's level counts even when no ego carries it; an ego
with no alters adds none), the cells are the unordered pairs `l1 ≤ l2` in
R's order — `(u₁,u₁), (u₁,u₂), (u₂,u₂), (u₁,u₃), …` — and the first cell is
dropped as the reference (R's `levels2 = -1`). `compute` of the unexpanded
term is the sum of its cells. `EgoMM(attr, l1, l2)` builds one cell
directly.

Not implemented, and refused: R's two-attribute form `mm(A ~ B)` (written
`EgoMM(:A, :B)`), its margins `mm(A ~ .)`, and the `levels=`/`levels2=`
selections. Every ego must carry `attr` for itself and its alters.
`ergm.ego` 1.1.4's `mm` stops with an error on a weighted design; `EgoMM`
weights its per-ego values as every other term does.

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [10, 11], zeros(Bool, 2, 2); ego_attrs=Dict{Symbol,Any}(:g => "A"),
                alter_attrs=Dict{Symbol,Vector}(:g => ["A", "B"]))
e2 = EgoNetwork(2, [12], zeros(Bool, 1, 1); ego_attrs=Dict{Symbol,Any}(:g => "B"),
                alter_attrs=Dict{Symbol,Vector}(:g => ["B"]))
ed = EgoData([e1, e2])
compute(EgoMM(:g, "A", "B"), ed)          # 0.25 — ego 1's one A–B alter, halved, averaged
ego_target_stats([EgoMM(:g)], ed, 10)     # [2.5, 2.5] — cells A–B and B–B; A–A is the reference
name(EgoMM(:g, "A", "B"))                 # "mm[g=A,g=B]"
```
"""
struct EgoMM <: EgoTerm
    attr::Symbol
    l1::Any
    l2::Any

    function EgoMM(attr::Symbol; levels=nothing, levels2=-1)
        levels === nothing || throw(ArgumentError(
            "EgoMM: R's mm(levels=) selection is not implemented; EgoMM(:$attr) " *
            "uses every level found at the ends of the reported ties"))
        levels2 == -1 || throw(ArgumentError(
            "EgoMM: R's mm(levels2=) cell selection other than the default -1 (the " *
            "first cell dropped) is not implemented; build single cells with " *
            "EgoMM(:$attr, l1, l2)"))
        return new(attr, nothing, nothing)
    end
    EgoMM(attr::Symbol, l1, l2) = new(attr, l1, l2)
end

EgoMM(row::Symbol, col::Symbol) = throw(ArgumentError(
    "EgoMM: two-attribute mixing (R's mm($row ~ $col)) and the margins of a mixing " *
    "matrix (mm(A ~ .)) are not implemented; EgoMM(:attr) is R's mm(~attr), and " *
    "EgoMM(:attr, l1, l2) one of its cells"))

name(t::EgoMM) = t.l1 === nothing ? "mm.$(t.attr)" :
    "mm[$(t.attr)=$(t.l1),$(t.attr)=$(t.l2)]"

# The levels of an `mm` specification: the sorted values at either end of a
# reported ego–alter tie (ergm.ego's choice, pinned by `ego_mixing_esp.toml`:
# a level only the alters carry is included, the level of an ego with no
# alters is not)
function _mm_levels(t::EgoMM, ed::EgoData)
    vals = Any[]
    for e in ed.egos
        _require_attribute(t, t.attr, e)
        n_alters(e) == 0 && continue
        push!(vals, e.ego_attrs[t.attr])
        append!(vals, e.alter_attrs[t.attr])
    end
    return sort!(unique(vals))
end

# The cells of an `mm` specification in R's order, the first dropped
function _mm_cells(t::EgoMM, ed::EgoData)
    lv = _mm_levels(t, ed)
    cells = [(lv[i], lv[j]) for j in eachindex(lv) for i in 1:j]
    length(cells) >= 2 || throw(ArgumentError(
        "$(name(t)): the reported ties carry $(length(lv)) level(s) of :$(t.attr) " *
        "($(lv)), so after the reference cell is dropped no cell remains"))
    return cells[2:end]
end

# A specification expands into its cells against the ego data
_expand_ego_term(t::EgoMM, ed::EgoData) =
    t.l1 === nothing ? EgoTerm[EgoMM(t.attr, a, b) for (a, b) in _mm_cells(t, ed)] :
                       EgoTerm[t]

function ego_contribution(t::EgoMM, e::EgoNetwork)
    t.l1 === nothing && throw(ArgumentError(
        "$(name(t)): a mixing-matrix specification has one statistic per cell; it " *
        "is expanded against the ego data (fit_ergm_ego, ego_target_stats, compute), " *
        "or build one cell with EgoMM(:$(t.attr), l1, l2)"))
    _require_attribute(t, t.attr, e)
    x = e.ego_attrs[t.attr]
    col = e.alter_attrs[t.attr]
    n = 0
    _is_level(x, t.l1)::Bool && (n += _count_matches(col, t.l2)::Int)
    (!isequal(t.l1, t.l2) && _is_level(x, t.l2)::Bool) &&
        (n += _count_matches(col, t.l1)::Int)
    return n / 2
end

function compute(t::EgoMM, ed::EgoData)
    t.l1 === nothing || return _wmean([ego_contribution(t, e) for e in ed.egos],
                                      ed.sampling_weights)
    return sum(compute(s, ed) for s in _expand_ego_term(t, ed))
end

"""
    compute(term::EgoTerm, ed::EgoData) -> Float64

The design-weighted mean per-ego contribution of the term (a per-capita
statistic; multiply by a network size to get a target sufficient
statistic — see [`ego_target_stats`](@ref)). A method of the shared
`NetworkCore.compute` statistic protocol (`ERGMEgo.compute === NetworkCore.compute`).

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
    _require_alter_ties(term, ed)
    h = [ego_contribution(term, e) for e in ed.egos]
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
    ERGMEgo.ergm_term(term::EgoTerm) -> AbstractERGMTerm

The ERGM.jl term whose sufficient statistic `m · compute(term, ed)`
estimates — the hook that makes an ego term **fittable** by
[`fit_ergm_ego`](@ref): `EgoEdges → Edges()`, `EgoNodeMatch(a) →
NodeMatch(a)`, `EgoNodeFactor(a; level=l) → NodeFactor(a; level=l)`,
`EgoNodeCov(a) → NodeCov(a)`, `EgoAbsDiff(a; pow) → AbsDiff(a; pow)`,
`EgoDegree(d) → Degree(d)`, `EgoTriangle → Triangle()`, `EgoGWDegree(α) →
GWDegree(α)`, `EgoGWESP(α) → GWESP(α)`, `EgoESP(k) → ESP(k)`,
`EgoMM(a, l1, l2) → NodeMix(a, l1, l2)`, `EgoConcurrent() → Concurrent()`.
A term without a method (a custom [`EgoTerm`](@ref) that has
not added one, or the goodness-of-fit statistics) is descriptive only, and
`fit_ergm_ego` refuses it with an `ArgumentError` naming the term. A
`public` (not exported) name, as [`ERGMEgo.ego_contribution`](@ref) is:
add a method to it (with the matching `ego_contribution`) to fit a custom
term, provided the per-ego contribution really is an unbiased per-capita
estimate of the ERGM term's statistic.

# Example
```julia
using ERGMEgo, ERGM
ERGMEgo.ergm_term(EgoEdges())                       # Edges()
name(ERGMEgo.ergm_term(EgoGWDegree(0.5)))           # "gwdeg.fixed.0.5" — R's label
name(ERGMEgo.ergm_term(EgoNodeFactor(:Sex; level="M")))   # "nodefactor.Sex.M"
struct EgoTwoStar <: EgoTerm end                     # per-capita 2-stars: C(degree, 2) — each 2-star has ONE centre, no divisor
ERGMEgo.name(::EgoTwoStar) = "kstar2"                # R's label for kstar(2)
ERGMEgo.ego_contribution(::EgoTwoStar, e::EgoNetwork) = Float64(binomial(ego_degree(e), 2))
ERGMEgo.ergm_term(::EgoTwoStar) = Kstar(2)          # now fittable
Base.ispublic(ERGMEgo, :ergm_term)                  # true
```
"""
ergm_term(::EgoEdges) = Edges()
ergm_term(t::EgoNodeMatch) = NodeMatch(t.attr)
ergm_term(::EgoTriangle) = Triangle()
ergm_term(t::EgoGWDegree) = GWDegree(t.decay)
ergm_term(t::EgoGWESP) = GWESP(t.decay)
ergm_term(t::EgoDegree) = Degree(t.d)
ergm_term(t::EgoNodeCov) = NodeCov(t.attr)
ergm_term(t::EgoAbsDiff) = AbsDiff(t.attr; pow=t.pow)
ergm_term(t::EgoESP) = ESP(t.k)
ergm_term(::EgoConcurrent) = Concurrent()
function ergm_term(t::EgoMM)
    t.l1 === nothing && throw(ArgumentError(
        "$(name(t)) is a mixing-matrix specification: expand it against the ego " *
        "data first (fit_ergm_ego does), or build one cell with EgoMM(:$(t.attr), l1, l2)"))
    return NodeMix(t.attr, t.l1, t.l2)
end
function ergm_term(t::EgoNodeFactor)
    t.level === nothing && throw(ArgumentError(
        "$(name(t)) is a multi-level term: expand it against the ego data first " *
        "(fit_ergm_ego does), or build one level with EgoNodeFactor(:$(t.attr); level=x)"))
    return NodeFactor(t.attr; level=t.level)
end
ergm_term(t::EgoTerm) =
    throw(ArgumentError("$(name(t)) is a descriptive statistic with no " *
                        "ERGM.jl counterpart; it cannot be used in ergm_ego"))

"""
    ego_target_stats(terms, ed::EgoData, m::Int) -> Vector{Float64}

Target sufficient statistics for a network of size `m`: `m` times the
design-weighted per-capita ego statistics — what `ergm.ego` passes to
`ergm` as `target.stats` (R's `summary(egor ~ …, scaleto = m)`). A
multi-level [`EgoNodeFactor`](@ref) gives one target per included level.
Under a census (every vertex an ego, unit weights, `m = nv`) they reduce
exactly to the network's own statistics.

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Sex], rng=Xoshiro(1))   # a census
ego_target_stats([EgoEdges(), EgoNodeMatch(:Grade)], ed, 205)   # [203.0, 163.0] — the network's own
ego_target_stats([EgoNodeFactor(:Sex), EgoAbsDiff(:Grade), EgoDegree(0)], ed, 205)   # [171.0, 79.0, 57.0] — R's nodefactor.Sex.M, absdiff.Grade, degree0
ego_target_stats([EgoEdges()], ed, 1000)                         # [990.24] — scaled to 1000 vertices
```
"""
ego_target_stats(terms, ed::EgoData, m::Int) =
    [m * compute(t, ed) for t in _expand_ego_terms(terms, ed)]

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
- `ergm_terms::Vector{AbstractERGMTerm}`: the terms of the ERGM simulated on
  the pseudo-population — the ERGM.jl term each ego term estimates, in
  order, followed by `Offset(GWESP(0.0), …)` when the network-size offset
  carries the triangle adjustment (see [`fit_ergm_ego`](@ref))
- `data::EgoData{T}`: the ego data the fit was made on
- `ppopsize::Int`: the realised pseudo-population size
- `popsize::Int`: the population size; `1` when it is unknown (`ergm.ego`'s
  default — the coefficients are then per capita)
- `targets::Vector{Float64}`: target statistics on the pseudo-population
  scale ([`ego_target_stats`](@ref))
- `ppop_counts::Vector{Int}`: how many pseudo-population vertices replicate
  each ego (`round(ppopsize · wᵢ/Σw)`; they sum to `ppopsize`)

# Example
```julia
using ERGMEgo, NetworkCore, Random
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
    ppop_counts::Vector{Int}
end

function Base.show(io::IO, m::EgoERGMModel{T}) where T
    print(io, "EgoERGMModel{$T}: ", _plural(length(m.data), "ego"),
          "; terms: ", join((name(t) for t in m.ego_terms), " + "),
          "; ppopsize $(m.ppopsize), popsize ",
          m.popsize == 1 ? "unknown (1: per-capita coefficients)" : string(m.popsize),
          "; targets [",
          join((_fmt3(x) for x in m.targets), ", "), "]")
    return nothing
end

"""
    EgoERGMResult{T}

Results from [`fit_ergm_ego`](@ref); `T` is the vertex-ID type of the ego
data (`EgoERGMModel{T}`).

# Fields
- `model::EgoERGMModel{T}`: the fitted specification
- `coefficients`: the coefficients for a population of `model.popsize`
  members — per capita when the population size is unknown (`popsize = 1`).
  They are the free coefficients of `ergm.ego`'s offset model, the numbers
  `ergm.ego` prints below its `netsize.adj` row
- `std_errors`: standard errors, `sqrt.(diag(vcov))`
- `vcov`: the covariance of the coefficients, `vcov_design + vcov_estimation`
  — `ergm.ego`'s decomposition (`vcov(fit, sources="all")`)
- `vcov_design`: the survey-design component `I⁻¹ Σ_design I⁻¹` (times
  `1 + n_egos/popsize` when `se_type == :superpopulation`), where
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
- `netsize_adjustment`: the coefficient of the network-size offset,
  `-log(ppopsize/popsize)` — R's `netsize.adj` row. The edges coefficient
  of the simulated pseudo-population is `coefficients[edges] +
  netsize_adjustment`. (Before 0.2 this field held the negative of R's
  value.)
- `converged`: whether the termination rule passed (`termination.rule`):
  R ergm's confidence test by default
- `termination`: `(rule, p_value, precision, confidence, n_samples)` — the
  rule, its p-value on the final sample (`:confidence` passes when it is
  below `1 − confidence`; `:hotelling` when it is above `hotelling_alpha`
  and every t-ratio is below `conv_threshold`), the size of that sample, and
  the coefficient step the equivalence test was evaluated at (`step`: the
  full Newton step taken from a sample that passed in the loop; zeros for a
  fresh sample drawn at the returned coefficients after `maxiter`, and
  under `:hotelling`, which tests the sample at its own coefficients)
- `mcmc_convergence::ERGM.MCMLEConvergence`: the diagnostics of the same
  sample — `(iterations, step_length, t_ratios, hotelling_p, n_eff)`: the
  iterations run, the last Hummel step length (1.0 for a full step), the
  per-statistic t-ratios, the Hotelling T² p-value and the Geyer effective
  sample size
- `sim_stats`: that sample (pseudo-population scale, one column per ego
  term) — the one `converged`, `termination`, `mcmc_convergence` and `vcov`
  describe. It is the sample that passed the test, from which the last step
  was taken (R's `ergm` design: no fresh draw after convergence), or, when
  `maxiter` was exhausted, a fresh sample at the returned coefficients that
  decided `converged`
- `se_type`: `:design` (`ergm.ego`'s standard errors), `:superpopulation`
  (the design component multiplied by `1 + n_egos/popsize`) or `:bootstrap`
  (a bootstrap over egos; `vcov_design` is then the bootstrap covariance
  less `vcov_estimation`); see [`fit_ergm_ego`](@ref) for what each does and
  does not cover
- `boot_replicates`: the `n_boot × p` refitted coefficients under
  `se=:bootstrap` (a `NaN` row for a refit that did not converge), `0 × p`
  otherwise

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
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                                  # true
fit.mcmc_convergence.n_eff > 100               # true
coeftable(fit)["edges"].estimate == coef(fit)[1]   # true — R's label
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
    termination::@NamedTuple{rule::Symbol, p_value::Float64, precision::Float64,
                             confidence::Float64, n_samples::Int, step::Vector{Float64}}
    se_type::Symbol
    boot_replicates::Matrix{Float64}
end

_fmt3(x::Real) = isfinite(x) ? string(round(x; sigdigits=3)) : string(x)

# One sentence naming the termination rule's verdict
function _termination_detail(t)
    if t.rule === :confidence
        return "$(round(Int, 100 * t.confidence))% equivalence test p " *
               "$(_fmt3(t.p_value)) (needs < $(_fmt3(1 - t.confidence)); tolerance " *
               "precision $(_fmt3(t.precision)), $(t.n_samples) draws)"
    else
        return "t-ratio + Hotelling rule, Hotelling p $(_fmt3(t.p_value)) " *
               "($(t.n_samples) draws)"
    end
end

# THE non-convergence sentence: printed by `show` under `Converged: false`,
# listed by `approximations` and quoted by the `@warn` in `fit_ergm_ego`,
# from the same numbers, so the three cannot disagree
function _nonconvergence_caveat(c::MCMLEConvergence, t, maxiter::Union{Nothing,Int}=nothing)
    cap = maxiter === nothing ? "" : " in maxiter=$maxiter iterations"
    # The numbers quoted are those of the sample at the RETURNED coefficients
    # — the one `converged` was decided on
    return "moment matching did not converge$cap ($(_termination_detail(t)); on " *
           "the final sample at the returned coefficients: max t-ratio " *
           "$(_fmt3(maximum(c.t_ratios))), Hotelling p $(_fmt3(c.hotelling_p)), " *
           "step length $(_fmt3(c.step_length)) after $(c.iterations) " *
           "iteration$(c.iterations == 1 ? "" : "s")): the estimates do not solve " *
           "the moment equations and the standard errors are unreliable — " *
           "increase maxiter/n_samples, or burnin/interval for a better-mixing chain"
end
_nonconvergence_caveat(result::EgoERGMResult) =
    _nonconvergence_caveat(result.mcmc_convergence, result.termination)

# R's "MCMC %" for an egocentric fit: `100 · (se − se_design)/se`, the share
# of the TOTAL standard error that the Monte-Carlo estimation term adds over
# the design part, rounded to an integer (NaN when the SE is NaN)
function _mcmc_percent(result::EgoERGMResult)
    se_design = sqrt.(max.(diag(result.vcov_design), 0.0))
    return [isfinite(se) && isfinite(d) && se > 0 ? round(Int, 100 * (se - d) / se) : NaN
            for (d, se) in zip(se_design, result.std_errors)]
end

# THE standard-error caveat: printed by `show`, listed by `approximations`
function _se_caveat(result::EgoERGMResult)
    m = result.model
    base = "the design variance component treats the egos as independent draws " *
           "with the given case weights (no strata, clusters, finite-population " *
           "correction, replicate weights or without-replacement inclusion " *
           "probabilities)"
    few = length(m.data) < 50 ? " With $(length(m.data)) egos, prefer `se=:bootstrap`" *
          " (it resamples the egos and refits, so it also carries the sampling" *
          " error of the pseudo-population's attribute composition)" :
          ". `se=:bootstrap` also carries the sampling error of the" *
          " pseudo-population's attribute composition, at the cost of n_boot refits"
    if result.se_type === :bootstrap
        B = size(result.boot_replicates, 1)
        n_ok = count(b -> all(isfinite, view(result.boot_replicates, b, :)), 1:B)
        known = m.popsize > 1 && length(m.data) <= m.popsize
        return "the standard errors are a bootstrap over egos (se=:bootstrap): " *
               "$n_ok of $B resamples of the egos, each with its sampling weight, " *
               "were refitted from a rebuilt pseudo-population and targets" *
               (n_ok < B ? " ($(B - n_ok) refits did not converge and are excluded. " *
                           _BOOT_EXCLUSION_BIAS * ")" : "") *
               (known ? ", and n_egos/popsize = $(_fmt3(length(m.data) / m.popsize)) " *
                        "times the design sandwich is added because a tie between " *
                        "two sampled egos is reported by both" :
                        "; the population size is unknown, so nothing is added for " *
                        "a tie between two sampled egos being reported by both") *
               ". The egos are resampled as independent draws: no strata, clusters, " *
               "finite-population correction or replicate weights"
    end
    if result.se_type === :superpopulation
        f = length(m.data) / m.popsize
        return base * ", multiplied by 1 + n_egos/popsize = $(_fmt3(1 + f)) " *
               "(se=:superpopulation) because a tie between two sampled egos is " *
               "reported by both: exact for tie-sum statistics (edges, nodematch) " *
               "under equal-probability sampling, approximate otherwise. In " *
               "simulation nominal 95 % intervals for edges cover 92–95 % with " *
               "50 or more egos; an attribute term (nodematch) covers 85–95 %, " *
               "and with about 20 egos they cover 87 % and 78 %, because the " *
               "pseudo-population's attribute composition is itself estimated " *
               "from the sample" * few * ". " *
               "`se=:design` gives ergm.ego's standard errors"
    end
    return base * ". These are ergm.ego's standard errors (se=:design" *
           (m.popsize == 1 ? "; the default when the population size is unknown" : "") *
           "). A tie between two sampled egos is reported by both, which " *
           "this ignores: against the model parameter, nominal " *
           "95 % intervals cover about 91 % at a 10 % sampling fraction and " *
           "80–85 % at a census, and they are not finite-population intervals " *
           "either. `se=:superpopulation`, the default when popsize is known, " *
           "corrects the tie double-count" * few
end

function Base.show(io::IO, result::EgoERGMResult)
    c = result.mcmc_convergence
    m = result.model
    percapita = m.popsize == 1
    println(io, "Egocentric ERGM Results")
    println(io, "=======================")
    println(io, "Egos: $(length(m.data)); pseudo-population: $(m.ppopsize); population: ",
            percapita ? "unknown (popsize = 1)" : string(m.popsize))
    note = _ppopsize_note(m)
    note === nothing || _print_wrapped(io, "Note: " * note * ".")
    println(io, "Network-size offset netsize.adj = -log(ppopsize/popsize): ",
            round(result.netsize_adjustment, digits=4),
            length(m.ergm_terms) > length(m.ego_terms) ?
                " on edges − transitiveties/3" : " on edges")
    println(io, "Converged: $(result.converged)")
    if !result.converged
        # The caveat sits right under the verdict: an unconverged fit must
        # never look like a fit with a footnote
        println(io, "  ", _nonconvergence_caveat(result))
    else
        println(io, "Termination: ", _termination_detail(result.termination))
    end
    bc = _boundary_caveat(result)
    bc === nothing || _print_wrapped(io, "Note: " * bc * ".")
    println(io)
    println(io, percapita ?
        "Coefficients (per capita: popsize unknown; for a population of N, edges − log N):" :
        "Coefficients (population scale, popsize = $(m.popsize)):")
    # Shared ecosystem presentation layer: the printed table IS
    # `coeftable(result)` (a NetworkCore.CoefficientTable rendered through
    # `print_coeftable`), so what is shown and what is inspected agree
    show(io, coeftable(result))

    # ergm.ego's standard-error decomposition, in R's "MCMC %" convention
    names = [name(t) for t in m.ego_terms]
    shares = _mcmc_percent(result)
    println(io)
    println(io, "Std.Error = design component ⊕ MCMC-estimation component " *
                "(n_eff = $(round(c.n_eff, digits=1)) on the final sample)")
    println(io, "MCMC % of the standard error (100·(se − se_design)/se): ",
            join(("$(names[k]) $(shares[k])" for k in eachindex(names)), ", "))

    # Honest-uncertainty caveat, the prose twin of `approximations(result)`
    println(io)
    _print_wrapped(io, "Note: " * _se_caveat(result) * ".")
end

# Print `text` wrapped at word boundaries to lines of at most `width` characters
function _print_wrapped(io::IO, text::AbstractString; width::Int=78)
    line = ""
    for word in split(text)
        if !isempty(line) && length(line) + 1 + length(word) > width
            println(io, line)
            line = String(word)
        else
            line = isempty(line) ? String(word) : line * " " * word
        end
    end
    isempty(line) || println(io, line)
    return nothing
end

# ============================================================================
# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`)
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

`:bootstrap` for a fit made with `se=:bootstrap` (a bootstrap over egos);
otherwise `:sandwich`: `V(θ̂) = I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff` — the survey-design
covariance of the target statistics sandwiched by the inverse MCMC information,
plus the Monte-Carlo estimation term of the moment equations (`ergm.ego`'s
`sources="model"` and `sources="estimation"`). `fit.se_type` says whether
`Σ_design` is `ergm.ego`'s (`:design`) or carries the `1 + n_egos/popsize`
factor (`:superpopulation`). See [`approximations`](@ref) for
what that design covariance does and does not encode.
"""
se_method(result::EgoERGMResult) = result.se_type === :bootstrap ? :bootstrap : :sandwich

# Egocentric data is a sample of egos and their reported alters, not a
# sociomatrix with a dyad mask: the missing-dyad concept does not arise.
missing_method(::EgoERGMResult) = :none

"""
    approximations(result::EgoERGMResult) -> Vector{String}

What the reported numbers do and do not account for, as one sentence per
item (the shared `NetworkCore.approximations` protocol, collected by
`fit_metadata`): the Monte-Carlo nature of the moment-matching fit with the
final sample's size, effective size, max t-ratio and Hotelling p; the
pseudo-population construction and the network-size offset; the
independent-egos assumption behind the design variance (no strata, clusters,
finite-population correction, replicate weights or without-replacement
inclusion probabilities) and the under-coverage that follows from ignoring
that a tie between two sampled egos is reported twice (under
`se=:bootstrap`: the resampling, the excluded refits and the robust
scale); and the design + MCMC-estimation composition of the standard
errors. A fit that did not converge lists the non-convergence caveat
first — the same sentence `show` prints and the `@warn` quoted.

# Example
```julia
using ERGMEgo, NetworkCore, Random
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
        "$(result.model.ppopsize), with ergm.ego's network-size offset " *
        "netsize.adj = $(round(result.netsize_adjustment, digits=4)) on " *
        (length(result.model.ergm_terms) > length(result.model.ego_terms) ?
             "edges − transitiveties/3" : "edges") *
        (result.model.popsize == 1 ?
             "; the population size is unknown (popsize = 1), so the " *
             "coefficients are per capita" :
             "; the coefficients are those of a population of " *
             "$(result.model.popsize)"),
        _se_caveat(result),
        result.se_type === :bootstrap ?
        "the bootstrap standard errors are the normalised interquartile ranges " *
        "(IQR/1.349) of $(size(result.boot_replicates, 1)) refits, with the " *
        "refits' correlations; each refit carries its own Monte-Carlo error, so " *
        "the spread includes it — increase n_boot to reduce the bootstrap's own noise" :
        "the standard errors are the design sandwich I⁻¹ Σ_design I⁻¹ plus the " *
        "Monte-Carlo estimation term I⁻¹/n_eff of the moment equations " *
        "(ergm.ego's decomposition); the information I is itself the covariance " *
        "of the statistics on a finite MCMC sample, so both components carry " *
        "Monte-Carlo noise — increase n_samples or interval to reduce it",
    ]
    note = _ppopsize_note(result.model)
    note === nothing || insert!(out, 3, note)
    bc = _boundary_caveat(result)
    bc === nothing || insert!(out, 2, bc)
    result.converged || pushfirst!(out, _nonconvergence_caveat(result))
    return out
end

# StatsAPI interface: methods on the shared statistics generics (mirroring
# ERGM.jl), so `coef(fit)` etc. work on egocentric fits too. `loglikelihood`,
# `aic` and `bic` are deliberately absent (see `EgoERGMResult`).

"""
    coef(result::EgoERGMResult) -> Vector{Float64}

The coefficients for a population of `result.model.popsize` members, in the
order of the model's ego terms (per capita when the population size is
unknown) — a method of `StatsAPI.coef`.

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
coef(fit) == fit.coefficients        # true
coef(fit)[1] + fit.netsize_adjustment    # the edges coefficient of the simulated pseudo-population
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
using ERGMEgo, NetworkCore, Random, LinearAlgebra
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
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
vcov(fit) == fit.vcov_design .+ fit.vcov_estimation       # true
size(vcov(fit))                                           # (2, 2)
```
"""
StatsAPI.vcov(result::EgoERGMResult) = result.vcov

"""
    coeftable(result::EgoERGMResult) -> NetworkCore.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `NetworkCore.CoefficientTable` — exactly the table
`show(result)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`); p-values come from the shared `NetworkCore.z_pvalues`,
except for a coefficient fixed at ∓Inf (its target at a bound of its
statistic: standard error 0, z = ∓Inf, p = 0, as R prints it). Rows can be
read by index or by their label ([`coefnames`](@ref), `ergm.ego`'s).

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
tbl = coeftable(fit)
tbl["edges"].estimate == coef(fit)[1]          # true
tbl[2].p_value == z_pvalues(coef(fit), stderror(fit)).p[2]   # true
```
"""
function StatsAPI.coeftable(result::EgoERGMResult)
    zp = z_pvalues(result.coefficients, result.std_errors)
    z, pv = copy(zp.z), copy(zp.p)
    # A coefficient fixed at ∓Inf (its target at a bound of its statistic)
    # has standard error 0: z = ∓Inf and p = 0, as R prints them
    for k in eachindex(result.coefficients)
        c = result.coefficients[k]
        isinf(c) && (z[k] = c; pv[k] = 0.0)
    end
    return CoefficientTable(coefnames(result), result.coefficients, result.std_errors;
                            z_values=z, p_values=pv)
end

"""
    coefnames(result::EgoERGMResult) -> Vector{String}

The coefficient labels, in `coef(result)` order — `ergm.ego`'s labels
(`names(coef(fit))` in R without its `offset(netsize.adj)` row, which is
`result.netsize_adjustment` here): `edges`, `degree0`,
`nodefactor.Race.Hisp`, `gwesp.fixed.0`, … They are the row labels of
[`coeftable`](@ref)`(result)`. A method of `StatsAPI.coefnames`; the vector
is a fresh copy.

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
coefnames(fit)                              # ["edges", "nodematch.Grade"]
coefnames(fit) == coeftable(fit).names      # true
```
"""
StatsAPI.coefnames(result::EgoERGMResult) = String[name(t) for t in result.model.ego_terms]

"""
    confint(result::EgoERGMResult; level=0.95) -> Matrix{Float64}

Normal-theory confidence intervals `θ̂ ± z_{1−α/2} · se`, one row per
coefficient (lower, upper) — a method of `StatsAPI.confint`. The standard
errors are the design + MCMC-estimation decomposition of
[`EgoERGMResult`](@ref); see [`fit_ergm_ego`](@ref), "Standard errors",
for their measured coverage (`se=:design`, `ergm.ego`'s, covers the model
parameter less often than `level` as the sampling fraction grows).

# Example
```julia
using ERGMEgo, NetworkCore, Random
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
using ERGMEgo, NetworkCore, Random
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
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
dof(fit)                             # 2
```
"""
StatsAPI.dof(result::EgoERGMResult) = count(isfinite, result.coefficients)

# ergm.ego's `ppop.wt = "round"`: ego i is replicated round(m·wᵢ/Σw) times.
# The realised pseudo-population size is the sum of the counts, which can
# differ from the requested `m` (R: "Constructed network has size … different
# from requested …"); the fit uses the realised size everywhere. The counts
# are a function of each ego's own weight only, so the composition of the
# pseudo-population cannot depend on the order of the egos (largest-remainder
# rounding, used before 0.2, gave the extra copies to whichever egos came
# first).
function _ppop_counts(weights::AbstractVector{<:Real}, m::Int)
    total = sum(weights)
    return Int[round(Int, m * w / total) for w in weights]
end

# The ego each pseudo-population vertex replicates (R's `ego.ind`)
function _ppop_ego_index(counts::Vector{Int})
    ind = Vector{Int}(undef, sum(counts))
    v = 0
    for (i, c) in enumerate(counts), _ in 1:c
        ind[v += 1] = i
    end
    return ind
end

# Build the pseudo-population network: vertices whose attributes are the ego
# attributes replicated proportionally to the sampling weights
# (`_ppop_counts`), with edges seeded at the target density
_pseudo_population(ed::EgoData, m::Int, target_density::Float64,
                   rng::Random.AbstractRNG) =
    _pseudo_population(ed, _ppop_counts(ed.sampling_weights, m), target_density, rng)

function _pseudo_population(ed::EgoData, counts::Vector{Int}, target_density::Float64,
                            rng::Random.AbstractRNG)
    ego_ind = _ppop_ego_index(counts)
    m = length(ego_ind)
    net = network(m; directed=false)

    # Assign ego attributes to pseudo-population vertices
    attrs = Dict{Symbol, Dict{Int, Any}}()
    for (v, i) in enumerate(ego_ind)
        for (attr, val) in ed.egos[i].ego_attrs
            get!(attrs, attr, Dict{Int, Any}())[v] = val
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

# Simulated annealing of the pseudo-population toward the target statistics
# (what R's `ergm(target.stats=)` does with `san` before it estimates): tie
# toggles that bring the model statistics closer to `targets`, in the scaled
# distance Σ (gₖ − tₖ)²/max(|tₖ|, 1), are accepted — a worse one with a
# probability that decays to zero. The result is the network the chains start
# from and the one the MPLE start is computed on, so the first sample is drawn
# near the solution instead of at a Bernoulli graph with every other
# coefficient at zero. `stat_idx` are the statistics to match (the offset
# statistic is not one). Every draw comes from `rng`.
function _san!(net::Network, ts, targets::Vector{Float64}, stat_idx::AbstractVector{Int},
               rng::Random.AbstractRNG; n_steps::Int)
    m = Int(nv(net))
    stats = compute_all(ts, net)[stat_idx]
    scale = max.(abs.(targets), 1.0)
    dist(x) = sum(abs2(x[k] - targets[k]) / scale[k] for k in eachindex(x))
    current = dist(stats)
    tie_list = Tuple{Int,Int}[(min(src(e), dst(e)), max(src(e), dst(e))) for e in edges(net)]
    trial = similar(stats)
    for step in 1:n_steps
        current < 1e-12 && break
        temperature = 0.05 * (1 - step / n_steps)
        remove = !isempty(tie_list) && rand(rng) < 0.5
        k = 0
        if remove
            k = rand(rng, 1:length(tie_list))
            i, j = tie_list[k]
        else
            i = rand(rng, 1:m)
            j = rand(rng, 1:(m - 1))
            j >= i && (j += 1)
            i, j = minmax(i, j)
            has_edge(net, i, j) && continue
        end
        delta = change_stat_all(ts, net, i, j)
        sgn = remove ? -1.0 : 1.0
        for (a, c) in enumerate(stat_idx)
            trial[a] = stats[a] + sgn * delta[c]
        end
        proposed = dist(trial)
        gain = current - proposed
        if gain > 0 || (temperature > 0 && rand(rng) < exp(gain / temperature))
            if remove
                rem_edge!(net, i, j)
                tie_list[k] = tie_list[end]
                pop!(tie_list)
            else
                add_edge!(net, i, j)
                push!(tie_list, (i, j))
            end
            stats .= trial
            current = proposed
        end
    end
    return net
end

# The ERGM on the pseudo-population the chains start from: attributes
# replicated by `counts`, ties seeded at the target density and annealed
# toward the targets (the first `p` statistics; a trailing offset statistic
# is not a target)
function _annealed_model(ed::EgoData, counts::Vector{Int}, ergm_terms, targets, p::Int,
                         edges_idx::Int, rng::Random.AbstractRNG)
    m = sum(counts)
    net = _pseudo_population(ed, counts, targets[edges_idx] / (m * (m - 1) / 2), rng)
    formula = ERGMFormula(ergm_terms)
    _san!(net, ERGMModel(formula, net).formula.terms, targets, 1:p, rng;
          n_steps=min(5_000_000, 100 * max(m, ceil(Int, targets[edges_idx]))))
    return ERGMModel(formula, net)
end

# The 0.95 quantile of χ²(p): the squared Mahalanobis radius within which the
# sampled statistic cloud is taken to cover the targets (the Hummel
# step-length rule of `ERGM.mcmle`)
_chisq95(p::Int) = 2 * first(gamma_inc_inv(p / 2, 0.95, 0.05))

# The squared distance of `x` on the tolerance scale of the confidence rule,
# x'(precision·Σ)⁻¹x with Σ the covariance of the sampled statistics
function _tolerance_d2(x::AbstractVector, samples::AbstractMatrix, precision::Float64)
    F = cholesky(Symmetric(precision .* cov(samples)); check=false)
    issuccess(F) || return Inf
    return max(dot(x, F \ x), 0.0)
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
  `ERGM.Extension.mcmc_defaults(n_dyads)`: `20·n_dyads` and `max(100, n_dyads ÷ 10)`
  toggles. On the 205-actor census (20 910 dyads) this is 3000 draws at
  interval 2091 after 418 200 burn-in toggles.

Under [`fit_ergm_ego`](@ref)'s default ESS-adaptive sampling
(`effective_size=64`) the moment-matching chain is burned in for `burnin`
toggles once, its interval starts at `interval ÷ 8` and adapts, the sample
size follows the effective-size target, and `n_samples` only sets the cap,
`max_n_samples = 4·n_samples`; with `effective_size=nothing` every iteration
draws `n_samples` statistics `interval` toggles apart after `burnin`
toggles.

Internal: the API reference describes the rule, and `burnin`/`interval`/
`n_samples` override it on every entry point.

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
    d = mcmc_defaults(n_dyads)
    return (n_samples = something(n_samples, max(400, min(3000, 20 * m))),
            burnin    = something(burnin, d.burnin),
            interval  = something(interval, d.interval))
end

"""
    fit_ergm_ego(ed::EgoData, terms::Vector{<:EgoTerm}; kwargs...) -> EgoERGMResult

Fit an ERGM to egocentrically sampled data, following `ergm.ego`:

1. Build a **pseudo-population** network: each ego's attributes replicated
   `round(ppopsize · wᵢ/Σw)` times (`ergm.ego`'s `ppop.wt = "round"`). The
   realised size `m` is the sum of those counts; it can differ from the
   requested `ppopsize` (an `@info` says so) and is what the fit uses and
   reports. The composition depends on each ego's own weight only, never on
   the order of the egos.
2. Compute design-weighted **target statistics** scaled to `m`.
3. Move the pseudo-population toward the targets by simulated annealing and
   take the pseudo-likelihood estimate on it as the start (what R's
   `ergm(target.stats=)` does with `san`), then fit by **MCMC moment
   matching** with ERGM.jl's MCMLE machinery: at every iteration a sample is
   drawn at the current coefficients and a Hummel-style partial Newton step
   `γ·Σ⁻¹(targets − ḡ)` is taken, with the step length `γ` growing to 1 as
   the sampled cloud comes to cover the targets. The sampler is `ERGM.mcmle`'s
   (`ERGM.Extension.mcmle_sampler`), with R ergm 4's defaults, which
   `ergm.ego` runs underneath: the SPDyad proposal, and **ESS-adaptive
   sampling** on one chain continued from iteration to iteration — each
   sample is extended, its thinning interval doubling, until its effective
   sample size reaches `effective_size` (R's `MCMLE.effectiveSize = 64`).
4. Stop by R ergm's **confidence rule** (`termination=:confidence`, as
   `ERGM.mcmle`): the moment equation at the updated coefficients must lie,
   with `conv_confidence` confidence, inside the tolerance region
   `x'(conv_precision·Σ)⁻¹x ≤ 1`; while it does not, the target effective
   sample size is raised (a fixed-size sample is enlarged), up to
   `max_n_samples` draws. `termination=:hotelling` is the pre-0.2 rule
   (every t-ratio below `conv_threshold` and a Hotelling T² test not
   rejected at `hotelling_alpha`). The sample that passes is the final
   sample: `converged`, `termination`, `mcmc_convergence`, `sim_stats` and
   the standard errors describe that one sample. When `maxiter` is
   exhausted, one more sample is drawn at the returned coefficients and
   that sample decides `converged`.
5. Report the coefficients with `ergm.ego`'s **network-size adjustment**
   (below).

# Network size: `popsize`, `ppopsize` and the offset

As in `ergm.ego`, the pseudo-population model carries the offset
`netsize.adj` with coefficient `-log(m/popsize)`, so the reported
coefficients are those of a population of `popsize` members:

- **`popsize` unknown** (not given, and `ed.population_size` is `nothing`):
  `popsize = 1`, R's default. The edges coefficient is then **per capita**
  (network-size invariant, Krivitsky, Handcock & Morris 2011): it does not
  depend on `ppopsize`, and the edges coefficient for a population of `N`
  members is `coef − log(N)`. (Before 0.2 an unknown population size was
  silently set to `ppopsize`, which put the coefficient on an arbitrary
  scale.)
- **`popsize = N`**: the coefficients are on the scale of a network of `N`
  vertices. `fit.netsize_adjustment` is the offset coefficient
  `-log(m/N)` (R's `netsize.adj` row), and the edges coefficient of the
  simulated pseudo-population is `coef(fit)[edges] + fit.netsize_adjustment`.
- **A triangle or gwesp term** (an order-3 statistic) makes the offset
  statistic `edges − transitiveties/3`
  (R's `offset(netsize.adj(edges=1, transitiveties=-1/3))`) whenever
  `m ≠ popsize`: the pseudo-population model gains
  `Offset(GWESP(0.0), netsize_adjustment · (−1/3))` (the last entry of
  `fit.model.ergm_terms`; `gwesp` at decay 0 counts the ties with at least
  one shared partner). With `popsize = 1` the triangle coefficient is
  therefore R's per-capita one, not the triangle coefficient of a plain
  ERGM on a network of size `m`.

# Targets at a bound

A target at an end of the attainable range of its statistic — `degree0`
when no ego is an isolate, `triangle` or `gwesp` when no alter–alter tie is
reported, a `nodefactor` level no ego or alter has — has no finite
coefficient. As `ergm.ego` does (ergm's check of `target.stats` under its
default `drop=TRUE`), its coefficient is fixed at `-Inf` (`+Inf` at the
top), with a warning in ergm's words; the sampler holds the statistic at
its bound, and the other coefficients are estimated given that. The fixed
coefficient has standard error 0 (p-value 0); `show` and
`approximations(fit)` name it, `dof` does not count it, and `sim_stats`
and `termination.step` hold the estimated statistics only. `drop=false`
refuses such a model with an `ArgumentError` instead, and `se=:bootstrap`
refuses it (the bootstrap needs finite point estimates).

# Standard errors

The standard errors are a sandwich, `V(θ̂) = I⁻¹ Σ I⁻¹ + I⁻¹/n_eff`: the
variance `Σ` of the targets sandwiched by the inverse information, plus the
Monte-Carlo estimation term (see [`EgoERGMResult`](@ref)). Two choices of
`Σ` are offered, and `fit.se_type` records which one a fit used:

- **`se=:design`** is `ergm.ego`'s: `Σ = Σ_design`, the survey-design
  variance of the targets, which treats the egos as independent. They are
  not: a tie between two sampled egos is reported by both. Against the
  model (superpopulation) parameter, nominal 95 % intervals therefore cover
  less often as the sampling fraction `f = n_egos/popsize` grows — 80–85 %
  at a census in simulation — and at a census they are not zero either, so
  they are not finite-population intervals. Use it to reproduce R.
- **`se=:superpopulation`** uses `Σ = (1 + f)·Σ_design`. For a statistic
  that is a sum over ties (edges, nodematch) under equal-probability
  sampling this is the exact variance of the targets under the model: each
  tie of a sampled ego is reported a second time with probability `f`. It
  is approximate for unequal weights and for triangle and degree
  statistics, and it needs a known `popsize` (an `ArgumentError`
  otherwise).

- **`se=:bootstrap`** resamples the egos with replacement (each drawn ego
  keeps its sampling weight), rebuilds the pseudo-population and the
  targets, and refits, `n_boot` times, through the shared
  `NetworkCore.bootstrap_cov` loop. The spread of the refits carries what no
  sandwich does — the sampling error of the pseudo-population's attribute
  composition — besides the design variance and the Monte-Carlo error of a
  fit; `f` times the design sandwich is added for the double-counted ties
  when `popsize` is known. The standard errors are the normalised
  interquartile ranges of the replicates (`IQR/1.349`, with the replicates'
  correlations): resamples with an extreme composition are outliers that
  inflate the plain standard deviation. It costs `n_boot` fits
  (refits run on the available threads; every resample and refit seed is
  drawn from `rng` up front, so the result does not depend on the thread
  count). A refit that does not converge is a `NaN` row of
  `fit.boot_replicates`, excluded, warned about once and listed by
  `approximations(fit)`; the standard errors are then conditional on a
  finite refit and biased downward (the excluded replicates are the extreme
  ones), which the warning, `show` and `approximations` say.

**The default** (`se=nothing`) is `:superpopulation` when the population
size is known and `:design` when it is not (then `f` is unknown; for a
small sampling fraction the two agree). Measured coverage of nominal 95 %
intervals for `edges` / `nodematch` on a 200-actor population, 300
replicates each:

| egos (f) | `:design` | `:superpopulation` |
|---|---|---|
| 200 (1.0) | 0.80 / 0.83 | 0.94 / 0.95 |
| 100 (0.5) | 0.86 / 0.85 | 0.92 / 0.92 |
| 50 (0.25) | 0.91 / 0.87 | 0.93 / 0.91 |
| 20 (0.1) | 0.85 / 0.78 | 0.87 / 0.78 |
| 100 of 1000 (0.1), mean degree 70 | 0.93 / 0.84 | 0.94 / 0.85 |

The correction removes the under-coverage that comes from the
double-counted ties. What remains affects attribute terms: the attribute
composition of the pseudo-population is itself estimated from the sample, a
second-order effect no sandwich captures. It matters with few egos (where
`nodematch` is also biased downward) and when the egos have many ties, so
that the targets are precise relative to the composition. (With the
composition held at its population value the 20-ego intervals cover
0.94 / 0.93.) `se=:bootstrap` captures it: on the hard cases it covers
0.94–0.97 / 0.98 (20 of 200 egos, mean degree 14, two runs of 150 and 100
replicates; default 0.85 / 0.80–0.84) and 0.98 / 0.99 (50 of 200 egos,
mean degree 56, 80 replicates; default 0.89 / 0.84) — conservatively, its standard errors
running 10–35 % above the sampling standard deviation. It costs `n_boot`
fits, so it is not the default; `show` recommends it below 50 egos.
`show` and `approximations(fit)` say what the reported standard errors
are.

A fit that does not converge is **loud**: a warning quoting the termination
test, the max t-ratio, the Hotelling p-value and the iteration count is
emitted, `converged` is `false`, `show` prints the caveat under
`Converged: false`, and `approximations(fit)` lists it. A singular
covariance of the sampled statistics (collinear terms, a degenerate model)
makes the fit step back toward the last coefficients that gave a regular
sample; if that fails too the iterations stop with a warning and the
standard errors are `NaN`.

[`ergm_ego`](@ref) is the same function under `ergm.ego`'s name. The
estimator and the network-size handling are `ergm.ego`'s; the default
standard errors are not when the population size is known (see "Standard
errors": pass `se=:design` for `ergm.ego`'s).

# Keyword Arguments
- `popsize::Union{Int,Nothing}`: population size (default:
  `ed.population_size`; when that is unknown too, 1 — per-capita
  coefficients)
- `ppopsize::Int`: requested pseudo-population size. The default is
  `ergm.ego`'s when the population size is unknown — the number of egos
  (with R's warning when the sampling weights are unequal: rounding
  `n·wᵢ/Σw` then drops or merges egos; pass a multiple of the number of
  egos) — and `popsize` when it is known and at most 1000. Above 1000 the
  default is ten times the number of egos, where R uses `popsize` itself
  (a pseudo-population of that size is slow to simulate; pass
  `ppopsize=popsize` for R's choice); `show` says so when this applies
- `n_samples`, `burnin`, `interval`: MCMC controls; `nothing` (the
  default) selects the package's dyad-scaled budget rule. Under ESS-adaptive
  sampling `n_samples` sets the cap `max_n_samples`, `burnin` the first
  burn-in and `interval` the starting interval (an eighth of it)
- `maxiter::Int=60`: maximum number of iterations
- `termination::Symbol=:confidence`, `conv_precision::Float64=0.1`,
  `conv_confidence::Float64=0.99`, `max_n_samples` (default `4·n_samples`):
  the stopping rule, as in `ERGM.mcmle`, and the cap on a sample's size
  (under ESS-adaptive sampling the target effective size is capped at a
  quarter of it)
- `conv_threshold::Float64=0.1`, `hotelling_alpha::Float64=0.05`: the
  t-ratio and Hotelling tests reported in `mcmc_convergence` (and the
  stopping rule under `termination=:hotelling`)
- `gamma0::Float64=0.1`, `max_step_norm::Float64=5.0`: initial step length
  and the cap on the Euclidean norm of a step
- `proposal::Symbol=:spdyad`: the Metropolis proposal, R ergm 4's default
  (`MCMC.prop = ~sparse + .triadic`): tie/no-tie mixed with a
  shared-partner-focused proposal. `:tnt` (tie/no-tie) and `:random`
  (uniform dyad toggles) are available
- `effective_size=64`: ESS-adaptive sampling (R's `MCMLE.effectiveSize`);
  `nothing` draws a fixed `n_samples` per iteration, each sample started
  afresh from the annealed pseudo-population (the pre-0.2 design;
  `proposal=:tnt, effective_size=nothing` reproduces it)
- `se=nothing`: `:design` (`ergm.ego`'s), `:superpopulation` or
  `:bootstrap`; the default is `:superpopulation` when the population size
  is known, else `:design` (see "Standard errors")
- `n_boot::Int=100`: bootstrap replicates under `se=:bootstrap`
- `drop::Bool=true`: fix a coefficient whose target is at a bound of its
  statistic at `∓Inf`, as `ergm.ego` does; `false` refuses such a model
  (see "Targets at a bound")
- `rng::AbstractRNG=Random.default_rng()`: source of every random draw
  (pseudo-population seeding, annealing and the MCMC chain); the same `rng`
  state gives a bit-identical fit whatever the global RNG holds

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)                 # 205 students, 203 ties
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))   # a census, population_size = 205
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
fit.converged                     # true
coef(fit)                         # ≈ [-6.03, 2.83] — ergm.ego with popsize = 205
stderror(fit)                     # ≈ [0.22, 0.25]: se=:superpopulation, the default when popsize is known
design = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; se=:design, rng=Xoshiro(1))
stderror(design)                  # ≈ [0.16, 0.18] — ergm.ego's (0.178, 0.196)
percap = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; popsize=1, ppopsize=205, rng=Xoshiro(1))
coef(percap)[1]                   # ≈ -0.71 = -6.03 + log(205): R's default output
percap.netsize_adjustment         # -5.323 = -log(205), R's netsize.adj
```
"""
function fit_ergm_ego(ed::EgoData{T}, terms::Vector{<:EgoTerm};
                      ppopsize::Union{Int, Nothing}=nothing,
                      popsize::Union{Int, Nothing}=nothing,
                      n_samples::Union{Int, Nothing}=nothing,
                      burnin::Union{Int, Nothing}=nothing,
                      interval::Union{Int, Nothing}=nothing,
                      maxiter::Int=60,
                      termination::Symbol=:confidence,
                      conv_precision::Float64=0.1,
                      conv_confidence::Float64=0.99,
                      max_n_samples::Union{Int, Nothing}=nothing,
                      conv_threshold::Float64=0.1,
                      hotelling_alpha::Float64=0.05,
                      gamma0::Float64=0.1,
                      max_step_norm::Float64=5.0,
                      proposal::Symbol=:spdyad,
                      effective_size::Union{Real, Nothing}=64,
                      se::Union{Symbol, Nothing}=nothing,
                      n_boot::Int=100,
                      drop::Bool=true,
                      rng::Random.AbstractRNG=Random.default_rng()) where T
    controls = _FitControls(ppopsize, popsize, n_samples, burnin, interval, maxiter,
                            termination, conv_precision, conv_confidence, max_n_samples,
                            conv_threshold, hotelling_alpha, gamma0, max_step_norm,
                            proposal,
                            effective_size === nothing ? nothing : Float64(effective_size),
                            se, n_boot, drop)
    # A multi-level EgoNodeFactor becomes its one-level terms here, once:
    # the bootstrap refits reuse the levels of the data, not of a resample
    return _fit_ergm_ego(ed, _expand_ego_terms(terms, ed), controls, rng)
end

# The attainable range of the statistic an ego term estimates is the range of
# the ERGM.jl term it maps to (`ergm_term`), on the pseudo-population
# network: this method of ERGM.jl's extension-API generic declares it, so
# the ranges are ERGM.jl's own (R ergm's `minval`/`maxval`), never a restated
# table. Only fittable terms reach it: a descriptive one has no ERGM.jl
# counterpart, and `ergm_term` refuses it with its explanation.
attainable_range(t::EgoTerm, net) = attainable_range(ergm_term(t), net)

# A mixing-matrix cell whose level no vertex of the pseudo-population carries
# (a level found only among the alters, or whose egos round to zero copies)
# is 0 on every network the model can simulate, so a positive target cannot
# be matched: refused, as a model that cannot be fitted, before any MCMC.
# (A zero target is at the bottom of the cell's range and is dropped.)
function _refuse_absent_levels(terms, targets, ed::EgoData, counts::Vector{Int})
    for (j, t) in enumerate(terms)
        t isa EgoMM || continue
        targets[j] > 0 || continue
        present = Set(e.ego_attrs[t.attr] for (e, c) in zip(ed.egos, counts) if c > 0)
        absent = [l for l in unique([t.l1, t.l2]) if !(l in present)]
        isempty(absent) || throw(ArgumentError(
            "fit_ergm_ego: $(name(t)) has a positive target, but no vertex of the " *
            "pseudo-population has level(s) $(join(repr.(absent), ", ")) of :$(t.attr) " *
            "(the pseudo-population replicates the egos, and no ego with a copy " *
            "carries it — it is found only among the alters). No network the model " *
            "simulates has such a tie. Sample egos of that level, merge it with " *
            "another level, or drop the cell."))
    end
    return nothing
end

# The ego terms whose target sits at an end of the attainable range of its
# statistic, as `(index, :min | :max)`: R's `ergm.checkextreme.model`, which
# ergm runs on `target.stats` as on observed statistics. Targets are
# weighted means scaled by `m`, so an end is matched to 1e-12 of its size
function _target_boundary(terms::AbstractVector, targets::AbstractVector, m::Int)
    out = Tuple{Int,Symbol}[]
    # the statistics are those of an undirected pseudo-population of m vertices
    ppop = network(m; directed=false)
    for (j, t) in enumerate(terms)
        lo, hi = attainable_range(t, ppop)
        at(b) = isfinite(b) && abs(targets[j] - b) <= 1e-12 * max(1.0, abs(b))
        if at(lo)
            push!(out, (j, :min))
        elseif at(hi)
            push!(out, (j, :max))
        end
    end
    return out
end

function _boundary_message(terms, boundary)
    parts = String[]
    for (side, word, at) in ((:min, "smallest", "-Inf"), (:max, "largest", "+Inf"))
        cols = [name(terms[j]) for (j, s) in boundary if s === side]
        isempty(cols) || push!(parts, "target statistic(s) $(join(cols, ", ")) are at " *
                                      "their $word attainable values (coefficient $at)")
    end
    return "fit_ergm_ego: " * join(parts, "; ")
end

# ergm's sentence for the statistics it fixes at ∓Inf (its default drop=TRUE)
function _warn_target_drop(terms, boundary)
    @warn _boundary_message(terms, boundary) * ". Their coefficients are fixed " *
          "there (no finite estimate exists; ergm.ego does the same, through ergm's " *
          "default drop=TRUE). The remaining coefficients are estimated with these " *
          "held fixed: the sampler never moves the statistic off its bound. Pass " *
          "drop=false to refuse such a model instead."
    return nothing
end

# The sentence `show` and `approximations` give a fit with a dropped term
function _boundary_caveat(result)
    terms = result.model.ego_terms
    lo = [name(terms[j]) for j in eachindex(terms) if result.coefficients[j] == -Inf]
    hi = [name(terms[j]) for j in eachindex(terms) if result.coefficients[j] == Inf]
    isempty(lo) && isempty(hi) && return nothing
    parts = String[]
    isempty(lo) || push!(parts, "$(join(lo, ", ")) fixed at -Inf (target at its " *
                                "smallest attainable value)")
    isempty(hi) || push!(parts, "$(join(hi, ", ")) fixed at +Inf (target at its " *
                                "largest attainable value)")
    return "coefficient(s) " * join(parts, "; ") * ", as ergm.ego's drop does: not " *
           "estimated (standard error 0), the statistic held at its bound in the " *
           "simulated model; the other coefficients are estimated given that"
end

# The default pseudo-population size. ergm.ego's `ppopsize = "auto"`: the
# population size when it is known, the number of egos when it is not
# (`popsize = 1`). One departure: a known population above 1000 gets ten
# times the number of egos, because the chains have to mix over ppopsize²/2
# dyads (`show` and `approximations` say so: `_ppopsize_note`)
_default_ppopsize(N::Integer, n_egos::Integer) =
    N <= 1 ? n_egos : N <= 1000 ? N : 10 * n_egos

# The sentence that discloses a pseudo-population smaller than a known
# population above 1000 (the one case where the default is not ergm.ego's)
_ppopsize_note(m::EgoERGMModel) =
    (m.popsize > 1000 && m.ppopsize != m.popsize) ?
    "the pseudo-population ($(m.ppopsize)) is smaller than the population " *
    "($(m.popsize)); ergm.ego's default pseudo-population is the population " *
    "itself — pass ppopsize=$(m.popsize) for it (slower: the chains mix over " *
    "ppopsize²/2 dyads)" : nothing

# The keyword values of one `fit_ergm_ego` call, as a concretely typed
# struct: the fit itself (`_fit_ergm_ego`) is then compiled once, not once
# per combination of `nothing`/`Int` keyword values, and the precompile
# workload covers every call
struct _FitControls
    ppopsize::Union{Int, Nothing}
    popsize::Union{Int, Nothing}
    n_samples::Union{Int, Nothing}
    burnin::Union{Int, Nothing}
    interval::Union{Int, Nothing}
    maxiter::Int
    termination::Symbol
    conv_precision::Float64
    conv_confidence::Float64
    max_n_samples::Union{Int, Nothing}
    conv_threshold::Float64
    hotelling_alpha::Float64
    gamma0::Float64
    max_step_norm::Float64
    proposal::Symbol
    effective_size::Union{Float64, Nothing}
    se::Union{Symbol, Nothing}
    n_boot::Int
    drop::Bool
end

function _fit_ergm_ego(ed::EgoData{T}, terms::Vector{EgoTerm}, controls::_FitControls,
                       rng::Random.AbstractRNG) where T
    (; ppopsize, popsize, n_samples, burnin, interval, maxiter, termination,
       conv_precision, conv_confidence, max_n_samples, conv_threshold, hotelling_alpha,
       gamma0, max_step_norm, proposal, effective_size, se, n_boot, drop) = controls
    maxiter >= 1 || throw(ArgumentError("fit_ergm_ego: maxiter must be ≥ 1 (got $maxiter)"))
    isempty(terms) && throw(ArgumentError("need at least one term"))
    any(t -> t isa EgoEdges, terms) ||
        throw(ArgumentError("the model must include EgoEdges() (as ergm.ego models include edges)"))
    termination in (:confidence, :hotelling) || throw(ArgumentError(
        "fit_ergm_ego: termination must be :confidence (R ergm's equivalence " *
        "test) or :hotelling (the pre-0.2 t-ratio + Hotelling rule); got " *
        "$(repr(termination))"))
    conv_precision > 0 || throw(ArgumentError(
        "fit_ergm_ego: conv_precision must be positive (got $conv_precision)"))
    0 < conv_confidence < 1 || throw(ArgumentError(
        "fit_ergm_ego: conv_confidence must be in (0, 1) (got $conv_confidence)"))
    0 < gamma0 <= 1 || throw(ArgumentError(
        "fit_ergm_ego: gamma0 must be in (0, 1] (got $gamma0)"))
    (effective_size === nothing || effective_size >= 8) || throw(ArgumentError(
        "fit_ergm_ego: effective_size must be ≥ 8 (R's MCMLE.effectiveSize is 64) or " *
        "nothing for fixed-size samples; got $effective_size"))
    isnothing(se) || check_se(se, (:design, :superpopulation, :bootstrap);
                              context="fit_ergm_ego")
    (se === :bootstrap && n_boot < 2) && throw(ArgumentError(
        "fit_ergm_ego: n_boot must be at least 2 to form a covariance (got $n_boot)"))

    n_egos = length(ed.egos)
    N_known = something(popsize, ed.population_size, Some(nothing))
    isnothing(N_known) || N_known >= 1 || throw(ArgumentError(
        "fit_ergm_ego: popsize must be a positive number of population members " *
        "(got $N_known); leave it out when the population size is unknown"))
    # ergm.ego's default: an unknown population size is popsize = 1, which
    # makes the reported coefficients per capita (network-size invariant)
    N = something(N_known, 1)
    m_requested = isnothing(ppopsize) ? _default_ppopsize(N, n_egos) : ppopsize
    m_requested >= 5 || throw(ArgumentError(
        "fit_ergm_ego: the pseudo-population must have at least 5 vertices (got " *
        "ppopsize=$m_requested, from " *
        (isnothing(ppopsize) ? "the default rule — the number of egos when the " *
                               "population size is unknown (ergm.ego's), popsize " *
                               "when it is known and ≤ 1000, else 10·n_egos — with " *
                               "popsize=$(something(N_known, "unknown")) " *
                               "and $(_plural(n_egos, "ego"))" :
                               "the ppopsize keyword") *
        "); pass ppopsize=<n> ≥ 5"))
    # ergm.ego's two warnings about the requested size, in its words
    if m_requested < n_egos
        @warn "fit_ergm_ego: using a smaller pseudo-population size ($m_requested) " *
              "than sample size ($n_egos egos) usually does not make sense."
    elseif m_requested == n_egos && n_egos > 1 && var(ed.sampling_weights) > sqrt(eps(Float64))
        @warn "fit_ergm_ego: using a pseudo-population size equal to the sample size " *
              "($n_egos) under weighted sampling: results may be highly biased " *
              "(ergm.ego's warning) — rounding n·wᵢ/Σw drops or merges egos. Pass " *
              "a larger ppopsize, e.g. a multiple of the number of egos."
    end
    # The default: the calibrated method when the sampling fraction is known
    se_type::Symbol = !isnothing(se) ? se : (N > 1 && n_egos <= N) ? :superpopulation : :design
    if se_type === :superpopulation && (N <= 1 || n_egos > N)
        throw(ArgumentError(
            "fit_ergm_ego: se=:superpopulation needs the population size — the " *
            "correction is 1 + n_egos/popsize — but popsize is " *
            (N <= 1 ? "unknown" : "$N, smaller than the $n_egos egos") *
            ". Pass popsize=<N>, or use se=:design."))
    end

    # The realised pseudo-population size (ergm.ego's `ppop.wt = "round"`)
    ppop_counts = _ppop_counts(ed.sampling_weights, m_requested)
    m = sum(ppop_counts)
    if m != m_requested
        m >= 5 || throw(ArgumentError(
            "fit_ergm_ego: replicating each ego round(ppopsize·wᵢ/Σw) times gives a " *
            "pseudo-population of $m vertices for ppopsize=$m_requested (the " *
            "pseudo-population must have at least 5); pass a larger ppopsize — " *
            "at least the number of egos, $n_egos"))
        @info "fit_ergm_ego: the constructed pseudo-population has $m vertices, not " *
              "the requested $m_requested (each ego is replicated " *
              "round(ppopsize·wᵢ/Σw) times, as ergm.ego's ppop.wt=\"round\"); the " *
              "fit uses and reports $m."
    end

    ego_terms = terms
    ergm_terms = AbstractERGMTerm[ergm_term(t) for t in ego_terms]
    p = length(ego_terms)

    # Target statistics on the pseudo-population scale
    targets = ego_target_stats(ego_terms, ed, m)
    _refuse_absent_levels(ego_terms, targets, ed, ppop_counts)

    edges_idx = findfirst(t -> t isa EgoEdges, ego_terms)
    n_dyads = m * (m - 1) / 2
    target_density = targets[edges_idx] / n_dyads
    target_density < 1 ||
        throw(ArgumentError("target mean degree implies density ≥ 1; increase ppopsize"))
    target_density > 0 || throw(ArgumentError(
        "fit_ergm_ego: no ego reports an alter, so the edges target is 0 and every " *
        "coefficient is at the boundary (the empty network is the only network " *
        "with these targets); there is nothing to estimate"))

    # R's drop (ergm's `ergm.checkextreme.model` on the target statistics,
    # which ergm.ego passes as `target.stats`): a target at an end of its
    # statistic's attainable range has no finite coefficient. It is fixed at
    # ∓Inf, the sampler holds the statistic at its bound, and the rest is
    # estimated on the other statistics; `drop=false` refuses instead
    boundary = _target_boundary(ego_terms, targets, m)
    if !isempty(boundary)
        drop || throw(ArgumentError(_boundary_message(ego_terms, boundary) *
            ". No finite estimate exists, and drop=false asks to keep such a " *
            "statistic in the model (R's `drop=FALSE`), which is not implemented. " *
            "Use the default drop=true — the coefficient fixed at ±Inf and the rest " *
            "estimated, as ergm.ego does — or remove the term(s)."))
        se === :bootstrap && throw(ArgumentError(
            "fit_ergm_ego: se=:bootstrap needs finite point estimates, but " *
            _boundary_message(ego_terms, boundary) * ". Use se=:design or " *
            "se=:superpopulation, or remove the term(s)."))
        _warn_target_drop(ego_terms, boundary)
    end
    dropped = Set(j for (j, _) in boundary)
    est = [j for j in 1:p if !(j in dropped)]    # the estimated coefficients
    q = length(est)

    # ergm.ego's network-size offset: coefficient -log(m/N) on the statistic
    # edges (+ transitiveties·(-1/3) when the model has a triadic term). The
    # edges part is a shift of the edges coefficient; the transitive-ties
    # part is an extra, fixed-coefficient statistic of the simulated model
    adjustment = N == m ? 0.0 : log(N / m)
    if adjustment != 0.0 && any(_is_triadic, ego_terms)
        push!(ergm_terms, Offset(GWESP(0.0), -adjustment / 3))
    end
    pm = length(ergm_terms)
    free = 1:p

    # The one MCMC budget rule (dyad-scaled; see `_mcmc_controls`)
    n_samples, burnin, interval = _mcmc_controls(m; n_samples, burnin, interval)::@NamedTuple{n_samples::Int, burnin::Int, interval::Int}
    n_samples >= 2 || throw(ArgumentError("fit_ergm_ego: n_samples must be ≥ 2 (got $n_samples)"))
    n_max = something(max_n_samples, 4 * n_samples)
    n_max >= n_samples || throw(ArgumentError(
        "fit_ergm_ego: max_n_samples ($n_max) must be ≥ n_samples ($n_samples)"))

    # Pseudo-population, annealed toward the targets; the chains start there
    model = _annealed_model(ed, ppop_counts, ergm_terms, targets, p, edges_idx, rng)
    # `mcmle`'s own sampler, as R's ergm.ego runs ergm's defaults underneath:
    # SPDyad proposals, and ESS-adaptive sampling (R's MCMLE.effectiveSize)
    # on one chain continued from iteration to iteration (MCMLE.sequential),
    # its thinning interval doubled until the sample carries the target
    # effective size; the stopping rule's boost raises that target.
    # `effective_size=nothing` draws a fixed `n_samples` per iteration from
    # the annealed network instead, boosted up to `max_n_samples`
    sampler = mcmle_sampler(model; effective_size=effective_size, n_samples=n_samples,
                            max_n_samples=n_max, n_free=q, burnin=burnin,
                            interval=interval, proposal=proposal, rng=rng)
    term_names = [name(t) for t in ego_terms]

    # Start: the MPLE on the annealed network (pseudo-population scale, the
    # offset — if any — at its fixed value); the edges-only logit start when
    # no MPLE is available.
    θ = zeros(pm)
    θ[edges_idx] = log(target_density / (1 - target_density))
    pm > p && (θ[pm] = -adjustment / 3)
    mple_start(mdl) = try
        st = with_logger(NullLogger()) do
            mple(mdl)
        end
        (st.converged && all(isfinite, coef(st)[est])) ? coef(st) : nothing
    catch err
        err isa ArgumentError || rethrow()
        nothing
    end
    start = mple_start(model)
    start === nothing || (θ[est] .= start[est])
    for (j, side) in boundary
        θ[j] = side === :min ? -Inf : Inf      # the sampler holds it at its bound
    end
    targets_est = targets[est]

    chisq_cut = _chisq95(q)
    converged = false
    iterations = 0
    γ = gamma0
    n_cur = sampler.n_samples
    term_test = nothing
    tested_step = zeros(q)      # the step the termination test was evaluated at
    final = nothing
    samples = Matrix{Float64}(undef, 0, q)
    not_improved = falses(4)
    prev_diff = nothing
    θ_good = copy(θ)            # the last coefficients that gave a regular sample
    n_singular = 0
    # The estimated statistics only: a dropped one is held at its bound
    draw_all(θ, n) = sampler.draw(θ, n).samples[:, free]
    draw(θ, n) = (S = draw_all(θ, n); isempty(dropped) ? S : S[:, est])
    for iter in 1:maxiter
        iterations = iter
        samples = draw(θ, n_cur)
        cov_stats = cov(samples)
        F = cholesky(Symmetric(cov_stats); check=false)
        if !issuccess(F)
            # A singular covariance: the last step went somewhere the sampler
            # collapses. Step back halfway toward the last regular
            # coefficients and try again; give up after three attempts, or
            # at once when there is no earlier point to return to
            n_singular += 1
            if iter == 1 || n_singular > 3 || iter == maxiter
                source = iter == 1 ? "the initial values" : "the last regular iterate"
                @warn "The covariance matrix of the sampled statistics is singular at " *
                      "iteration $iter (collinear statistics, a degenerate model, or a " *
                      "collapsed sampler; terms $(join(term_names, ", "))). Moment " *
                      "matching cannot take further Newton steps; the returned " *
                      "coefficients are $source, unrefined, and standard errors will " *
                      "be NaN. Check the model for degeneracy or redundant terms."
                iter == 1 || (θ .= θ_good)
                break
            end
            θ .= (θ .+ θ_good) ./ 2
            γ = max(γ / 2, 0.01)
            continue
        end
        n_singular = 0
        θ_good .= θ

        diff = targets_est .- vec(mean(samples, dims=1))
        d2 = max(dot(diff, F \ diff), 0.0)
        # Hummel step length: the largest fraction of the way from the sampled
        # mean to the targets that stays inside the cloud, at most doubling
        # per iteration; 1 once the cloud covers the targets
        γ = d2 <= chisq_cut ? 1.0 : clamp(min(sqrt(chisq_cut / d2), 2.0 * γ), 0.01, 1.0)
        delta = F \ (γ .* diff)
        step_norm = norm(delta)
        step_norm > max_step_norm && (delta .*= max_step_norm / step_norm)

        if termination === :confidence
            term_test = confidence_test(samples, [n_cur], targets_est, delta;
                                              precision=conv_precision,
                                              confidence=conv_confidence)
            tested_step = copy(delta)
            passed = γ == 1.0 && term_test.converged
        else
            tests = mcmc_convergence(samples, targets_est; conv_threshold=conv_threshold,
                                     hotelling_alpha=hotelling_alpha,
                                     chain_lengths=[n_cur])
            term_test = (converged=tests.converged, p_value=tests.hotelling_p,
                         d2=NaN, boost=1.0)
            passed = tests.converged
            # The pre-0.2 rule tests the sample at θ itself: no step is taken
            # from a sample that passes
            passed && fill!(delta, 0.0)
            tested_step = zeros(q)
        end

        θ[est] .+= delta
        if passed
            converged = true
            final = samples
            break
        end

        # R's sample-size boost under the confidence rule (see `ERGM.mcmle`)
        if termination === :confidence
            boost = γ == 1.0 && term_test.d2 < 2 ? term_test.boost : 1.0
            if prev_diff !== nothing
                popfirst!(not_improved)
                push!(not_improved,
                      term_test.d2 >= _tolerance_d2(prev_diff, samples, conv_precision))
                if sum(not_improved) > 1
                    boost = max(boost, 2.0)
                    fill!(not_improved, false)
                end
            end
            prev_diff = diff
            boost > 1.0 && (n_cur = sampler.resize(n_cur, boost))
        end
    end

    # The final sample: the one the converged estimate was stepped from, or —
    # when the loop ended unconverged — a fresh sample at the returned
    # coefficients, which then decides
    if final === nothing
        if n_singular == 0
            samples = draw(θ, n_cur)
            tested_step = zeros(q)
            if termination === :confidence
                term_test = confidence_test(samples, [n_cur], targets_est, tested_step;
                                                  precision=conv_precision,
                                                  confidence=conv_confidence)
                converged = term_test.converged
            end
        end
        final = samples
    end
    tests = mcmc_convergence(final, targets_est; conv_threshold=conv_threshold,
                             hotelling_alpha=hotelling_alpha,
                             chain_lengths=[size(final, 1)])
    if termination === :hotelling && !converged && n_singular == 0
        converged = tests.converged
        term_test = (converged=tests.converged, p_value=tests.hotelling_p, d2=NaN, boost=1.0)
    end
    convergence = MCMLEConvergence((iterations, γ, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))
    termination_report = (rule=termination,
                          p_value=term_test === nothing ? NaN : Float64(term_test.p_value),
                          precision=conv_precision, confidence=conv_confidence,
                          n_samples=size(final, 1), step=tested_step)

    # Variance — ergm.ego's decomposition (vcov.ergm.ego, sources="model" /
    # "estimation"): the design component I⁻¹ Σ_t I⁻¹, where Σ_t is the
    # survey variance of the target statistics and I the information
    # (covariance of the sampled statistics), plus the Monte-Carlo estimation
    # component I⁻¹ Σ_mc I⁻¹ with Σ_mc = I/n_eff the variance of the sampled
    # mean. No standalone I⁻¹: the population network is not a draw from the
    # model, the egos are the sample. `se=:superpopulation` inflates Σ_t by
    # 1 + n/N: a tie between two sampled egos is reported twice.
    # A coefficient fixed at its bound is not estimated: standard error 0,
    # no covariance (its rows and columns of `vcov` are 0)
    I_mat = cov(final)
    Σ_t = _design_cov(isempty(dropped) ? ego_terms : ego_terms[est], ed, m)
    se_type === :superpopulation && (Σ_t .*= 1 + n_egos / N)
    F = cholesky(Symmetric(I_mat); check=false)
    embed(M) = isempty(dropped) ? M : (out = zeros(p, p); out[est, est] .= M; out)
    vcov_design, vcov_estimation = if issuccess(F)
        Iinv = Matrix(inv(F))
        Vd = Iinv * Σ_t * Iinv
        (embed((Vd .+ Vd') ./ 2), embed(Iinv ./ tests.n_eff))
    else
        n_singular == 0 && @warn "The covariance matrix of the statistics sampled at the fitted " *
              "coefficients is singular (collinear statistics or a collapsed " *
              "sampler; terms $(join(term_names, ", "))): no standard errors are " *
              "available — `stderror`, `vcov`, z-values and p-values are NaN. " *
              "The point estimates are unaffected."
        (embed(fill(NaN, q, q)), embed(fill(NaN, q, q)))
    end
    # `se=:bootstrap`: resample the egos, rebuild the pseudo-population and
    # the targets, refit. The spread of the refits carries what no sandwich
    # does — the sampling error of the pseudo-population's composition —
    # besides the design variance and the Monte-Carlo error of a fit. A
    # resample of egos treats them as independent, so the tie double-count is
    # added as f·(I⁻¹ Σ_design I⁻¹) when the sampling fraction f is known.
    boot_replicates = Matrix{Float64}(undef, 0, p)
    if se_type === :bootstrap
        boot = _ego_bootstrap(ed, ego_terms, controls, m_requested, N_known, n_boot, rng)
        boot_replicates = boot.replicates
        f = (N > 1 && n_egos <= N) ? n_egos / N : 0.0
        V = boot.vcov
        (f > 0 && all(isfinite, vcov_design)) && (V = V .+ f .* vcov_design)
        all(isfinite, vcov_estimation) || (vcov_estimation = zeros(p, p))
        # the refits carry their own Monte-Carlo error, so the estimation
        # term is part of V already: `vcov_design` is the remainder
        vcov_design = V .- vcov_estimation
    end
    vcov_θ = vcov_design .+ vcov_estimation
    std_errors = sqrt.(max.(diag(vcov_θ), 0.0))

    # Non-convergence is loud: warned here with the diagnostics of the
    # returned estimate, recorded in `converged`/`mcmc_convergence` (hence in
    # `approximations(fit)` and `show`) so a reader and a machine both see it.
    converged || @warn _nonconvergence_caveat(convergence, termination_report, maxiter)

    # Reported scale: the free edges coefficient of the offset model
    coefficients = θ[free]
    coefficients[edges_idx] -= adjustment

    ego_model = EgoERGMModel(ego_terms, ergm_terms, ed, m, N, targets, ppop_counts)
    return EgoERGMResult(ego_model, coefficients, std_errors, vcov_θ, vcov_design,
                         vcov_estimation, adjustment, converged, convergence, final,
                         termination_report, se_type, boot_replicates)
end

# The sentence every bootstrap of the ERGM family uses, verbatim, to say what
# excluding failed refits does to the standard errors (the warning, `show`
# and `approximations`)
const _BOOT_EXCLUSION_BIAS =
    "The standard errors are conditional on a finite refit: the excluded " *
    "replicates are the extreme ones, so the standard errors are biased downward."

# The bootstrap over egos behind `se=:bootstrap`, through the ONE shared
# resampling loop `NetworkCore.bootstrap_cov`. `simulate` draws every resample —
# `n` ego indices with replacement and one seed for the refit — up front from
# `rng`; `refit` rebuilds the ego data (each drawn ego keeps its sampling
# weight, so the weighted design is resampled as it was drawn), and with it
# the pseudo-population and the targets, and fits with the caller's controls
# from the replicate's own seed. A refit is therefore a pure function of its
# replicate and the threaded loop is thread-count independent. A refit that
# does not converge, has a non-finite coefficient or cannot be set up (a
# resample whose homophily target is 0, say) is a NaN row: excluded from the
# covariance, warned about once, kept in `boot_replicates`.
function _ego_bootstrap(ed::EgoData{T}, terms::Vector{EgoTerm}, c::_FitControls,
                        m_requested::Int, N_known::Union{Int, Nothing}, n_boot::Int,
                        rng::Random.AbstractRNG) where T
    n = length(ed.egos)
    p = length(terms)
    refit_controls = _FitControls(m_requested, N_known, c.n_samples, c.burnin, c.interval,
                                  c.maxiter, c.termination, c.conv_precision,
                                  c.conv_confidence, c.max_n_samples, c.conv_threshold,
                                  c.hotelling_alpha, c.gamma0, c.max_step_norm, c.proposal,
                                  c.effective_size, :design, 2, c.drop)
    simulate(r, B) = [(rand(r, 1:n, n), rand(r, UInt64)) for _ in 1:B]
    function refit(replicate)
        idx, seed = replicate
        try
            resampled = EgoData(ed.egos[idx]; population_size=ed.population_size,
                                sampling_weights=ed.sampling_weights[idx], design=ed.design)
            fit = with_logger(NullLogger()) do
                _fit_ergm_ego(resampled, terms, refit_controls, Random.Xoshiro(seed))
            end
            return (fit.converged && all(isfinite, fit.coefficients)) ?
                   fit.coefficients : fill(NaN, p)
        catch err
            err isa ArgumentError || rethrow()
            return fill(NaN, p)
        end
    end
    boot = bootstrap_cov(refit, simulate, zeros(p); n_boot=n_boot, rng=rng)
    ok = [all(isfinite, view(boot.replicates, b, :)) for b in 1:n_boot]
    n_ok = count(ok)
    n_ok < n_boot && @warn "fit_ergm_ego: $(n_boot - n_ok) of $n_boot bootstrap refits did " *
        "not converge or could not be fitted (a resample of the egos can put a " *
        "statistic on its boundary); they are excluded from the bootstrap " *
        "covariance and are NaN rows of `boot_replicates`. $_BOOT_EXCLUSION_BIAS"
    V = n_ok >= 2 ? _robust_cov(boot.replicates[ok, :]) : fill(NaN, p, p)
    return (vcov=V, replicates=boot.replicates)
end

# The covariance of the bootstrap replicates with a robust scale: the
# correlations are the replicates', the standard deviations are their
# normalised interquartile ranges, IQR/1.349 (which is the standard deviation
# for a normal distribution). A resample of the egos that happens to have an
# extreme attribute composition gives an outlying refit, and with few egos
# such resamples are common enough to inflate the plain standard deviation
# well beyond the sampling standard deviation of the estimate (20 of 200
# egos: +35 %, coverage 0.99–1.00, against +14 % and 0.97–0.98 with this
# scale). A coefficient whose interquartile range is 0 keeps its plain
# standard deviation.
function _robust_cov(B::AbstractMatrix{<:Real})
    p = size(B, 2)
    V = Matrix{Float64}(cov(B))
    sd = sqrt.(max.(diag(V), 0.0))
    scale = map(1:p) do j
        iqr = (quantile(view(B, :, j), 0.75) - quantile(view(B, :, j), 0.25)) / 1.349
        iqr > 0 ? iqr : sd[j]
    end
    for j in 1:p, k in 1:p
        V[j, k] = sd[j] > 0 && sd[k] > 0 ? V[j, k] * scale[j] * scale[k] / (sd[j] * sd[k]) : 0.0
    end
    return V
end

"""
    ergm_ego(ed::EgoData, terms; kwargs...)

R-faithful alias for [`fit_ergm_ego`](@ref) (the same function, so
`ergm_ego === fit_ergm_ego`), matching the R `ergm.ego` package name — the
ecosystem convention of one `fit_<model>` name and one statnet-style name.

# Example
```julia
using ERGMEgo, NetworkCore, Random
ergm_ego === fit_ergm_ego            # true
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))   # R: ergm.ego(egor ~ edges + nodematch("Grade"))
fit.converged                        # true
```
"""
const ergm_ego = fit_ergm_ego

# The per-ego contributions of ONE (concretely typed) term into column `j`
function _contribution_column!(H::Matrix{Float64}, j::Int, t::EgoTerm, egos::Vector)
    @inbounds for (i, e) in enumerate(egos)
        H[i, j] = ego_contribution(t, e)
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
- `:capture_recapture`: Lincoln–Petersen (Chapman's form) on **nominated
  alters who are themselves sampled egos**. The egos are the first capture
  — an equal-probability sample, so every member of the population,
  isolates included, is in it with the same probability — and the `R`
  ego–alter nominations are the second. A nominated alter is one of the
  other `n − 1` sampled egos with probability `(n − 1)/(N − 1)`, whatever
  its degree, so with `M` the nominations whose alter is a sampled ego,
  `N̂ = 1 + (n − 1)(R + 2)/(M + 2)` (the `+2` is Chapman's small-sample
  correction: a tie between two sampled egos is two nominations). It requires ego and alter IDs in one ID space
  (as `simulate_ego_sample` and `as_egodata` keep them) and an
  equal-probability sample: unequal sampling weights are an
  `ArgumentError`, as is `M = 0` (no nominated alter is a sampled ego).
  The estimate is noisy when `M` is small (its relative standard error is
  about `√(2/M)`).

  Before 0.2 this method split the egos into halves by input order and
  compared the two halves' alter sets; alters are a degree-biased sample
  and isolates are never named, so it estimated the number of non-isolates
  (about 25 % low on `faux.mesa.high`). The present estimator is within
  about 1 % of the truth on average there for 40 or more egos.

An unknown `method` is an `ArgumentError` naming it and listing the two
methods above (`:horvitz_thompson`, `:capture_recapture`).

# Example
```julia
using ERGMEgo
e1 = EgoNetwork(1, [2, 11, 12], zeros(Bool, 3, 3))    # names ego 2
e2 = EgoNetwork(2, [1, 13], zeros(Bool, 2, 2))        # names ego 1
e3 = EgoNetwork(3, [10, 14], zeros(Bool, 2, 2))
e4 = EgoNetwork(4, [12, 15], zeros(Bool, 2, 2))
ed = ego_design(EgoData([e1, e2, e3, e4]); weights=[10.0, 10.0, 5.0, 5.0])
estimate_popsize(ed)                                  # 30.0 — the weight sum
estimate_popsize(EgoData([e1, e2, e3, e4]); method=:capture_recapture)   # 9.25 — 4 egos, 9 nominations, 2 of them of sampled egos: 1 + 3·11/4
try estimate_popsize(ed; method=:capture_recapture) catch e; e isa ArgumentError end   # true — unequal weights
try estimate_popsize(ed; method=:lincoln_petersen) catch e; e isa ArgumentError end   # true — ArgumentError: unknown method :lincoln_petersen; expected :horvitz_thompson ... or :capture_recapture ...
```
"""
function estimate_popsize(ed::EgoData; method::Symbol=:horvitz_thompson)
    if method == :horvitz_thompson
        return sum(ed.sampling_weights)
    elseif method == :capture_recapture
        n = length(ed.egos)
        n >= 2 || throw(ArgumentError("estimate_popsize: need at least two egos"))
        w = ed.sampling_weights
        all(x -> isapprox(x, first(w)), w) || throw(ArgumentError(
            "estimate_popsize: method=:capture_recapture assumes an " *
            "equal-probability sample of egos (every population member equally " *
            "likely to be a sampled ego), but the sampling weights are unequal. " *
            "Use method=:horvitz_thompson, whose estimate is the weight sum."))
        ego_ids = Set(e.ego for e in ed.egos)
        length(ego_ids) == n || throw(ArgumentError(
            "estimate_popsize: the ego IDs are not distinct; capture–recapture " *
            "matches nominated alters against the sampled egos by ID"))
        nominations = 0
        recaptured = 0
        for e in ed.egos, a in e.alters
            nominations += 1
            recaptured += a in ego_ids
        end
        nominations > 0 || throw(ArgumentError(
            "estimate_popsize: no ego names an alter; there is nothing to recapture"))
        recaptured > 0 || throw(ArgumentError(
            "estimate_popsize: no nominated alter is a sampled ego (no alter " *
            "overlap with the ego sample); capture–recapture requires alter IDs " *
            "in the same ID space as the ego IDs, and a sample large enough " *
            "for some alters to be sampled egos"))
        return 1 + (n - 1) * (nominations + 2) / (recaptured + 2)
    else
        throw(ArgumentError(
            "estimate_popsize: unknown method :$method; expected :horvitz_thompson " *
            "(the weight sum) or :capture_recapture (Lincoln–Petersen on nominated " *
            "alters who are sampled egos)"))
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
the conversion contract (`NetworkCore.ConversionReport`) and the missing-data
contract (`NetworkCore.require_observed`):

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
using ERGMEgo, NetworkCore, Random
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
    # `EgoNodeMatch` then miscounted (a silent zero-fill)
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
    egos = _ego_networks(net, ego_ids, columns)

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

# The ego networks of the vertices `ego_ids` of `net`, each with its alters,
# the ties among them and the attribute `columns` (one value per vertex)
function _ego_networks(net::Network, ego_ids, columns::Dict{Symbol, Vector})
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
        for (attr, col) in columns
            e_attrs[attr] = col[eid]
            a_attrs[attr] = col[alters]      # same eltype as `col`, empty or not
        end

        push!(egos, EgoNetwork(Int(eid), Int.(alters), ties;
                               ego_attrs=e_attrs, alter_attrs=a_attrs))
    end
    return egos
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
NetworkCore.supports_missing(::typeof(simulate_ego_sample)) = true
NetworkCore.missing_policies(::typeof(simulate_ego_sample)) = (:error, :face)

# =============================================================================
# Diagnostics
# =============================================================================

# The coefficients of the ERGM simulated on the pseudo-population: the fitted
# coefficients with the network-size offset applied to edges, followed by the
# fixed transitive-ties offset coefficient when the model carries one
function _pseudo_theta(result::EgoERGMResult)
    model = result.model
    θ = zeros(length(model.ergm_terms))
    θ[1:length(result.coefficients)] .= result.coefficients
    edges_idx = findfirst(t -> t isa EgoEdges, model.ego_terms)
    θ[edges_idx] += result.netsize_adjustment
    length(θ) > length(result.coefficients) &&
        (θ[end] = -result.netsize_adjustment / 3)
    return θ
end

# The ONE simulation engine behind `gof` (and hence `ego_gof`): simulate
# pseudo-population networks at the fitted coefficients, draw from each an
# ego sample that mirrors the observed design, and evaluate every statistic
# on it exactly as it is evaluated on the data.
#
# The simulated ego sample: one vertex per observed ego, drawn uniformly
# among the pseudo-population vertices that replicate that ego, carrying the
# ego's sampling weight. The simulated statistics are therefore
# design-weighted estimates from a sample of the observed size and weight
# distribution — the same estimator as the observed column, with the same
# design effect — instead of unweighted means over a simple random sample,
# whose spread is too small under unequal weights. (An ego whose weight
# rounds to no pseudo-population vertex has no replicate and is left out.)
#
# Every draw flows through `rng`: the pseudo-population seeding and
# annealing, the `sample_networks` chains (seeded per chain from `rng`,
# concatenated in order, so the result is thread-count independent) and each
# ego sample. The MCMC budget is the package's one dyad-scaled rule
# (`_mcmc_controls`).
function _gof_simulations(result::EgoERGMResult, n_sim::Int,
                          rng::Random.AbstractRNG;
                          burnin::Union{Int, Nothing}=nothing,
                          interval::Union{Int, Nothing}=nothing,
                          n_chains::Int=min(n_sim, 4),
                          proposal::Symbol=:tnt)
    n_sim >= 1 || throw(ArgumentError("gof: n_sim must be ≥ 1 (got $n_sim)"))
    n_chains >= 1 || throw(ArgumentError("gof: n_chains must be ≥ 1 (got $n_chains)"))
    model = result.model
    ed = model.data
    m = model.ppopsize
    ego_terms = model.ego_terms
    p = length(ego_terms)
    edges_idx = findfirst(t -> t isa EgoEdges, ego_terms)

    ctl = _mcmc_controls(m; burnin=burnin, interval=interval)
    ergm_model = _annealed_model(ed, model.ppop_counts, model.ergm_terms, model.targets,
                                 p, edges_idx, rng)
    sims = sample_networks(ergm_model, _pseudo_theta(result); n_sim=n_sim,
                           burnin=ctl.burnin, interval=ctl.interval, rng=rng,
                           n_chains=n_chains, proposal=proposal)

    # The replicates of each ego, and the attribute columns of the
    # pseudo-population (the same for every simulated network)
    members = [Int[] for _ in 1:length(ed)]
    for (v, i) in enumerate(_ppop_ego_index(model.ppop_counts))
        push!(members[i], v)
    end
    present = findall(!isempty, members)
    weights = ed.sampling_weights[present]
    ppop = ergm_model.network
    columns = Dict{Symbol, Vector}()
    for attr in list_vertex_attributes(ppop)
        vals = get_vertex_attribute(ppop, attr)
        length(vals) == m && (columns[attr] = [vals[v] for v in 1:m])
    end

    # The statistics: every model statistic per capita (gof.ergm.ego's
    # GOF="model"), the degree distribution over R's bins (GOF="degree"), the
    # edgewise-shared-partner distribution (GOF="espartners") when the ties
    # among alters were collected, and the two descriptive means
    aaties = _alter_ties_observed(ed)
    K = maximum(ego_degree(e) for e in ed.egos)
    degree_terms, degree_labels = _degree_gof_bins(K, m)
    # gof.ergm.ego's GOF="espartners": R's esp(k) on the egor, the fittable
    # `EgoESP` term
    esp_terms = aaties ? EgoTerm[EgoESP(k) for k in 0:(2 * (max(K, 3) - 1))] : EgoTerm[]
    summary_terms = aaties ? EgoTerm[_EgoMeanDegree(), _EgoMeanAlterTies()] :
                             EgoTerm[_EgoMeanDegree()]
    panels = (ego_terms, degree_terms, esp_terms, summary_terms)
    observed = [Float64[compute(t, ed) for t in terms] for terms in panels]
    simulated = [Matrix{Float64}(undef, length(sims), length(terms)) for terms in panels]
    for (k, s) in enumerate(sims)
        sample_ed = _design_sample(rng, s, members, present, weights, columns)
        for (terms, sim) in zip(panels, simulated), (j, t) in enumerate(terms)
            sim[k, j] = compute(t, sample_ed)
        end
    end
    esp_labels = ["esp $k" for k in 0:(length(esp_terms) - 1)]
    return (; observed, simulated, degree_labels, esp_labels, aaties)
end

# One simulated ego sample with the observed design: for each observed ego
# that has replicates in the pseudo-population (`present`), one of its
# replicating vertices, drawn uniformly, with that ego's sampling weight
function _design_sample(rng::Random.AbstractRNG, net::Network, members::Vector{Vector{Int}},
                        present::Vector{Int}, weights::Vector{Float64},
                        columns::Dict{Symbol, Vector})
    ids = [members[i][rand(rng, 1:length(members[i]))] for i in present]
    return EgoData(_ego_networks(net, ids, columns); sampling_weights=weights)
end

# The upper-tail bin of the degree GOF: the design-weighted proportion of
# egos with degree ≥ d (R's `degrange(d)` on an egor). Descriptive only, like
# EgoDegree — never fittable — and internal: it exists so every simulated
# ego lands in exactly one row of the "degree distribution" statistic.
struct _EgoDegreeAtLeast <: EgoTerm
    d::Int
end
name(t::_EgoDegreeAtLeast) = "degree$(t.d)+"
ego_contribution(t::_EgoDegreeAtLeast, e::EgoNetwork) = Float64(ego_degree(e) >= t.d)

# The two descriptive means of the "ego summary statistics" panel
struct _EgoMeanDegree <: EgoTerm end
name(::_EgoMeanDegree) = "mean degree"
ego_contribution(::_EgoMeanDegree, e::EgoNetwork) = Float64(ego_degree(e))
struct _EgoMeanAlterTies <: EgoTerm end
name(::_EgoMeanAlterTies) = "mean alter ties"
_needs_alter_ties(::_EgoMeanAlterTies) = true
ego_contribution(::_EgoMeanAlterTies, e::EgoNetwork) = Float64(n_alter_ties(e))

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
        burnin=nothing, interval=nothing, n_chains=min(n_sim, 4),
        proposal=:tnt) -> GOFResult

Goodness-of-fit assessment of a fitted egocentric ERGM: pseudo-population
networks are simulated at the fitted coefficients, an ego sample mirroring
the observed design is drawn from each, and the observed design-weighted
statistics are compared with their simulated distributions. The statistics
are `gof.ergm.ego`'s three diagnostics, plus two descriptive means:

1. `"model statistics"` — **every statistic of the model, per capita**
   (`GOF = "model"`), labelled by the ego terms. These are the fitted
   targets: their p-values say whether the simulation reproduces what was
   fitted (a small one means the fit did not converge, or the chain does not
   mix), not whether the model fits.
2. `"degree distribution"` — the proportion of egos at each degree, over
   `gof.ergm.ego`'s bins (`GOF = "degree"`): `degree 0 … degree maxdeg−1`
   plus an upper tail `degree ≥ maxdeg`, with `maxdeg = 2·max(K, 3)` and
   `K` the largest **observed** ego degree — R's `degree(0:(maxdeg−1)) +
   degrange(maxdeg)` — so every simulated ego falls in exactly one row.
   When `maxdeg ≥ ppopsize − 1` every attainable degree is its own bin and
   there is no tail, as in R. An edges-only model matches the mean degree
   by construction but not the distribution around it.
3. `"edgewise shared partners"` — the per-capita number of ties whose ends
   have `k` shared partners, `k = 0 … 2·(max(K, 3) − 1)` (`GOF =
   "espartners"`, R's `esp(0:maxesp)`): where a model without a triangle
   term shows its lack of clustering. Present only when the ties among
   alters were collected (`ed.design[:alter_ties_observed]`).
4. `"ego summary statistics"` — the mean degree and, when the ties among
   alters were collected, the mean number of alter–alter ties per ego (what
   [`ego_gof`](@ref) reads).

**The simulated ego samples respect the design.** From each simulated
network one vertex is drawn for each observed ego, uniformly among the
pseudo-population vertices that replicate it, and it carries that ego's
sampling weight. The simulated column is therefore the same design-weighted
estimator as the observed one, from a sample of the same size and weight
distribution. (`gof.ergm.ego` compares the observed estimates with the
statistics of the whole simulated network instead, which leaves the
sampling variation of the observed column out of the reference
distribution.)

This is a method of the shared `NetworkCore.gof` generic; it returns the
shared `NetworkCore.GOFResult` (observed value, simulation envelope, and
two-sided Monte-Carlo p-value per level from `NetworkCore.mc_pvalue`, the
`(1 + k)/(N + 1)` estimator, so it is never exactly zero).

Every random draw — the pseudo-population, the MCMC chains and the ego
samples — flows through `rng`, so two calls from the same `rng` state give
identical simulated statistics, and the multi-chain sampler is seeded per
chain from `rng` so the result does not depend on the thread count.

# Keyword Arguments
- `n_sim::Int=50`: number of simulated networks
- `rng::AbstractRNG=Random.default_rng()`: source of every random draw
- `burnin`, `interval`: MCMC controls of the network sampler; `nothing`
  (the default) selects the dyad-scaled rule the fit uses
  (the package's budget rule, ERGM.jl's `20·n_dyads` and `max(100, n_dyads ÷ 10)`)
- `n_chains::Int=min(n_sim, 4)`: chains the `n_sim` networks are split over
  (`ERGM.sample_networks`, one seed per chain drawn from `rng`); for a given
  `n_chains` and `rng` state the result is bit-identical whatever the thread
  count (changing `n_chains` changes the chain seeding, hence the draws)
- `proposal::Symbol=:tnt`: the Metropolis proposal (`:tnt` or `:random`)

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
g = gof(fit; n_sim=20, rng=Xoshiro(5))
[s.name for s in g.statistics]                  # ["model statistics", "degree distribution", "edgewise shared partners", "ego summary statistics"]
g.statistics[1].labels                          # ["edges", "nodematch.Grade"]
g.statistics[1].observed ≈ fit.model.targets ./ 205   # true — the per-capita targets
g.statistics[2].labels[end]                     # "degree ≥ 26" — the tail bin, maxdeg = 2·max(13, 3); K = 13 observed
all(sum(g.statistics[2].simulated; dims=2) .≈ 1) # true — every simulated ego lands in one row
g.statistics[3].labels[1:2]                     # ["esp 0", "esp 1"]
g.statistics[1].simulated == gof(fit; n_sim=20, rng=Xoshiro(5)).statistics[1].simulated   # true
g.statistics[1].p_values[1] == mc_pvalue(g.statistics[1].simulated[:, 1], g.statistics[1].observed[1])   # true
```
"""
function gof(result::EgoERGMResult; n_sim::Int=50,
             rng::Random.AbstractRNG=Random.default_rng(),
             burnin::Union{Int, Nothing}=nothing,
             interval::Union{Int, Nothing}=nothing,
             n_chains::Int=min(n_sim, 4),
             proposal::Symbol=:tnt)
    sim = _gof_simulations(result, n_sim, rng; burnin, interval, n_chains, proposal)
    obs, s = sim.observed, sim.simulated
    # p-values are the shared `NetworkCore.mc_pvalue` (GOFStatistic's default)
    stats = GOFStatistic[
        GOFStatistic("model statistics", [name(t) for t in result.model.ego_terms],
                     obs[1], s[1]),
        GOFStatistic("degree distribution", sim.degree_labels, obs[2], s[2])]
    sim.aaties && push!(stats, GOFStatistic("edgewise shared partners", sim.esp_labels,
                                            obs[3], s[3]))
    push!(stats, GOFStatistic("ego summary statistics",
                              sim.aaties ? ["mean degree", "mean alter ties"] :
                                           ["mean degree"], obs[4], s[4]))
    return GOFResult(stats; model="Egocentric ERGM")
end

"""
    ego_gof(result::EgoERGMResult; n_sim=50, rng=Random.default_rng(),
            burnin=nothing, interval=nothing, n_chains=min(n_sim, 4),
            proposal=:tnt) -> NamedTuple

Goodness of fit for an egocentric ERGM as a NamedTuple
`(observed, simulated, p_values, n_sim)` keyed by `mean_degree` /
`mean_alter_ties`: `observed` the design-weighted sample statistics,
`simulated` the means over the simulated ego samples, `p_values` the
two-sided Monte-Carlo p-values (`NetworkCore.mc_pvalue`). `mean_alter_ties`
is `NaN` throughout when the ties among alters were not collected.

A thin wrapper over [`gof`](@ref) — the same simulations, the same
keywords, the same p-value estimator, read out of the `GOFResult`'s
`"ego summary statistics"` — so the two cannot disagree. Prefer `gof`,
which returns the shared `NetworkCore.GOFResult` and also carries every model
statistic, the degree distribution and the shared-partner distribution.

# Example
```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade], rng=Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng=Xoshiro(1))
e = ego_gof(fit; n_sim=20, rng=Xoshiro(5))
g = gof(fit; n_sim=20, rng=Xoshiro(5))
e.p_values.mean_degree == g.statistics[end].p_values[1]   # true
e.observed.mean_degree == summary_stats(ed).mean_degree   # true
```
"""
function ego_gof(result::EgoERGMResult; n_sim::Int=50,
                 rng::Random.AbstractRNG=Random.default_rng(),
                 burnin::Union{Int, Nothing}=nothing,
                 interval::Union{Int, Nothing}=nothing,
                 n_chains::Int=min(n_sim, 4),
                 proposal::Symbol=:tnt)
    g = gof(result; n_sim, rng, burnin, interval, n_chains, proposal)
    stat = g.statistics[end]
    sim = stat.simulated
    two = length(stat.labels) == 2
    return (
        observed = (mean_degree = stat.observed[1],
                    mean_alter_ties = two ? stat.observed[2] : NaN),
        simulated = (mean_degree = mean(view(sim, :, 1)),
                     mean_alter_ties = two ? mean(view(sim, :, 2)) : NaN),
        p_values = (mean_degree = stat.p_values[1],
                    mean_alter_ties = two ? stat.p_values[2] : NaN),
        n_sim = n_sim
    )
end

# =============================================================================
# Precompile workload
# =============================================================================
#
# A small census fit, its printed table and a goodness-of-fit run, so the
# first `fit_ergm_ego` of a session does not pay for compiling the sampler,
# the moment-matching loop and the presentation layer. Every draw comes from
# a local rng; the warnings of a deliberately tiny fit are discarded.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        with_logger(NullLogger()) do
            rng = Random.Xoshiro(1)
            net = network(12; directed=false)
            for i in 1:12, j in (i+1):12
                rand(rng) < 0.25 && add_edge!(net, i, j)
            end
            set_vertex_attribute!(net, :g, Dict(v => (isodd(v) ? "a" : "b") for v in 1:12))
            set_vertex_attribute!(net, :k, Dict(v => mod1(v, 3) for v in 1:12))
            ed = simulate_ego_sample(net, 12; ego_attrs=[:g, :k], rng=rng)
            summary_stats(ed)
            ego_target_stats([EgoNodeFactor(:g), EgoNodeCov(:k), EgoAbsDiff(:k),
                              EgoDegree(1)], ed, 12)
            # The sampler specializes on the model's terms (and an attribute
            # term on the attribute's type): the edges-only model and
            # nodematch on a string and on an integer attribute are compiled
            for terms in ([EgoEdges()], [EgoEdges(), EgoNodeMatch(:k)],
                          [EgoEdges(), EgoNodeMatch(:g)])
                fit = fit_ergm_ego(ed, terms; n_samples=60, maxiter=3, rng=rng)
                show(devnull, fit)
                coeftable(fit); confint(fit); approximations(fit)
                show(devnull, gof(fit; n_sim=2, rng=rng))
            end
            fit_ergm_ego(ed, [EgoEdges()]; ppopsize=12, popsize=24, n_samples=60, rng=rng)
        end
    end
end

end # module
