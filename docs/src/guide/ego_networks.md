# Ego Networks

## EgoNetwork

An [`EgoNetwork`](@ref) records one ego's local view:

- `alters`: the alter IDs, preserved from the source data,
- `alter_ties`: a symmetric Bool matrix of ties among the alters,
- `ego_attrs` / `alter_attrs`: attribute dictionaries.

Helpers: [`ego_degree`](@ref), [`n_alters`](@ref),
[`alter_degree`](@ref), [`n_alter_ties`](@ref).

## EgoData

[`EgoData`](@ref) bundles the ego networks with per-ego
`sampling_weights` (design weights) and an optional `population_size`.
[`ego_design`](@ref) attaches or replaces design information
(`popsize` — the size of the population the egos were sampled from — and
`weights`);
[`summary_stats`](@ref) gives design-weighted descriptives (mean **and**
median degree honour the same weights); [`ego_mixing_matrix`](@ref)
tabulates weighted ego–alter attribute mixing, and refuses an ego that
lacks the attribute rather than skipping it.

`show` is one informative line per object, in the style of ERGM.jl's
`ERGMModel`:

```julia
using ERGMEgo
e = EgoNetwork(1, [10, 11, 12], Bool[0 1 0; 1 0 0; 0 0 0];
               ego_attrs = Dict{Symbol,Any}(:group => "A"),
               alter_attrs = Dict{Symbol,Vector}(:group => ["A", "B", "A"]))
e                    # EgoNetwork{Int64}: ego 1, 3 alters, 1 alter–alter tie; ego attributes: group; alter attributes: group
EgoData([e])         # EgoData{Int64}: 1 ego, weighted mean degree 3.0; population size unknown; unit weights
```

## Sampling from a complete network

[`simulate_ego_sample`](@ref) is the `Network → EgoData` conversion
adapter: undirected one-mode networks only, masked dyads refused unless
`missing = :face`, every `ego_attrs` entry required on every vertex, and
`report = true` returning a `NetworkCore.ConversionReport` of what an ego
sample cannot hold (ties between unsampled vertices, unrequested vertex
attributes, edge and network attributes, loops, face-read masked dyads).

## Population size

[`estimate_popsize`](@ref) supports:

- `:horvitz_thompson` — the sum of the sampling weights (meaningful only
  when weights are inverse inclusion probabilities);
- `:capture_recapture` — Lincoln–Petersen on **nominated alters who are
  themselves sampled egos**. The ``n`` egos are the first capture: an
  equal-probability sample, so every member of the population — isolates
  included — is in it with the same probability. The ``R`` ego–alter
  nominations are the second. A nominated alter is one of the other
  ``n - 1`` sampled egos with probability ``(n-1)/(N-1)`` whatever its
  degree, so with ``M`` nominations of sampled egos,
  ``\hat N = 1 + (n-1)(R+2)/(M+2)`` (Chapman's correction, applied to
  ties). It needs ego and alter IDs in one ID space and equal sampling
  weights (unequal weights are an `ArgumentError`), and it is noisy when
  ``M`` is small.

  On `faux.mesa.high` (205 students, 57 of them isolates) the mean estimate
  from 80 sampled egos is 207. The estimator this replaces compared the
  alter sets of two halves of the sample; alters are a degree-biased sample
  and isolates are never named, so it averaged 153.

Feed the estimate to `fit_ergm_ego(…; popsize = round(Int, N̂))`, or leave
the population size out for per-capita coefficients.
