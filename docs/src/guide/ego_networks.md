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
[`ego_design`](@ref) attaches or replaces design information;
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
`report = true` returning a `Networks.ConversionReport` of what an ego
sample cannot hold (ties between unsampled vertices, unrequested vertex
attributes, edge and network attributes, loops, face-read masked dyads).

## Population size

[`estimate_popsize`](@ref) supports:

- `:horvitz_thompson` — the sum of the sampling weights (meaningful only
  when weights are inverse inclusion probabilities);
- `:capture_recapture` — two-sample Lincoln–Petersen on alter overlap:
  the egos are split in half, and with ``n_1``, ``n_2`` distinct alters
  named per half and ``m`` shared, ``\hat N = n_1 n_2 / m``. Requires
  globally meaningful alter IDs.
