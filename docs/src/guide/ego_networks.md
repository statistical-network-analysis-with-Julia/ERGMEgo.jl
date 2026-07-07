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
[`summary_stats`](@ref) gives design-weighted descriptives;
[`ego_mixing_matrix`](@ref) tabulates weighted ego–alter attribute mixing.

## Population size

[`estimate_popsize`](@ref) supports:

- `:horvitz_thompson` — the sum of the sampling weights (meaningful only
  when weights are inverse inclusion probabilities);
- `:capture_recapture` — two-sample Lincoln–Petersen on alter overlap:
  the egos are split in half, and with ``n_1``, ``n_2`` distinct alters
  named per half and ``m`` shared, ``\hat N = n_1 n_2 / m``. Requires
  globally meaningful alter IDs.
