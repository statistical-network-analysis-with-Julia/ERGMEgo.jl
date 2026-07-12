# Changelog

All notable changes to ERGMEgo.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: placeholder estimation is
replaced by real Krivitsky–Morris method-of-moments fitting on a pseudo
population (via ERGM.jl's public `mh_sample` API instead of a private
function), ego data ingestion is rebuilt on the egodata two/three-table
format, and the package adopts the ecosystem-wide naming/StatsAPI/GOF
conventions.

### Breaking

- **Estimation is real now — coefficients change entirely.** The old
  `ergm_ego` ran a placeholder gradient loop with fabricated standard errors
  (`0.1` everywhere); `fit_ergm_ego` performs MCMC moment matching against
  design-weighted target statistics on a pseudo population, applies the
  `−log(popsize/ppopsize)` network-size offset to the edges coefficient, and
  reports model + survey-design variance. *Migration:* expect different (now
  meaningful) coefficients; the edges coefficient is population-scaled.
- **`fit_ergm_ego(data, terms::Vector{<:EgoTerm})` requires an
  `EgoEdges()` term and typed ego terms;** the `method=:mple` and `maxiter`
  keywords are gone (now `ppopsize`, `popsize`, `n_samples`, `burnin`,
  `interval`, `max_iter`, `tol`, `rng`). *Migration:* include `EgoEdges()`;
  rename `maxiter` to `max_iter`; drop `method=`.
- **`as_egodata` switched to the egodata two/three-table format:**
  `as_egodata(ego_df, alter_df; aatie_df=..., population_size=...)` replaces
  the single long-format DataFrame with `ego_id`/`alter_id`/`tie_col`. The
  old version silently produced all-zero alter-alter ties and renumbered
  alters (destroying cross-ego overlap); the new one reads real ties from
  `aatie_df` and preserves alter IDs. *Migration:* split your input into an
  ego table, an alter table, and (optionally) an alter-alter tie table.
- **`EgoNetwork` dropped its per-alter `weights` field/keyword** (sampling
  weights live on `EgoData` only), and `alter_ties` must be symmetric
  (throws otherwise). *Migration:* remove `weights=` from `EgoNetwork`.
- **`EgoMixingMatrix` term removed** — use the new
  `ego_mixing_matrix(ed, attr)` function returning `(levels, matrix)`.
- **`EgoDegree` redefined** as a descriptive proportion-of-egos statistic
  with a mandatory degree (`EgoDegree(d)`); it can no longer be used in
  `fit_ergm_ego` (throws). *Migration:* use `EgoEdges()` in models and
  `summary_stats` for mean degree.
- **`ego_gof` return restructured** to
  `(observed, simulated, p_values, n_sim)` NamedTuples keyed by
  `mean_degree`/`mean_alter_ties`; the `statistics=` keyword is gone.
  *Migration:* prefer the new `gof(result)` returning a `Network.GOFResult`.
- **`estimate_popsize` returns `Float64`** (was rounded `Int`), and
  `:capture_recapture` is a real two-sample Lincoln–Petersen estimator on
  alter-ID overlap (requires globally meaningful alter IDs).
- **Removed exports:** `EgoSample`, `read_ego_data`, `merge_ego_data`,
  `compare_ego_population` (no replacements). *Migration:* ingest data via
  `as_egodata` from DataFrames.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- `fit_ergm_ego` as the canonical entry point; `ergm_ego` (R-faithful) and
  `fit_ego_ergm` (legacy) kept as `const` aliases.
- `gof(::EgoERGMResult; n_sim, rng)` extending the ecosystem-wide
  `Network.gof` generic; StatsAPI `coef`/`stderror`/`vcov` accessors.
- Survey-design machinery: design-weighted `summary_stats`,
  `ego_target_stats`, combined model + design covariance
  (`I⁻¹ + I⁻¹ Σ_design I⁻¹`).
- New exported helpers: `n_alters`, `ego_degree`, `alter_degree`,
  `n_alter_ties`, `ego_mixing_matrix`, `EgoERGMModel`, `EgoERGMResult`.

### Changed

- Simulation and moment matching use ERGM.jl's public `mh_sample` /
  `sample_networks` APIs (removing the layering violation on the private
  `ERGM._mcmc_sample`).
- Ego terms reformulated as per-capita design-weighted contributions with
  R-style labels (`ego.edges`, `ego.nodematch.<attr>`, `ego.triangle`,
  `ego.gwdegree.<decay>`); `EgoGWDegree` validates `decay > 0`.
- `simulate_ego_sample` gains `rng` and `ego_attrs` keywords, preserves
  vertex IDs as alter IDs, and drops
  `with_replacement`/`include_alter_ties`.

### Fixed

- `as_egodata` no longer fabricates empty alter-alter tie matrices or
  destroys alter identities (see Breaking) — capture-recapture population
  estimates are now meaningful.
- Standard errors are real observed-information + design variances instead
  of the hard-coded `0.1` placeholders.

## [0.1.0] - 2026-02-09

Initial release: ego network data structures and prototype ego-ERGM
estimation.
