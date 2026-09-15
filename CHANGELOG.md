# Changelog

All notable changes to ERGMEgo.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 and 2026-09 expert-panel reviews: placeholder
estimation is replaced by real Krivitsky–Morris method-of-moments fitting on
a pseudo population (via ERGM.jl's public `mh_sample` API instead of a
private function) with ERGM.jl's MCMLE convergence tests and `ergm.ego`'s
standard-error decomposition, ego data ingestion is rebuilt on the egodata
two/three-table format, and the package adopts the ecosystem-wide
naming/StatsAPI/GOF conventions.

### Breaking

- **Estimation is real now — coefficients change entirely.** The old
  `ergm_ego` ran a placeholder gradient loop with fabricated standard errors
  (`0.1` everywhere); `fit_ergm_ego` performs MCMC moment matching against
  design-weighted target statistics on a pseudo population, applies the
  `−log(popsize/ppopsize)` network-size offset to the edges coefficient, and
  reports model + survey-design variance. *Migration:* expect different (now
  meaningful) coefficients; the edges coefficient is population-scaled.
- **`fit_ergm_ego(data, terms::Vector{<:EgoTerm})` requires an
  `EgoEdges()` term and typed ego terms;** the `method=:mple` keyword is
  gone. The keywords are `ppopsize`, `popsize`, `n_samples`, `burnin`,
  `interval`, `maxiter` (default 80), `conv_threshold` (0.1),
  `hotelling_alpha` (0.05) and `rng` — the ecosystem-wide vocabulary
  (panel 2026-09, item 16). `max_iter` is accepted as a **deprecated
  spelling** of `maxiter` (honoured, warns once); `tol` is **deprecated and
  ignored** (warns once) because convergence is no longer a relative-change
  rule (see below). *Migration:* include `EgoEdges()`; drop `method=` and
  `tol=`; spell the iteration cap `maxiter`.
- **Convergence is ERGM.jl's MCMLE rule, and non-convergence is loud.**
  Moment matching now stops when every per-statistic t-ratio
  `|target − mean|/sd` is below `conv_threshold` AND a Hotelling T² test on
  the Geyer effective sample size is not rejected at `hotelling_alpha`
  (`ERGM.mcmc_convergence`, evaluated on the sample of every iteration;
  report stored as `fit.mcmc_convergence`). **The sample that passes is the
  final sample** (R's `ergm` design): `converged`, `fit.mcmc_convergence`,
  `fit.sim_stats` and the standard errors all describe that one draw, so
  `converged` is exactly the rule applied to the recorded report; only when
  `maxiter` is exhausted after a Newton step is a fresh sample drawn at the
  returned coefficients, and then that sample decides `converged` (a loop
  that "ran out one step early" whose post-step sample passes is a
  converged fit, not a warning). The pre-0.2
  rule — every target matched to within 1 % — never looked at the
  Monte-Carlo sd and so stopped at the noise level: on the faux.mesa census
  it left a seed-to-seed sd of 0.040/0.044 and a 0.052 gap to `ergm.ego`;
  under the new rule they are 0.026/0.026 and 0.010. A fit that reaches
  `maxiter` without converging now **warns** (max t-ratio, Hotelling p,
  iterations), records `converged == false`, prints the caveat under
  `Converged: false`, and lists it in `approximations(fit)`; a singular
  covariance of the sampled statistics warns instead of silently stopping.
  *Migration:* fits take a few more iterations; read `fit.mcmc_convergence`
  for the diagnostics; treat a warning as what it says.
- **Standard errors change: the spurious `I⁻¹` term is gone.** `vcov(fit)` is
  now `ergm.ego`'s decomposition `I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff` — the
  survey-design sandwich (`fit.vcov_design`, R's
  `vcov(fit, sources="model")`) plus the Monte-Carlo estimation term of the
  moment equations (`fit.vcov_estimation`, R's `sources="estimation"`). The
  previous `I⁻¹ + I⁻¹ Σ_design I⁻¹` added a standalone inverse-information
  term that `ergm.ego` does not have (the population network is not modelled
  as a draw from the ERGM; the egos are the sample), which inflated an
  `ergm.ego` standard error of 0.178 to ≈ 0.19 on the faux.mesa census. The
  golden fixture now pins the decomposition against R (`mle_std_errors`,
  `mle_se_design_component`, and the exact-information sandwich
  `design_se_exact` to 1e-6). *Migration:* standard errors are smaller and
  now match `ergm.ego`; `show(fit)` prints R's "MCMC %" share.
- **MCMC budget follows ERGM.jl's one dyad-scaled rule.** `burnin` and
  `interval` default to `ERGM._mcmc_defaults(n_dyads)` — `20·n_dyads` and
  `max(100, n_dyads ÷ 10)` toggles — through the single helper
  `ERGMEgo._mcmc_controls(m; n_samples, burnin, interval)`, replacing the
  July `3·n_dyads/2` / `n_dyads/70` constants whose final sample had an
  effective size of ≈ 23 on the 205-actor census (now ≈ 150, at 0.3 s per
  3000-draw sample; five census fits take 16 s). *Migration:* fits on large
  pseudo-populations take longer per iteration and converge in fewer;
  explicit `burnin`/`interval` are honoured as before.
- **`EgoERGMResult` gained fields and both result types are parametric.**
  `EgoERGMModel{T}` (`data::EgoData{T}`) and `EgoERGMResult{T}`; new fields
  `vcov_design`, `vcov_estimation` and `mcmc_convergence::ERGM.MCMLEConvergence`.
  *Migration:* positional construction of either type needs the new fields;
  `EgoERGMResult` in a type annotation still matches every fit.
- **`ERGMEgo.name`/`ERGMEgo.compute` are `Networks.name`/`Networks.compute`**
  (imported from the shared statistic protocol rather than through ERGM);
  the private `_z_pvalues` copy is deleted in favour of `Networks.z_pvalues`
  and the `Distributions` dependency is dropped (`SpecialFunctions` provides
  the normal quantile `confint` needs). *Migration:* none for users; a
  package that reached into `ERGMEgo._z_pvalues` must use `Networks.z_pvalues`.
- **`as_egodata` switched to the egodata two/three-table format:**
  `as_egodata(ego_df, alter_df; aatie_df=..., population_size=...)` replaces
  the single long-format DataFrame with `ego_id`/`alter_id`/`tie_col`. The
  old version silently produced all-zero alter-alter ties and renumbered
  alters (destroying cross-ego overlap); the new one reads real ties from
  `aatie_df` and preserves alter IDs. *Migration:* split your input into an
  ego table, an alter table, and (optionally) an alter-alter tie table.
- **`simulate_ego_sample` honours the missing-data and conversion contracts
  and refuses what it used to misread.** The positional is typed
  `net::Network` (the docstring always said so); a network with masked
  (unobserved) dyads is refused under the default `missing=:error` with
  `Networks.require_observed` (it used to read the face value of a masked
  dyad as an observed ego–alter or alter–alter tie and feed it to
  `summary_stats` and the targets; panel 2026-09, item 4); a **directed**
  network is refused (`neighbors` was silently read as out-neighbours only)
  and so is a **two-mode** one (`Network` with `bipartite` metadata or a
  `BipartiteNetwork`); an `ego_attrs` entry the network does not carry, or
  carries for only some vertices, is an `ArgumentError` listing the
  attributes the network has (the old code filled `missing`, which
  `EgoNodeMatch` then miscounted); `n_egos` outside `0:nv(net)` names both
  numbers. An ego's own self-loop is no longer recorded as an alter.
  *Migration:* symmetrise a directed network first; pass `missing=:face` to
  read masked dyads at face value (audited in the report); set the
  attribute on every vertex or drop it from `ego_attrs`.
- **Missing attributes are errors, not zeros.** `compute(EgoNodeMatch(:x),
  ed)` (hence `ego_target_stats`, `_design_cov` and `fit_ergm_ego`, which
  now fails at the targets before any MCMC) throws an `ArgumentError`
  naming the attribute, the ego and whether the ego or its alters lack it;
  the old code returned `0.0` for such an ego — the same silent zero-fill
  class as ERGM's `NodeCov` (panel P1-11) — biasing the homophily target
  toward zero. `ego_mixing_matrix` likewise throws instead of silently
  skipping the ego. `as_egodata` checks every column its keywords name
  (`ego_id`, `alter_id`, `ego_attrs`, `alter_attrs`, `weight_col`,
  `source_col`, `target_col`) up front and names the column, the keyword,
  the frame and the frame's columns, instead of a `KeyError` from the row
  loop (only `weight_col` was checked before). *Migration:* build the data
  with the attribute on both sides, or drop the term.
- **`summary_stats(ed).median_degree` is design-weighted.** It is the 0.5
  quantile of the weight-estimated degree distribution (midpoint
  convention, so it equals `Statistics.median` under unit weights) — the
  same weights `mean_degree` already honoured; the unweighted value was a
  second estimand under the same heading. *Migration:* none under unit
  weights; with design weights the median moves.
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
  *Migration:* prefer the new `gof(result)` returning a `Networks.GOFResult`.
- **`estimate_popsize` returns `Float64`** (was rounded `Int`), and
  `:capture_recapture` is a real two-sample Lincoln–Petersen estimator on
  alter-ID overlap (requires globally meaningful alter IDs).
- **Removed exports:** `EgoSample`, `read_ego_data`, `merge_ego_data`,
  `compare_ego_population` (no replacements). *Migration:* ingest data via
  `as_egodata` from DataFrames.
- **Minimum Julia raised to 1.12**; package UUID regenerated. *Migration:*
  upgrade Julia and re-resolve environments pinning the old UUID.

### Added

- **A second golden fixture for the weighted, non-census design — the
  estimator's actual use case.** `test/fixtures/fauxmesa_ego_weighted.toml`
  (regenerable with `Rscript test/fixtures/r/fauxmesa_ego_weighted.R > …`,
  ergm.ego 1.1.4) freezes, for egos 1, 4, …, 205 of `faux.mesa.high` (69
  egos) with case weights `Grade − 6` and ppopsize = popsize = 205:
  deterministically (asserted at 1e-9) the Hájek targets of `edges +
  nodematch("Grade") + gwdegree(0.5, fixed=TRUE) + triangle` scaled to 205
  and their full 4×4 design covariance — the first R pin of `EgoTriangle`'s
  `/3`, of `EgoGWDegree`, of the design variance of a *weighted* mean and of
  the weights entering at all — the pseudo-population's composition by grade
  (largest-remainder rounding equals R's `ppop.wt = "round"`), the exact
  information of the dyad-independent model at R's estimate and the design
  SE it implies; and, Monte-Carlo, a seeded weighted `ergm.ego` fit's
  population-scale coefficients (a single default Julia fit within 0.05,
  4× the combined single-fit sd; the 3-seed mean within 0.03 of R's 5-seed
  mean) and its SE decomposition (0.08, > 3× Julia's per-fit sd). The census
  fixture could pin none of this: its weights are all 1 and its targets are
  the network's own statistics.
- **`gof` carries `gof.ergm.ego`'s degree-distribution diagnostic, over
  R's bins.** The `GOFResult` now has two statistics: `"ego summary
  statistics"` (mean degree, mean alter ties — R's `GOF = "model"`) and
  `"degree distribution"` — `degree 0` … `degree maxdeg−1` plus the tail
  `degree ≥ maxdeg`, with `maxdeg = 2·max(K, 3)` and K the largest
  **observed** ego degree, exactly R's `degree(0:(maxdeg−1)) +
  degrange(maxdeg)` (and R's no-tail branch when `maxdeg ≥ ppopsize − 1`):
  the design-weighted proportion of egos in each bin, `EgoDegree(d)` on
  every simulated ego sample. The bins reach twice the observed maximum and
  close with the tail so that every simulated ego lands in exactly one row —
  observed and every simulated row sum to 1 (pinned) — and a model that
  over-produces high degrees is caught in the tail, which is what the
  diagnostic exists for. (The round-2 statistic stopped at `degree K`; on a
  40-vertex Bernoulli census 5 % of simulated egos then fell in no row and
  the rows summed to 0.95 with nothing in the output to say so.) The docs
  say what the mean-degree row can and cannot tell you: every model includes
  `EgoEdges()`, so it is a fitted target and its p-value is uninformative by
  construction; the degree distribution is what an edges-only model fails.
  `ego_gof` reads the first statistic only, as before. *Migration:* code
  indexing `g.statistics[1]` is unchanged; `length(g.statistics)` is 2; the
  degree statistic has `maxdeg + 1` rows, not `K + 1`, and its last label is
  `"degree ≥ <maxdeg>"`.
- **Sampling weights are validated when the data is built.** The `EgoData`
  constructor (hence `ego_design(…; weights)` and `as_egodata(…;
  weight_col)`) throws an `ArgumentError` naming the ego (index and id) and
  the value for a `NaN`, `Inf` or negative weight, and for weights that sum
  to zero — `ergm.ego`/`survey` refuse the same. Round 2 checked only the
  length: a zero-sum, `NaN` or `Inf` vector gave `NaN` targets silently and
  then the unrelated "target mean degree implies density ≥ 1" error from
  the fit, and a negative weight ran the whole MCMC and **returned
  coefficients** computed from a negative-weight Hájek target (the `w²` of
  the design variance hid the sign). A single zero weight remains legal; the
  weighted median keeps its own sum guard as a second line of defence.
- **Duplicate alters and alter self-ties are refused.** `EgoNetwork` throws
  an `ArgumentError` naming the ego and the alter when an alter is listed
  twice (the wide-to-long survey reshape mistake: round 2 accepted
  `EgoNetwork(5, [10, 10], …)` with `ego_degree == 2`, doubling the ego's
  degree and homophily count in every target and in the design variance)
  or when `alter_ties` has a `true` on its diagonal (round 2 accepted it and
  `n_alter_ties` dropped it by `1 ÷ 2 == 0`). `as_egodata` surfaces the
  same two mistakes in the frames' terms — "alter_df has N duplicate (ego,
  alter) rows for ego E (alter a appears more than once)" and "aatie_df row
  (E, a, a) is a self-tie". `simulate_ego_sample` never produced either.
- **`estimate_popsize`'s unknown-method error names the valid methods**
  (`:horvitz_thompson` — the weight sum — and `:capture_recapture` — the
  two-sample Lincoln–Petersen on alter overlap); it said only "Unknown
  method: foo".
- **`ERGMEgo._ergm_term` is `public`** (beside `_mcmc_controls`), with a
  docstring and an API-reference entry: it is the hook a custom `EgoTerm`
  extends to become fittable, and the `EgoTerm` docstring, the terms guide
  and the README now say so instead of naming a private function. The
  README "Known limitations" and the terms guide state the term coverage —
  `EgoEdges`, `EgoNodeMatch`, `EgoTriangle`, `EgoGWDegree` are the whole
  fittable vocabulary; `ergm.ego`'s `nodefactor`, `nodecov`, `absdiff`,
  `gwesp`, `mm`, `degree`/`concurrent` have no ego counterpart yet — and
  that there is no `simulate(fit)` (`simulate.ergm.ego`): `gof` is the only
  routine that draws from a fitted model.
- **`EgoTerm` is exported** (as ERGM.jl exports `AbstractERGMTerm`) with a
  docstring showing how a custom ego term is defined, and
  **`ERGMEgo._mcmc_controls` is declared `public`** (THE MCMC budget rule,
  documented in the API reference). The "Shared contracts" testset scans the
  source for every `ERGM._name` reach-in and pins `Base.ispublic(ERGM, …)`
  for each (`_mcmc_defaults`), plus `mcmc_convergence`/`MCMLEConvergence`.
- **A `missing` attribute value is an actionable error.** `EgoNodeMatch` and
  `ego_mixing_matrix` refuse an ego whose own value, or any of whose alters'
  values, is `missing` (an unknown alter attribute in a real survey, which
  `as_egodata` passes through from a `Union{T,Missing}` column) with an
  `ArgumentError` naming the attribute, the ego, how many alters lack a
  value and the fix; it used to be `TypeError: non-boolean (Missing) used in
  boolean context` from inside the matching loop. The check is a
  compile-time no-op on a `Vector{String}` column, so the per-ego
  contribution stays allocation-free.
- **Two more guards name the thing and the fix:** the pseudo-population
  minimum (`ArgumentError` with the size, where it came from — the
  `ppopsize` keyword, or the `popsize`/`10·n_egos` default rule with the
  numbers it saw — and `pass ppopsize=<n> ≥ 5`; it said "pseudo-population
  size too small"), and `summary_stats` on an `EgoData` with no egos (it
  threw `mean`'s "reducing over an empty collection").
- **Provenanced golden fixture against a real `ergm.ego` fit under a stated
  sampling design** (issue #8), and with it **the answer to issue ERGMEgo#1**.
  `test/fixtures/fauxmesa_ego_census.toml` freezes an ergm.ego 1.1.4 fit of
  `egor ~ edges + nodematch("Grade")` on `faux.mesa.high` under a **census**
  design (all 205 actors are egos, unit weights, ppopsize = popsize = 205),
  regenerable with `Rscript test/fixtures/r/fauxmesa_ego_census.R >
  test/fixtures/fauxmesa_ego_census.toml`.

  A census is chosen because it makes two of the three compared quantities
  *deterministic*, so neither can be excused as Monte-Carlo noise:

  - **Target statistics** reduce to the observed network's own (edges = 203,
    nodematch.Grade = 163). ERGMEgo.jl reproduces them **exactly**.
  - **The design variance of the targets** is a function of the 205 per-ego
    contributions and the weights and nothing else. And here is the finding:

    > **ERGMEgo.jl's design variance is too small by exactly the factor (n−1)/n.**

    Not approximately — exactly, in every entry of the covariance matrix.
    `_design_cov` divided the sum of squared deviations by `n`; the survey
    (SRS/Horvitz–Thompson) variance of a mean that `ergm.ego` computes divides by
    `n−1`. At n = 205 the standard errors came out **0.24% narrow**. The Bessel
    factor is now applied (see Fixed), the design standard errors are asserted
    against R at 1e-9, and the correction is pinned by an independent
    `Statistics.cov` reference and a hand-computed weighted three-ego case so it
    cannot be removed silently. This is the concrete, numeric form of the
    ERGMEgo#1 warning that the design variance is "narrower than advertised" —
    and the fixture also shows why it matters: on this fit the design component
    is **17×** the estimation component, so an ergm.ego standard error
    essentially *is* its design variance.

  - **The design standard errors of the coefficients with the exact
    information** (`design_se_exact`): the model is dyad-independent, so a plain
    ERGM's MPLE is the MLE and its `vcov` is the exact `I⁻¹`; the sandwich
    `I⁻¹ Σ_design I⁻¹` is deterministic on both sides and agrees to 1e-6. R's
    own MCMC estimate of the information (`r_DtDe`, frozen) is 12 % off it in
    `I⁻¹[1,1]`, which sizes the Monte-Carlo effect on a standard error and
    justifies the 0.06 tolerance on `mle_std_errors`.

  **A parameterization difference, now mapped rather than assumed away.**
  `ergm.ego` splits the population edges parameter into a fixed offset
  `netsize.adj = −log(popsize) = −5.3230` plus a free `edges` coefficient
  (−0.6974); ERGMEgo.jl reports it as one number on the pseudo-population scale.
  R's −0.697 and ERGMEgo.jl's −6.07 are the same parameter in different clothes.
  The comparable quantity is the sum (−6.0204), frozen as
  `mle_coefficients_population`, and it is independently anchored by a plain
  (non-egocentric) ERGM MPLE of the same model on the same network (−6.034), which
  a census fit must reduce to — and both packages do.

- **Pinned: the defaults converge at realistic network size.** The pre-0.2
  defaults (`n_samples=400, burnin=2000, interval=20`) did not scale with the
  pseudo-population and returned `edges ≈ −21.9` on the 205-actor census where
  the answer is `−6.02` — off by a factor of three. The dyad-scaled budget (see
  Breaking) converges there under the defaults and lands within 0.1 of
  `ergm.ego`; a testset pins it, together with the loud non-convergence at
  `maxiter=1`.
- StatsAPI surface completed: `coeftable` (a `Networks.CoefficientTable` —
  the table `show` prints, p-values via `Networks.z_pvalues`), `confint`
  (normal theory), `nobs` (the number of egos; R's `nobs()` on the underlying
  `ergm` object counts pseudo-population dyads) and `dof` (finite
  coefficients), all exported beside `coef`/`stderror`/`vcov` and pinned by
  `Networks.check_statsapi`. `loglikelihood`, `aic` and `bic` deliberately
  have no methods (`objective(fit) == :moment`: no likelihood is evaluated).
- `show(fit)` prints the coefficient table through `coeftable`, the
  non-convergence caveat under `Converged: false`, and `ergm.ego`'s
  standard-error decomposition in R's "MCMC %" convention with the effective
  sample size; `approximations(fit)` reports the final sample's size,
  effective size, max t-ratio and Hotelling p.
- `EgoGWDegree(decay::Real=0.5)` accepts any `Real` and `decay = 0`
  (statnet's `gwdegree(0, fixed=TRUE)`; a negative decay is an
  `ArgumentError` "decay must be non-negative"); its ERGM counterpart is
  labelled `gwdeg.fixed.<decay>`, R's name.
- A reproducibility testset: two fits from the same `rng` state are
  bit-identical whatever the global RNG holds, and the global RNG is left
  untouched.
- `simulate_ego_sample(net, n; missing=:error, report=false)`: the
  `Network → EgoData` adapter's `missing=` keyword (`:error`/`:face`;
  `supports_missing(simulate_ego_sample) == true`,
  `missing_policies(simulate_ego_sample) == (:error, :face)`) and
  `report=true` returning `(ed, ::Networks.ConversionReport)` that names
  `:edges` (ties between unsampled vertices, when `n_egos < nv`),
  `:vertex_attrs` (each attribute not in `ego_attrs`), `:edge_attrs`,
  `:network_attrs`, `:loops` and `:missing_dyads` (masked dyads read at
  face value under `missing=:face`). A census with every vertex attribute
  requested is `is_lossless`.
- `gof`/`ego_gof` keywords `burnin`, `interval` (default `nothing` → the
  fit's dyad-scaled `_mcmc_controls` rule) and `n_chains` (`min(n_sim, 4)`,
  ERGM's default), pinned by a fresh-process thread-count-independence
  test; CI runs the ubuntu / Julia `1` cell with `JULIA_NUM_THREADS=4` as
  ERGM.jl does.
- One-line `show` methods for `EgoNetwork` (ego id, alter and alter-tie
  counts, attribute names), `EgoData` (egos, weighted mean degree,
  population size, unit or summed weights) and `EgoERGMModel` (terms,
  ppopsize/popsize, targets), in the style of ERGM.jl's `ERGMModel`.
- **Every export carries a docstring with a runnable example** — every
  type, term, helper, entry point and StatsAPI method (`coef`, `stderror`,
  `vcov` gained ERGMEgo docstrings; `nobs`/`dof`/`ergm_ego`/`fit_ego_ergm`/
  `estimate_popsize`/`ego_target_stats`/`compute`/`name` and the data
  types gained examples), the module docstring included — pinned by the
  testset "Every exported docstring carries a runnable example", which
  evaluates every ```julia block of every ERGMEgo-owned docstring in a
  fresh module after `using ERGMEgo` (ERGM.jl's testset, adapted). Error
  demonstrations are written `try f(...) catch e; e isa ArgumentError end`.
  A README "Actionable errors" table.
- **`compute` and `name` are exported** (the shared Networks.jl statistic
  generics, re-exported as ERGM.jl re-exports them), so
  `compute(EgoEdges(), ed)` works with `using ERGMEgo` alone; the getting-
  started page no longer needs `using ERGM: compute`.
- **Benchmark environment and allocation gates** (panel 2026-09, item 7):
  `benchmark/Project.toml` (`[sources]` to `..`, `../../ERGM.jl`,
  `../../Networks.jl`, mirroring ERGM.jl), `benchmark/benchmarks.jl`
  (`BENCHJL` rows for `compute` of the four fittable terms on 2000 egos,
  `_design_cov`, a 500-vertex `simulate_ego_sample` census and a small
  `fit_ergm_ego`, consumed by the site's `tools/run_benchmarks.jl`) and
  `benchmark/regression_tests.jl` pinning `@allocated
  _ego_contribution(term, ego) == 0` for `EgoEdges`/`EgoNodeMatch`/
  `EgoTriangle`/`EgoGWDegree` and `_design_cov` at ≤ 4·n·p·8 bytes + 4096.
  The same pins live in the test suite ("Hot paths are allocation-free")
  and CI's `'1'`/ubuntu cell runs the benchmark environment's gate.
- A fresh-process co-loading testset: `using ERGM, ERGMEgo` (and, beside
  the monorepo workspace, the nine-package model family) leaves `compute`,
  `name`, `gof`, `coef`, `coeftable` and `Network` defined and identical to
  Networks.jl's generics.
- README "Known limitations": the survey design is independent egos with
  case weights and nothing richer (no strata, clusters, finite-population
  correction, replicate weights, without-replacement inclusion probabilities
  or alter dependence — the open half of ERGMEgo#1, so the standard errors
  are narrower than for any richer design, as `show` and `approximations`
  already say); no likelihood; single-chain moment matching; no directed or
  two-mode models. The getting-started page gains a faux.mesa.high census
  walkthrough on `load_dataset(:faux_mesa_high)` and the keyword
  vocabulary (`maxiter`, `missing=`, `n_chains`); the API page notes
  `supports_missing`/`missing_policies` of `simulate_ego_sample` and
  documents `coef`/`stderror`/`vcov`/`name`.

- `fit_ergm_ego` as the canonical entry point; `ergm_ego` (R-faithful) and
  `fit_ego_ergm` (legacy) kept as `const` aliases.
- `gof(::EgoERGMResult; n_sim, rng)` extending the ecosystem-wide
  `Networks.gof` generic; StatsAPI `coef`/`stderror`/`vcov` accessors.
- Survey-design machinery: design-weighted `summary_stats`,
  `ego_target_stats`, and the design covariance of the targets
  (`_design_cov`) behind `vcov_design` (see Breaking for the final
  `I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff` form).
- New exported helpers: `n_alters`, `ego_degree`, `alter_degree`,
  `n_alter_ties`, `ego_mixing_matrix`, `EgoERGMModel`, `EgoERGMResult`.

### Changed

- Documentation uses the default Documenter themes, with a new package-specific
  SVG icon and browser favicon in the official Julia logo colors.
- **Graphs.jl is no longer a dependency.** Every graph primitive the package
  uses (`nv`, `neighbors`, `has_edge`, `vertices`) is Networks.jl's
  re-export; `using Graphs` and the `[deps]`/`[compat]` entries were dead.
  Graphs moves to `[extras]` for the test suite's documented-elsewhere check
  only.
- `simulate_ego_sample` builds one concretely typed attribute column per
  attribute and slices it per ego, so an isolate's empty `alter_attrs[attr]`
  has the same element type as its neighbours' (`Vector{String}`, say); it
  used to be a `Vector{Any}` beside runtime-widened `Vector{String}`s, which
  cost a second specialisation of the matching barrier and gave one ego a
  different column type for no data reason.
- The non-convergence caveat (the `@warn`, `show` under `Converged: false`,
  `approximations`) labels its numbers "on the final sample at the returned
  coefficients".
- Simulation and moment matching use ERGM.jl's public `mh_sample` /
  `sample_networks` APIs (removing the layering violation on the private
  `ERGM._mcmc_sample`).
- Ego terms reformulated as per-capita design-weighted contributions with
  R-style labels (`ego.edges`, `ego.nodematch.<attr>`, `ego.triangle`,
  `ego.gwdegree.<decay>`); `EgoGWDegree` validates `decay >= 0`.
- The golden testset builds the census egodata from the bundled
  `load_dataset(:faux_mesa_high)` (checked against the fixture's frozen edge
  list) instead of hand loops, and its `[tolerance]` prose describes the
  current state: design SEs exact (1e-9), `design_se_exact`/`plain_ergm_vcov`
  at 1e-6, coefficients at 0.05 (≥ 3× the combined Monte-Carlo sd of the two
  means and ≥ 2× the observed gap), standard errors at 0.06 (R's DtDe offset
  from the exact information + 3× the Julia per-fit sd). The fixture was
  regenerated on 2026-09-12 (R 4.6.1, ergm.ego 1.1.4); every previously
  frozen value is unchanged to 15 significant digits.
- `simulate_ego_sample` gains `rng` and `ego_attrs` keywords, preserves
  vertex IDs as alter IDs, and drops
  `with_replacement`/`include_alter_ties`.
- **GOF simulations use the fit's dyad-scaled MCMC budget.** `gof` and
  `ego_gof` sample the pseudo-population networks with the same
  `_mcmc_controls` rule the fit uses (burn-in `20·n_dyads`, interval
  `max(100, n_dyads ÷ 10)`) instead of the literals `burnin=2000,
  interval=200`, which on the 205-actor census were 0.1 % of the fit's
  burn-in — the GOF envelope came from chains that had not left the seed
  network; `burnin`/`interval` remain overridable.
- **`_ego_contribution(::EgoNodeMatch, ego)` is allocation-free** through
  a function barrier (`_count_matches`) over the abstractly typed
  `Dict{Symbol,Vector}` attribute column, and **`_design_cov` allocates
  O(n·p) bytes and nothing per ego** (a per-term `_contribution_column!`
  barrier and an explicit triple loop replace the per-row `H[i, :] .- h̄`
  and `d * d'` temporaries; measured 112 KB → 81 KB for 2000 egos × 4 terms,
  same numbers to 1e-11). Neither changes any result.
- CI comments that the sibling clone list (`Networks ERGM`) is derived from
  `Project.toml`'s `[sources]`.
- The getting-started "Fitting" block fits the 30-ego sample with
  `ppopsize = 50` and lets the recorded `population_size = 100` supply
  `popsize` (with the comment saying so and the resulting
  `netsize_adjustment = −log(2)`); it used to override both with
  `ppopsize = 200, popsize = 1000` — a pseudo-population twice the actual
  population and a population ten times it — on the page a new user reads
  first, contradicting its own closing paragraph.
- **`ego_gof` is a thin wrapper over `gof`** and its p-values are the shared
  `Networks.mc_pvalue` (`(1 + k)/(N + 1)`, two-sided, never exactly zero);
  the local closure `min(1, 2·min(mean(sim .>= o), mean(sim .<= o)))`,
  which could return exactly 0 and could disagree with `gof` on the same
  fit, is gone.

### Fixed

- **`converged == true` could sit next to a recorded convergence report that
  rejects.** The round-1 loop declared convergence on the sample that passed
  and then drew a *fresh* 3000-draw sample at the same coefficients for the
  standard errors and `fit.mcmc_convergence`; on about one default fit in
  four that second draw failed the Hotelling test (p = 0.0005 was observed),
  so `show` printed `Converged: true` beside an `approximations` sentence
  quoting diagnostics above the threshold — and, symmetrically, a fit that
  ran out of `maxiter` warned "did not converge" while quoting a fresh
  sample that *passed*. The passing sample is now the final sample and the
  one the report describes (see Breaking); the suite pins
  `converged == (all(t_ratios .< 0.1) && hotelling_p > 0.05)` on the
  recorded report and that `ERGM.mcmc_convergence(fit.sim_stats, targets)`
  reproduces it exactly, on every golden fit. Point estimates are unchanged
  in distribution (census seed-to-seed sd 0.012/0.012 over six seeds, gap to
  `ergm.ego` 0.01); each fit is one 0.3 s sample cheaper.
- `as_egodata` no longer fabricates empty alter-alter tie matrices or
  destroys alter identities (see Breaking) — capture-recapture population
  estimates are now meaningful.
- Standard errors are real design + Monte-Carlo-estimation variances
  (`ergm.ego`'s decomposition) instead of the hard-coded `0.1` placeholders,
  and no longer carry the spurious `I⁻¹` term (see Breaking).
- `_design_cov` applies the with-replacement factor `n/(n−1)` of a weighted
  mean's variance; it was too small by exactly `(n−1)/n` (issue #1, found by
  the golden fixture).
- `fit.netsize_adjustment` is `0.0`, not `-0.0`, when `popsize == ppopsize`.
- The five `Random.seed!` calls in the test suite that claimed "ERGM's MCMC
  sampler draws from the global RNG" are gone — it has not since July; every
  draw flows through `rng`.
- **`gof`/`ego_gof` dropped `rng` on the way to `sample_networks`** (panel
  2026-09, item 6, confirmed major: two `gof(res; rng=Xoshiro(5))` calls
  differed and the global RNG was consumed). The pseudo-population, the
  network chains (`rng`, `n_chains`) and each ego sample now all draw from
  the caller's `rng`; two calls from the same state are bit-identical and
  the global RNG is untouched.

## [0.1.0] - 2026-02-09

Initial release: ego network data structures and prototype ego-ERGM
estimation.
