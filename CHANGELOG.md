# Changelog

All notable changes to ERGMEgo.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

First public release of ERGMEgo.jl, a Julia port of R's `ergm.ego`
(statnet): exponential-family random graph models fitted to egocentrically
sampled data by moment matching on a pseudo-population, with targets,
design variance, network-size handling and fitted coefficients checked
against `ergm.ego` 1.1.4 by four provenanced fixtures. The changes below
are relative to 0.1.0, a development version, never released.

**Dependency renamed:** the foundation package is now `NetworkCore` (developed as `Networks`); write `using NetworkCore` where code said `using Networks`. Types and functions keep their names.

### Highlights

- **Real estimation.** `fit_ergm_ego` (alias `ergm_ego`)
  fits by MCMC moment matching against design-weighted target statistics,
  with ERGM.jl's MCMLE machinery: a start annealed to the targets, Hummel
  step lengths and R ergm's confidence stopping rule. 0.1 ran a placeholder
  loop with fabricated standard errors.
- **`ergm.ego`'s network-size handling.** The offset `netsize.adj =
  −log(ppopsize/popsize)`, per-capita coefficients when the population size
  is unknown (`popsize = 1`, R's default), and the `transitiveties = −1/3`
  part of the offset for triangle models.
- **Standard errors**: `ergm.ego`'s decomposition `I⁻¹ Σ_design I⁻¹ +
  I⁻¹/n_eff` (`se = :design`), and — the default when the population size
  is known — the same with the design component multiplied by
  `1 + n_egos/popsize` (`se = :superpopulation`), which allows for a tie
  between two sampled egos being reported twice.
- **`ergm.ego`'s attribute, degree, mixing and shared-partner terms**:
  `EgoNodeFactor`, `EgoNodeCov`, `EgoAbsDiff`, a fittable `EgoDegree`,
  `EgoGWESP`, `EgoESP`, `EgoMM` and `EgoConcurrent`, so the `ergm.ego`
  example model can be fitted as its help page writes it, `gwesp(0, fixed
  = TRUE)` included — at the defaults, in seconds; targets, labels and fits
  pinned against `ergm.ego`.
- **`gof.ergm.ego`'s three diagnostics** — every model statistic, the degree
  distribution, edgewise shared partners — on simulated ego samples that
  carry the observed design.
- **Loud failures**: missing attributes, bad weights, duplicated alters,
  uncollected alter–alter ties, masked dyads and unconverged fits are
  refused or reported, never computed around.

### Breaking

- **The custom-term hooks lose their underscore and stay `public`:**
  `ERGMEgo._ergm_term` is `ERGMEgo.ergm_term` and `ERGMEgo._ego_contribution`
  is `ERGMEgo.ego_contribution` (now `public` and documented: it is the
  per-ego contribution a custom `EgoTerm` defines). `ERGMEgo._mcmc_controls`
  is no longer `public`: the budget rule is described in the API reference
  and overridden by the `n_samples`/`burnin`/`interval` keywords. No
  underscore name of ERGMEgo is `public`; no deprecation aliases.

- **Coefficients change entirely**: estimation is real (see Highlights).
  `fit_ergm_ego(ed, terms::Vector{<:EgoTerm})` requires `EgoEdges()` and
  typed ego terms; `method=` is gone. Keywords: `popsize`, `ppopsize`,
  `n_samples`, `burnin`, `interval`, `maxiter` (60), `termination`
  (`:confidence` or `:hotelling`), `conv_precision`, `conv_confidence`,
  `max_n_samples`, `conv_threshold`, `hotelling_alpha`, `gamma0`,
  `max_step_norm`, `proposal`, `effective_size`, `se`, `rng`. The development spellings
  `max_iter` and `tol`, and the alias `fit_ego_ergm`, are removed (a table
  in the API reference lists them).
- **An unknown population size means per-capita coefficients.** With
  neither `popsize` nor `ed.population_size`, `popsize = 1` as in R: the
  edges coefficient is network-size invariant, and for a population of `N`
  it is `coef − log(N)`. `simulate_ego_sample` records the population size,
  so its samples are fitted on that scale; pass `popsize = 1` for R's
  default output.
- **The default pseudo-population size is `ergm.ego`'s**: the number of
  egos when the population size is unknown (it was ten times that), with
  R's warnings for a size below the number of egos and for a size equal to
  it under unequal weights; the population size when it is known, up to
  1000. Above 1000 it stays ten times the number of egos (R uses the
  population), which `show` and `approximations` now say.
- **`netsize_adjustment` is R's `netsize.adj`**, `−log(ppopsize/popsize)`:
  the pseudo-population's edges coefficient is `coef + netsize_adjustment`.
- **Triangle models carry the offset on `edges − transitiveties/3`** when
  `ppopsize ≠ popsize`, as `ergm.ego` does, so the triangle coefficient
  depends on `popsize`.
- **The pseudo-population is built as `ergm.ego` builds it**: each ego is
  replicated `round(ppopsize · wᵢ/Σw)` times. Its realised size can differ
  from the request (an `@info` says so) and is what the fit uses and
  reports; its composition no longer depends on the order of the egos.
- **The stopping rule is R ergm's confidence test**; the result gained the
  fields `termination` and `se_type`, and `EgoERGMModel` the field
  `ppop_counts`. A chain too short to mix no longer "converges".
  `termination = :hotelling` selects the t-ratio + Hotelling rule.
- **The moment matching samples as R ergm 4 does underneath `ergm.ego`**:
  ERGM.jl's `mcmle` sampler (`ERGM.Extension.mcmle_sampler`), with the
  SPDyad proposal (`proposal = :spdyad`, the new default) and ESS-adaptive
  sampling (`effective_size = 64`, R's `MCMLE.effectiveSize`) on one chain
  continued between iterations; the stopping rule's boost raises the target
  effective size. `ergm.ego`'s help-page model, `gwesp(0, fixed = TRUE)`
  included, now converges at the defaults in about 5–15 s; the fixed-size
  tie/no-tie sampler could not reach the effective sample size its 14
  statistics need within the 4·`n_samples` cap and did not converge in 60
  iterations. `effective_size = nothing` draws a fixed `n_samples` per
  iteration (`proposal = :tnt, effective_size = nothing` is the pre-0.2
  sampler); `proposal = :tnt` and `:random` remain available.
- **`estimate_popsize(ed; method = :capture_recapture)` is a different
  estimator**: Lincoln–Petersen on nominated alters who are sampled egos,
  `1 + (n − 1)(R + 2)/(M + 2)`. It needs equal sampling weights and at
  least one recapture. The estimator it replaces (alter overlap of two
  halves of the sample) was about 25 % low. The function returns `Float64`.
- **`gof` returns different statistics**: `"model statistics"`, `"degree
  distribution"`, `"edgewise shared partners"` and `"ego summary
  statistics"`, in that order; the summary means that were first are now
  last. `ego_gof` returns a NamedTuple `(observed, simulated, p_values,
  n_sim)` read from the last.
- **`ego_design(ed; popsize = N)`**: the keyword that sets the population
  size was spelled `ppopsize` in development versions; that spelling is
  removed.
- **Data built by `as_egodata` without `aatie_df` has unobserved alter–alter
  ties**: `EgoTriangle` refuses it and `summary_stats(ed).mean_alter_ties`
  is `NaN`. Pass an empty `aatie_df` when the ties were collected and there
  are none.
- **`as_egodata` takes `ergm.ego`-style frames** (egos, ego–alter rows,
  alter–alter ties) and preserves alter IDs. `EgoNetwork` dropped its
  per-alter `weights`.
- **`simulate_ego_sample` follows the missing-data and conversion
  contracts**: masked networks are refused unless `missing = :face`;
  directed and two-mode networks and unknown or partial `ego_attrs` are
  refused; `report = true` returns a `ConversionReport`.
- **Missing attributes and `missing` values are errors, not zeros**
  (`EgoNodeMatch`, `ego_mixing_matrix`).
- **`EgoDegree(d)` is `ergm.ego`'s `degree(d)`**, fittable (it maps to
  ERGM.jl's `Degree(d)`), and refuses `d < 0`. `EgoMixingMatrix` is
  replaced by `ego_mixing_matrix`.
  `summary_stats(ed).median_degree` is design-weighted.
- **Coefficient labels are `ergm.ego`'s** (`edges`, `degree0`,
  `nodefactor.Race.Hisp`, `nodematch.Race`, `gwdeg.fixed.0.5`,
  `gwesp.fixed.0`, …): `name(term)`, `coefnames(fit)` and the rows of
  `coeftable(fit)` no longer carry an `ego.` prefix (`ego.edges`,
  `ego.degree.0`, `ego.gwdegree.0.5`), so a coefficient is found by its R
  name. Pinned against `ergm.ego`'s labels.
- **A target at a bound of its statistic is fixed at ∓Inf, as in
  `ergm.ego`**: `degree0` when no ego is an isolate, for example, used to
  get a finite, unidentified coefficient reported as converged (−5.6 with
  standard error 0.17 where `ergm.ego` reports `-Inf`). It is now fixed at
  `-Inf` with ergm's warning, held at its bound by the sampler, given
  standard error 0 and p-value 0, and named by `show` and
  `approximations`; the other coefficients are estimated given it.
  `drop = false` refuses such a model, and so does `se = :bootstrap`.
- **`ERGMEgo.name` and `ERGMEgo.compute` are the shared NetworkCore.jl
  generics.** Result and model types are parametric on the ego ID type.
- **Removed exports**: `EgoSample`, `read_ego_data`, `merge_ego_data`.
- Julia 1.12 is required. Graphs.jl is no longer a dependency.

### Added

- `EgoGWESP(decay)`: `ergm.ego`'s `gwesp(decay, fixed = TRUE)`, from each
  alter's degree among the alter–alter ties (a clean-room implementation of
  the published definition, validated against `ergm.ego`'s output); fittable (ERGM.jl's `GWESP`),
  and, as an order-3 statistic, it carries the `transitiveties = −1/3` part
  of the network-size offset. It has no shared-partner cutoff: `ergm.ego`
  leaves out ties with more than `cutoff = 30` shared partners, while
  `EgoGWESP`, like `ergm`'s `gwesp` and ERGM.jl's `GWESP`, counts them.
- `EgoESP(k)`, `EgoMM(attr)` and `EgoConcurrent()`: `ergm.ego`'s `esp(k)`,
  `mm(attr)` (its default form: every level found at an end of a reported
  tie, the first cell dropped; `EgoMM(attr, l1, l2)` is one cell) and
  `concurrent`, with its per-ego values and labels (`esp1`,
  `mm[Race=Black,Race=Hisp]`, `concurrent`); fittable through ERGM.jl's
  `ESP`, `NodeMix` and `Concurrent`. `EgoESP` is an order-3 statistic, so
  it carries the `transitiveties = −1/3` part of the network-size offset.
  Implemented from the terms' documented definitions and checked against
  `ergm.ego`'s outputs (`test/fixtures/ego_mixing_esp.toml`). `mm`'s
  two-attribute form, margins and `levels=`/`levels2=`, and
  `concurrent(by=)`, are refused with an explanation.
- `fit_ergm_ego(...; effective_size = 64)`: ESS-adaptive sampling (see
  Breaking).
- A target at the TOP of its attainable range is fixed at `+Inf` (pinned on
  `gwdegree(0)` with no isolate, against `ergm.ego`); a mixing-matrix cell
  whose level no ego of the pseudo-population carries is refused when its
  target is positive (no simulated network can match it).
- `coefnames(fit)` (StatsAPI) returns the coefficient labels.
- `fit_ergm_ego(...; drop = true)`: `false` refuses a model with a target at
  a bound of its statistic instead of fixing its coefficient at ∓Inf.
- `EgoNodeFactor(attr; levels, base)`, `EgoNodeCov(attr)` and
  `EgoAbsDiff(attr; pow)`: `ergm.ego`'s `nodefactor`, `nodecov` and
  `absdiff`, with its per-ego values and labels; a multi-level
  `EgoNodeFactor` expands into one statistic per level of the egos' values
  (the first dropped by default, as in R).
- `fit.termination.step`: the coefficient step R's equivalence test was
  evaluated at, so `converged` can be recomputed from the recorded sample.
- `se = :bootstrap`: a bootstrap over egos. Each replicate resamples the
  egos with replacement (with their sampling weights), rebuilds the
  pseudo-population and the targets and refits, through
  `NetworkCore.bootstrap_cov`; reproducible from `rng` and independent of the
  thread count. Standard errors are the replicates' normalised
  interquartile ranges, plus `n_egos/popsize` times the design sandwich
  when the population size is known; failed refits are `NaN` rows of
  `fit.boot_replicates`, warned about once and listed by `approximations`,
  each saying that the standard errors are then conditional on a finite
  refit and biased downward.
  Coverage of nominal 95 % intervals for edges / nodematch: 0.94–0.97 /
  0.98 with 20 of 200 egos and 0.98 / 0.99 with 50 egos of mean degree 56,
  where the default covers 0.85 / 0.80–0.84 and 0.89 / 0.84. `show` recommends
  it when there are fewer than 50 egos.
- `se = :superpopulation` (default when `popsize` is known; `se = :design`
  gives `ergm.ego`'s): coverage of nominal 95 % intervals on a 200-actor
  population is 0.94 / 0.95 at a census (`:design`: 0.80 / 0.83), 0.92 /
  0.92 at 100 egos, 0.93 / 0.91 at 50.
- The edgewise-shared-partner goodness-of-fit panel, and design-respecting
  simulated ego samples (one replicate per observed ego, with its weight).
- `fit.termination`, `fit.se_type`, `fit.vcov_design`,
  `fit.vcov_estimation`, `fit.mcmc_convergence`, `fit.sim_stats`.
- The StatsAPI surface `coef`, `stderror`, `vcov`, `confint`, `coeftable`,
  `nobs` (egos), `dof`; the result-metadata protocol (`estimand`,
  `objective == :moment`, `is_exact`, `se_method`, `approximations`).
  `loglikelihood`, `aic` and `bic` deliberately have no methods.
- `EgoTerm` (exported) and the `public` hooks `ERGMEgo.ergm_term` and
  `ERGMEgo.ego_contribution`; `EgoGWDegree(decay)` for any `decay ≥ 0`.
- The ego terms declare the attainable range of the statistic they estimate
  as a method of ERGM.jl's `ERGM.Extension.attainable_range` (the range of
  `ergm_term(t)`), instead of restating ERGM.jl's table; the sampler budget
  and the stopping rule come from `ERGM.Extension` too.
- `ego_design`, `ego_target_stats`, `ego_mixing_matrix`, `n_alters`,
  `ego_degree`, `alter_degree`, `n_alter_ties`, `summary_stats`.
- Validation at construction: sampling weights (finite, non-negative, not
  all zero), duplicated alters, alter self-ties, an ego listed as its own
  alter, alter attributes of the wrong length, unknown columns.
- One-line `show` for `EgoNetwork`, `EgoData` and `EgoERGMModel`; `show` of
  a fit prints the offset, the termination test, the coefficient table, the
  standard-error decomposition and what the standard errors do not cover.
- Four golden fixtures against `ergm.ego` 1.1.4 (`test/fixtures/`): a
  census, a weighted sub-design, network-size handling (triangle model
  under three population sizes, a weighted `gwdegree` fit, per-capita
  coefficients at two pseudo-population sizes, `popsize ≠ ppopsize`), and
  the attribute, degree and shared-partner terms (targets under a census
  and a weighted design, the `ergm.ego` example model without and with
  `gwesp`, `gwesp` under the network-size offset, a target at its bound,
  R's default pseudo-population). Every Monte-Carlo tolerance is derived
  from R's own seed-to-seed spread, with a seed that failed or timed out
  counted in the band.
- A precompile workload; a benchmark environment with allocation gates.
- Every export has a docstring with a runnable example, executed by the
  test suite; Aqua.jl checks.

### Fixed

- Moment matching no longer diverges on `gwdegree` models (about a third of
  seeds before): the start is annealed to the targets and the steps are
  Hummel-damped. A singular sampled covariance makes the fit step back
  instead of stopping.
- The design variance has the with-replacement factor `n/(n − 1)` and
  matches `ergm.ego` exactly; the spurious standalone `I⁻¹` term is gone.
- The MCMC budget scales with the pseudo-population's dyad count (fixed
  budgets stopped mixing at 205 vertices).
- `converged` and the recorded diagnostics describe one sample.
- `gof` draws everything from `rng`, uses the fit's MCMC budget, and its
  degree distribution has R's bins with an upper tail.
- `fit.netsize_adjustment` is `0.0`, not `-0.0`, when `popsize == ppopsize`.

### Known limitations

The same nine items, in the same order, are in the README under "Not
implemented".

- **Richer survey designs.** One sampling weight per ego; no strata,
  clusters, finite-population correction, replicate weights or
  without-replacement inclusion probabilities, and no port of `ergm.ego`'s
  bootstrap or jackknife variance estimators of the targets (`stats.est`).
- **Calibrated default standard errors for small samples or an unknown
  population size.** With about 20 egos the default intervals cover 0.87
  (edges) and 0.78 (nodematch), and an attribute term can cover as little
  as 0.85 with 100 egos that have many ties: the pseudo-population's
  attribute composition is itself estimated. `se = :bootstrap` covers
  there (conservatively) but costs `n_boot` refits and is not the default.
  Without `popsize` the default standard errors are `ergm.ego`'s, which
  under-cover as the sampling fraction grows.
- **Several of `ergm.ego`'s terms.** The curved `gwesp(fixed = FALSE)`
  (refused: ERGM.jl's curved MCMLE fits a whole network, not target
  statistics); `nodemix`, `absdiffcat`, `degrange`, `concurrentties`,
  `degree1.5`, `transitiveties`, `cyclicalties` and `meandeg`; `mm`'s
  two-attribute form, margins and level selections; `concurrent(by=)`;
  `degree`'s `by=`/`homophily=` forms; attribute terms on data without
  alter attributes.
- **`simulate(fit)`** is not implemented.
- **`ergm.ego`'s other controls**: `ppop.wt = "sample"`, `stats.wt`,
  constraints, user offsets, `drop = FALSE` (a refusal here), and R's
  default `ppopsize` for a known population above 1000 (here ten times the
  number of egos).
- **Directed and two-mode ego data** are refused.
- **A likelihood.** `loglikelihood`, `aic` and `bic` have no methods.
- **Multi-chain fitting.** `fit_ergm_ego` samples with one chain.
- **The goodness of fit of `gof.ergm.ego` exactly.** R's three statistics,
  computed on design-respecting ego samples of each simulated network
  where R uses the whole simulated network.

## [0.1.0] - 2026-02-09

Development version, never released.
