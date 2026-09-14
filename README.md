# ERGMEgo.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMEgo.jl/stable/)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGMEgo.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGMEgo.jl icon" width="160">
</p>

ERGMs for egocentrically sampled network data in Julia — a port of the R
`ergm.ego` package (Krivitsky & Morris 2017).

## Installation

Requires Julia 1.12+. ERGMEgo.jl depends on the unregistered
[Networks.jl](https://github.com/statistical-network-analysis-with-Julia/Networks.jl) and [ERGM.jl](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl) packages, which must be added first (in this order):

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Networks.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl")
```

For development, you can instead clone all ecosystem repositories side by
side (the monorepo layout) and start Julia with the root workspace project
(`julia --project=.` in the clone root): the `[sources]` path dependencies
then wire the packages together with no ordered installs needed.

## Methodology

Given a sample of egos with their local networks (alters and alter–alter
ties), `ergm_ego`:

1. computes design-weighted **target statistics** scaled to a
   pseudo-population of size `ppopsize` (`ego_target_stats`);
2. builds a **pseudo-population network** whose vertex attributes are the
   egos' attributes replicated proportionally to the sampling weights;
3. fits coefficients by **MCMC moment matching** on the targets (the
   method-of-moments estimator that `ergm` uses for `target.stats`), with
   ERGM.jl's MCMLE convergence tests — per-statistic t-ratios below
   `conv_threshold` and a Hotelling T² test on the Geyer effective sample
   size (`ERGM.mcmc_convergence`) — evaluated on the sample drawn at every
   Newton iteration; the sample that passes **is** the final sample, so
   `converged`, `fit.mcmc_convergence`, `fit.sim_stats` and the standard
   errors describe one draw and cannot disagree (only when `maxiter` runs
   out after a step is one more sample drawn at the returned coefficients,
   and it decides). A fit that does not converge **warns**, records
   `converged == false`, and says so in `show` and `approximations`;
4. applies the **network-size adjustment** `−log(popsize/ppopsize)` to the
   edges coefficient, putting it on the population scale;
5. reports `ergm.ego`'s standard-error decomposition: the **survey-design**
   variance of the targets sandwiched by the inverse information, plus the
   Monte-Carlo **estimation** term of the moment equations,
   `V(θ̂) = I⁻¹ Σ_design I⁻¹ + I⁻¹/n_eff` — `vcov_design` and
   `vcov_estimation` on the result, `vcov(fit)` their sum. There is no
   standalone `I⁻¹` term: the estimand is a population parameter estimated
   from a sample of egos, and the population network is not modelled as a
   draw from the ERGM.

The MCMC budget scales with the pseudo-population's dyad count (ERGM.jl's
one rule: burn-in `20·n_dyads`, interval `max(100, n_dyads ÷ 10)`, 3000
draws at most); `n_samples`, `burnin`, `interval` and `maxiter` override it.
Every random draw flows through the `rng` keyword, so a fit is bit-identical
for the same `rng` state whatever the global RNG holds.

## Ego statistics and their ERGM counterparts

| Ego term | Per-ego contribution | Estimates |
|----------|---------------------|-----------|
| `EgoEdges()` | degree/2 | `edges` |
| `EgoNodeMatch(attr)` | matching alters/2 | `nodematch(attr)` |
| `EgoTriangle()` | alter–alter ties/3 | `triangle` |
| `EgoGWDegree(decay)` | `e^α(1−(1−e^{−α})^d)`, `decay ≥ 0` | `gwdegree(decay, fixed=TRUE)` (`gwdeg.fixed.<decay>`) |
| `EgoDegree(d)` | `1[degree = d]` | descriptive only |

With a census ego sample these mappings are exact:
`n · compute(EgoEdges(), ed) == edges(network)` (tested).

## Quick Start

```julia
using ERGMEgo, DataFrames, Random

ego_df   = DataFrame(ego_id = [1, 2], group = ["A", "B"], w = [2.0, 1.0])
alter_df = DataFrame(ego_id = [1, 1, 2], alter_id = [10, 11, 10],
                     group = ["A", "B", "B"])
aatie_df = DataFrame(ego_id = [1], src = [10], dst = [11])

ed = as_egodata(ego_df, alter_df; aatie_df = aatie_df,
                ego_attrs = [:group], alter_attrs = [:group],
                weight_col = :w)

# Fit an egocentric ERGM (population of 500, pseudo-population of 100).
# fit_ergm_ego is the standardized entry point (fit_<model> naming);
# ergm_ego is the R-faithful alias of the same function.
result = ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:group)];
                  ppopsize = 100, popsize = 500)

# Goodness of fit against simulated ego samples (gof returns the shared
# Networks.GOFResult with gof.ergm.ego's two diagnostics — "ego summary
# statistics" (GOF = "model": mean degree, mean alter ties; the mean degree
# is a fitted target) and "degree distribution" (GOF = "degree": R's bins,
# degree 0 … 2·max(K, 3) − 1 plus a "degree ≥ 2·max(K, 3)" tail, K the
# largest observed ego degree, so every simulated ego lands in one row);
# ego_gof gives the first as a NamedTuple). Every draw flows through rng;
# the MCMC budget is the fit's dyad-scaled rule.
g = gof(result; n_sim = 50, rng = Xoshiro(1))
g.statistics[2].labels[1:2]                       # ["degree 0", "degree 1"]
g.statistics[2].labels[end]                       # "degree ≥ 6" — the tail bin (K = 2 here)
ego_gof(result; n_sim = 50, rng = Xoshiro(1))

# Population size estimation
estimate_popsize(ed)                              # Horvitz-Thompson
estimate_popsize(ed; method = :capture_recapture) # Lincoln-Petersen on alter overlap
```

## Simulating ego samples

`simulate_ego_sample` is the ecosystem's `Network → EgoData` conversion
adapter, and it follows the shared conversion and missing-data contracts:

```julia
using ERGMEgo, Networks, Random

net = load_dataset(:faux_mesa_high)                    # undirected; :Grade, :Race, :Sex
ed = simulate_ego_sample(net, 30; ego_attrs = [:Grade], rng = Xoshiro(1))
ed                                                     # EgoData{Int64}: 30 egos, weighted mean degree …

# What the conversion dropped, named (the conversion contract)
ed, rep = simulate_ego_sample(net, 30; ego_attrs = [:Grade], rng = Xoshiro(1), report = true)
dropped_fields(rep)            # [:edges, :vertex_attrs, :vertex_attrs] — unsampled ties, :Race, :Sex
census, rep = simulate_ego_sample(net, 205; ego_attrs = [:Grade, :Race, :Sex], report = true)
is_lossless(rep)               # true

# Masked (unobserved) dyads are refused unless asked for in writing:
# `simulate_ego_sample(masked, 30)` throws the shared ecosystem ArgumentError
# whose bullets name the opt-in, `missing = :face`
masked = copy(net)
set_missing_dyad!(masked, 1, 2)
ed, rep = simulate_ego_sample(masked, 30; missing = :face, report = true)
:missing_dyads in dropped_fields(rep)                  # true
supports_missing(simulate_ego_sample), missing_policies(simulate_ego_sample)   # (true, (:error, :face))
```

Refused with an `ArgumentError` that says what to do instead: a masked
network under the default `missing = :error`, a **directed** network (ego
networks are undirected; symmetrise first), a **two-mode** network, and an
`ego_attrs` entry the network does not carry, or carries for only some
vertices — the pre-0.2 code silently filled `missing`, which a homophily
term then miscounted. Alter IDs are the network's vertex IDs, so cross-ego
overlap (needed for capture-recapture) is preserved.

**Missing dyads.** A masked dyad is *unobserved*, not absent, and an ego
sample would record its stored face value as an observed ego–alter or
alter–alter tie. `simulate_ego_sample` therefore refuses a masked network
(`missing = :error`, the default; the message names the masked count and
the opt-in) and reads the face values only when asked in writing with
`missing = :face`, recording the number of face-read dyads under
`:missing_dyads` in the `ConversionReport` that `report = true` returns.
`supports_missing(simulate_ego_sample) == true` and
`missing_policies(simulate_ego_sample) == (:error, :face)` declare exactly
that. `fit_ergm_ego` itself takes `EgoData` and never sees a mask — the
egos are a sample of reported local networks, not a sociomatrix — so the
missing-dyad question arises only at the adapter.

## Convergence, standard errors and the StatsAPI surface

A census of statnet's `faux.mesa.high` (bundled as a Networks.jl teaching
dataset) reproduces `ergm.ego`'s fit — two golden fixtures in
`test/fixtures/` pin the package against R: the **census** (unit weights,
where the targets and their design variance are the network's own and are
asserted exactly) and a **weighted sub-design** (every third actor, case
weights `Grade − 6`), which pins what the census cannot — the Hájek
targets of `edges`, `nodematch`, `gwdegree(0.5)` and `triangle`, the design
variance of a weighted mean, the weight-proportional pseudo-population,
the exact information at R's estimate, and a weighted `ergm.ego` fit:

```julia
using ERGMEgo, Networks, Random

net = load_dataset(:faux_mesa_high)                   # 205 students, 203 ties
ed = simulate_ego_sample(net, 205; ego_attrs = [:Grade], rng = Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng = Xoshiro(1))

fit.converged                        # true
fit.mcmc_convergence                 # (iterations, step_length, t_ratios, hotelling_p, n_eff)
coef(fit)                            # ≈ [-6.02, 2.82]: ergm.ego's netsize.adj + edges, nodematch.Grade
stderror(fit)                        # ≈ [0.16, 0.19] (ergm.ego: 0.178, 0.196 — Monte-Carlo noise in I on both sides)
sqrt.(vcov(fit)[1, 1])               # = stderror(fit)[1]
fit.vcov_design                      # ergm.ego's vcov(fit, sources = "model")
fit.vcov_estimation                  # ergm.ego's vcov(fit, sources = "estimation")
coeftable(fit)["ego.edges"]          # the printed row, inspectable
confint(fit)                         # normal-theory 95 % intervals
nobs(fit), dof(fit)                  # (205, 2): egos, finite coefficients
approximations(fit)                  # what the numbers do and do not account for
```

`coef`, `stderror`, `vcov`, `confint`, `coeftable` (a
`Networks.CoefficientTable` — the table `show(fit)` prints), `nobs` (the
number of egos; R's `nobs()` on the underlying `ergm` object counts
pseudo-population dyads) and `dof` are defined. `loglikelihood`, `aic` and
`bic` deliberately have **no** methods: the fit is moment matching
(`objective(fit) == :moment`) and no likelihood is ever evaluated. An
unconverged fit (`maxiter` too small, a degenerate model) is loud — a
warning with the max t-ratio, Hotelling p-value and iteration count, the
same sentence under `Converged: false` in `show`, and an entry in
`approximations(fit)`.

## Actionable errors

The classic mistakes fail early, with the fix in the message, instead of
producing a number:

| Mistake | What happens |
|---------|--------------|
| `EgoNodeMatch(:x)` on data whose egos or alters lack `:x` | `ArgumentError` naming the attribute, the ego and the side that lacks it — from `compute`, `ego_target_stats` and `fit_ergm_ego` (before any MCMC). The pre-0.2 code scored the ego as 0. |
| `EgoNodeMatch(:x)` (or `ego_mixing_matrix`) where an ego's or an alter's `:x` is `missing` — an unknown alter attribute from a survey, passed through by `as_egodata` | `ArgumentError` naming the attribute, the ego and how many of its alters have no value, with the fix (drop those alters, recode, or drop the term); it used to be a `TypeError` from inside the matching loop. |
| `ego_mixing_matrix(ed, :x)` with an ego lacking `:x` | `ArgumentError` (it used to skip the ego silently). |
| `as_egodata(…; ego_attrs = [:x])`, or any other column keyword, naming a column its frame lacks | `ArgumentError` naming the column, the keyword, the frame and the columns the frame has. |
| `simulate_ego_sample` on a directed, two-mode or masked network, or with an unknown/partial `ego_attrs` entry | `ArgumentError` (see above); `missing = :face` is the written opt-in for masked dyads. |
| `fit_ergm_ego(…; ppopsize = 1)`, or a default pseudo-population below 5 vertices | `ArgumentError` naming the size, where it came from (the keyword, or the `popsize`/`10·n_egos` default rule with the numbers it saw) and the fix `ppopsize=<n> ≥ 5`. |
| `summary_stats` on an `EgoData` with no egos (an ego frame that filtered to zero rows) | `ArgumentError` saying so, not `mean`'s "reducing over an empty collection". |
| A sampling weight that is `NaN`, `Inf` or negative (an `NA` inclusion probability, a `1/0`), or weights that sum to zero — through `EgoData`, `ego_design(…; weights)` or `as_egodata(…; weight_col)` | `ArgumentError` from the `EgoData` constructor naming the ego (index and id) and the value, before any statistic is computed. Round 2 returned `NaN` targets (then the unrelated "density ≥ 1" error) for a zero-sum/`NaN`/`Inf` vector, and ran the whole MCMC and **returned coefficients** from a negative-weight Hájek mean. A single zero weight is legal. |
| An ego listing the same alter twice (a duplicated `(ego, alter)` row from a wide-to-long reshape), or an alter tied to itself (a `true` on the diagonal of `alter_ties`, or an `aatie_df` row `(ego, a, a)`) | `ArgumentError` from `EgoNetwork` naming the ego and the alter; `as_egodata` says how many duplicate rows and which alter. Round 2 accepted both: the duplicate doubled the ego's degree and homophily count in every target, the self-tie was dropped by `sum ÷ 2`. |
| `estimate_popsize(ed; method = :lincoln_petersen)` | `ArgumentError` naming the unknown method and the two valid ones, `:horvitz_thompson` (the weight sum) and `:capture_recapture` (two-sample Lincoln–Petersen on alter overlap). |
| A fit that does not converge | a warning quoting the final sample's diagnostics (which fail the rule, by construction), `converged == false`, the caveat in `show` and `approximations`. |

## Known limitations

- **The survey design is "independent egos with case weights", and nothing
  richer.** `EgoData` carries one sampling weight per ego; `_design_cov`
  computes the with-replacement variance of the weighted mean of the
  per-ego contributions (`n/(n−1)` factor, pinned exactly against
  `ergm.ego` under the census design). There is no input for **strata,
  clusters, a finite-population correction, replicate weights,
  without-replacement inclusion probabilities, or alter dependence**, and
  the design variance encodes none of them — so for any richer sampling
  design the standard errors are **narrower** than "survey-design variance"
  implies. This is the open half of
  [ERGMEgo#1](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl/issues/1);
  the fit tells you so: `show(fit)` prints the note, `approximations(fit)`
  lists it, and the ecosystem's capabilities page quotes it. If your design
  is not a weighted simple random sample of egos, treat the standard errors
  as a lower bound.
- **No likelihood.** The estimator is method-of-moments; `loglikelihood`,
  `aic` and `bic` have no methods (see above).
- **Moment matching runs one chain.** `fit_ergm_ego` samples with ERGM.jl's
  single-chain `mh_sample`; `n_chains` is honoured by `gof`/`ego_gof` (whose
  `sample_networks` is multi-chain) but not by the fit. Fits are reproducible
  and thread-count independent either way (every draw comes from `rng`).
- **`ERGMEgo` has no two-mode or directed counterpart** of `ergm.ego`'s
  models: `simulate_ego_sample` refuses such networks, and `EgoNetwork`
  requires a symmetric `alter_ties`.
- **Four fittable terms, and no more.** Only `EgoEdges`, `EgoNodeMatch`,
  `EgoTriangle` and `EgoGWDegree` can be fitted; `ergm.ego`'s `nodefactor`,
  `nodecov`, `absdiff`, `gwesp`, `mm` and `degree`/`concurrent` have **no
  ego counterpart yet** — a model written as
  `ergm.ego(egor ~ edges + nodefactor("Race") + absdiff("Grade") +
  gwesp(0.5, fixed=TRUE) + mm("Sex"))` has no Julia spelling: there is no
  `EgoNodeFactor`/`EgoNodeCov`/`EgoAbsDiff`/`EgoGWESP`/`EgoMixing` name to
  type (an undefined name, not a fit), and a descriptive `EgoDegree(d)` in a
  model is an `ArgumentError`. A custom fittable term extends the `public`
  hook `ERGMEgo._ergm_term` (and `_ego_contribution`); see the `EgoTerm`
  docstring.
- **There is no `simulate(fit)`** (`ergm.ego`'s `simulate.ergm.ego`, which
  draws networks or ego samples from a fitted model): `gof`/`ego_gof` is
  the only routine that draws from the fitted model, and it returns the
  summary statistics of those draws, not the networks or the ego samples.

## References

1. Krivitsky, P.N. & Morris, M. (2017). Inference for social network models
   from egocentrically sampled data, with application to understanding
   persistent racial disparities in HIV prevalence in the US. *Annals of
   Applied Statistics*, 11(1), 427-455.

2. Krivitsky, P.N., et al. ergm.ego: Fit, Simulate and Diagnose
   Exponential-Family Random Graph Models to Egocentrically Sampled Network
   Data. R package.
   [https://cran.r-project.org/package=ergm.ego](https://cran.r-project.org/package=ergm.ego)

## Citation

If you use ERGMEgo.jl in your work, please cite it using the entry in
[`CITATION.bib`](CITATION.bib):

```biblatex
@misc{SNWJERGMEgoJL,
  author = {{Statistical Network Analysis with Julia}},
  title = {ERGMEgo.jl: Exponential Random Graph Models for Egocentrically Sampled Network Data in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMEgo.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
