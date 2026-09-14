# Population Inference

## From ego statistics to targets

For a pseudo-population of size ``m``, the target for each term is
``m \cdot \bar h_w`` where ``\bar h_w`` is the design-weighted mean
per-ego contribution. This is what `ergm.ego` passes to `ergm` as
`target.stats`.

## Pseudo-population and moment matching

[`fit_ergm_ego`](@ref) (alias [`ergm_ego`](@ref)) builds an undirected
network of `ppopsize` vertices whose
attributes replicate the egos proportionally to the sampling weights
(largest-remainder rounding), seeds it near the target density, and then
iterates Newton steps

```math
\theta \leftarrow \theta + \widehat{\operatorname{Cov}}_\theta(g)^{-1}
\, (\text{targets} - \bar g_\theta)
```

with ``\bar g_\theta`` and the covariance estimated from MCMC samples at
the current ``\theta`` (a step is damped so that no coefficient moves by
more than 1 per iteration). Convergence is ERGM.jl's MCMLE rule
(`ERGM.mcmc_convergence`): every per-statistic t-ratio
``|t_j - \bar g_j| / \mathrm{sd}(g_j)`` below `conv_threshold` (0.1) **and**
a Hotelling ``T^2`` test of ``\bar g = t`` on the Geyer effective sample
size not rejected at `hotelling_alpha` (0.05). It is evaluated on the
sample drawn at every iteration, and **the sample that passes is the final
sample** (R's `ergm` design): it is what `converged`, the report stored as
`fit.mcmc_convergence` (`(iterations, step_length, t_ratios, hotelling_p,
n_eff)`), `fit.sim_stats` and the standard errors all describe, so
`converged` is exactly the rule applied to the recorded report and the four
cannot disagree. Only when `maxiter` (80) is exhausted after a Newton step —
the returned coefficients have no sample yet — is one more sample drawn at
them, and then that sample decides `converged`. A fit whose final sample
fails is **loud**: a warning quoting that sample's max t-ratio, Hotelling
p-value and the iteration count (labelled "on the final sample at the
returned coefficients", so the numbers quoted always fail the rule),
`converged == false`, the same sentence under `Converged: false` in `show`,
and an entry in `approximations(fit)`. The pre-0.2 rule — stop when every target is matched
to within 1 % — never looked at the Monte-Carlo sd, so it stopped at the
noise level; `tol` is now a deprecated, ignored keyword.

The MCMC budget per sample scales with the pseudo-population's dyad count
through ERGM.jl's one rule (`ERGMEgo._mcmc_controls`): burn-in
``20 \cdot n_{dyads}`` toggles, interval ``\max(100, n_{dyads}/10)``, and
``\min(3000, 20m)`` draws. On the 205-actor faux.mesa census that is an
effective sample size of about 150 per draw at 0.3 s.

## The network-size adjustment

A size-invariant ERGM's edges coefficient scales as
``\theta_{edges}(N) = \theta^* - \log N``. Fitting on a
pseudo-population of size `ppopsize` while targeting a population of size
`popsize` therefore requires the offset

```math
\theta_{edges}^{pop} = \hat\theta_{edges} - \log(popsize / ppopsize),
```

which `fit_ergm_ego` applies to the reported coefficient (stored in
`netsize_adjustment`). With `popsize == ppopsize` the adjustment is 0.

## Variance

The standard errors are `ergm.ego`'s decomposition (its
`vcov(fit, sources = "model")` and `sources = "estimation"`). Two sources
of uncertainty enter, and **the population network is not one of them**:

1. **Design**: the survey variance of the targets,
   ``\Sigma_t = m^2 \, \widehat{\operatorname{Var}}_w(\bar h)`` (with the
   ``n/(n-1)`` with-replacement factor of a weighted mean), propagated to
   the coefficients through the moment equations by the delta method —
   ``V_{design} = I^{-1} \Sigma_t I^{-1}`` with
   ``I = \operatorname{Cov}_\theta(g)`` from the final MCMC sample
   (`fit.vcov_design`);
2. **Estimation**: the Monte-Carlo error of solving the moment equations
   on a finite sample, ``V_{est} = I^{-1} \Sigma_{mc} I^{-1}`` with
   ``\Sigma_{mc} = I / n_{eff}`` the variance of the sampled mean and
   ``n_{eff}`` the Geyer effective sample size of the final sample
   (`fit.vcov_estimation`).

``V(\hat\theta) = V_{design} + V_{est}`` is `vcov(fit)`. There is
deliberately **no** standalone ``I^{-1}`` term: the estimand is a population
parameter estimated from a *sample of egos*, and the population network is
treated as fixed, not as a draw from the ERGM (Krivitsky & Morris 2017,
§4). An earlier version added ``I^{-1}`` anyway, which inflated an
`ergm.ego` standard error of 0.178 to about 0.19; the golden fixture now
pins the decomposition against R. `show(fit)` prints R's "MCMC %" — the
share of each standard error that the estimation term adds — and, on a
well-mixed chain, it is 0-2 %. The noise that remains is the Monte-Carlo
noise in ``I`` itself (one MCMC estimate of the information moves a
standard error by ~10 %, on both sides): increase `n_samples` or `interval`
for more stable standard errors.

## Goodness of fit

[`gof`](@ref) — a method of the shared `Networks.gof` generic returning
the shared `Networks.GOFResult` — simulates pseudo-population networks at
the fitted coefficients, draws ego samples of the observed size, and
compares observed design-weighted statistics to their simulated
distributions with two-sided Monte Carlo p-values (`Networks.mc_pvalue`,
the ``(1 + k)/(N + 1)`` estimator, never exactly zero). It carries the two
diagnostics of R's `gof.ergm.ego`:

| `GOFResult` statistic | rows | `gof.ergm.ego` |
|---|---|---|
| `"ego summary statistics"` | mean degree, mean alter ties | `GOF = "model"` |
| `"degree distribution"` | `degree 0` … `degree maxdeg−1` plus the tail `degree ≥ maxdeg`, with `maxdeg = 2·max(K, 3)` and K the largest **observed** ego degree — R's `degree(0:(maxdeg−1)) + degrange(maxdeg)`: the design-weighted proportion of egos in each bin, [`EgoDegree`](@ref) on each simulated sample. Every simulated ego lands in exactly one row, so observed and every simulated row sum to 1; the observed tail is 0 by construction and a model that over-produces high degrees shows up there. (No tail when `maxdeg ≥ ppopsize − 1`, as in R: every attainable degree is then a bin.) | `GOF = "degree"` |

Read the first with care: every model includes `EgoEdges()`, so the **mean
degree is a fitted target** — its p-value reports whether the chain
reproduced the target (a converged fit gives a large one by construction),
not whether the model fits. Mean alter ties is the informative row of the
pair (a model without a triangle term fails it), and the degree
distribution is what an edges-only model does *not* match: it reproduces
the mean degree but not the spread around it, which is exactly what R's
`GOF = "degree"` was designed to show. [`ego_gof`](@ref) is a thin wrapper
returning the first statistic as a NamedTuple, so the two cannot disagree;
it does not carry the degree distribution.

The keywords are the fit's: `n_sim` (50), `rng` (every draw — the
pseudo-population, the MCMC chains and the ego samples — flows through it,
so two calls from the same `rng` state give identical envelopes), `burnin`
and `interval` (`nothing` selects the dyad-scaled rule of
`ERGMEgo._mcmc_controls`, the same budget the fit uses — the pre-0.2
literals `2000`/`200` were 0.1 % of the fit's burn-in on the 205-actor
census), and `n_chains` (`min(n_sim, 4)`; the chains are seeded in order
from `rng`, so the result is bit-identical whatever the thread count).

```julia
using ERGMEgo, Networks, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs = [:Grade], rng = Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng = Xoshiro(1))
g = gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2)
g.statistics[1].labels                          # ["mean degree", "mean alter ties"]  (GOF = "model")
g.statistics[2].labels[1:3]                     # ["degree 0", "degree 1", "degree 2"]  (GOF = "degree")
g.statistics[2].labels[end]                     # "degree ≥ 26" — the tail bin: K = 13 observed, maxdeg = 2·13
sum(g.statistics[2].observed)                   # 1.0 — a distribution over the bins
all(sum(g.statistics[2].simulated; dims = 2) .≈ 1)   # true — and so is every simulated row
g.statistics[1].simulated == gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2).statistics[1].simulated   # true
ego_gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2).p_values.mean_degree == g.statistics[1].p_values[1]   # true
```
