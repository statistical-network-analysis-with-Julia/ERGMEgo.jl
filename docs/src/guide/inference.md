# Population Inference

## From ego statistics to targets

For a pseudo-population of size ``m``, the target for each term is
``m \cdot \bar h_w`` where ``\bar h_w`` is the design-weighted mean
per-ego contribution. This is what `ergm.ego` passes to `ergm` as
`target.stats`.

## Pseudo-population and moment matching

[`fit_ergm_ego`](@ref) (alias [`ergm_ego`](@ref)) builds an undirected
network whose attributes replicate each ego ``\operatorname{round}(m_0 w_i /
\sum w)`` times, ``m_0`` the requested `ppopsize` — `ergm.ego`'s
`ppop.wt = "round"`. The realised size ``m`` is the sum of those counts and
can differ from the request (69 unit-weight egos and `ppopsize = 100` give
69 vertices; R's fit of the weighted sub-design at `ppopsize = 500` has
508); the fit says so and uses ``m`` throughout. Because an ego's count
depends on its own weight only, the composition of the pseudo-population
does not depend on the order of the egos. Without `ppopsize` the size is
`ergm.ego`'s default: the number of egos when the population size is
unknown (R warns, and so does `fit_ergm_ego`, when the weights are then
unequal), the population size when it is known — up to 1000; above that
the default is ten times the number of egos, which `show` points out.

The network is seeded at the target density and then **annealed** toward
the target statistics (tie toggles that reduce the scaled distance to the
targets), which is what R's `ergm(target.stats=)` does with `san`. The
pseudo-likelihood estimate on that network is the start. Each iteration
then draws a sample at the current coefficients and takes a Hummel-style
partial Newton step, exactly as `ERGM.mcmle` does:

```math
\theta \leftarrow \theta + \gamma\,\widehat{\operatorname{Cov}}_\theta(g)^{-1}
\, (\text{targets} - \bar g_\theta),
```

with the step length ``\gamma`` growing to 1 as the sampled cloud comes to
cover the targets (the squared Mahalanobis distance of the targets from the
sample mean below the 0.95 quantile of ``\chi^2_p``). A singular covariance
(a collapsed sampler) makes the fit step back halfway toward the last
coefficients that gave a regular sample.

The stopping rule is R ergm's **confidence** test (`termination =
:confidence`): the moment equation at the updated coefficients — estimated
by importance-reweighting the sample — must lie, with `conv_confidence`
(0.99) confidence, inside the tolerance region
``x'(\texttt{conv\_precision}\cdot\Sigma)^{-1}x \le 1``. While it does not,
the sample's target effective size is raised (by the factor the test asks
for, at most doubled), up to a quarter of `max_n_samples` (four times
`n_samples`).
Unlike a fixed threshold on the t-ratios this is attainable under
Monte-Carlo noise, and unlike a non-significant Hotelling test it cannot
pass because the sample is too small to see a difference: a chain that does
not mix does not converge. `termination = :hotelling` selects the previous
rule (every t-ratio below `conv_threshold` and a Hotelling ``T^2`` test not
rejected at `hotelling_alpha`).

**The sample that passes is the final sample** (R's `ergm` design): it is
what `converged`, `fit.termination` (`(rule, p_value, precision,
confidence, n_samples)`), `fit.mcmc_convergence` (`(iterations,
step_length, t_ratios, hotelling_p, n_eff)`), `fit.sim_stats` and the
standard errors all describe. Only when `maxiter` (60) is exhausted is one
more sample drawn at the returned coefficients, and then that sample
decides `converged`. A fit that does not converge is **loud**: a warning
quoting the termination test and that sample's max t-ratio, Hotelling
p-value and iteration count, `converged == false`, the same sentence under
`Converged: false` in `show`, and an entry in `approximations(fit)`.

With this start and step rule the textbook fits converge in one or two
iterations. A weighted `edges + nodematch + gwdegree(0.5)` fit — which the
earlier damped Newton iteration from a Bernoulli start lost on about a
third of the seeds — converges on every seed tried and matches `ergm.ego`
(`test/fixtures/ego_netsize.toml`; R's own fit fails on one of its six
seeds there).

**The sampler is `ERGM.mcmle`'s** (`ERGM.Extension.mcmle_sampler`), with
the defaults R ergm 4 runs underneath `ergm.ego`: the SPDyad proposal
(`proposal = :spdyad`, tie/no-tie mixed with a shared-partner-focused
proposal; R's `MCMC.prop = ~sparse + .triadic`) and **ESS-adaptive
sampling** (`effective_size = 64`, R's `MCMLE.effectiveSize`): one chain,
started from the annealed pseudo-population, burned in once (``20 \cdot
n_{dyads}`` toggles, ERGM.jl's rule `ERGM.Extension.mcmc_defaults`) and then
continued from iteration to iteration (R's `MCMLE.sequential`); each sample
of a few hundred draws is extended, its thinning interval doubling, until
its effective sample size reaches the target, and the stopping rule's boost
raises the target. What the test needs is an effective sample size, not a
draw count: with 14 statistics and the default precision and confidence it
needs several hundred effective draws, so the sample grows exactly as far
as the chain's autocorrelation requires.

`effective_size = nothing` restores the fixed-size design: each iteration
draws ``\min(3000, 20m)`` statistics ``\max(100, n_{dyads}/10)`` toggles
apart after a fresh burn-in from the annealed network, boosted up to
`max_n_samples` draws; `proposal = :tnt, effective_size = nothing` is the
pre-0.2 sampler. That design does not converge on `ergm.ego`'s help-page
model (its example model plus `gwesp(0, fixed = TRUE)`, 14 statistics on
205 vertices): with tie/no-tie proposals an interval of 2091 toggles leaves
about one effective draw in twenty for the `gwesp` and `nodematch`
statistics, so the 12 000-draw cap is about 530 effective draws, and the
test did not pass in 60 iterations (about 6 minutes on one seed). Raising
the cap to 48 000 draws let it pass in 9 iterations (47 s) on the same
seed, which locates the cause in the effective size of the sample, not in
the model or the test. At the defaults the same model converges in 5–15
seconds.

### A target at a bound

A target at an end of its statistic's attainable range has no finite
coefficient: `degree0` when no ego is an isolate, `triangle` or `gwesp`
when no alter–alter tie is reported, a `nodefactor` level no ego or alter
has. `ergm.ego` passes its targets to ergm as `target.stats`, and ergm's
check of them under its default `drop=TRUE` fixes such a coefficient at
`-Inf` (`+Inf` at the top of the range). `fit_ergm_ego` does the same:
ergm's warning, the coefficient fixed, the sampler holding the statistic at
its bound (a move off it is never accepted), and the other coefficients
estimated given that. The fixed coefficient has standard error 0 and
p-value 0, `dof` does not count it, and `show` and `approximations` name
it. On `faux.mesa.high` without its isolates, `edges + degree(0) +
nodematch("Grade")` gives `degree0 = -Inf` and the other two coefficients
of `ergm.ego` (pinned by `test/fixtures/ego_terms.toml`), and `edges +
nodematch("Grade") + gwdegree(0, fixed = TRUE)` gives `gwdeg.fixed.0 =
+Inf` — every ego reports an alter, so the number of non-isolates is at its
largest value, the network size — with `ergm.ego`'s other two coefficients
(`test/fixtures/ego_mixing_esp.toml`). `drop=false`
refuses such a model, and `se=:bootstrap` refuses it too (a bootstrap needs
finite point estimates).

## The network-size offset

A size-invariant ERGM's edges coefficient scales as
``\theta_{edges}(N) = \theta^* - \log N`` (Krivitsky, Handcock & Morris
2011). `ergm.ego` therefore fits the pseudo-population model with an
offset term `netsize.adj` whose coefficient is fixed at

```math
\texttt{netsize.adj} = -\log(m / N),
```

``m`` the pseudo-population size and ``N`` the population size `popsize`.
`fit_ergm_ego` does the same, and reports the free coefficients — the ones
R prints below its `netsize.adj` row — with the offset coefficient in
`fit.netsize_adjustment`:

- **`popsize` unknown.** Neither the keyword nor `ed.population_size`:
  ``N = 1``, R's default. The edges coefficient is ``\theta^*``, **per
  capita**: it does not depend on the pseudo-population size, and the edges
  coefficient of a population of ``N`` members is ``\theta^* - \log N``.
  `show` says "per capita".
- **`popsize = N`.** The coefficients are those of a network of ``N``
  vertices; with ``m = N`` the offset is 0.
- **Triangles.** For a model with [`EgoTriangle`](@ref) the offset
  statistic is ``\text{edges} - \text{transitiveties}/3`` (R's
  `offset(netsize.adj(edges = 1, transitiveties = -1/3))`), transitive ties
  being the ties with at least one shared partner. Whenever ``m \ne N`` the
  simulated model therefore carries the extra fixed-coefficient statistic
  `Offset(GWESP(0.0), -netsize.adj/3)` (the last entry of
  `fit.model.ergm_terms`), and the triangle coefficient depends on
  `popsize`: on the fixture's 100-vertex network it is 0.08 for
  ``N = m``, 0.65 for ``N = 2m`` and −4.19 per capita, in R and here.

Before 0.2 an unknown population size was set to the pseudo-population size
(so the edges coefficient moved by ``\log 2`` when `ppopsize` doubled), only
the edges coefficient was shifted (so triangle models with ``m \ne N`` were a
different model from R's), and `netsize_adjustment` held the negative of R's
value.

## Variance

The standard errors are `ergm.ego`'s decomposition (its
`vcov(fit, sources = "model")` and `sources = "estimation"`):

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

``V(\hat\theta) = V_{design} + V_{est}`` is `vcov(fit)`. `show(fit)` prints
R's "MCMC %" — the share of each standard error that the estimation term
adds — and on a well-mixed chain it is 0–2 %. The information ``I`` is one
MCMC estimate, which moves a standard error by about 10 % from seed to
seed, in R and here.

### What these standard errors cover

``\Sigma_t`` treats the egos as **independent** draws. They are not: a tie
between two sampled egos is reported by both, so their contributions are
positively correlated. For a statistic that is a sum over ties (edges,
nodematch) and an equal-probability sample of ``n`` of ``N`` actors, the
variance of the target under the model is the independent-egos variance
times ``1 + f``, ``f = n/N``: each tie of a sampled ego is reported a
second time with probability ``f``. Consequently:

- against the **model** parameter, nominal 95 % intervals cover less often
  as ``f`` grows — about 91 % at ``f = 0.1`` and 80–85 % at a census, where
  the reported standard error is ``1/\sqrt 2`` of the sampling standard
  deviation of the estimate (the test suite reproduces this);
- they are not finite-population intervals either: at a census the
  finite-population design variance is 0, and the reported one is not.

`se = :superpopulation` multiplies ``\Sigma_t`` by ``1 + n/N``. **It is the
default whenever the population size is known**; `se = :design` — the
default when it is not, since ``f`` is then unknown — reproduces
`ergm.ego`. `fit.se_type` records the method, and `show(fit)` and
`approximations(fit)` say what it does and does not cover.

Coverage of nominal 95 % intervals for `edges` / `nodematch` on a 200-actor
population with two equal groups, 300 replicates per row:

| egos (sampling fraction) | `se = :design` (`ergm.ego`) | `se = :superpopulation` |
|---|---|---|
| 200 (1.0) | 0.80 / 0.83 | 0.94 / 0.95 |
| 100 (0.5) | 0.86 / 0.85 | 0.92 / 0.92 |
| 50 (0.25) | 0.91 / 0.87 | 0.93 / 0.91 |
| 20 (0.1) | 0.85 / 0.78 | 0.87 / 0.78 |
| 100 of 1000 actors (0.1), mean degree 70 | 0.93 / 0.84 | 0.94 / 0.85 |

The correction restores the coverage where the double-counting is the
problem. What remains affects attribute terms: the attribute composition of
the pseudo-population (the group sizes) is estimated from the sample, and
at a balanced composition its effect on the coefficients is second-order,
so no sandwich captures it. It shows with few egos, where `nodematch` is
also biased downward, and when the egos have many ties, so that the targets
are precise relative to the composition (last row).
With the composition held at its population value the 20-ego intervals
cover 0.94 / 0.93. The correction is exact only for tie-sum statistics under
equal-probability sampling; it is approximate for unequal weights and for
triangle and degree statistics, and neither method knows about strata,
clusters, replicate weights or without-replacement inclusion probabilities,
which `EgoData` cannot express.

### A bootstrap over egos

`se = :bootstrap` captures the composition error no sandwich does. Each of
`n_boot` replicates (default 100) resamples the egos with replacement —
each drawn ego keeping its sampling weight, so the weighted design is
resampled as it was drawn — rebuilds the pseudo-population and the targets
from the resample, and refits with the same controls. The replicates run
through the shared `NetworkCore.bootstrap_cov` loop: every resample and every
refit seed is drawn from `rng` before the threaded refits start, so the
result is reproducible from `rng` alone and does not depend on the thread
count. The spread of the refits includes the design variance, the
composition error and each fit's Monte-Carlo error; it treats the egos as
independent, so when `popsize` is known ``f \, V_{design}`` is added for the
double-counted ties. The standard errors are the replicates' normalised
interquartile ranges (`IQR/1.349`, with the replicates' correlations): a
resample with an extreme composition is an outlier that would inflate the
plain standard deviation (by 35 % with 20 of 200 egos). A refit that does
not converge or cannot be set up is a `NaN` row of `fit.boot_replicates`,
excluded from the covariance, warned about once and listed by
`approximations(fit)`. The warning, `show` and `approximations` then say
what the exclusion does: "The standard errors are conditional on a finite
refit: the excluded replicates are the extreme ones, so the standard errors
are biased downward."

Coverage of nominal 95 % intervals for `edges` / `nodematch`:

| design | `se = :superpopulation` | `se = :bootstrap` |
|---|---|---|
| 20 of 200 egos, mean degree 14 (two runs, 150 and 100 replicates) | 0.85 / 0.80–0.84 | 0.94–0.97 / 0.98 |
| 50 of 200 egos, mean degree 56 (80 replicates) | 0.89 / 0.84 | 0.98 / 0.99 |

The bootstrap is calibrated or conservative (its standard errors run 10–35 %
above the sampling standard deviation of the estimate) where the default is
not. It costs `n_boot` fits, so it is not the default; `show` recommends it
when there are fewer than 50 egos. With an unknown population size it cannot
add the double-counting term (``f`` is unknown), which is small when the
sampling fraction is.

## Goodness of fit

[`gof`](@ref) — a method of the shared `NetworkCore.gof` generic returning
the shared `NetworkCore.GOFResult` — simulates pseudo-population networks at
the fitted coefficients, draws from each an ego sample with the observed
design, and compares observed design-weighted statistics to their simulated
distributions with two-sided Monte Carlo p-values (`NetworkCore.mc_pvalue`,
the ``(1 + k)/(N + 1)`` estimator, never exactly zero). It carries the
three diagnostics of R's `gof.ergm.ego` and two descriptive means:

| `GOFResult` statistic | rows | `gof.ergm.ego` |
|---|---|---|
| `"model statistics"` | every statistic of the model, per capita, labelled by the ego terms | `GOF = "model"` |
| `"degree distribution"` | `degree 0` … `degree maxdeg−1` plus the tail `degree ≥ maxdeg`, with `maxdeg = 2·max(K, 3)` and K the largest **observed** ego degree — R's `degree(0:(maxdeg−1)) + degrange(maxdeg)`. Every simulated ego lands in exactly one row. (No tail when `maxdeg ≥ ppopsize − 1`, as in R.) | `GOF = "degree"` |
| `"edgewise shared partners"` | `esp 0` … `esp 2·(max(K, 3) − 1)`: the per-capita number of ties whose two ends have `k` shared partners — R's `esp(0:maxesp)`. Present only when the ties among alters were collected. | `GOF = "espartners"` |
| `"ego summary statistics"` | mean degree and mean alter–alter ties per ego | — |

Read the first with care: the model statistics are the **fitted targets**,
so their p-values report whether the simulation reproduces what was fitted
(a small one means the fit did not converge or the chain does not mix), not
whether the model fits. The degree distribution is what an edges-only model
does *not* match, and the shared-partner distribution is where a model
without a triangle term shows its lack of clustering.

**The simulated ego samples have the observed design.** From each simulated
network one vertex is drawn for each observed ego, uniformly among the
pseudo-population vertices that replicate it, and it carries that ego's
sampling weight. The simulated column is therefore the same design-weighted
estimator as the observed one, from a sample of the same size and weight
distribution, and its spread includes the design effect of unequal weights.
(Drawing an unweighted simple random sample instead, as the package did
before 0.2, gives an envelope that is too narrow under unequal weights.
`gof.ergm.ego` uses the statistics of the whole simulated network, which
leaves the sampling variation of the observed column out altogether.)

[`ego_gof`](@ref) is a thin wrapper returning the `"ego summary
statistics"` as a NamedTuple.

The keywords are the fit's: `n_sim` (50), `rng` (every draw — the
pseudo-population, the MCMC chains and the ego samples — flows through it,
so two calls from the same `rng` state give identical envelopes), `burnin`
and `interval` (`nothing` selects the dyad-scaled rule of
`ERGM.Extension.mcmc_defaults`, the same budget the fit uses — the pre-0.2
literals `2000`/`200` were 0.1 % of the fit's burn-in on the 205-actor
census), and `n_chains` (`min(n_sim, 4)`; the chains are seeded in order
from `rng`, so the result is bit-identical whatever the thread count).

```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs = [:Grade], rng = Xoshiro(1))
fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng = Xoshiro(1))
g = gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2)
[s.name for s in g.statistics]                  # ["model statistics", "degree distribution", "edgewise shared partners", "ego summary statistics"]
g.statistics[1].labels                          # ["edges", "nodematch.Grade"]  (GOF = "model")
g.statistics[2].labels[end]                     # "degree ≥ 26" — the tail bin: K = 13 observed, maxdeg = 2·13
all(sum(g.statistics[2].simulated; dims = 2) .≈ 1)   # true — every simulated row is a distribution
g.statistics[3].observed[2]                     # 0.34 ties per capita with one shared partner; the model simulates ≈ 0
g.statistics[1].simulated == gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2).statistics[1].simulated   # true
ego_gof(fit; n_sim = 20, rng = Xoshiro(5), n_chains = 2).p_values.mean_degree == g.statistics[end].p_values[1]   # true
```
