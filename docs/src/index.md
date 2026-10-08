# ERGMEgo.jl

Estimate complete-network properties from egocentric observations: sampled actors, their alters, optional alter–alter ties, and sampling weights. ERGMEgo.jl turns these local observations into target statistics and fits an ERGM by simulation-based moment matching.

| First analysis | Learn the model or data | Reference and detail |
|:--|:--|:--|
| [Build ego data and fit](getting_started.md) | [Understand the sampling data](guide/ego_networks.md) | [Interpret inference](guide/inference.md) |

!!! note "Supported scope"

    The implemented ego models use undirected, one-mode networks. Sampling uncertainty treats egos as independent with the supplied case weights; arbitrary clustered or stratified survey designs are not represented. The result is a moment fit, with MCMC diagnostics, rather than a fitted full-network likelihood.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

## Quick Start

A census of a small known network makes the target-statistic calculation easy to inspect before working with a sample:

```julia
using NetworkCore, ERGMEgo, Random

net = load_dataset(:florentine_marriage)
egos = simulate_ego_sample(net, nv(net); rng=Xoshiro(1))
println((ego_target=nv(net) * compute(EgoEdges(), egos), observed_edges=ne(net)))
fit = fit_ergm_ego(egos, [EgoEdges()]; ppopsize=nv(net), rng=Xoshiro(2))
display(fit)
```

With every actor sampled, the edge target equals the known edge count. A real ego survey instead requires a defensible sampling scheme, a population size, and compatible alter information. The [first tutorial](getting_started.md) shows the data-frame input and a sample of egos.

## The Method

Egocentric samples observe, for each sampled ego, its alters, ties among
those alters, and attributes. ERGMEgo.jl turns this into an ERGM fit in
four steps:

1. **Target statistics.** Each ego term contributes a design-weighted
   per-capita statistic (e.g. mean degree / 2 for edges); multiplied by
   the pseudo-population size these are the target sufficient statistics
   ([`ego_target_stats`](@ref)).
2. **Pseudo-population.** A network whose attributes replicate each ego
   `round(ppopsize · wᵢ/Σw)` times (`ergm.ego`'s rounding), annealed toward
   the targets.
3. **Moment matching.** Coefficients solve `E_θ[g] = targets` by MCMC with
   ERGM.jl's MCMLE machinery — Hummel-style partial Newton steps and R
   ergm's confidence stopping rule; non-convergence is a warning, a
   `converged == false`, and a caveat in `show`.
4. **Network-size offset.** `ergm.ego`'s `netsize.adj`, with coefficient
   `−log(ppopsize/popsize)`: the coefficients are those of a population of
   `popsize` members, and **per capita when the population size is
   unknown** (`popsize = 1`, R's default).

Standard errors are `ergm.ego`'s decomposition — the survey-design
component plus the Monte-Carlo estimation component, with no standalone
model-based term:
``V(\hat\theta) = I^{-1} \Sigma_{design} I^{-1} + I^{-1}/n_{eff}``.
`ergm.ego`'s design component treats the egos as independent and
under-covers the model parameter as the sampling fraction grows; when the
population size is known the default multiplies it by `1 + n/N`
(`se = :superpopulation`). `se = :bootstrap` resamples the egos and refits,
which also carries the sampling error of the pseudo-population's attribute
composition. See [Population Inference](guide/inference.md).

## Not implemented

- Survey designs beyond independent egos with case weights (strata,
  clusters, finite-population corrections, replicate weights), and
  `ergm.ego`'s bootstrap and jackknife variance estimators of the targets
  (`se = :bootstrap` is a bootstrap over egos that refits).
- Default standard errors calibrated for small samples (about 20 egos) or
  an unknown population size (`se = :bootstrap` covers there, at the cost
  of `n_boot` refits).
- `ergm.ego`'s terms other than `edges`, `nodematch`, `nodefactor`,
  `nodecov`, `absdiff`, `degree`, `triangle`, `gwdegree` and `gwesp` (`esp`,
  the curved `gwesp`, `mm`, `concurrent`, …), and the `by=`/`homophily=`
  forms of `degree`.
- `simulate(fit)`, constraints, user offsets, `ppop.wt = "sample"`.
- Directed and two-mode ego data; a likelihood (`loglikelihood`, `aic`,
  `bic`); multi-chain fitting.
- `gof.ergm.ego`'s reference distribution: the three statistics are R's,
  computed here on design-respecting ego samples of each simulated network.

The README's "Not implemented" section has the details.

## Contents

```@contents
Pages = [
    "getting_started.md",
    "guide/ego_networks.md",
    "guide/terms.md",
    "guide/inference.md",
    "api/types.md",
    "api/terms.md",
    "api/estimation.md",
]
Depth = 2
```

## Package overview

```@docs
ERGMEgo.ERGMEgo
```

## References

1. Krivitsky, P.N. & Morris, M. (2017). Inference for social network
   models from egocentrically sampled data. *Annals of Applied
   Statistics*, 11(1), 427-455.
2. Krivitsky, P.N., Handcock, M.S. & Morris, M. (2011). Adjusting for
   network size and composition effects in exponential-family random graph
   models. *Statistical Methodology*, 8(4), 319-339.


## Citation

If you use ERGMEgo.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl/blob/main/CITATION.bib).
Please also cite the R package it ports, `ergm.ego`, and the methods paper
(Krivitsky & Morris 2017); the ecosystem's
[How to cite](https://statistical-network-analysis-with-julia.github.io/citing/)
page lists the references.

```biblatex
@misc{SNWJERGMEgoJL,
  author = {Santoni, Simone},
  title = {ERGMEgo.jl: Exponential Random Graph Models for Egocentrically Sampled Network Data in Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGMEgo.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## Module

```@docs
ERGMEgo
```
