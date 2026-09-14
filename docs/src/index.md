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
using Networks, ERGMEgo, Random

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
2. **Pseudo-population.** A network of `ppopsize` vertices whose
   attributes replicate the egos proportionally to their sampling
   weights.
3. **Moment matching.** Coefficients solve `E_θ[g] = targets` by MCMC
   Newton iterations — the estimator behind `ergm`'s `target.stats` —
   with ERGM.jl's MCMLE convergence tests (per-statistic t-ratios and a
   Hotelling T² test, `ERGM.mcmc_convergence`); non-convergence is a
   warning, a `converged == false`, and a caveat in `show`.
4. **Network-size adjustment.** The edges coefficient receives
   `−log(popsize/ppopsize)` so it refers to the population scale.

Standard errors are `ergm.ego`'s decomposition — the survey-design
component plus the Monte-Carlo estimation component, with no standalone
model-based term:
``V(\hat\theta) = I^{-1} \Sigma_{design} I^{-1} + I^{-1}/n_{eff}``.

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


## Citation

If you use ERGMEgo.jl in your work, please cite it using the entry in
[`CITATION.bib`](https://github.com/statistical-network-analysis-with-Julia/ERGMEgo.jl/blob/main/CITATION.bib):

```biblatex
@misc{SNWJERGMEgoJL,
  author = {{Statistical Network Analysis with Julia}},
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
