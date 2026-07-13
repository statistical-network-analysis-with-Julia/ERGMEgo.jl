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
   method-of-moments estimator that `ergm` uses for `target.stats`);
4. applies the **network-size adjustment** `−log(popsize/ppopsize)` to the
   edges coefficient, putting it on the population scale;
5. reports standard errors that combine the model-based
   (inverse-information) and **survey-design** variance components:
   `V(θ̂) = I⁻¹ + I⁻¹ Σ_design I⁻¹`.

## Ego statistics and their ERGM counterparts

| Ego term | Per-ego contribution | Estimates |
|----------|---------------------|-----------|
| `EgoEdges()` | degree/2 | `edges` |
| `EgoNodeMatch(attr)` | matching alters/2 | `nodematch(attr)` |
| `EgoTriangle()` | alter–alter ties/3 | `triangle` |
| `EgoGWDegree(decay)` | `e^α(1−(1−e^{−α})^d)` | `gwdegree(decay)` |
| `EgoDegree(d)` | `1[degree = d]` | descriptive only |

With a census ego sample these mappings are exact:
`n · compute(EgoEdges(), ed) == edges(network)` (tested).

## Quick Start

```julia
using ERGMEgo, DataFrames

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

# Goodness of fit against simulated ego samples
ego_gof(result)

# Population size estimation
estimate_popsize(ed)                              # Horvitz-Thompson
estimate_popsize(ed; method = :capture_recapture) # Lincoln-Petersen on alter overlap
```

## Simulating ego samples

```julia
using Networks
net = network(100; directed = false)
# ... add edges/attributes ...
ed = simulate_ego_sample(net, 30; ego_attrs = [:group])
```

Alter IDs are the network's vertex IDs, so cross-ego overlap (needed for
capture-recapture) is preserved.

## References

1. Krivitsky, P.N. & Morris, M. (2017). Inference for social network models
   from egocentrically sampled data, with application to understanding
   persistent racial disparities in HIV prevalence in the US. *Annals of
   Applied Statistics*, 11(1), 427-455.

2. Krivitsky, P.N., et al. ergm.ego: Fit, Simulate and Diagnose
   Exponential-Family Random Graph Models to Egocentrically Sampled Network
   Data. R package.
   [https://cran.r-project.org/package=ergm.ego](https://cran.r-project.org/package=ergm.ego)

## License

MIT License - see [LICENSE](LICENSE) for details.
