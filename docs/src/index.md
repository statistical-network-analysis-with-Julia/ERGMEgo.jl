# ERGMEgo.jl

ERGMs for egocentrically sampled network data — inference about
complete-network properties from a sample of egos and their local
networks, following R `ergm.ego` (Krivitsky & Morris 2017).

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
   Newton iterations — the estimator behind `ergm`'s `target.stats`.
4. **Network-size adjustment.** The edges coefficient receives
   `−log(popsize/ppopsize)` so it refers to the population scale.

Standard errors combine the model-based and survey-design components:
``V(\hat\theta) = I^{-1} + I^{-1} \Sigma_{design} I^{-1}``.

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

## References

1. Krivitsky, P.N. & Morris, M. (2017). Inference for social network
   models from egocentrically sampled data. *Annals of Applied
   Statistics*, 11(1), 427-455.
