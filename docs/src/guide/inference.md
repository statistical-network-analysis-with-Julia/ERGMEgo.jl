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
the current ``\theta``. Convergence is declared when every target is
matched to within `tol` (relative).

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

Two sources of uncertainty enter:

1. **Model**: the inverse information ``I^{-1}`` with
   ``I = \operatorname{Cov}_\theta(g)`` from the final MCMC sample;
2. **Design**: the survey variance of the targets,
   ``\Sigma_t = m^2 \, \widehat{\operatorname{Var}}_w(\bar h)``.

These combine as ``V(\hat\theta) = I^{-1} + I^{-1} \Sigma_t I^{-1}``
(the design-based sandwich of Krivitsky & Morris). Pseudo-likelihood-style
understatement does not arise here, but the MCMC noise in ``I`` does —
increase `n_samples` for more stable standard errors.

## Goodness of fit

[`gof`](@ref) — a method of the shared `Networks.gof` generic returning
the shared `Networks.GOFResult` — simulates pseudo-population networks at
the fitted coefficients, draws ego samples of the observed size, and
compares the observed design-weighted mean degree and mean alter-tie
count to their simulated distributions with two-sided Monte Carlo
p-values. [`ego_gof`](@ref) is the legacy NamedTuple-returning form.
