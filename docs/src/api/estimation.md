# Estimation

[`fit_ergm_ego`](@ref) is the primary entry point; [`ergm_ego`](@ref) is
the same function under `ergm.ego`'s name.

```@docs
fit_ergm_ego
ergm_ego
estimate_popsize
simulate_ego_sample
```

Every sampler of the package draws on one MCMC budget, scaled by the
pseudo-population of `m` vertices: `n_samples = max(400, min(3000, 20m))`
draws per sample, and `burnin`/`interval` from ERGM.jl's dyad-scaled rule
(`ERGM.Extension.mcmc_defaults`: `20·n_dyads` and `max(100, n_dyads ÷ 10)`
toggles over the `m(m−1)/2` dyads). Pass `n_samples`, `burnin` or `interval`
to override it. Under `fit_ergm_ego`'s default ESS-adaptive sampling
(`effective_size = 64`, through `ERGM.Extension.mcmle_sampler`) the chain is
burned in once, its interval starts at an eighth of `interval` and adapts,
each sample grows until it reaches the target effective size, and
`n_samples` sets the cap `max_n_samples = 4·n_samples`.

`simulate_ego_sample` is the one routine of the package that can meet a
masked dyad, and it declares its missing-data policy through the shared
NetworkCore.jl traits: `supports_missing(simulate_ego_sample) == true` and
`missing_policies(simulate_ego_sample) == (:error, :face)`. `fit_ergm_ego`
takes `EgoData` — a sample of reported local networks, not a sociomatrix —
so no `missing=` keyword exists on it.

## StatsAPI

`coef`, `coefnames`, `stderror`, `vcov`, `confint`, `coeftable`, `nobs` and
`dof` are the surface the fit can honestly answer; `coefnames` and the rows of
`coeftable` are `ergm.ego`'s labels. `loglikelihood`, `aic` and `bic`
deliberately have no methods (the fit is moment matching; no likelihood is
evaluated).

```@docs
coef(::EgoERGMResult)
coefnames(::EgoERGMResult)
stderror(::EgoERGMResult)
vcov(::EgoERGMResult)
coeftable(::EgoERGMResult)
confint(::EgoERGMResult)
nobs(::EgoERGMResult)
dof(::EgoERGMResult)
```

## Result metadata

The shared `NetworkCore.fit_metadata` protocol: what the fit actually did.

```@docs
ERGMEgo.objective(::EgoERGMResult)
ERGMEgo.is_exact(::EgoERGMResult)
ERGMEgo.se_method(::EgoERGMResult)
ERGMEgo.approximations(::EgoERGMResult)
```

## Diagnostics

`gof` is a method of the shared `NetworkCore.gof` generic and returns the
shared `NetworkCore.GOFResult`; [`ego_gof`](@ref) is the legacy
NamedTuple-returning form.

```@docs
gof(::EgoERGMResult)
ego_gof
```

## Renamed and removed names

Names and keywords of the development versions that 0.2.0 does not carry.
Each is gone, not deprecated: the old spelling is an error.

| Old | Now |
|-----|-----|
| `fit_ego_ergm(ed, terms)` | `fit_ergm_ego(ed, terms)` or `ergm_ego(ed, terms)` |
| `fit_ergm_ego(…; max_iter=k)` | `fit_ergm_ego(…; maxiter=k)` |
| `fit_ergm_ego(…; tol=x)` | removed (it was ignored): convergence is the `termination` rule, tuned by `conv_precision` and `conv_confidence` |
| `ego_design(ed; ppopsize=N)` | `ego_design(ed; popsize=N)` — it sets the population size; the pseudo-population size is `fit_ergm_ego`'s `ppopsize` |
