# Estimation

[`fit_ergm_ego`](@ref) is the primary entry point; [`ergm_ego`](@ref) is
the R-faithful alias and [`fit_ego_ergm`](@ref) a legacy alias.

```@docs
fit_ergm_ego
ergm_ego
fit_ego_ergm
ERGMEgo._mcmc_controls
estimate_popsize
simulate_ego_sample
```

`simulate_ego_sample` is the one routine of the package that can meet a
masked dyad, and it declares its missing-data policy through the shared
Networks.jl traits: `supports_missing(simulate_ego_sample) == true` and
`missing_policies(simulate_ego_sample) == (:error, :face)`. `fit_ergm_ego`
takes `EgoData` — a sample of reported local networks, not a sociomatrix —
so no `missing=` keyword exists on it.

## StatsAPI

`coef`, `stderror`, `vcov`, `confint`, `coeftable`, `nobs` and `dof` are the
surface the fit can honestly answer. `loglikelihood`, `aic` and `bic`
deliberately have no methods (the fit is moment matching; no likelihood is
evaluated).

```@docs
coef(::EgoERGMResult)
stderror(::EgoERGMResult)
vcov(::EgoERGMResult)
coeftable(::EgoERGMResult)
confint(::EgoERGMResult)
nobs(::EgoERGMResult)
dof(::EgoERGMResult)
```

## Result metadata

The shared `Networks.fit_metadata` protocol: what the fit actually did.

```@docs
ERGMEgo.objective(::EgoERGMResult)
ERGMEgo.is_exact(::EgoERGMResult)
ERGMEgo.se_method(::EgoERGMResult)
ERGMEgo.approximations(::EgoERGMResult)
```

## Diagnostics

`gof` is a method of the shared `Networks.gof` generic and returns the
shared `Networks.GOFResult`; [`ego_gof`](@ref) is the legacy
NamedTuple-returning form.

```@docs
gof(::EgoERGMResult)
ego_gof
```
