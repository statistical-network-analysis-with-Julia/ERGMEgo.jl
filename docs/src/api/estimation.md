# Estimation

[`fit_ergm_ego`](@ref) is the primary entry point; [`ergm_ego`](@ref) is
the R-faithful alias and [`fit_ego_ergm`](@ref) a legacy alias.

```@docs
fit_ergm_ego
ergm_ego
fit_ego_ergm
estimate_popsize
simulate_ego_sample
```

## Diagnostics

`gof` is a method of the shared `Network.gof` generic and returns the
shared `Network.GOFResult`; [`ego_gof`](@ref) is the legacy
NamedTuple-returning form.

```@docs
gof(::EgoERGMResult)
ego_gof
```
