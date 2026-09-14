# Terms

```@docs
EgoTerm
EgoEdges
EgoNodeMatch
EgoTriangle
EgoGWDegree
EgoDegree
ERGMEgo.compute(::ERGMEgo.EgoTerm, ::ERGMEgo.EgoData)
ERGMEgo.name(::ERGMEgo.EgoEdges)
ego_mixing_matrix
ego_target_stats
```

## Making a custom term fittable

`ERGMEgo._ergm_term` is the `public` hook (like `ERGMEgo._mcmc_controls`)
that maps an ego term to the ERGM.jl term whose sufficient statistic it
estimates; the four built-in fittable terms are its only methods, and a
custom [`EgoTerm`](@ref) becomes fittable by adding one.

```@docs
ERGMEgo._ergm_term
```
