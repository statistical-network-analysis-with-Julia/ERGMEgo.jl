# Terms

```@docs
EgoTerm
EgoEdges
EgoNodeMatch
EgoNodeFactor
EgoNodeCov
EgoAbsDiff
EgoDegree
EgoTriangle
EgoGWDegree
EgoGWESP
EgoESP
EgoMM
EgoConcurrent
ERGMEgo.compute(::ERGMEgo.EgoTerm, ::ERGMEgo.EgoData)
ERGMEgo.name(::ERGMEgo.EgoEdges)
ego_mixing_matrix
ego_target_stats
```

## Making a custom term fittable

A custom [`EgoTerm`](@ref) extends two `public` (not exported) hooks.
`ERGMEgo.ego_contribution` gives one ego's contribution to the per-capita
statistic. `ERGMEgo.ergm_term` maps the ego term to the ERGM.jl term whose
sufficient statistic it estimates; the built-in fittable terms are its
methods, and a custom term becomes fittable by adding one.

```@docs
ERGMEgo.ego_contribution
ERGMEgo.ergm_term
```
