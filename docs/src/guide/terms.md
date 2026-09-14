# Ego Terms

Every ego term is a **per-capita statistic**: `compute(term, ed)` returns
the design-weighted mean per-ego contribution, and `ppopsize × compute`
is the target sufficient statistic used in fitting. Each fittable term
maps to the ERGM.jl term whose sufficient statistic it estimates.

| Term | Per-ego contribution | ERGM term |
|------|---------------------|-----------|
| [`EgoEdges`](@ref) | ``d_i / 2`` | `Edges()` |
| [`EgoNodeMatch`](@ref) | matching alters ``/ 2`` | `NodeMatch(attr)` |
| [`EgoTriangle`](@ref) | alter–alter ties ``/ 3`` | `Triangle()` |
| [`EgoGWDegree`](@ref) | ``e^\alpha(1-(1-e^{-\alpha})^{d_i})``, ``\alpha \ge 0`` | `GWDegree(α)` (labelled `gwdeg.fixed.α`, R's `gwdegree(α, fixed=TRUE)`) |
| [`EgoDegree`](@ref) | ``1[d_i = d]`` | *descriptive only* |

The divisors correct for multiple counting: every population edge is seen
by both endpoints (÷2), and every triangle appears as an alter–alter tie
in exactly three egos' local views (÷3). With a census ego sample the
mappings are exact, which the test suite verifies.

**This table is the whole fittable vocabulary.** Only `EgoEdges`,
`EgoNodeMatch`, `EgoTriangle` and `EgoGWDegree` can be fitted; `ergm.ego`'s
`nodefactor`, `nodecov`, `absdiff`, `gwesp`, `mm` and `degree`/`concurrent`
have no ego counterpart yet — a model using them has no Julia spelling (an
undefined name such as `EgoNodeFactor`, not a fit). A custom fittable term
subtypes [`EgoTerm`](@ref) and adds methods to `name`,
`ERGMEgo._ego_contribution` and the `public` hook
[`ERGMEgo._ergm_term`](@ref) (the ERGM.jl term whose statistic it
estimates); without the last it is descriptive only.

`EgoDegree` cannot be used in [`ergm_ego`](@ref) because ERGM.jl has no
degree-count term; including it raises an informative error.

`EgoGWDegree(decay)` accepts any non-negative `Real` (`EgoGWDegree(0.0)` is
statnet's `gwdegree(0, fixed=TRUE)`: each ego with at least one alter
contributes exactly 1, so the statistic is the proportion of non-isolates);
a negative decay is an `ArgumentError`.

For descriptive mixing structure use [`ego_mixing_matrix`](@ref), which
returns the full weighted mixing matrix rather than a scalar.
