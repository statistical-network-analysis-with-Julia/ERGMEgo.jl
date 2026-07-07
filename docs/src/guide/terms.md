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
| [`EgoGWDegree`](@ref) | ``e^\alpha(1-(1-e^{-\alpha})^{d_i})`` | `GWDegree(α)` |
| [`EgoDegree`](@ref) | ``1[d_i = d]`` | *descriptive only* |

The divisors correct for multiple counting: every population edge is seen
by both endpoints (÷2), and every triangle appears as an alter–alter tie
in exactly three egos' local views (÷3). With a census ego sample the
mappings are exact, which the test suite verifies.

`EgoDegree` cannot be used in [`ergm_ego`](@ref) because ERGM.jl has no
degree-count term; including it raises an informative error.

For descriptive mixing structure use [`ego_mixing_matrix`](@ref), which
returns the full weighted mixing matrix rather than a scalar.
