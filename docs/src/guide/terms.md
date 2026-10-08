# Ego Terms

Every ego term is a **per-capita statistic**: `compute(term, ed)` returns
the design-weighted mean per-ego contribution, and `ppopsize × compute`
is the target sufficient statistic used in fitting. Each fittable term
maps to the ERGM.jl term whose sufficient statistic it estimates.

| Term | Per-ego contribution | ERGM term |
|------|---------------------|-----------|
| [`EgoEdges`](@ref) | ``d_i / 2`` | `Edges()` |
| [`EgoNodeMatch`](@ref) | matching alters ``/ 2`` | `NodeMatch(attr)` |
| [`EgoNodeFactor`](@ref) | ``(1[x_i = l]\,d_i + \#\{\text{alters with } x = l\}) / 2``, one statistic per level ``l`` | `NodeFactor(attr; level=l)` (R's `nodefactor(attr)`, labelled `nodefactor.<attr>.<l>`) |
| [`EgoNodeCov`](@ref) | ``(x_i d_i + \sum_{\text{alters}} x) / 2`` | `NodeCov(attr)` |
| [`EgoAbsDiff`](@ref) | ``\sum_{\text{alters}} \lvert x_i - x \rvert^{p} / 2`` | `AbsDiff(attr; pow=p)` |
| [`EgoDegree`](@ref) | ``1[d_i = d]`` | `Degree(d)` (R's `degree(d)`, labelled `degree<d>`) |
| [`EgoTriangle`](@ref) | alter–alter ties ``/ 3`` | `Triangle()` |
| [`EgoGWDegree`](@ref) | ``e^\alpha(1-(1-e^{-\alpha})^{d_i})``, ``\alpha \ge 0`` | `GWDegree(α)` (labelled `gwdeg.fixed.α`, R's `gwdegree(α, fixed=TRUE)`) |
| [`EgoGWESP`](@ref) | ``\sum_{\text{alters } a} e^\alpha(1-(1-e^{-\alpha})^{s_a}) / 2``, ``s_a`` the alter's degree among the alter–alter ties | `GWESP(α)` (labelled `gwesp.fixed.α`, R's `gwesp(α, fixed=TRUE)`) |
| [`EgoESP`](@ref) | ``\#\{\text{alters } a : s_a = k\} / 2`` | `ESP(k)` (R's `esp(k)`, labelled `esp<k>`) |
| [`EgoMM`](@ref) | alters ``a`` with ``\{x_i, x_a\} = \{l_1, l_2\}``, ``/ 2``, one statistic per cell | `NodeMix(attr, l1, l2)` (R's `mm(attr)`, labelled `mm[<attr>=<l1>,<attr>=<l2>]`) |
| [`EgoConcurrent`](@ref) | ``1[d_i \ge 2]`` | `Concurrent()` (R's `concurrent`) |

Each term's label — `name(term)`, the rows of `coeftable(fit)` and
`coefnames(fit)` — is `ergm.ego`'s, the label R ergm gives the term it
estimates: `edges`, `nodematch.Grade`, `nodefactor.Race.Hisp`, `degree0`,
`absdiff.Grade`, `gwdeg.fixed.0.5`, `gwesp.fixed.0`, `esp1`,
`mm[Race=Black,Race=Hisp]`, `concurrent`, …

The divisors correct for multiple counting: every population edge is seen
by both endpoints (÷2), and every triangle appears as an alter–alter tie
in exactly three egos' local views (÷3); a degree is a property of one
vertex, so it is not divided. Each per-ego value is derived from the
term's definition and agrees with `ergm.ego`'s; the test suite pins them three ways: under a
census they reduce exactly to ERGM.jl's statistics of the network; on
`faux.mesa.high`, under the census and under a weighted sub-design, the
targets and their design covariance equal `ergm.ego`'s at 1e-9; and the
`ergm.ego` example model (without its `gwesp` term), a `gwesp` model on the
census and a per-capita `gwesp` model fit to `ergm.ego`'s coefficients
within tolerances derived from `ergm.ego`'s own seed spread
(`test/fixtures/ego_terms.toml`); so do `ergm.ego`'s help-page model with
its `gwesp` term and models with `esp`, `mm` and `concurrent`
(`test/fixtures/ego_mixing_esp.toml`).

**Levels of `EgoNodeFactor`** follow `ergm.ego`: the sorted distinct values
of the **egos**, with the first dropped as the reference category (R's
default `levels = -1`). `base=0` keeps every level (R's `levels = TRUE`),
`base=[1, 2]` drops the first two (R's `levels = -(1:2)`), and `levels=[…]`
names the included values. The term expands into one statistic per level
when the model is fitted, so `coef(fit)` has one row per level. An alter
whose value no ego has counts for no level, as in R.

```julia
using ERGMEgo, NetworkCore, Random
net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs=[:Grade, :Race, :Sex], rng=Xoshiro(1))
ego_target_stats([EgoNodeFactor(:Race)], ed, 205)   # [178.0, 156.0, 1.0, 45.0] — Hisp, NatAm, Other, White
ego_target_stats([EgoDegree(0), EgoNodeCov(:Grade)], ed, 205)   # [57.0, 3491.0] — R's degree0, nodecov.Grade
```

**Levels of `EgoMM`** follow `ergm.ego`'s `mm(attr)`: the sorted values
found at either end of a reported ego–alter tie — a level that only alters
carry is included, the level of an ego with no alters is not — and the
cells are the unordered pairs `l1 ≤ l2` in R's order, the first dropped
(R's `levels2 = -1`). R's two-attribute form `mm(A ~ B)`, the margins
`mm(A ~ .)` and the `levels=`/`levels2=` selections are refused. A cell
whose level no ego of the pseudo-population carries cannot be matched by
any simulated network, so a fit with a positive target there is refused
(with a zero target the cell is at its bound and fixed at `-Inf`).
`ergm.ego` 1.1.4's `mm` stops with an error on weighted data; `EgoMM`
weights its per-ego values like every other term.

```julia
ego_target_stats([EgoMM(:Sex), EgoESP(1), EgoConcurrent()], ed, 205)   # [71.0, 50.0, 70.0, 97.0] — R's mm[Sex=F,Sex=M], mm[Sex=M,Sex=M], esp1, concurrent
```

`ergm.ego`'s `degree(1:3)` is `EgoDegree.(1:3)` and its `esp(0:3)` is
`EgoESP.(0:3)`; `degree`'s `by=` and
`homophily=` forms, and the fallback of `nodefactor`/`nodecov` for data
without alter attributes, are not ported (every attribute term needs the
attribute on the ego and on its alters).

**What is not fittable.** The curved `gwesp(fixed=FALSE)`, whose decay
is estimated, is refused (`EgoGWESP(decay; fixed=false)`): ERGM.jl's curved
MCMLE fits the statistics of a whole observed network, and has no form that
matches target statistics, which the egocentric fit needs. `ergm.ego`'s
other terms (`nodemix`, `absdiffcat`, `degrange`, `concurrentties`,
`degree1.5`, `transitiveties`, `cyclicalties`, `meandeg`) have no ego
counterpart yet. A custom fittable term subtypes
[`EgoTerm`](@ref) and adds methods to `name`, `ERGMEgo.ego_contribution`
and the `public` hook [`ERGMEgo.ergm_term`](@ref) (the ERGM.jl term whose
statistic it estimates); without the last it is descriptive only.

`EgoTriangle`, `EgoGWESP` and `EgoESP` need the ties among alters: data built
without them (`as_egodata` with no `aatie_df`) is refused. Seen from an
ego, the shared partners of its tie to alter `a` are its other alters tied
to `a`, so `a`'s degree among the alter–alter ties is that tie's
shared-partner count. `EgoGWESP` weights each of these counts as the
gwesp definition does and halves the sum, since every tie is seen from both
of its ends; it was derived from that definition and checked against
`ergm.ego`'s outputs. `EgoESP(k)` counts the alters whose count is exactly
`k`, halved for the same reason. Unlike `ergm.ego`'s `gwesp`, which leaves out ties with
more than `cutoff = 30` shared partners, `EgoGWESP` has no cutoff, like `ergm`'s
`gwesp` and ERGM.jl's `GWESP`. All three are order-3 statistics, and they change the
network-size offset, as in `ergm.ego`: when the pseudo-population size
differs from the population size the offset statistic is
`edges − transitiveties/3`, not `edges`, so the triangle coefficient depends
on `popsize` (see [The network-size offset](@ref)); so does a gwesp
coefficient (and an esp one). `ergm.ego`'s help-page model — its example
model plus `gwesp(0, fixed=TRUE)` — fits at the defaults in about 5–15
seconds to `ergm.ego`'s coefficients (`ergm.ego` itself takes 24–75 seconds
on six of eight seeds and runs past 5 minutes on two): the moment matching samples as R
ergm 4 does underneath `ergm.ego` (SPDyad proposals, ESS-adaptive sampling
on a chain continued between iterations; see
[Pseudo-population and moment matching](@ref)). Triangle models are
prone to degeneracy in any ERGM; on `faux.mesa.high`, `ergm.ego` itself
cannot fit `edges + nodematch + triangle`.

`EgoGWDegree(decay)` accepts any non-negative `Real` (`EgoGWDegree(0.0)` is
statnet's `gwdegree(0, fixed=TRUE)`: each ego with at least one alter
contributes exactly 1, so the statistic is the proportion of non-isolates);
a negative decay is an `ArgumentError`.

For descriptive mixing structure use [`ego_mixing_matrix`](@ref), which
returns the full weighted mixing matrix rather than a scalar.
