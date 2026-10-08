# Getting Started

Start from ego and alter tables, compute target statistics, and fit a pseudo-population model. A simulated sample and a bundled census then show what is observed locally and what the model must infer about the complete network.

!!! note "Before you begin"

    The implemented ego models use undirected, one-mode networks. Sampling uncertainty treats egos as independent with the supplied case weights; arbitrary clustered or stratified survey designs are not represented. The result is a moment fit, with MCMC diagnostics, rather than a fitted full-network likelihood.

## Installation

```@raw html
<p>Use Julia <strong>1.12 or newer</strong> and the <a href="/getting-started/">shared workspace installation guide</a>. These development packages are not yet registered; the guide prepares the required sibling checkouts and a Julia environment for the examples.</p>
```

Run the blocks below in order in that environment. They build on variables from earlier steps; stochastic examples use seeded random number generators where shown.

## Building ego data

From `ergm.ego`-style data frames — one frame of egos, one of ego–alter
rows, and (optionally) one of alter–alter ties:

```julia
using ERGMEgo, DataFrames

ego_df   = DataFrame(ego_id = [1, 2], group = ["A", "B"], w = [2.0, 1.0])
alter_df = DataFrame(ego_id = [1, 1, 2], alter_id = [10, 11, 10],
                     group = ["A", "B", "B"])
aatie_df = DataFrame(ego_id = [1], src = [10], dst = [11])

ed = as_egodata(ego_df, alter_df; aatie_df = aatie_df,
                ego_attrs = [:group], alter_attrs = [:group],
                weight_col = :w)

summary_stats(ed)
```

Alter IDs are preserved exactly as given, so an alter who is also a sampled
ego remains identifiable (used by capture-recapture population estimation).
Leave `aatie_df` out only when the ties among alters were **not collected**:
the data then records that (`ed.design[:alter_ties_observed] == false`),
and `EgoTriangle` refuses it instead of reading "no triangles". Pass an
empty frame when they were collected and there are none.

Every column a keyword names must exist in its frame: `as_egodata(…;
ego_attrs = [:nope])` is an `ArgumentError` naming `:nope`, `ego_df` and
the columns `ego_df` does have.

Or by sampling egos from a complete network. `simulate_ego_sample` is the
`Network → EgoData` conversion adapter and honours the ecosystem's
conversion and missing-data contracts: the network must be **undirected**
and **one-mode**, its masked (unobserved) dyads are refused unless you pass
`missing = :face`, and it must actually carry — for every vertex — the
attributes named in `ego_attrs` (an unknown or partial attribute is an
`ArgumentError` listing the attributes the network has, instead of a
silent `missing` fill). `report = true` returns a `ConversionReport` naming
what an `EgoData` could not hold:

```julia
using NetworkCore, Random

rng = Xoshiro(1)
net = network(100; directed = false)
for _ in 1:300
    i, j = rand(rng, 1:100), rand(rng, 1:100)
    i == j || add_edge!(net, i, j)
end
set_vertex_attribute!(net, :group,
    Dict(v => (isodd(v) ? "A" : "B") for v in 1:100))

ed = simulate_ego_sample(net, 30; ego_attrs = [:group], rng = rng)
ed, rep = simulate_ego_sample(net, 30; ego_attrs = [:group], rng = rng, report = true)
dropped_fields(rep)        # [:edges] — the ties between the 70 unsampled vertices
```

A homophily term on an attribute the ego data does not carry —
`EgoNodeMatch(:nope)` — is an `ArgumentError` from `compute` and from
`fit_ergm_ego` (raised at the targets, before any MCMC) naming the
attribute, the ego and whether it is the ego or the alters that lack it;
the pre-0.2 code scored such an ego as 0.

## Ego statistics

`compute` and `name` are the shared NetworkCore.jl statistic generics
(re-exported, as ERGM.jl re-exports them):

```julia
compute(EgoEdges(), ed)          # weighted mean degree / 2 (per capita)
compute(EgoNodeMatch(:group), ed)
compute(EgoTriangle(), ed)
ego_target_stats([EgoEdges()], ed, 500)   # scaled to a network of size 500
```

With a census ego sample the scaling is exact:
`n * compute(EgoEdges(), ed) == edges(net)`.

## Fitting

```julia
# `ed` is the 30-ego sample drawn from the 100-vertex network above, so
# popsize = 100 comes from the sample (`ed.population_size`). ppopsize = 60
# replicates each ego twice, and the offset netsize.adj is −log(60/100)
result = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:group)];
                      ppopsize = 60, rng = Xoshiro(3))
result.model.popsize, result.netsize_adjustment      # (100, 0.511)
println(result)

# Model checking: gof returns the shared NetworkCore.GOFResult; every draw
# flows through rng, so the same rng state gives the same envelope
gof(result; n_sim = 50, rng = Xoshiro(2))
```

`ergm_ego` is the same function as `fit_ergm_ego`, under `ergm.ego`'s
name; `ego_gof` is the NamedTuple-returning form of `gof`.

The reported coefficients are those of a population of `popsize` members:
the pseudo-population model carries `ergm.ego`'s offset `netsize.adj` with
coefficient `−log(ppopsize/popsize)`, reported in
`result.netsize_adjustment`. When the population size is unknown — neither
`popsize` nor `ed.population_size` — it is taken as 1, as in R, and the
coefficients are **per capita**: the edges coefficient does not depend on
`ppopsize`, and for a population of `N` it is `coef − log(N)`. The
pseudo-population is built by replicating each ego `round(ppopsize ·
wᵢ/Σw)` times, so its realised size can differ from the request (an `@info`
says so; `ppopsize = 50` with 30 equal-weight egos gives 60). Without
`ppopsize` the default is `ergm.ego`'s: the population size when it is
known (up to 1000), and the number of egos when it is not.

The keywords follow the ecosystem vocabulary: `maxiter` (60) caps the
iterations, `n_samples`/`burnin`/`interval` override the
dyad-scaled MCMC budget, `termination`/`conv_precision`/`conv_confidence`
set the stopping rule (R ergm's confidence test by default), `proposal`
the sampler's proposal (R ergm's SPDyad by default), `effective_size` the
effective sample size each sample is grown to (64, R's
`MCMLE.effectiveSize`; `nothing` for fixed-size samples), `se` the
standard-error method, and `rng` is the
source of every draw. `gof` adds `n_sim` and `n_chains`. `missing=` belongs
to `simulate_ego_sample`, the only routine here that can meet a masked
dyad. A fit that does not converge warns, sets `result.converged == false`
and prints the caveat — treat the warning as what it says.

## A census of faux.mesa.high

The bundled teaching dataset (statnet's `faux.mesa.high`, 205 students,
203 friendship ties, `:Grade`/`:Race`/`:Sex`) as an egocentric **census**
— every student an ego — reproduces the plain ERGM's answer and `ergm.ego`'s
fit of `egor ~ edges + nodematch("Grade")`, which the package's golden
fixture pins against R:

```julia
using ERGMEgo, NetworkCore, Random

net = load_dataset(:faux_mesa_high)
ed = simulate_ego_sample(net, 205; ego_attrs = [:Grade], rng = Xoshiro(1))
ego_target_stats([EgoEdges(), EgoNodeMatch(:Grade)], ed, 205)   # [203.0, 163.0] — the network's own

fit = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:Grade)]; rng = Xoshiro(1))
fit.converged                    # true, in one or two iterations
coef(fit)                        # ≈ [-6.03, 2.83]  (ergm.ego: netsize.adj + edges = -6.020, nodematch.Grade = 2.822)
stderror(fit)                    # ≈ [0.22, 0.25]  (se = :superpopulation; with se = :design ≈ [0.16, 0.18], ergm.ego: 0.178, 0.196)
fit.mcmc_convergence.n_eff       # > 1000 on the final sample
gof(fit; n_sim = 20, rng = Xoshiro(2), n_chains = 2)
```

Take a 30-ego sample instead of the census and the population size is what
`popsize` is for: `simulate_ego_sample(net, 30; …)` records
`population_size = 205`, and `fit_ergm_ego(ed30, …; ppopsize = 90)` reports
the coefficients of the 205-actor population through the offset
`netsize_adjustment = −log(90/205)`. Pass `popsize = 1` for R's default,
per-capita output.

`ergm.ego`'s standard errors (`se = :design`) treat the egos as
independent; at a census they are about `1/√2` of the sampling standard
deviation of the estimate. Because the population size is known here, the
default is `se = :superpopulation`, which corrects that — see
[Population Inference](guide/inference.md).
