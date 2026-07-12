# Getting Started

## Building ego data

From `ergm.ego`-style data frames — one frame of egos, one of ego–alter
rows, and (optionally) one of alter–alter ties:

```julia
using ERGMEgo, DataFrames
using ERGM: compute   # generic statistic evaluator shared with ERGM.jl

ego_df   = DataFrame(ego_id = [1, 2], group = ["A", "B"], w = [2.0, 1.0])
alter_df = DataFrame(ego_id = [1, 1, 2], alter_id = [10, 11, 10],
                     group = ["A", "B", "B"])
aatie_df = DataFrame(ego_id = [1], src = [10], dst = [11])

ed = as_egodata(ego_df, alter_df; aatie_df = aatie_df,
                ego_attrs = [:group], alter_attrs = [:group],
                weight_col = :w)

summary_stats(ed)
```

Alter IDs are preserved exactly as given, so alters named by several egos
remain identifiable (used by capture-recapture population estimation).

Or by sampling egos from a complete network. The network must actually
carry the attributes named in `ego_attrs` — fitting a term on an
attribute the network does not have raises an `ArgumentError` (from
ERGM.jl's model validation) instead of silently fitting a zero statistic:

```julia
using Network, Random

Random.seed!(1)
net = network(100; directed = false)
for _ in 1:300
    i, j = rand(1:100), rand(1:100)
    i == j || add_edge!(net, i, j)
end
set_vertex_attribute!(net, :group,
    Dict(v => (isodd(v) ? "A" : "B") for v in 1:100))

ed = simulate_ego_sample(net, 30; ego_attrs = [:group])
```

## Ego statistics

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
result = fit_ergm_ego(ed, [EgoEdges(), EgoNodeMatch(:group)];
                      ppopsize = 200, popsize = 1000)
println(result)

# Model checking: gof returns the shared Network.GOFResult
gof(result)
```

`ergm_ego` is an R-faithful alias for `fit_ergm_ego` (and `fit_ego_ergm`
a legacy one); `ego_gof` is the legacy NamedTuple-returning form of
`gof`.

The reported edges coefficient is on the *population* scale — the
`−log(popsize/ppopsize)` adjustment is already applied (and reported in
`result.netsize_adjustment`).
