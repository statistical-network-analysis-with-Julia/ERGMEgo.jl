# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERGMEgo.jl is a Julia port of the R `ergm.ego` package (StatNet collection) for fitting Exponential Random Graph Models (ERGMs) to egocentrically sampled network data. It enables inference about population-level network properties from ego samples where each sampled "ego" reports their local network (alters and alter-alter ties).

## Development Commands

- **Run tests**: `julia --project -e 'using Pkg; Pkg.test()'`
- **Install dependencies**: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- **Load package locally**: `julia --project -e 'using ERGMEgo'`
- **Build docs**: `julia --project=docs docs/make.jl`

## Architecture

The entire package lives in a single module file: `src/ERGMEgo.jl`. It is organized into these sections (in order):

1. **Data Structures** — `EgoNetwork{T}` (single ego observation with alters, alter ties, attributes, weights) and `EgoData{T}` (collection of ego networks with population size, sampling weights, survey design info). Helper functions: `n_alters`, `alter_degree`, `ego_degree`, `n_alter_ties`, `summary_stats`.
2. **Data Preparation** — `as_egodata(ego_df, alter_df; aatie_df=...)`: ergm.ego-style frames (egos, ego-alter rows, alter-alter ties); alter IDs are preserved (never relabeled) so cross-ego overlap works; weight column looked up string-safely on the EGO frame. `ego_design` attaches ppopsize/weights.
3. **Ego-Specific Terms** — subtype `EgoTerm <: AbstractERGMTerm` (`import ERGM: name, compute, summary_stats`). `compute(term, ed)` returns the design-weighted per-capita statistic; `_ego_contribution(term, ego)` is the per-ego value; `_ergm_term(term)` maps to the ERGM.jl term whose sufficient statistic it estimates (EgoEdges→Edges, EgoNodeMatch→NodeMatch, EgoTriangle→Triangle with /3 for triple counting, EgoGWDegree→GWDegree). `EgoDegree(d)` is descriptive-only (no ERGM.jl degree-count term). `ego_mixing_matrix` is a descriptive function returning (levels, matrix).
4. **Model and Estimation** — `EgoERGMModel`, `EgoERGMResult`, `ergm_ego` (alias `fit_ego_ergm`): design-weighted target statistics scaled to `ppopsize`, pseudo-population network with weight-proportional attribute replication, MCMC moment matching via `ERGM._mcmc_sample`, netsize adjustment `−log(popsize/ppopsize)` on the edges coefficient, and SEs `V(θ̂) = I⁻¹ + I⁻¹Σ_design I⁻¹`. Models must include `EgoEdges()`.
5. **Population Size Estimation** — `estimate_popsize` with Horvitz-Thompson and capture-recapture methods.
6. **Simulation** — `simulate_ego_sample` generates ego samples from a complete `Network`.
7. **Diagnostics** — `ego_gof` for goodness-of-fit checks.

## Key Dependencies

- **ERGM.jl** — Provides `AbstractERGMTerm` base type that all ego terms extend
- **Network.jl** — Network data structure used in simulation (`simulate_ego_sample`)
- **DataFrames.jl** — Used in `as_egodata` for DataFrame ingestion
- **Graphs.jl** — Graph primitives (`nv`, `neighbors`, `has_edge`) used in simulation
- **StatsBase.jl** — Weighted statistics and sampling

## Conventions

- All code resides in a single file (`src/ERGMEgo.jl`); there are no `include` statements.
- Types are parametric on vertex ID type `T` (e.g., `EgoNetwork{T}`, `EgoData{T}`).
- Each ERGM term is a struct subtyping `AbstractERGMTerm` with two required methods: `name(t)::String` and `compute(t, ed::EgoData)::Float64`.
- Weighted statistics use `ed.sampling_weights` paired with `ed.egos` via `zip`.
- Networks are assumed undirected (alter ties divided by 2 in `n_alter_ties`).
- Julia docstrings are used on all exported functions and types.
- Requires Julia >= 1.12.
