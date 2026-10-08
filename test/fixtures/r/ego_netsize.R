# Golden fixture: statnet `ergm.ego` network-size handling and curved-degree
# fits -- what the census and weighted fixtures do not pin.
#
# Regenerate from the package root (~4 min: 40 ergm.ego fits):
#
#   Rscript test/fixtures/r/ego_netsize.R > test/fixtures/ego_netsize.toml
#
# WHAT IT PINS
#
# (a) THE OFFSET STATISTIC. ergm.ego fits `offset(netsize.adj(edges = 1,
#     transitiveties = -1/3))` whenever the model has an order-3 statistic
#     (triangle), with coefficient -log(ppopsize/popsize). Deterministic:
#     `transitiveties`, and the offset statistic edges - transitiveties/3, on
#     faux.mesa.high and on a frozen 100-vertex network.
#
# (b) A TRIANGLE MODEL UNDER THREE POPULATION SIZES. `egor ~ edges +
#     triangle` on the census of the frozen 100-vertex network (a Bernoulli
#     graph, so the triangle model is not degenerate; on faux.mesa.high R's
#     own fit of edges + nodematch + triangle fails with the density guard),
#     pseudo-population 100, with popsize = 100 (offset 0), popsize = 200
#     (offset log 2 on edges - transitiveties/3) and R's default popsize = 1
#     (per-capita coefficients, offset -log 100). The triangle coefficient
#     moves from 0.09 to 0.68 to -4.2 between them: a port that shifts only
#     the edges coefficient is a different model, and fails here.
#
# (c) A gwdegree FIT UNDER A WEIGHTED DESIGN. `edges + nodematch("Grade") +
#     gwdegree(0.5, fixed = TRUE)` on the weighted sub-design of
#     fauxmesa_ego_weighted.R (egos 1, 4, ..., 205, weights Grade - 6),
#     popsize = ppopsize = 205.
#
# (d) PER-CAPITA COEFFICIENTS DO NOT DEPEND ON THE PSEUDO-POPULATION SIZE.
#     `edges + nodematch("Grade")` on the unweighted sub-design (69 egos) with
#     R's default popsize = 1, at ppopsize = 207 (3 x 69) and 414 (6 x 69).
#
# (e) popsize != ppopsize ON A WEIGHTED DESIGN: popsize = 2050,
#     ppopsize = 500 requested (R constructs 508 vertices; netsize.adj =
#     -log(508/2050)).
#
# All fits are Monte-Carlo on both sides: R's mean and seed-to-seed sd over
# `rep_seeds` are frozen, and the Julia side compares seed means at a
# multiple of the combined Monte-Carlo sd (see [tolerance]).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.ego)
})

seed <- 20261002
rep_seeds <- c(101, 202, 303, 404, 505, 606)

# A fit, or NULL when ergm stops with an error (a triangle model can trip
# ergm's density guard on an unlucky seed; such seeds are counted, not hidden)
fit_once <- function(f, s, popsize, ppopsize) {
  set.seed(s)
  out <- NULL
  invisible(capture.output(
    out <- tryCatch(suppressWarnings(suppressMessages(
      if (is.null(popsize))
        ergm.ego(f, control = control.ergm.ego(ppopsize = ppopsize,
                                               ergm = control.ergm(seed = s)))
      else
        ergm.ego(f, popsize = popsize,
                 control = control.ergm.ego(ppopsize = ppopsize,
                                            ergm = control.ergm(seed = s))))),
      error = function(e) NULL),
    type = "output"))
  out
}

# coefficients and standard errors of the free (non-offset) terms, the
# offset coefficient and the constructed pseudo-population size, over seeds
replicate_fit <- function(f, popsize, ppopsize) {
  rows <- lapply(rep_seeds, function(s) {
    ft <- fit_once(f, s, popsize, ppopsize)
    if (is.null(ft)) return(NULL)
    cf <- coef(ft)
    off <- grep("netsize.adj", names(cf))
    free <- setdiff(seq_along(cf), off)
    list(coef = as.numeric(cf[free]), se = sqrt(diag(vcov(ft)))[free],
         adj = if (length(off)) as.numeric(cf[off]) else 0,
         m = ft$ppopsize, names = names(cf)[free])
  })
  n_failed <- sum(sapply(rows, is.null))
  rows <- rows[!sapply(rows, is.null)]
  stopifnot(length(rows) >= 4)
  cf <- do.call(rbind, lapply(rows, `[[`, "coef"))
  se <- do.call(rbind, lapply(rows, `[[`, "se"))
  stopifnot(length(unique(sapply(rows, `[[`, "m"))) == 1)
  list(mean = colMeans(cf), sd = apply(cf, 2, sd),
       se_mean = colMeans(se), se_sd = apply(se, 2, sd),
       adj = rows[[1]]$adj, m = rows[[1]]$m, names = rows[[1]]$names,
       n_fits = length(rows), n_failed = n_failed)
}

# --- (a) the frozen 100-vertex network and the offset statistic -------------
set.seed(seed)
n <- 100
A <- matrix(0, n, n)
for (i in 1:(n - 1)) for (j in (i + 1):n) if (runif(1) < 0.04) A[i, j] <- A[j, i] <- 1
g <- network(A, directed = FALSE)
el <- as.edgelist(g)
g_stats <- summary(g ~ edges + triangle + transitiveties)
g_offset <- as.numeric(summary(g ~ netsize.adj(edges = 1, transitiveties = -1/3)))

data(faux.mesa.high)
fmh <- faux.mesa.high
fmh_stats <- summary(fmh ~ edges + triangle + transitiveties)
fmh_offset <- as.numeric(summary(fmh ~ netsize.adj(edges = 1, transitiveties = -1/3)))
stopifnot(abs(g_offset - (g_stats[1] - g_stats[3] / 3)) < 1e-9,
          abs(fmh_offset - (fmh_stats[1] - fmh_stats[3] / 3)) < 1e-9)

# --- (b) the triangle model under three population sizes --------------------
ged <- as.egor(g)
ft <- NULL
for (s in rep_seeds) if (is.null(ft)) ft <- fit_once(ged ~ edges + triangle, s, 200, n)
offset_formula <- paste(deparse(ft$ergm.formula, width.cutoff = 500), collapse = " ")
stopifnot(grepl("transitiveties = -0.333", offset_formula))
tri_same <- replicate_fit(ged ~ edges + triangle, n, n)
tri_double <- replicate_fit(ged ~ edges + triangle, 2 * n, n)
tri_percap <- replicate_fit(ged ~ edges + triangle, NULL, n)
stopifnot(abs(tri_same$adj) < 1e-12, abs(tri_double$adj - log(2)) < 1e-12,
          abs(tri_percap$adj + log(n)) < 1e-12)

# --- (c), (d), (e): faux.mesa.high sub-designs ------------------------------
nf <- network.size(fmh)
grade <- fmh %v% "Grade"
ed <- as.egor(fmh)
idx <- seq(1, nf, 3)
sub <- ed[idx, ]
stopifnot(all(sub$ego$.egoID == idx))
subw <- sub
subw$ego$w <- grade[idx] - 6
ego_design(subw) <- list(weights = "w")

gw <- replicate_fit(subw ~ edges + nodematch("Grade") + gwdegree(0.5, fixed = TRUE), nf, nf)
pc207 <- replicate_fit(sub ~ edges + nodematch("Grade"), NULL, 207)
pc414 <- replicate_fit(sub ~ edges + nodematch("Grade"), NULL, 414)
big <- replicate_fit(subw ~ edges + nodematch("Grade"), 2050, 500)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
block <- function(key, r) {
  cat(sprintf("%s_names = [%s]\n", key, strs(r$names)))
  cat(sprintf("%s_mean = [%s]\n", key, num(r$mean)))
  cat(sprintf("%s_sd = [%s]\n", key, num(r$sd)))
  cat(sprintf("%s_se_mean = [%s]\n", key, num(r$se_mean)))
  cat(sprintf("%s_se_sd = [%s]\n", key, num(r$se_sd)))
  cat(sprintf("%s_netsize_adj = %.17g\n", key, r$adj + 0))
  cat(sprintf("%s_ppopsize = %d\n", key, as.integer(r$m)))
  cat(sprintf("%s_n_fits = %d\n", key, r$n_fits))
  cat(sprintf("%s_n_failed = %d\n", key, r$n_failed))
}

cat('name = "ego_netsize"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_ego_version = "%s"\n', as.character(packageVersion("ergm.ego"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/ego_netsize.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "a frozen 100-vertex Bernoulli(0.04) network (edge list below) and ergm::faux.mesa.high"\n')
cat('model = "ergm.ego fits with offset(netsize.adj(edges = 1, transitiveties = -1/3)): edges + triangle under popsize = ppopsize, 2 x ppopsize and the default 1; edges + nodematch(Grade) + gwdegree(0.5, fixed = TRUE) on the weighted sub-design; per-capita edges + nodematch(Grade) at two pseudo-population sizes; popsize = 2050 with ppopsize = 500"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# DETERMINISTIC counts on fixed networks (integers and thirds).\n")
cat("offset_statistic = 1e-9\n")
cat("#\n")
cat("# FITTED COEFFICIENTS -- MONTE-CARLO ON BOTH SIDES. The Julia side compares\n")
cat("# the mean of three seeded fits with R's mean over the `rep_seeds` that\n")
cat("# fitted (*_n_fits; a seed on which ergm stopped with an error is counted in\n")
cat("# *_n_failed), at\n")
cat("#   seed_mean_sds * sqrt(sd_R^2/n_fits + sd_J^2/3)  (sd_J = sd_R: both packages\n")
cat("#   are moment-matching by MCMC from the same targets; measured Julia\n")
cat("#   seed sds are of R's size or smaller),\n")
cat("# floored at coef_floor_se standard errors: the two packages build\n")
cat("# different pseudo-population networks and estimate E[g] by different\n")
cat("# chains, which leaves an O(1/ESS) difference that R's seed spread (as low\n")
cat("# as 0.003 here) does not measure. 0.15 SE is the same floor ERGM.jl's\n")
cat("# MCMLE fixture argues for (0.1 SE there; the ego sandwich SE is itself\n")
cat("# Monte-Carlo, with a seed sd of about 10 %).\n")
cat("# The triangle model's three fits differ from one another by 0.6 to 4.2 in\n")
cat("# the triangle coefficient, 30 to 200 times this tolerance.\n")
cat("seed_mean_sds = 4.0\n")
cat("coef_floor_se = 0.15\n")
cat("#\n")
cat("# STANDARD ERRORS: each package sandwiches the same design covariance with\n")
cat("# its own Monte-Carlo estimate of the information; the three-seed Julia\n")
cat("# mean is held to R's six-seed mean within this relative tolerance (R's own\n")
cat("# seed-to-seed sd of an SE is 5-12 % here; see the *_se_sd rows).\n")
cat("std_errors_rel = 0.25\n")
cat("\n")

cat("[values]\n")
cat("# --- (a) the frozen 100-vertex network and the offset statistic ---------\n")
cat(sprintf("g_n = %d\n", n))
cat(sprintf("g_edge_src = [%s]\n", paste(el[, 1], collapse = ", ")))
cat(sprintf("g_edge_dst = [%s]\n", paste(el[, 2], collapse = ", ")))
cat(sprintf("g_edges = %d\n", as.integer(g_stats[1])))
cat(sprintf("g_triangles = %d\n", as.integer(g_stats[2])))
cat(sprintf("g_transitiveties = %d\n", as.integer(g_stats[3])))
cat(sprintf("g_offset_statistic = %.17g\n", g_offset))
cat(sprintf("fmh_transitiveties = %d\n", as.integer(fmh_stats[3])))
cat(sprintf("fmh_offset_statistic = %.17g\n", fmh_offset))
cat(sprintf('offset_formula = "%s"\n', gsub('"', '\\\\"', offset_formula)))
cat("\n# --- (b) egor ~ edges + triangle on the census of g, ppopsize = 100 -----\n")
cat("# popsize = 100: offset coefficient 0\n")
block("tri_same", tri_same)
cat("# popsize = 200: offset coefficient log 2 on edges - transitiveties/3\n")
block("tri_double", tri_double)
cat("# popsize = 1 (R's default): per-capita coefficients, offset -log 100\n")
block("tri_percap", tri_percap)
cat("\n# --- (c) weighted sub-design, edges + nodematch + gwdegree(0.5) ---------\n")
cat(sprintf("ego_ids = [%s]\n", paste(idx, collapse = ", ")))
cat(sprintf("weights = [%s]\n", paste(grade[idx] - 6, collapse = ", ")))
block("gwdeg", gw)
cat("\n# --- (d) unweighted sub-design, default popsize = 1, two ppopsizes ------\n")
block("percap207", pc207)
block("percap414", pc414)
cat("\n# --- (e) weighted sub-design, popsize = 2050, ppopsize = 500 requested --\n")
block("big", big)
