# Golden fixture: statnet `ergm.ego`'s attribute, degree and shared-partner
# terms -- nodefactor, nodecov, absdiff, degree and gwesp -- its coefficient
# labels, its default pseudo-population size when the population size is
# unknown, and its drop of a statistic whose target sits at the bound of its
# attainable range.
#
# Regenerate from the package root (~8 min: 40 ergm.ego fits, each stopped
# after 5 minutes at most):
#
#   Rscript test/fixtures/r/ego_terms.R > test/fixtures/ego_terms.toml
#
# WHAT IT PINS
#
# (a) THE TARGETS, deterministic (1e-9): summary(egor ~ ..., scaleto = 205)
#     of edges + nodefactor("Race") + nodefactor("Sex") + nodecov("Grade") +
#     absdiff("Grade") + absdiff("Grade", pow = 2) + degree(0:3) +
#     nodefactor("Grade", levels = -(1:2)) + nodefactor("Grade", levels = TRUE)
#     + gwesp(0, fixed = TRUE) + gwesp(0.5, fixed = TRUE) on faux.mesa.high
#       * under a CENSUS (every actor an ego), where they must equal the
#         network's own statistics (asserted here with stopifnot), and
#       * under the WEIGHTED sub-design of fauxmesa_ego_weighted.R (egos 1,
#         4, ..., 205, case weights Grade - 6), with ergm.ego's design
#         covariance of the targets (attr(., "var")).
#     The labels R prints are frozen too: nodefactor drops the first sorted
#     level of the EGOS' values by default.
#
# (b) THE ergm.ego EXAMPLE MODEL without its gwesp term: edges + degree(0:3)
#     + nodefactor("Race") + nodematch("Race") + nodefactor("Sex") +
#     nodematch("Sex") + absdiff("Grade") on the census, popsize = 205
#     (ppopsize = 205, ergm.ego's default). Monte-Carlo: R's mean and sd over
#     the `rep_seeds` that fitted are frozen; a seed on which ergm.ego errors
#     or runs past `time_limit` is counted (*_n_failed, of which *_n_timeout
#     ran out of time).
#
# (c) R'S DEFAULT PSEUDO-POPULATION SIZE WITH THE POPULATION SIZE UNKNOWN:
#     edges + nodefactor("Sex") + absdiff("Grade") on the UNWEIGHTED 69-ego
#     sub-design, popsize and ppopsize left at ergm.ego's defaults (popsize =
#     1, ppopsize = "auto" = the number of egos). The constructed size and
#     netsize.adj are frozen, and the per-capita coefficients over the seeds.
#
# (d) gwesp ON THE CENSUS: edges + nodematch("Grade") + gwesp(0, fixed =
#     TRUE) -- the help page's gwesp term -- popsize = 205. (The help-page
#     model itself, the example model of (b) plus gwesp(0), is block (b) of
#     ego_mixing_esp.R: ergm.ego runs past 5 minutes on some of its seeds.)
#
# (e) gwesp UNDER THE NETWORK-SIZE OFFSET: edges + gwesp(0.5, fixed = TRUE)
#     on the unweighted 69-ego sub-design at ergm.ego's defaults (per
#     capita). gwesp is an order-3 statistic, so ergm.ego's offset carries
#     transitiveties = -1/3.
#
# (f) A TARGET AT ITS BOUND: edges + degree(0) + nodematch("Grade") on the
#     census of faux.mesa.high without its 57 isolates (148 egos, popsize
#     148). No ego has degree 0, so the degree0 target is at its smallest
#     attainable value: ergm.ego (through ergm's target.stats check) fixes
#     its coefficient at -Inf and fits the rest. The -Inf is asserted on
#     every seed and frozen.
#
# TOLERANCES ARE DERIVED HERE FROM R'S OWN SEED SPREAD (the [tolerance]
# block states each rule); each block records, per coefficient, R's sd over
# the seeds of the estimate, of its standard error and of the design
# component of that standard error (vcov(fit, sources = "model")).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.ego)
})

seed <- 20261006
rep_seeds <- c(101, 202, 303, 404, 505, 606, 707, 808)

data(faux.mesa.high)
fmh <- faux.mesa.high
n <- network.size(fmh)
grade <- fmh %v% "Grade"
ed <- as.egor(fmh)
idx <- seq(1, n, 3)
sub <- ed[idx, ]
stopifnot(all(sub$ego$.egoID == idx))
subw <- sub
subw$ego$w <- grade[idx] - 6
ego_design(subw) <- list(weights = "w")

# --- (a) targets ------------------------------------------------------------
rhs <- ~ edges + nodefactor("Race") + nodefactor("Sex") + nodecov("Grade") +
  absdiff("Grade") + absdiff("Grade", pow = 2) + degree(0:3) +
  nodefactor("Grade", levels = -(1:2)) + nodefactor("Grade", levels = TRUE) +
  gwesp(0, fixed = TRUE) + gwesp(0.5, fixed = TRUE)
census <- summary(statnet.common::nonsimp_update.formula(rhs, ed ~ .), scaleto = n)
net_stats <- summary(statnet.common::nonsimp_update.formula(rhs, fmh ~ .))
stopifnot(max(abs(as.numeric(census) - as.numeric(net_stats))) < 1e-9,
          identical(names(census), names(net_stats)))
weighted <- summary(statnet.common::nonsimp_update.formula(rhs, subw ~ .), scaleto = n)
stopifnot(identical(names(weighted), names(census)))
wcov <- attr(weighted, "var")

# --- fits -------------------------------------------------------------------
# One seeded fit, run in a forked child so that it can be stopped: on some
# seeds ergm's MCMLE keeps enlarging its sample under the confidence rule and
# runs for hours (seed 202 of the example model: 10 iterations, then an
# eleventh that had not finished after 20 minutes). A fit that errors, or
# that has not finished within `time_limit` seconds, returns NULL and is
# COUNTED (*_n_failed, *_n_timeout), never hidden.
time_limit <- 300
fit_once <- function(f, s, popsize) {
  t0 <- Sys.time()
  job <- parallel::mcparallel({
    set.seed(s)
    out <- NULL
    invisible(capture.output(
      out <- tryCatch(suppressWarnings(suppressMessages(
        if (is.null(popsize))
          ergm.ego(f, control = control.ergm.ego(ergm = control.ergm(seed = s)))
        else
          ergm.ego(f, popsize = popsize,
                   control = control.ergm.ego(ergm = control.ergm(seed = s))))),
        error = function(e) NULL),
      type = "output"))
    if (is.null(out)) NULL else {
      cf <- coef(out)
      off <- grep("netsize.adj", names(cf))
      free <- setdiff(seq_along(cf), off)
      list(coef = as.numeric(cf[free]), se = sqrt(diag(vcov(out)))[free],
           se_model = sqrt(diag(vcov(out, sources = "model")))[free],
           adj = if (length(off)) as.numeric(cf[off]) else 0,
           m = network.size(out$network), names = names(cf)[free])
    }
  }, silent = TRUE)
  res <- parallel::mccollect(job, wait = FALSE, timeout = time_limit)
  timed_out <- is.null(res)
  if (timed_out) {
    tools::pskill(job$pid)
    parallel::mccollect(job, wait = TRUE)
  }
  row <- if (timed_out) NULL else res[[1]]
  if (inherits(row, "try-error")) row <- NULL
  message(sprintf("fit %s seed %d: %.0f s%s", deparse(f[[3]])[1], s,
                  as.numeric(Sys.time() - t0, units = "secs"),
                  if (timed_out) " (stopped at the time limit)" else if (is.null(row)) " (error)" else ""))
  list(row = row, timed_out = timed_out)
}

replicate_fit <- function(f, popsize) {
  runs <- lapply(rep_seeds, function(s) fit_once(f, s, popsize))
  rows <- lapply(runs, `[[`, "row")
  n_timeout <- sum(sapply(runs, `[[`, "timed_out"))
  n_failed <- sum(sapply(rows, is.null))
  rows <- rows[!sapply(rows, is.null)]
  stopifnot(length(rows) >= 4)
  cf <- do.call(rbind, lapply(rows, `[[`, "coef"))
  se <- do.call(rbind, lapply(rows, `[[`, "se"))
  sem <- do.call(rbind, lapply(rows, `[[`, "se_model"))
  stopifnot(length(unique(sapply(rows, `[[`, "m"))) == 1)
  # a coefficient fixed at -Inf/+Inf must be so on every seed (its sd is
  # then reported as 0, its standard error as R prints it)
  for (k in seq_len(ncol(cf))) stopifnot(all(is.finite(cf[, k])) || length(unique(cf[, k])) == 1)
  sdf <- function(c) if (all(is.finite(c))) sd(c) else 0
  list(mean = colMeans(cf), sd = apply(cf, 2, sdf),
       se_mean = colMeans(se), se_sd = apply(se, 2, sd),
       se_model_mean = colMeans(sem), se_model_sd = apply(sem, 2, sd),
       adj = rows[[1]]$adj, m = rows[[1]]$m, names = rows[[1]]$names,
       n_fits = length(rows), n_failed = n_failed, n_timeout = n_timeout)
}

# (b) the example model on the census, popsize = 205
vig <- replicate_fit(ed ~ edges + degree(0:3) + nodefactor("Race") + nodematch("Race") +
                       nodefactor("Sex") + nodematch("Sex") + absdiff("Grade"), n)
stopifnot(vig$m == n, abs(vig$adj) < 1e-12)

# (c) R's default pseudo-population size, popsize unknown
pc <- replicate_fit(sub ~ edges + nodefactor("Sex") + absdiff("Grade"), NULL)
stopifnot(pc$m == length(idx), abs(pc$adj + log(length(idx))) < 1e-12)

# (d) gwesp(0) on the census
gw <- replicate_fit(ed ~ edges + nodematch("Grade") + gwesp(0, fixed = TRUE), n)
stopifnot(gw$m == n, abs(gw$adj) < 1e-12)

# (e) gwesp under the network-size offset (per capita, 69 egos)
gwpc <- replicate_fit(sub ~ edges + gwesp(0.5, fixed = TRUE), NULL)
stopifnot(gwpc$m == length(idx), abs(gwpc$adj + log(length(idx))) < 1e-12)

# (f) a target at its bound: degree0 on the census without isolates
deg <- sna::degree(fmh, gmode = "graph")
noniso <- fmh
delete.vertices(noniso, which(deg == 0))
edn <- as.egor(noniso)
n_noniso <- network.size(noniso)
bnd <- replicate_fit(edn ~ edges + degree(0) + nodematch("Grade"), n_noniso)
stopifnot(bnd$m == n_noniso, bnd$n_failed == 0)

# TOML spellings of the non-finite values (a dropped coefficient is -Inf)
num <- function(x) paste(ifelse(is.na(x), "nan", ifelse(x == Inf, "inf",
                         ifelse(x == -Inf, "-inf", sprintf("%.17g", x)))), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
mat <- function(M) paste(apply(M, 1, function(r) paste0("[", num(r), "]")), collapse = ", ")
block <- function(key, r) {
  cat(sprintf("%s_names = [%s]\n", key, strs(r$names)))
  cat(sprintf("%s_mean = [%s]\n", key, num(r$mean)))
  cat(sprintf("%s_sd = [%s]\n", key, num(r$sd)))
  cat(sprintf("%s_se_mean = [%s]\n", key, num(r$se_mean)))
  cat(sprintf("%s_se_sd = [%s]\n", key, num(r$se_sd)))
  cat(sprintf("%s_se_model_mean = [%s]\n", key, num(r$se_model_mean)))
  cat(sprintf("%s_se_model_sd = [%s]\n", key, num(r$se_model_sd)))
  cat(sprintf("%s_netsize_adj = %.17g\n", key, r$adj + 0))
  cat(sprintf("%s_ppopsize = %d\n", key, r$m))
  cat(sprintf("%s_n_fits = %d\n", key, r$n_fits))
  cat(sprintf("%s_n_failed = %d\n", key, r$n_failed))
  cat(sprintf("%s_n_timeout = %d\n", key, r$n_timeout))
}

cat('name = "ego_terms"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_ego_version = "%s"\n', as.character(packageVersion("ergm.ego"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/ego_terms.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::faux.mesa.high: 205 students, 203 undirected friendship ties; attributes Grade (7-12), Race, Sex"\n')
cat('model = "targets of edges + nodefactor(Race) + nodefactor(Sex) + nodecov(Grade) + absdiff(Grade) + absdiff(Grade, pow = 2) + degree(0:3) + nodefactor(Grade, levels = -(1:2)) + nodefactor(Grade, levels = TRUE) + gwesp(0, fixed = TRUE) + gwesp(0.5, fixed = TRUE) (census and weighted sub-design); fits of the ergm.ego example model without gwesp and of edges + nodematch(Grade) + gwesp(0) on the census (popsize 205), of edges + nodefactor(Sex) + absdiff(Grade) and edges + gwesp(0.5) on the unweighted 69-ego sub-design at ergm.ego default popsize and ppopsize, and of edges + degree(0) + nodematch(Grade) on the census without isolates (degree0 at its bound)"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# TARGETS: deterministic weighted means of per-ego contributions, scaled\n")
cat("# to 205; 1e-9 is agreement to the floor of two summation orders.\n")
cat("targets = 1e-9\n")
cat("# DESIGN COVARIANCE of the weighted targets: 1e-9 RELATIVE TO ITS LARGEST\n")
cat("# ENTRY (the nodecov variance, ~7e4; the smallest entries are ~1e-2), i.e.\n")
cat("# max|S_J - S_R| <= design_cov_rel * max|S_R|.\n")
cat("design_cov_rel = 1e-9\n")
cat("#\n")
cat("# EVERYTHING FITTED IS MONTE-CARLO ON BOTH SIDES, and every band below is\n")
cat("# built from R's own seed-to-seed spread, recorded per coefficient in each\n")
cat("# block (n_R = *_n_fits seeds; n_J = the Julia seeds of the test):\n")
cat("#\n")
cat("# COEFFICIENTS: |mean_J - mean_R| <= seed_mean_sds * sd_R * sqrt(1/n_R + 1/n_J)\n")
cat("#   + seed_mean_sds * sd_R * n_failed / (n_fits + n_failed).\n")
cat("# The first term is the sd of the difference of two seed means when one\n")
cat("# Julia fit is no noisier than one R fit; the test ASSERTS that premise\n")
cat("# (its seeds differ pairwise by at most seed_mean_sds * sqrt(2) * sd_R).\n")
cat("# The second term is what a seed that failed or timed out can have moved\n")
cat("# R's mean, had its estimate lain anywhere inside R's seed_mean_sds band:\n")
cat("# the failed seeds are counted, never treated as if they had not been run.\n")
cat("# A coefficient fixed at -Inf/+Inf is compared exactly.\n")
cat("seed_mean_sds = 4.0\n")
cat("#\n")
cat("# STANDARD ERRORS: ergm.ego's standard error is the design component\n")
cat("# (vcov(fit, sources = \"model\"), *_se_model_*) plus its MCMC-estimation\n")
cat("# component. ERGMEgo's se=:design has the same design component and a\n")
cat("# smaller estimation component (a longer final sample), so its standard\n")
cat("# error lies between R's design component and R's total. Held to that\n")
cat("# interval, widened at each end by the seed-mean band above applied to\n")
cat("# *_se_model_sd and *_se_sd respectively; the Julia design component\n")
cat("# (sqrt(diag(vcov_design))) is held to R's design component by the same\n")
cat("# band.\n")
cat("\n")

cat("[values]\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ego_ids = [%s]\n", paste(idx, collapse = ", ")))
cat(sprintf("weights = [%s]\n", paste(grade[idx] - 6, collapse = ", ")))
cat("\n# --- (a) targets, scaled to 205 ------------------------------------------\n")
cat(sprintf("term_names = [%s]\n", strs(names(census))))
cat(sprintf("census_targets = [%s]\n", num(as.numeric(census))))
cat(sprintf("weighted_targets = [%s]\n", num(as.numeric(weighted))))
cat(sprintf("weighted_design_cov = [%s]\n", mat(wcov)))
cat("\n# --- (b) the example model on the census, popsize 205 --------------------\n")
block("example", vig)
cat("\n# --- (c) R's default ppopsize, popsize unknown, 69 unweighted egos -------\n")
block("default_ppop", pc)
cat("\n# --- (d) edges + nodematch(Grade) + gwesp(0), census, popsize 205 --------\n")
block("gwesp_census", gw)
cat("\n# --- (e) edges + gwesp(0.5), popsize unknown, 69 unweighted egos ---------\n")
block("gwesp_percap", gwpc)
cat("\n# --- (f) degree0 at its bound: the census without its isolates ----------\n")
cat(sprintf("noniso_ego_ids = [%s]\n", paste(which(deg > 0), collapse = ", ")))
block("boundary", bnd)
