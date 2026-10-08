# Golden fixture: statnet `ergm.ego`'s mixing-matrix, edgewise-shared-partner
# and concurrency terms -- mm, esp and concurrent -- the help-page model with
# its gwesp term, and the drop of a target at the TOP of its attainable range.
#
# Regenerate from the package root (~15-40 min: 40 ergm.ego fits, each
# stopped after 5 minutes at most; the help-page model is the slow block):
#
#   Rscript test/fixtures/r/ego_mixing_esp.R > test/fixtures/ego_mixing_esp.toml
#
# Only R's OUTPUTS are recorded here (summary(), coef(), vcov()); the ERGMEgo
# terms these numbers pin are written from the terms' published definitions.
#
# WHAT IT PINS
#
# (a) THE TARGETS, deterministic (1e-9): summary(egor ~ esp(0:3) +
#     mm("Race") + mm("Sex") + concurrent, scaleto = 205) on faux.mesa.high
#       * under a CENSUS (every actor an ego), where they must equal the
#         network's own statistics (asserted here with stopifnot);
#       * under the WEIGHTED sub-design of fauxmesa_ego_weighted.R (egos 1,
#         4, ..., 205, case weights Grade - 6), with ergm.ego's design
#         covariance of the targets (attr(., "var")) -- esp and concurrent
#         only: ergm.ego 1.1.4's mm stops with an error on a weighted design
#         (its message is recorded);
#       * mm's LEVELS, which are the levels found at either end of a reported
#         ego-alter tie: on the unweighted sub-design (egos 1, 4, ..., 205,
#         where the one "Other" ego reports no alter and no alter is "Other",
#         so there is no Other row), on the egos who are not Black (Black
#         occurs among the alters only: its rows are there), and on the egos
#         without the "Other" student's one friend (that "Other" ego reports
#         an alter but no alter is "Other": its rows are there). The first
#         cell is dropped (levels2 = -1).
#     The labels R prints are frozen too (esp0, mm[Race=Black,Race=Hisp],
#     concurrent).
#
# (b) THE ergm.ego HELP-PAGE MODEL, gwesp(0, fixed = TRUE) included: edges +
#     degree(0:3) + nodefactor("Race") + nodematch("Race") + nodefactor("Sex")
#     + nodematch("Sex") + absdiff("Grade") + gwesp(0, fixed = TRUE) on the
#     census, popsize = 205. Monte-Carlo: R's mean and sd over the
#     `rep_seeds` that fitted are frozen, with each seed's elapsed time; a
#     seed on which ergm.ego errors or runs past `time_limit` is counted
#     (*_n_failed, of which *_n_timeout ran out of time).
#
# (c) mm: edges + mm("Sex") on the census, popsize = 205.
#
# (d) concurrent: edges + nodematch("Grade") + concurrent on the census.
#
# (e) esp: edges + nodematch("Grade") + esp(1) on the census.
#
# (f) A TARGET AT THE TOP OF ITS RANGE: edges + nodematch("Grade") +
#     gwdegree(0, fixed = TRUE) on the census of faux.mesa.high without its
#     57 isolates (148 egos, popsize 148). Every ego has an alter, so the
#     gwdeg.fixed.0 target (the number of non-isolates) is at its largest
#     attainable value, the network size: ergm.ego (through ergm's
#     target.stats check) fixes its coefficient at +Inf and fits the rest.
#     The +Inf is asserted on every seed and frozen.
#
# (g) esp UNDER THE NETWORK-SIZE OFFSET: edges + esp(1) on the unweighted
#     69-ego sub-design at ergm.ego's defaults (per capita). esp is an
#     order-3 statistic, so ergm.ego's offset carries transitiveties = -1/3
#     (fit$ergm.formula reads `netsize.adj(edges = 1, mutual = 0,
#     transitiveties = -1/3)`; recorded below).
#
# TOLERANCES follow ego_terms.R's rules, derived from R's own seed spread
# (the [tolerance] block states them); each fitted block records, per
# coefficient, R's sd over the seeds of the estimate, of its standard error
# and of the design component of that standard error (vcov(fit, sources =
# "model")).

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.ego)
})

seed <- 20261007
rep_seeds <- c(101, 202, 303, 404, 505, 606, 707, 808)

data(faux.mesa.high)
fmh <- faux.mesa.high
n <- network.size(fmh)
grade <- fmh %v% "Grade"
race <- fmh %v% "Race"
ed <- as.egor(fmh)
idx <- seq(1, n, 3)
sub <- ed[idx, ]
stopifnot(all(sub$ego$.egoID == idx))
subw <- sub
subw$ego$w <- grade[idx] - 6
ego_design(subw) <- list(weights = "w")
nonblack <- which(race != "Black")
ednb <- ed[nonblack, ]
stopifnot(!("Black" %in% ednb$ego$Race))
# the one tie of an "Other" student, and the egos without its other end
el <- as.edgelist(fmh)
oth <- which(race == "Other")
otie <- el[el[, 1] %in% oth | el[, 2] %in% oth, , drop = FALSE]
stopifnot(nrow(otie) == 1)
oth_friend <- setdiff(as.integer(otie), oth)
nofriend <- setdiff(seq_len(n), c(oth_friend, setdiff(oth, as.integer(otie))))
ednf <- ed[nofriend, ]
stopifnot(!("Other" %in% ednf$alter$Race), sum(ednf$ego$Race == "Other") == 1)

# --- (a) targets ------------------------------------------------------------
rhs <- ~ esp(0:3) + mm("Race") + mm("Sex") + concurrent
census <- summary(statnet.common::nonsimp_update.formula(rhs, ed ~ .), scaleto = n)
net_stats <- summary(statnet.common::nonsimp_update.formula(rhs, fmh ~ .))
stopifnot(max(abs(as.numeric(census) - as.numeric(net_stats))) < 1e-9,
          identical(names(census), names(net_stats)))
rhs_w <- ~ esp(0:3) + concurrent
weighted <- summary(statnet.common::nonsimp_update.formula(rhs_w, subw ~ .), scaleto = n)
wcov <- attr(weighted, "var")
err_mm_weighted <- tryCatch({ summary(subw ~ mm("Race"), scaleto = n); "" },
                            error = function(e) conditionMessage(e))
stopifnot(nchar(err_mm_weighted) > 0)
mm_sub <- summary(sub ~ mm("Race") + mm("Sex"), scaleto = n)
mm_nb <- summary(ednb ~ mm("Race"), scaleto = n)
mm_nf <- summary(ednf ~ mm("Race"), scaleto = n)
stopifnot(!any(grepl("Other", names(mm_sub))), any(grepl("Race=Black", names(mm_nb))),
          any(grepl("Other", names(mm_nf))))

# --- fits -------------------------------------------------------------------
# One seeded fit, run in a forked child so that it can be stopped (as in
# ego_terms.R): a fit that errors, or that has not finished within
# `time_limit` seconds, returns NULL and is COUNTED, never hidden.
time_limit <- 300
fit_once <- function(f, s, popsize) {
  t0 <- Sys.time()
  job <- parallel::mcparallel({
    set.seed(s)
    out <- NULL
    t1 <- Sys.time()
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
           m = network.size(out$network), names = names(cf)[free],
           secs = as.numeric(Sys.time() - t1, units = "secs"))
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
  for (k in seq_len(ncol(cf))) stopifnot(all(is.finite(cf[, k])) || length(unique(cf[, k])) == 1)
  sdf <- function(c) if (all(is.finite(c))) sd(c) else 0
  list(mean = colMeans(cf), sd = apply(cf, 2, sdf),
       se_mean = colMeans(se), se_sd = apply(se, 2, sd),
       se_model_mean = colMeans(sem), se_model_sd = apply(sem, 2, sd),
       adj = rows[[1]]$adj, m = rows[[1]]$m, names = rows[[1]]$names,
       secs = sapply(rows, `[[`, "secs"),
       n_fits = length(rows), n_failed = n_failed, n_timeout = n_timeout)
}

# (b) the help-page model, gwesp(0) included
hp <- replicate_fit(ed ~ edges + degree(0:3) + nodefactor("Race") + nodematch("Race") +
                      nodefactor("Sex") + nodematch("Sex") + absdiff("Grade") +
                      gwesp(0, fixed = TRUE), n)
stopifnot(hp$m == n, abs(hp$adj) < 1e-12)

# (c) mm
mmf <- replicate_fit(ed ~ edges + mm("Sex"), n)
stopifnot(mmf$m == n)

# (d) concurrent
conc <- replicate_fit(ed ~ edges + nodematch("Grade") + concurrent, n)
stopifnot(conc$m == n)

# (e) esp
espf <- replicate_fit(ed ~ edges + nodematch("Grade") + esp(1), n)
stopifnot(espf$m == n)

# (f) a target at the top of its range: gwdegree(0) with no isolate
deg <- sna::degree(fmh, gmode = "graph")
noniso <- fmh
delete.vertices(noniso, which(deg == 0))
edn <- as.egor(noniso)
n_noniso <- network.size(noniso)
top <- replicate_fit(edn ~ edges + nodematch("Grade") + gwdegree(0, fixed = TRUE), n_noniso)
stopifnot(top$m == n_noniso, top$n_failed == 0, all(top$mean[3] == Inf))

# (g) esp per capita: the offset statistic is edges - transitiveties/3
esppc <- replicate_fit(sub ~ edges + esp(1), NULL)
stopifnot(esppc$m == length(idx), abs(esppc$adj + log(length(idx))) < 1e-12)
off_formula <- {
  set.seed(1)
  fit1 <- suppressWarnings(suppressMessages(ergm.ego(sub ~ edges + esp(1),
            control = control.ergm.ego(ergm = control.ergm(seed = 1, MCMLE.maxit = 1)))))
  paste(deparse(fit1$ergm.formula[[3]][[2]]), collapse = " ")
}
stopifnot(grepl("transitiveties = -0.333", off_formula, fixed = TRUE))

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
  cat(sprintf("%s_seconds = [%s]\n", key, paste(sprintf("%.1f", r$secs), collapse = ", ")))
  cat(sprintf("%s_n_fits = %d\n", key, r$n_fits))
  cat(sprintf("%s_n_failed = %d\n", key, r$n_failed))
  cat(sprintf("%s_n_timeout = %d\n", key, r$n_timeout))
}

cat('name = "ego_mixing_esp"\n\n')
cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_ego_version = "%s"\n', as.character(packageVersion("ergm.ego"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/ego_mixing_esp.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::faux.mesa.high: 205 students, 203 undirected friendship ties; attributes Grade (7-12), Race, Sex"\n')
cat('model = "targets of esp(0:3) + mm(Race) + mm(Sex) + concurrent (census), of esp(0:3) + concurrent (weighted sub-design), of mm(Race) + mm(Sex) (unweighted sub-design) and of mm(Race) (the egos who are not Black; the egos without the Other student friend); fits on the census (popsize 205) of the ergm.ego help-page model with gwesp(0, fixed = TRUE), of edges + mm(Sex), edges + nodematch(Grade) + concurrent and edges + nodematch(Grade) + esp(1), of edges + nodematch(Grade) + gwdegree(0, fixed = TRUE) on the census without isolates (gwdeg.fixed.0 at its largest value), and of edges + esp(1) on the unweighted 69-ego sub-design at ergm.ego default popsize and ppopsize"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# TARGETS: deterministic weighted means of per-ego contributions, scaled\n")
cat("# to 205; 1e-9 is agreement to the floor of two summation orders.\n")
cat("targets = 1e-9\n")
cat("# DESIGN COVARIANCE of the weighted targets: 1e-9 RELATIVE TO ITS LARGEST\n")
cat("# ENTRY, i.e. max|S_J - S_R| <= design_cov_rel * max|S_R|.\n")
cat("design_cov_rel = 1e-9\n")
cat("#\n")
cat("# FITS: the rules of ego_terms.toml, from R's own seed spread.\n")
cat("# COEFFICIENTS: |mean_J - mean_R| <= seed_mean_sds * sd_R * sqrt(1/n_R + 1/n_J)\n")
cat("#   + seed_mean_sds * sd_R * n_failed / (n_fits + n_failed),\n")
cat("# with the premise that one Julia fit is no noisier than one R fit\n")
cat("# ASSERTED by the test (its seeds differ pairwise by at most\n")
cat("# seed_mean_sds * sqrt(2) * sd_R). A coefficient fixed at -Inf/+Inf is\n")
cat("# compared exactly.\n")
cat("seed_mean_sds = 4.0\n")
cat("# STANDARD ERRORS: the Julia design component (sqrt(diag(vcov_design)),\n")
cat("# se=:design) is held to R's design component (*_se_model_*) by the same\n")
cat("# band on its sd.\n")
cat("\n")

cat("[values]\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ego_ids = [%s]\n", paste(idx, collapse = ", ")))
cat(sprintf("weights = [%s]\n", paste(grade[idx] - 6, collapse = ", ")))
cat(sprintf("nonblack_ego_ids = [%s]\n", paste(nonblack, collapse = ", ")))
cat(sprintf("nofriend_ego_ids = [%s]\n", paste(nofriend, collapse = ", ")))
cat("\n# --- (a) targets, scaled to 205 ------------------------------------------\n")
cat(sprintf("term_names = [%s]\n", strs(names(census))))
cat(sprintf("census_targets = [%s]\n", num(as.numeric(census))))
cat(sprintf("weighted_names = [%s]\n", strs(names(weighted))))
cat(sprintf("weighted_targets = [%s]\n", num(as.numeric(weighted))))
cat(sprintf("weighted_design_cov = [%s]\n", mat(wcov)))
cat(sprintf("r_error_mm_weighted = \"%s\"\n", gsub('"', '\\\\"', gsub("\n", " ", err_mm_weighted))))
cat(sprintf("sub_mm_names = [%s]\n", strs(names(mm_sub))))
cat(sprintf("sub_mm_targets = [%s]\n", num(as.numeric(mm_sub))))
cat(sprintf("nonblack_mm_names = [%s]\n", strs(names(mm_nb))))
cat(sprintf("nonblack_mm_targets = [%s]\n", num(as.numeric(mm_nb))))
cat(sprintf("nofriend_mm_names = [%s]\n", strs(names(mm_nf))))
cat(sprintf("nofriend_mm_targets = [%s]\n", num(as.numeric(mm_nf))))
cat("\n# --- (b) the help-page model with gwesp(0), census, popsize 205 ----------\n")
block("helppage", hp)
cat("\n# --- (c) edges + mm(Sex), census, popsize 205 ----------------------------\n")
block("mm", mmf)
cat("\n# --- (d) edges + nodematch(Grade) + concurrent, census, popsize 205 ------\n")
block("concurrent", conc)
cat("\n# --- (e) edges + nodematch(Grade) + esp(1), census, popsize 205 ----------\n")
block("esp", espf)
cat("\n# --- (f) gwdeg.fixed.0 at its top: the census without its isolates -------\n")
cat(sprintf("noniso_ego_ids = [%s]\n", paste(which(deg > 0), collapse = ", ")))
block("top", top)
cat("\n# --- (g) edges + esp(1), popsize unknown, 69 unweighted egos -------------\n")
cat(sprintf("esp_percap_offset_term = \"%s\"\n", gsub('"', "'", off_formula)))
block("esp_percap", esppc)
