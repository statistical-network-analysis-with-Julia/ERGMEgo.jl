# Golden fixture: statnet `ergm.ego` on faux.mesa.high under a WEIGHTED,
# NON-CENSUS design -- the estimator's actual use case.
#
# Regenerate from the package root (~15 s: seven ergm.ego fits):
#
#   Rscript test/fixtures/r/fauxmesa_ego_weighted.R > test/fixtures/fauxmesa_ego_weighted.toml
#
# WHY A SECOND FIXTURE. The census fixture (fauxmesa_ego_census.R) pins the
# design variance and the netsize parameterisation under a design where the
# targets collapse to the network's own statistics and the sampling weights
# never enter: every ego has weight 1, and only edges + nodematch are fitted.
# Everything that makes ERGMEgo.jl an EGOCENTRIC estimator -- the
# case-weighted (Hajek) targets, the with-replacement design variance of a
# WEIGHTED mean, the weight-proportional pseudo-population, and the two other
# fittable terms (EgoTriangle with its /3, EgoGWDegree) -- was pinned by
# nothing R produced. This fixture pins exactly that.
#
# THE DESIGN: a DETERMINISTIC subsample with unequal case weights.
#
#   * Egos: every third actor, ids 1, 4, 7, ..., 205 (69 egos), as
#     `ed[seq(1, 205, 3), ]` of as.egor(faux.mesa.high). Their alters and the
#     alter-alter ties are the census's, so the Julia side rebuilds the same
#     EgoData from load_dataset(:faux_mesa_high) with no random draw.
#   * Weights: Grade - 6, i.e. 1 for grade 7 up to 6 for grade 12. Unequal,
#     integer, and a function of an ego attribute, so the weight-proportional
#     pseudo-population replicates egos 1:6 by grade -- the composition is
#     frozen (`ppop_grade_counts`) and asserted.
#   * ppopsize = popsize = 205, so ergm.ego's netsize.adj offset is exactly 0
#     and the free `edges` coefficient IS the population-scale parameter; the
#     Julia side reports the same scale with adjustment -log(N/m) = 0.
#
# DETERMINISTIC (no estimator, no MCMC; asserted at 1e-9):
#   * the four weighted targets of edges + nodematch("Grade") +
#     gwdegree(0.5, fixed=TRUE) + triangle, scaled to 205
#     (summary(egor ~ ..., scaleto = 205));
#   * their full 4x4 design covariance (attr(., "var"));
#   * the pseudo-population's composition by grade;
#   * the EXACT information of the dyad-independent edges + nodematch model
#     at ergm.ego's estimate on that pseudo-population (a closed-form dyad
#     sum: the model has two dyad types, same-grade and different-grade), and
#     the design SE of the coefficients it implies (`design_se_exact`).
# MONTE-CARLO (tolerances from both packages' seed-to-seed spread):
#   * the fitted population-scale coefficients of edges + nodematch("Grade");
#   * ergm.ego's SE decomposition, with its MCMC estimate of the information
#     (fit$DtDe) frozen so the Julia side can see how far R's own estimate
#     sits from the exact one.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.ego)
})

seed <- 20260912
data(faux.mesa.high)
fmh <- faux.mesa.high
n <- network.size(fmh)
grade <- fmh %v% "Grade"

ed <- as.egor(fmh)
idx <- seq(1, n, 3)
sub <- ed[idx, ]
stopifnot(nrow(sub$ego) == length(idx), all(sub$ego$.egoID == idx))
w <- grade[idx] - 6
sub$ego$w <- w
ego_design(sub) <- list(weights = "w")

# --- DETERMINISTIC: weighted targets and their design covariance -----------
f4 <- sub ~ edges + nodematch("Grade") + gwdegree(0.5, fixed = TRUE) + triangle
s4 <- summary(f4, scaleto = n)
targets <- as.numeric(s4)
term_names <- names(s4)
design_cov <- attr(s4, "var")
design_se <- sqrt(diag(design_cov))

# --- MONTE-CARLO: the fit of edges + nodematch("Grade") ---------------------
f <- sub ~ edges + nodematch("Grade")

fit_once <- function(s) {
  set.seed(s)
  out <- NULL
  invisible(capture.output(
    out <- suppressWarnings(suppressMessages(
      ergm.ego(f, popsize = n,
               control = control.ergm.ego(ppopsize = n,
                                          ergm = control.ergm(seed = s))))),
    type = "output"))
  out
}

fit <- fit_once(seed)
cf <- coef(fit)
free <- setdiff(seq_along(cf), grep("netsize.adj", names(cf)))
mle_coef <- as.numeric(cf[free])
netsize_adj <- as.numeric(cf[grep("netsize.adj", names(cf))])
stopifnot(abs(netsize_adj) < 1e-12)          # ppopsize == popsize
pop_coef <- c(netsize_adj + mle_coef[1], mle_coef[-1])

se_all <- sqrt(diag(vcov(fit)))[free]
se_model <- sqrt(diag(vcov(fit, sources = "model")))[free]
se_est <- sqrt(diag(vcov(fit, sources = "estimation")))[free]
DtDe <- fit$DtDe
design_cov2 <- design_cov[1:2, 1:2]
stopifnot(max(abs(sqrt(diag(solve(DtDe) %*% design_cov2 %*% solve(DtDe))) - se_model)) < 1e-12)

# The pseudo-population ergm.ego built: 205 vertices, egos replicated in
# proportion to their weights (control ppop.wt = "round"). Its composition by
# grade is deterministic and is all the exact information below depends on.
ppop <- fit$network
stopifnot(network.size(ppop) == n)
pg <- ppop %v% "Grade"
ppop_counts <- as.numeric(table(factor(pg, levels = 7:12)))

# EXACT information of edges + nodematch("Grade") at the estimate, on that
# pseudo-population. The model is dyad-independent with two dyad types:
#   same-grade dyads    x = (1, 1), p1 = plogis(theta1 + theta2)
#   different-grade     x = (1, 0), p0 = plogis(theta1)
#   I = n_mis p0(1-p0) [1 0; 0 0] + n_mat p1(1-p1) [1 1; 1 1]
# No MCMC anywhere: with the frozen counts and coefficients the Julia side
# reproduces it to machine precision, and so the design SE it implies.
n_mat <- sum(ppop_counts * (ppop_counts - 1) / 2)
n_mis <- n * (n - 1) / 2 - n_mat
p0 <- plogis(pop_coef[1])
p1 <- plogis(pop_coef[1] + pop_coef[2])
I_exact <- n_mis * p0 * (1 - p0) * matrix(c(1, 0, 0, 0), 2) +
           n_mat * p1 * (1 - p1) * matrix(1, 2, 2)
design_se_exact <- sqrt(diag(solve(I_exact) %*% design_cov2 %*% solve(I_exact)))

rep_seeds <- c(101, 202, 303, 404, 505)
reps <- t(sapply(rep_seeds, function(s) {
  ft <- fit_once(s)
  cfs <- coef(ft)
  fr <- setdiff(seq_along(cfs), grep("netsize.adj", names(cfs)))
  c(as.numeric(cfs[fr]), sqrt(diag(vcov(ft)))[fr])
}))
seed_sd <- apply(reps[, 1:2], 2, sd)
seed_mean <- colMeans(reps[, 1:2])
se_seed_sd <- apply(reps[, 3:4], 2, sd)
se_seed_mean <- colMeans(reps[, 3:4])

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")
mat <- function(M) paste(apply(M, 1, function(r) paste0("[", num(r), "]")), collapse = ", ")

cat('name = "fauxmesa_ego_weighted"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_ego_version = "%s"\n', as.character(packageVersion("ergm.ego"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/fauxmesa_ego_weighted.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::faux.mesa.high: 205 students, 203 undirected friendship ties, Grade 7-12"\n')
cat('sampling_design = "WEIGHTED SUBSAMPLE: egos 1, 4, 7, ..., 205 (every third actor, 69 egos) of as.egor(faux.mesa.high), case weights Grade - 6 (1 to 6), ppopsize = popsize = 205 (netsize.adj = 0). Deterministic ego set, unequal weights: the Hajek targets, the design variance of a weighted mean and the weight-proportional pseudo-population all enter, which the census fixture cannot exercise."\n')
cat('model = "egor ~ edges + nodematch(\\"Grade\\") fitted by ergm.ego (popsize = 205, control.ergm.ego(ppopsize = 205)); targets and design covariance frozen for edges + nodematch(\\"Grade\\") + gwdegree(0.5, fixed = TRUE) + triangle"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# WEIGHTED TARGETS (edges, nodematch.Grade, gwdeg.fixed.0.5, triangle,\n")
cat("# scaled to 205). Hajek estimates: 205 * sum(w_i h_i) / sum(w_i) with the\n")
cat("# per-ego contributions degree/2, matches/2, e^a(1-(1-e^-a)^degree) and\n")
cat("# alter-alter ties/3. Deterministic; 1e-9 on numbers of size 56-210 is\n")
cat("# agreement to 11 significant digits, the floor of two summation orders.\n")
cat("targets = 1e-9\n")
cat("#\n")
cat("# DESIGN STANDARD ERRORS OF THE TARGETS, sqrt(diag(Sigma_design)), and the\n")
cat("# full 4x4 Sigma_design (off-diagonal included): the with-replacement\n")
cat("# variance of a WEIGHTED mean, m^2 n/(n-1) sum_i w_i^2 (h_i - hbar)(h_i - hbar)'\n")
cat("# with normalised weights -- the survey variance ergm.ego computes for a\n")
cat("# design with unequal case weights. Deterministic; asserted exactly.\n")
cat("design_std_errors = 1e-9\n")
cat("#\n")
cat("# THE PSEUDO-POPULATION'S COMPOSITION BY GRADE (ppop_grade_counts): egos\n")
cat("# replicated in proportion to their weights, 205 vertices in all.\n")
cat("# ergm.ego's ppop.wt = \"round\" and ERGMEgo.jl's largest-remainder rounding\n")
cat("# give the same counts here; asserted exactly (integers).\n")
cat("ppop_grade_counts = 0.5\n")
cat("#\n")
cat("# EXACT INFORMATION at ergm.ego's estimate on that pseudo-population and\n")
cat("# the design SE it implies (design_se_exact): a closed-form dyad sum with\n")
cat("# no estimator in it, reproducible in Julia from the frozen counts and\n")
cat("# coefficients to machine precision.\n")
cat("exact_information = 1e-6\n")
cat("design_se_exact = 1e-9\n")
cat("#\n")
cat("# FITTED POPULATION-SCALE COEFFICIENTS -- MONTE-CARLO ON BOTH SIDES.\n")
cat("# Both packages are moment-matching by MCMC from the same targets. Measured\n")
cat("# spreads on this design:\n")
cat(sprintf("#   ergm.ego seed-to-seed sd (5 seeds)   : %.4f (edges), %.4f (nodematch)\n", seed_sd[1], seed_sd[2]))
cat("#   ERGMEgo.jl seed-to-seed sd (10 seeds): 0.0093, 0.0103\n")
cat(sprintf("#   ergm.ego 5-seed mean                 : %.4f, %.4f\n", seed_mean[1], seed_mean[2]))
cat("#   ERGMEgo.jl 10-seed mean              : -5.6032, 2.2906\n")
cat("# The two means differ by 0.0003 / 0.001 -- well inside the combined\n")
cat("# Monte-Carlo sd of the two means (sqrt(0.0070^2/5 + 0.0093^2/10) = 0.004):\n")
cat("# no systematic difference is detectable. A SINGLE Julia fit is held to the\n")
cat("# single frozen R fit at 0.05: the combined single-fit sd is\n")
cat("# sqrt(0.0070^2 + 0.0093^2) = 0.012, so 0.05 is 4x it (and 0.2 of the\n")
cat("# smaller fitted standard error; the largest single-fit gap seen over ten\n")
cat("# Julia seeds was 0.026). The 3-seed Julia mean is held to ergm.ego's 5-seed\n")
cat("# mean at 0.03 (5x the combined sd of the two means, 0.006). Both sides are\n")
cat("# seeded: deterministic, cannot flake.\n")
cat("mle_coefficients_population = 0.05\n")
cat("mcmle_seed_mean = 0.03\n")
cat("#\n")
cat("# STANDARD ERRORS -- MONTE-CARLO ON BOTH SIDES. Each package sandwiches the\n")
cat("# (exact) design covariance with ITS OWN MCMC estimate of the information.\n")
cat(sprintf("# R's DtDe puts the design SE at %.4f / %.4f against the exact %.4f / %.4f\n", se_model[1], se_model[2], design_se_exact[1], design_se_exact[2]))
cat(sprintf("# (ergm.ego 5-seed SE mean %.4f / %.4f, sd %.4f / %.4f); ERGMEgo.jl's\n", se_seed_mean[1], se_seed_mean[2], se_seed_sd[1], se_seed_sd[2]))
cat("# per-fit SE has mean 0.243 / 0.262 and sd 0.024 / 0.022 over 10 seeds, i.e.\n")
cat("# both scatter around the exact value (the census fixture's 12 % offset of\n")
cat("# R's DtDe does not recur here). The information is the covariance of the\n")
cat("# ONE sample that passed the convergence tests (n_eff ~ 110-170), so the\n")
cat("# Julia per-fit SE carries that sample's noise: the largest |Julia - exact|\n")
cat("# over ten seeds was 0.050 / 0.042 and the largest |Julia - R| 0.048 / 0.038.\n")
cat("# 0.08 is > 3x the Julia per-fit sd (0.072) and R's offset from the exact\n")
cat("# value + 3x that sd, rounded up; it is 0.3 of the standard error. Applied\n")
cat("# per fit (three seeds) to the design component and to the total; the\n")
cat("# estimation component is not held to a tolerance (I^-1/n_eff is a\n")
cat("# property of each package's own chain) beyond being < 0.3x the design part.\n")
cat("mle_std_errors = 0.08\n")
cat("mle_se_design_component = 0.08\n")
cat("\n")

cat("[values]\n")
cat("# --- the design, frozen. Julia rebuilds the EgoData from the bundled\n")
cat("# load_dataset(:faux_mesa_high): egos `ego_ids` with weights `weights`. ---\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("ego_ids = [%s]\n", paste(idx, collapse = ", ")))
cat(sprintf("weights = [%s]\n", paste(w, collapse = ", ")))
cat(sprintf("n_egos = %d\n", length(idx)))
cat(sprintf("ppopsize = %d\n", n))
cat(sprintf("popsize = %d\n", n))
cat("\n# --- DETERMINISTIC: weighted targets and their design covariance --------\n")
cat(sprintf("term_names = [%s]\n", strs(term_names)))
cat(sprintf("targets = [%s]\n", num(targets)))
cat(sprintf("design_std_errors = [%s]\n", num(design_se)))
cat(sprintf("design_cov = [%s]\n", mat(design_cov)))
cat("\n# --- DETERMINISTIC: the pseudo-population and the exact information -----\n")
cat("# vertices per grade 7..12 of the 205-vertex pseudo-population\n")
cat(sprintf("ppop_grade_counts = [%s]\n", paste(ppop_counts, collapse = ", ")))
cat(sprintf("n_same_grade_dyads = %d\n", as.integer(n_mat)))
cat("# I at mle_coefficients_population, row-major\n")
cat(sprintf("exact_information = [%s]\n", num(as.numeric(t(I_exact)))))
cat(sprintf("design_se_exact = [%s]\n", num(design_se_exact)))
cat("\n# --- ergm.ego's fit of edges + nodematch(\"Grade\") -------------------------\n")
cat(sprintf("mle_coefficients = [%s]\n", num(mle_coef)))
cat(sprintf("netsize_adjustment = %.17g\n", netsize_adj + 0))   # + 0: -0 prints as 0
cat("# netsize.adj + edges, nodematch: the population-scale parameter both\n")
cat("# packages estimate (the offset is 0 here, so this equals mle_coefficients)\n")
cat(sprintf("mle_coefficients_population = [%s]\n", num(pop_coef)))
cat(sprintf("mle_std_errors = [%s]\n", num(as.numeric(se_all))))
cat(sprintf("mle_se_design_component = [%s]\n", num(as.numeric(se_model))))
cat(sprintf("mle_se_estimation_component = [%s]\n", num(as.numeric(se_est))))
cat("# ergm.ego's MCMC estimate of the information (fit$DtDe, row-major)\n")
cat(sprintf("r_DtDe = [%s]\n", num(as.numeric(t(DtDe)))))
cat("\n# ergm.ego over five further seeds: the Monte-Carlo floor under the\n")
cat("# coefficient tolerance\n")
cat(sprintf("mcmle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mcmle_seed_mean = [%s]\n", num(seed_mean)))
cat(sprintf("mcmle_se_seed_sd = [%s]\n", num(se_seed_sd)))
cat(sprintf("mcmle_se_seed_mean = [%s]\n", num(se_seed_mean)))
