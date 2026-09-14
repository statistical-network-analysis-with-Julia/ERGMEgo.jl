# Golden fixture: statnet `ergm.ego` on faux.mesa.high under a KNOWN sampling
# design (a complete census, unit weights, ppopsize = n).
#
# Regenerate from the package root (~3 min: six ergm.ego fits and one plain ergm):
#
#   Rscript test/fixtures/r/fauxmesa_ego_census.R > test/fixtures/fauxmesa_ego_census.toml
#
# THIS FIXTURE EXISTS TO SETTLE ISSUE ERGMEgo#1.
#
# ERGMEgo.jl reports standard errors V(theta) = I^-1 Sigma_design I^-1 + I^-1/n_eff
# -- ergm.ego's own decomposition into a design component (vcov(fit,
# sources="model")) and an MCMC-estimation component (sources="estimation") --
# and calls the first term a "survey-design variance". The package's own
# `approximations()` already warns that it is narrower than that name implies --
# it encodes no strata, no clusters, no finite-population correction, no
# replicate weights, no without-replacement inclusion probabilities, no alter
# dependence. But a warning in prose is not a measurement. Nobody knew, in
# numbers, how far ERGMEgo.jl's design variance sits from what `ergm.ego`
# computes under a design both packages CAN express. That is what this fixture
# measures, and the whole design of the fixture is arranged around making that
# one number visible.
#
# THE DESIGN: A CENSUS.
#
#   as.egor(faux.mesa.high) -- every one of the 205 actors is an ego, reporting
#   their alters and the alter-alter ties. Unit weights. ppopsize = 205 = n.
#
# A census is chosen deliberately, and not because it is easy:
#
#   * The TARGET STATISTICS become exactly the observed network's statistics
#     (edges = 203, nodematch.Grade = 163). They are a deterministic function of
#     the data with no sampling and no Monte Carlo anywhere, so they are asserted
#     at machine precision. If ERGMEgo.jl's design-weighted per-capita scaling
#     were wrong by even a constant, this catches it -- and no amount of
#     Monte-Carlo hand-waving can excuse a failure.
#
#   * The DESIGN VARIANCE of those targets is likewise deterministic: it is a
#     function of the 205 per-ego contributions and the weights, with no
#     estimator in it. `summary(egor ~ ...)` reports it (`design_std_errors`
#     below: 15.202 for edges, 13.631 for nodematch). This is the number
#     ERGMEgo.jl's `_design_cov` is claiming to compute, stripped of the I^-1
#     sandwich and of every source of noise that could be blamed for a
#     disagreement. It is the cleanest possible test of issue #1, and the Julia
#     testset asserts against it directly rather than against the SEs it feeds.
#
#   * Only the FITTED COEFFICIENTS remain Monte-Carlo on both sides, and those
#     get a tolerance measured from ergm.ego's own seed-to-seed spread
#     (`mcmle_seed_sd`), refit here under five further seeds.
#
# ergm.ego decomposes its standard errors, and the decomposition is frozen too:
# `vcov(fit, sources="model")` is the egocentric-design component and
# `sources="estimation"` is the MCMC error. On this fit the design component is
# ~17x the estimation component (0.133 vs 0.0079 on edges) -- i.e. essentially
# ALL of an ergm.ego standard error is the design variance. That is exactly why
# getting it right matters, and why issue #1 is not a footnote.

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(ergm.ego)
})

seed <- 20260713
data(faux.mesa.high)
fmh <- faux.mesa.high
n <- network.size(fmh)

grade <- fmh %v% "Grade"
el <- as.matrix(fmh, matrix.type = "edgelist")

# The census egodata. Unit weights; every actor is an ego.
ed <- as.egor(fmh)

f <- ed ~ edges + nodematch("Grade")

# Design-weighted target statistics scaled to the pseudo-population, AND their
# design standard errors. Both deterministic -- no estimator, no MCMC.
# `summary.egor` returns an `svystat`: a named vector of the scaled targets
# carrying the full design COVARIANCE matrix in attr(., "var"). Freeze the whole
# matrix, not just its diagonal -- the off-diagonal is part of the claim.
s <- summary(f, scaleto = n)
targets <- as.numeric(s)
term_names <- names(s)
design_cov <- attr(s, "var")
design_se <- sqrt(diag(design_cov))

# ergm.ego prints MCMLE chatter (and an lpSolveAPI note) to stdout, and this
# script's stdout IS the TOML fixture.
fit_once <- function(s) {
  set.seed(s)
  out <- NULL
  invisible(capture.output(
    out <- suppressWarnings(suppressMessages(
      ergm.ego(f, control = control.ergm.ego(ppopsize = "samp",
                                             ergm = control.ergm(seed = s))))),
    type = "output"))
  out
}

fit <- fit_once(seed)
# The netsize.adj offset is coefficient 1 and is not estimated; the free
# parameters are edges and nodematch.Grade.
cf <- coef(fit)
free <- setdiff(seq_along(cf), grep("netsize.adj", names(cf)))
mle_coef <- as.numeric(cf[free])
netsize_adj <- as.numeric(cf[grep("netsize.adj", names(cf))])

se_all <- sqrt(diag(vcov(fit)))[free]
se_model <- sqrt(diag(vcov(fit, sources = "model")))[free]
se_est <- sqrt(diag(vcov(fit, sources = "estimation")))[free]

# ergm.ego's estimate of the information: DtDe is the covariance of the
# sufficient statistics on the final MCMC sample (the Jacobian of the moment
# equations). Its inverse sandwiches the design covariance:
#   vcov(fit, sources = "model") == solve(DtDe) %*% design_cov %*% solve(DtDe)
# (verified below to 1e-12 before freezing). Frozen so the Julia side can see
# how far ONE MCMC estimate of the information sits from the exact one.
DtDe <- fit$DtDe
stopifnot(max(abs(sqrt(diag(solve(DtDe) %*% design_cov %*% solve(DtDe))) - se_model)) < 1e-12)

# The same sandwich with the EXACT information. edges + nodematch("Grade") is
# dyad-independent, so the plain-ergm MPLE is the MLE and vcov(ergm(...)) is
# the exact inverse Fisher information I^-1 (no MCMC anywhere). Hence
#   design_se_exact = sqrt(diag(I^-1 design_cov I^-1))
# is DETERMINISTIC: the design standard error of the coefficients with every
# Monte-Carlo excuse removed, on both sides.
plain <- suppressMessages(ergm(fmh ~ edges + nodematch("Grade")))
plain_coef <- as.numeric(coef(plain))
plain_vcov <- vcov(plain)
I_mple <- solve(plain_vcov)
design_se_exact <- sqrt(diag(solve(I_mple) %*% design_cov %*% solve(I_mple)))

rep_seeds <- c(101, 202, 303, 404, 505)
reps <- t(sapply(rep_seeds, function(s) as.numeric(coef(fit_once(s))[free])))
seed_sd <- apply(reps, 2, sd)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
strs <- function(x) paste(sprintf('"%s"', x), collapse = ", ")

cat('name = "fauxmesa_ego_census"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('ergm_ego_version = "%s"\n', as.character(packageVersion("ergm.ego"))))
cat(sprintf('ergm_version = "%s"\n', as.character(packageVersion("ergm"))))
cat(sprintf('network_version = "%s"\n', as.character(packageVersion("network"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/fauxmesa_ego_census.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "ergm::faux.mesa.high: 205 students, 203 undirected friendship ties, Grade 7-12"\n')
cat('sampling_design = "CENSUS: as.egor(faux.mesa.high) -- every actor is an ego, unit sampling weights, ppopsize = popsize = 205. A stated, known design, chosen so the target statistics and their design variance are both DETERMINISTIC and can be asserted exactly (issue ERGMEgo#1)."\n')
cat('model = "egor ~ edges + nodematch(\\"Grade\\"), fitted by ergm.ego with an offset netsize.adj term"\n')
cat(sprintf('replication_seeds = "%s"\n', paste(rep_seeds, collapse = ",")))
cat("\n")

cat("[tolerance]\n")
cat("# TARGET STATISTICS. Under a census these are the observed network's own\n")
cat("# statistics (edges = 203, nodematch.Grade = 163): a deterministic function\n")
cat("# of the data, with no sampling and no Monte Carlo. Machine precision. A\n")
cat("# failure here means the design-weighted per-capita scaling is wrong.\n")
cat("targets = 1e-9\n")
cat("#\n")
cat("# DESIGN STANDARD ERRORS OF THE TARGETS -- THE POINT OF THIS FIXTURE, AND\n")
cat("# THE DIRECT TEST OF ISSUE ERGMEgo#1.\n")
cat("#\n")
cat("# This is `sqrt(diag(Sigma_design))`, the survey variance of the target\n")
cat("# statistics: a function of the 205 per-ego contributions and their weights\n")
cat("# and NOTHING else -- no estimator, no MCMC, no I^-1 sandwich to blame a\n")
cat("# disagreement on. ERGMEgo.jl's `_design_cov` is compared to it directly,\n")
cat("# and to the full covariance matrix `design_cov` (off-diagonal included).\n")
cat("#\n")
cat("# HISTORY (issue ERGMEgo#1, answered by this fixture and FIXED): the first\n")
cat("# run of this fixture found ERGMEgo.jl's design variance too small by\n")
cat("# exactly the factor (n-1)/n -- `_design_cov` divided the sum of squared\n")
cat("# deviations by n where the survey (SRS / Horvitz-Thompson) variance of a\n")
cat("# mean, which ergm.ego computes, divides by n-1. The Bessel factor is now\n")
cat("# applied, and the Julia testset pins it with an independent Bessel\n")
cat("# reference (Statistics.cov) and a hand-computed weighted case, so the\n")
cat("# correction cannot be removed silently.\n")
cat("#\n")
cat("# The design standard errors are therefore asserted EXACTLY: 1e-9 on\n")
cat("# numbers of size 15.2 and 13.6 is agreement to 11 significant digits, the\n")
cat("# floating-point floor of two different summation orders. There is no gap\n")
cat("# to admit and no tolerance to justify.\n")
cat("design_std_errors = 1e-9\n")
cat("#\n")
cat("# DESIGN STANDARD ERRORS OF THE COEFFICIENTS WITH THE EXACT INFORMATION.\n")
cat("#\n")
cat("# `design_se_exact = sqrt(diag(I^-1 Sigma_design I^-1))` with I the exact\n")
cat("# Fisher information of the plain (non-egocentric) ERGM of the same model\n")
cat("# on the same network -- dyad-independent, so MPLE = MLE and\n")
cat("# `vcov(ergm(...))` is I^-1 with no MCMC in it. Both factors are\n")
cat("# deterministic. ERGM.jl's MPLE reproduces R's vcov to ~1e-6 (the Newton\n")
cat("# convergence tolerance of two different optimisers), and the Julia\n")
cat("# testset rebuilds the sandwich from `inv(vcov(fit_ergm(...)))` and its own\n")
cat("# `_design_cov`; 1e-6 on numbers of size 0.16 is what those two\n")
cat("# optimisers leave.\n")
cat("design_se_exact = 1e-6\n")
cat("plain_ergm_vcov = 1e-6\n")
cat("plain_ergm_mple = 1e-6\n")
cat("#\n")
cat("# FITTED COEFFICIENTS -- AND A PARAMETERIZATION DIFFERENCE THAT MUST BE\n")
cat("# MAPPED, NOT ASSUMED AWAY.\n")
cat("#\n")
cat("# ergm.ego splits the population edges parameter into a FIXED OFFSET\n")
cat("# `netsize.adj` = -log(popsize) = -5.3230 plus a free `edges` coefficient\n")
cat("# (-0.6974). ERGMEgo.jl reports a single edges coefficient on the\n")
cat("# pseudo-population scale, with its own adjustment -log(N/m), which under\n")
cat("# this census (N = m = 205) is 0. The two numbers are therefore NOT\n")
cat("# comparable term-by-term: R's -0.6974 and Julia's -6.03 are the same\n")
cat("# parameter in different parameterizations. The quantity both packages\n")
cat("# estimate is netsize.adj + edges = -6.0204, frozen below as\n")
cat("# `mle_coefficients_population`, and it is what the Julia testset compares\n")
cat("# against. (Cross-check: a plain ERGM.jl MPLE of edges + nodematch on the\n")
cat("# same 205-node network gives -6.034 / 2.831 -- which is what a census\n")
cat("# SHOULD reduce to, and it does.)\n")
cat("#\n")
cat("# Both sides are moment-matching by MCMC. Measured spreads (ERGMEgo.jl\n")
cat("# under its 2026-09 defaults: ERGM.jl's t-ratio + Hotelling convergence\n")
cat("# rule, dyad-scaled burnin 20*n_dyads / interval n_dyads/10, 3000 draws,\n")
cat("# effective sample size 120-180 per draw):\n")
cat("#   ergm.ego seed-to-seed sd : 0.0100 (edges), 0.0098 (nodematch)\n")
cat("#   ERGMEgo.jl seed-to-seed sd: 0.026, 0.026 (was 0.040/0.044 under the\n")
cat("#                               pre-panel 1 % relative-change rule;\n")
cat("#                               0.012/0.012 over six seeds since round 2\n")
cat("#                               made the passing sample the final one)\n")
cat("#   |mean(ERGMEgo.jl, 5 seeds) - R| : 0.010, 0.012 (was 0.052/0.047)\n")
cat("# The combined Monte-Carlo sd of the two means is\n")
cat("# sqrt(0.026^2/5 + 0.010^2) = 0.015, so the observed gap is 0.7-0.8 of it:\n")
cat("# no systematic difference is detectable any more (the earlier 0.052 gap\n")
cat("# was the relative-change rule stopping at the noise level, not the\n")
cat("# pseudo-population construction).\n")
cat("#\n")
cat("# 0.05 is >= 3x the combined Monte-Carlo sd of the two means (0.046) and\n")
cat("# >= 2x the larger observed gap (0.024); it is 0.28 of the smaller fitted\n")
cat("# standard error. Both sides are seeded, so the comparison is\n")
cat("# deterministic and cannot flake.\n")
cat("mle_coefficients_population = 0.05\n")
cat("#\n")
cat("# STANDARD ERRORS OF THE COEFFICIENTS -- MONTE-CARLO ON BOTH SIDES.\n")
cat("#\n")
cat("# `mle_std_errors` and `mle_se_design_component` sandwich the (exact)\n")
cat("# design covariance with an MCMC ESTIMATE of the information, on both\n")
cat("# sides: ergm.ego's DtDe (frozen as `r_DtDe`) and ERGMEgo.jl's covariance\n")
cat("# of its final sample. Neither is the exact information: R's DtDe puts\n")
cat("# solve(DtDe)[1,1] = 0.02853 against the exact I^-1[1,1] = 0.02506 (12 %\n")
cat("# high), so R's design SE (0.1779 / 0.1954) sits 0.022 / 0.019 above the\n")
cat("# exact `design_se_exact` (0.1563 / 0.1760); ERGMEgo.jl's per-fit design SE\n")
cat("# has mean 0.160 / 0.177 and seed-to-seed sd 0.013 / 0.009 over the five\n")
cat("# replication seeds, i.e. it scatters around the exact value with R's DtDe\n")
cat("# 1.4 sd above it. The largest per-fit |Julia - R| observed is 0.039.\n")
cat("#\n")
cat("# 0.06 = R's 0.022 offset from the exact value + 3 x ERGMEgo.jl's 0.013\n")
cat("# per-fit sd. It is applied per fit (every one of the five seeds must\n")
cat("# pass), and separately the five-seed MEAN SE is held to it as well. The\n")
cat("# estimation component is NOT held to a tolerance: it is I^-1/n_eff and\n")
cat("# n_eff is a property of each package's own chain (ERGMEgo.jl's is\n")
cat("# ~150 per 3000-draw sample; ergm.ego's larger sample gives 0.0083 /\n")
cat("# 0.0103); the testset asserts it is below 0.3 x the design component.\n")
cat("mle_std_errors = 0.06\n")
cat("mle_se_design_component = 0.06\n")
cat("\n")

cat("[values]\n")
cat("# --- the network, frozen. Julia rebuilds the census egodata from it. ----\n")
cat(sprintf("n_actors = %d\n", n))
cat("directed = false\n")
cat(sprintf("grade = [%s]\n", paste(grade, collapse = ", ")))
cat(sprintf("edge_src = [%s]\n", paste(el[, 1], collapse = ", ")))
cat(sprintf("edge_dst = [%s]\n", paste(el[, 2], collapse = ", ")))
cat(sprintf("n_edges = %d\n", nrow(el)))
cat(sprintf("ppopsize = %d\n", n))
cat(sprintf("popsize = %d\n", n))
cat("\n# --- DETERMINISTIC: targets and their design variance -------------------\n")
cat(sprintf("term_names = [%s]\n", strs(term_names)))
cat(sprintf("targets = [%s]\n", num(targets)))
cat("# sqrt(diag(Sigma_design)) -- the survey standard error of each TARGET\n")
cat("# statistic. This is ERGMEgo.jl's `_design_cov` with every excuse removed.\n")
cat(sprintf("design_std_errors = [%s]\n", num(design_se)))
cat("# ...and the full design covariance matrix, row-major. The off-diagonal is\n")
cat("# part of the claim too: an implementation that got the marginal variances\n")
cat("# right and the covariance wrong would still produce wrong standard errors\n")
cat("# once sandwiched by I^-1, which is not diagonal.\n")
cat(sprintf("design_cov = [%s]\n",
            paste(apply(design_cov, 1, function(r)
                        paste0("[", num(r), "]")), collapse = ", ")))
cat("\n# --- ergm.ego's fit -----------------------------------------------------\n")
cat(sprintf("mle_coefficients = [%s]\n", num(mle_coef)))
cat(sprintf("mle_std_errors = [%s]\n", num(as.numeric(se_all))))
cat(sprintf("netsize_adjustment = %.17g\n", netsize_adj))
cat("\n# THE COMPARABLE QUANTITY. ergm.ego puts the population edges parameter in\n")
cat("# TWO pieces -- the fixed offset netsize.adj = -log(popsize) and the free\n")
cat("# `edges` coefficient -- while ERGMEgo.jl reports it as one number on the\n")
cat("# pseudo-population scale. Their SUM is the same parameter, and it is what\n")
cat("# the Julia testset is held to. Comparing R's -0.697 against Julia's -6.07\n")
cat("# term-by-term would be comparing two different parameterizations and would\n")
cat("# fail for no reason at all.\n")
cat(sprintf("mle_coefficients_population = [%s]\n",
            num(c(netsize_adj + mle_coef[1], mle_coef[-1]))))
cat("# Sanity: a plain (non-egocentric) ERGM MPLE of edges + nodematch(\"Grade\")\n")
cat("# on the same 205-node network. A CENSUS ego fit must reduce to this, and\n")
cat("# both packages do. Frozen as an independent anchor on the number above.\n")
cat(sprintf("plain_ergm_mple = [%s]\n", num(plain_coef)))
cat("# ...and its covariance vcov(ergm(...)), row-major: the EXACT inverse Fisher\n")
cat("# information (dyad-independent model, MPLE = MLE, no MCMC).\n")
cat(sprintf("plain_ergm_vcov = [%s]\n", num(as.numeric(t(plain_vcov)))))
cat("# sqrt(diag(I^-1 Sigma_design I^-1)) with that EXACT I: the design standard\n")
cat("# error of the coefficients with every Monte-Carlo excuse removed.\n")
cat(sprintf("design_se_exact = [%s]\n", num(design_se_exact)))
cat("\n# ergm.ego's OWN decomposition of its standard errors. Note the ratio: the\n")
cat("# design component is ~17x the estimation component, so essentially ALL of\n")
cat("# an ergm.ego standard error IS the design variance. That is why issue #1\n")
cat("# is not a footnote.\n")
cat(sprintf("mle_se_design_component = [%s]\n", num(as.numeric(se_model))))
cat(sprintf("mle_se_estimation_component = [%s]\n", num(as.numeric(se_est))))
cat("# ergm.ego's MCMC estimate of the information (fit$DtDe, row-major): the\n")
cat("# matrix whose inverse sandwiches design_cov to give the design component\n")
cat("# above. Compare its inverse with plain_ergm_vcov to see how much one MCMC\n")
cat("# estimate of I moves a standard error (~12 % here).\n")
cat(sprintf("r_DtDe = [%s]\n", num(as.numeric(t(DtDe)))))
cat("\n# ergm.ego disagreeing with ITSELF over five further seeds: the Monte-Carlo\n")
cat("# floor under the coefficient tolerance.\n")
cat(sprintf("mcmle_seed_sd = [%s]\n", num(seed_sd)))
cat(sprintf("mcmle_seed_mean = [%s]\n", num(colMeans(reps))))
