# msgarch.R — Estimate MSGARCH(1,1) with Student-t innovations for all series.
#
# Usage (called from Makefile):
#   Rscript R/msgarch.R config/config.yaml
#
# Requirements:
#   install.packages(c("MSGARCH", "yaml", "jsonlite"))
#
# Output (per series):
#   outputs/models/MSGARCH/<series>/sigma2_test.npy  (via reticulate or write.csv)
#   outputs/models/MSGARCH/<series>/params.json
#   outputs/models/MSGARCH/<series>/fit_info.json
#
# If MSGARCH is not installed, the script writes an error to
# logs/msgarch_error.log and exits with code 1 (Makefile catches this).

# ── Load libraries ──────────────────────────────────────────────────────────
required_pkgs <- c("MSGARCH", "yaml", "jsonlite")
missing_pkgs  <- required_pkgs[!vapply(required_pkgs, requireNamespace,
                                        quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_pkgs) > 0) {
  msg <- paste("Missing R packages:", paste(missing_pkgs, collapse = ", "),
               "\nInstall with: install.packages(c(",
               paste0('"', missing_pkgs, '"', collapse = ", "), "))")
  cat(msg, "\n", file = stderr())
  dir.create("logs", showWarnings = FALSE, recursive = TRUE)
  writeLines(msg, "logs/msgarch_error.log")
  quit(status = 1)
}

library(MSGARCH)
library(yaml)
library(jsonlite)

# ── Config ───────────────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
cfg_path <- if (length(args) >= 1) args[1] else "config/config.yaml"
cfg      <- yaml::read_yaml(cfg_path)

processed_dir <- cfg$paths$processed_data
models_dir    <- cfg$paths$models
logs_dir      <- cfg$paths$logs
seed          <- cfg$seed

dir.create(logs_dir,  showWarnings = FALSE, recursive = TRUE)
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(seed)

# ── Helper: write numpy-compatible CSV then convert in Python ─────────────────
save_results <- function(out_dir, sigma2_test, params_list, fit_info) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # sigma2_test as CSV (Python eval step reads this)
  write.csv(data.frame(sigma2_test = sigma2_test),
            file.path(out_dir, "sigma2_test.csv"), row.names = FALSE)

  # params
  writeLines(toJSON(params_list, auto_unbox = TRUE, digits = 8),
             file.path(out_dir, "params.json"))

  # fit_info
  writeLines(toJSON(fit_info, auto_unbox = TRUE, digits = 8),
             file.path(out_dir, "fit_info.json"))

  cat("Saved to", out_dir, "\n")
}

# ── MSGARCH(1,1) spec — two-state Markov, GARCH(1,1)-t per state ─────────────
make_spec <- function() {
  MSGARCH::CreateSpec(
    variance.spec  = list(model = c("sGARCH", "sGARCH")),
    distribution.spec = list(distribution = c("std", "std")),
    switch.spec    = list(do.mix = FALSE, K = 2),
    constraint.spec = list(regime.const = "nu")
  )
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for (sc in cfg$series) {
  series <- sc$name
  cat("\n══ MSGARCH:", series, "═══════════════════════════════\n")

  train_eps_f <- file.path(processed_dir, series, "train_eps.csv")
  test_eps_f  <- file.path(processed_dir, series, "test_eps.csv")

  if (!file.exists(train_eps_f)) {
    cat("WARN: train_eps.csv not found for", series, "— skipping.\n")
    next
  }

  train_eps <- read.csv(train_eps_f)[, 1]
  test_eps  <- read.csv(test_eps_f)[, 1]

  spec <- make_spec()

  # Fit on training data
  t0 <- proc.time()["elapsed"]
  fit <- tryCatch(
    MSGARCH::FitML(spec = spec, data = train_eps, do.se = TRUE,
                   ctr = list(num.init.cand = 10)),
    error = function(e) {
      msg <- paste("MSGARCH FitML failed for", series, ":", conditionMessage(e))
      cat("ERROR:", msg, "\n")
      writeLines(msg, file.path(logs_dir, paste0("msgarch_error_", series, ".log")))
      NULL
    }
  )
  fit_secs <- proc.time()["elapsed"] - t0

  if (is.null(fit)) next

  cat("Fit time:", round(fit_secs, 2), "s\n")

  # OOS conditional variance on test data
  # Use Predict() with 1-step ahead for each test observation
  tryCatch({
    # Get filtered variance on full series (train + test)
    full_eps <- c(train_eps, test_eps)
    cond_vol  <- MSGARCH::FilteredVolatility(fit, data = full_eps)
    sigma2_test <- cond_vol[(length(train_eps) + 1):length(full_eps)]^2

    # Parameters
    coef_vals <- coef(fit)
    params_list <- as.list(coef_vals)

    # Log-likelihood
    ll_val <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA)

    fit_info <- list(
      LL_insample  = ll_val,
      convergence  = 0,
      fit_seconds  = round(fit_secs, 3),
      n_obs        = length(train_eps),
      n_params     = length(coef_vals)
    )

    out_dir <- file.path(models_dir, "MSGARCH", series)
    save_results(out_dir, sigma2_test, params_list, fit_info)

    # Print summary
    cat("LL =", round(ll_val, 2), "  n_test =", length(sigma2_test), "\n")

  }, error = function(e) {
    msg <- paste("MSGARCH predict failed for", series, ":", conditionMessage(e))
    cat("ERROR:", msg, "\n")
    writeLines(msg, file.path(logs_dir, paste0("msgarch_pred_error_", series, ".log")))
  })
}

cat("\n══ R/msgarch.R complete ═════════════════════════════════════\n")
