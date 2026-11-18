# Run the screen production impact model from R.
#
# The model is written in Python; reticulate runs it inside the R session and returns pandas tables
# as R data frames. Run this script from the repository folder, or set REPO below.
#
#   install.packages("reticulate")   # once
#   Rscript run_in_r.R
#
# The first run creates a Python environment called "screen-impact" with numpy, pandas and
# matplotlib, which takes a minute or two. Later runs reuse it. Run the script in a fresh R session
# (in RStudio: Session > Restart R), because reticulate can only connect to one Python per session.

library(reticulate)

REPO <- getwd()                       # path to the screen-impact-model folder
ENV  <- "screen-impact"
PKGS <- c("numpy", "pandas", "matplotlib")

if (!virtualenv_exists(ENV)) {
  # Needs a Python 3.9 or later on the machine; if none is found, run reticulate::install_python() first
  virtualenv_create(ENV, packages = PKGS)
}
use_virtualenv(ENV, required = TRUE)
if (!all(vapply(PKGS, py_module_available, logical(1)))) {
  stop("The '", ENV, "' environment is missing packages. Run reticulate::virtualenv_install('", ENV,
       "', c('numpy', 'pandas', 'matplotlib')), then restart R.")
}

py_run_string(sprintf("import sys; sys.path.insert(0, r'%s')", REPO))
io     <- import("io_footprint")
slate  <- import("slate")
fiscal <- import("fiscal")

# Convert a pandas table to an R data frame, keeping the row labels. Recent versions of reticulate
# do this automatically; this helper also works with older versions and pandas 3.
to_r <- function(df) {
  if (!inherits(df, "python.builtin.object")) return(df)
  d <- df$reset_index()
  cols <- d$to_dict("list")
  out <- as.data.frame(lapply(cols, function(v) unlist(lapply(v, function(e) if (is.null(e)) NA else e))),
                       check.names = FALSE)
  rownames(out) <- out[[1]]
  out[-1]
}

# 1. Check the engine reproduces the ONS published results
validation <- to_r(io$FootprintModel()$validate())
print(table(validation$status))                        # 16 PASS and 1 info line

# 2. Run a slate of productions from a cost-report file, with direct jobs reported from payroll
SLATE   <- file.path(REPO, "examples", "illustrative_slate.csv")
PAYROLL <- file.path(REPO, "examples", "illustrative_payroll.csv")
run   <- slate$run_slate(SLATE, payroll = PAYROLL)
prods <- run[[1]]
model <- run[[3]]$model
summary_tbl <- to_r(slate$summary_table(prods, model))   # one row per production
print(round(summary_tbl[, c("uk_spend", "gva_total", "fte_total", "net_cost_of_relief",
                            "gross_receipts_per_1_relief")], 2))

# 3. From gross receipts to the net fiscal impact of the credit (eligible productions only)
elig <- summary_tbl[summary_tbl$avec_eligible == TRUE & rownames(summary_tbl) != "Slate total", ]
for (a in c(0.5, 0.75, 1)) {
  nf <- fiscal$net_fiscal_impact(sum(elig$gross_receipts), sum(elig$net_cost_of_relief),
                                 additionality = a, displacement = 0.25)
  cat(sprintf("Additionality %.0f%%, displacement 25%%: net fiscal impact %.1fm\n",
              100 * a, nf[["Net fiscal impact"]]))
}

# 4. Change a setting: 2024 household saving ratio, and the 2024 tax wedge for labour receipts
cfg <- fiscal$load_config("uk")
cfg$exchequer$labour_tax_rate_on_compensation <- 0.294
alt <- slate$run_slate(SLATE, settings = slate$default_settings(cfg, saving_ratio = 0.101),
                       cfg = cfg, payroll = PAYROLL)
alt_tbl <- to_r(slate$summary_table(alt[[1]], alt[[3]]$model))
print(round(alt_tbl["Slate total", "gross_receipts_per_1_relief"], 2))

# 5. Run any spending pattern directly (£m by ONS industry code)
res <- model$run(in_house_pay = list(`J59 & J60` = 10),
                 procurement  = list(N77 = 4, I55 = 1),
                 mode = 2L)
print(round(to_r(res$summary()), 2))

# 6. Rebuild the full output pack (tables, charts and summary in outputs/)
py_run_file(file.path(REPO, "make_outputs.py"))
