#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- normalizePath(sub(file_arg, "", args[grep(file_arg, args)]), mustWork = FALSE)
script_dir <- dirname(script_path)
repo_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = TRUE)

alpha_raw_path <- file.path(repo_root, "reports/generated_outputs/06_neurips_end_to_end_diagnostics/predictive_alpha_raw.csv")
alpha_summary_path <- file.path(repo_root, "reports/generated_outputs/06_neurips_end_to_end_diagnostics/predictive_alpha_summary.csv")
selection_summary_path <- file.path(repo_root, "reports/generated_outputs/07_drift_incorporated_selection/selection_summary.csv")
out_dir <- file.path(repo_root, "NeurIPS_Paper", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_pdf <- file.path(out_dir, "06_drift_extension_evidence.pdf")

alpha_raw <- read.csv(alpha_raw_path, stringsAsFactors = FALSE)
alpha_summary <- read.csv(alpha_summary_path, stringsAsFactors = FALSE)
selection_summary <- read.csv(selection_summary_path, stringsAsFactors = FALSE)

label_for_sample <- function(x) {
  labels <- c(
    "real_eval" = "Real",
    "iid_signs" = "IID",
    "markov_signs_p0.65" = "p=.65",
    "markov_signs_p0.75" = "p=.75",
    "markov_signs_p0.85" = "p=.85",
    "markov_signs_p0.90" = "p=.90"
  )
  unname(labels[x])
}

sample_order <- c("real_eval", "iid_signs", "markov_signs_p0.65", "markov_signs_p0.75", "markov_signs_p0.85", "markov_signs_p0.90")
sample_cols <- c(
  "real_eval" = "#4D4D4D",
  "iid_signs" = "#6BAED6",
  "markov_signs_p0.65" = "#74C476",
  "markov_signs_p0.75" = "#FDB863",
  "markov_signs_p0.85" = "#E34A33",
  "markov_signs_p0.90" = "#B2182B"
)

selector_order <- c("oracle_test_mse", "drift_penalized", "stylized_only", "random")
selector_labels <- c(
  "oracle_test_mse" = "Oracle",
  "drift_penalized" = "Drift",
  "stylized_only" = "Stylized",
  "random" = "Random"
)
selector_cols <- c(
  "oracle_test_mse" = "#969696",
  "drift_penalized" = "#2B8CBE",
  "stylized_only" = "#F16913",
  "random" = "#756BB1"
)

pdf(out_pdf, width = 7.0, height = 5.15, useDingbats = FALSE)
op <- par(
  mfrow = c(2, 2),
  mar = c(4.0, 3.6, 2.0, 0.8),
  oma = c(0.1, 0.1, 0.1, 0.1),
  mgp = c(2.0, 0.65, 0),
  tcl = -0.25,
  family = "Helvetica",
  cex = 0.78,
  cex.main = 0.94
)

plot(
  alpha_raw$pooled_delta,
  alpha_raw$strategy_sharpe,
  pch = 16,
  cex = 0.32,
  col = adjustcolor(sample_cols[alpha_raw$sample_id], alpha.f = 0.25),
  xlab = expression(hat(delta)),
  ylab = "Held-out Sharpe",
  main = "A. Drift tracks artificial alpha"
)
fit <- lm(strategy_sharpe ~ pooled_delta, data = alpha_raw)
abline(fit, col = "#252525", lwd = 1.4)
usr <- par("usr")
text(
  usr[1] + 0.05 * diff(usr[1:2]),
  usr[4] - 0.10 * diff(usr[3:4]),
  sprintf("corr = %.3f", cor(alpha_raw$pooled_delta, alpha_raw$strategy_sharpe)),
  adj = c(0, 1),
  cex = 0.86
)
legend(
  "bottomright",
  legend = label_for_sample(sample_order),
  col = sample_cols[sample_order],
  pch = 16,
  pt.cex = 0.8,
  bty = "n",
  ncol = 2,
  cex = 0.70
)

alpha_summary <- alpha_summary[match(sample_order, alpha_summary$sample_id), ]
x <- seq_along(sample_order)
means <- alpha_summary$strategy_sharpe_mean
sds <- alpha_summary$strategy_sharpe_std
bp_alpha <- barplot(
  means,
  col = sample_cols[sample_order],
  border = NA,
  ylim = c(0, max(means + sds) * 1.08),
  ylab = "Held-out Sharpe",
  main = "B. Hidden drift separates from controls",
  axisnames = FALSE
)
arrows(bp_alpha, pmax(0, means - sds), bp_alpha, means + sds, angle = 90, code = 3, length = 0.035, lwd = 0.8)
mtext(label_for_sample(sample_order), side = 1, at = bp_alpha, line = 0.8, cex = 0.68)

selection_summary <- selection_summary[match(selector_order, selection_summary$selector), ]
nmse <- selection_summary$normalized_mse_mean
nmse_sd <- selection_summary$normalized_mse_std
plot(
  seq_along(nmse),
  nmse,
  pch = 16,
  cex = 1.25,
  col = selector_cols[selector_order],
  ylim = c(0.96, 1.11),
  xlim = c(0.6, length(nmse) + 0.4),
  xaxt = "n",
  xlab = "",
  ylab = "Held-out NMSE",
  main = "C. Selection improves held-out error"
)
arrows(seq_along(nmse), nmse - nmse_sd, seq_along(nmse), nmse + nmse_sd, angle = 90, code = 3, length = 0.035, lwd = 0.9)
abline(h = 1.0, lty = 2, col = "#737373")
axis(1, at = seq_along(nmse), labels = selector_labels[selector_order], tick = FALSE, cex.axis = 0.72)

rates <- rbind(
  null = selection_summary$null_violation_rate,
  highp = selection_summary$high_persistence_rate
)
bp2 <- barplot(
  rates,
  beside = TRUE,
  col = c("#9ECAE1", "#FDD0A2"),
  border = NA,
  ylim = c(0, 1.0),
  ylab = "Selection rate",
  main = "D. Diagnostic avoids drift artifacts",
  axisnames = FALSE
)
legend("topleft", legend = c("Null violation", "p >= .75"), fill = c("#9ECAE1", "#FDD0A2"), bty = "n", cex = 0.76)
text(bp2, rates + 0.035, labels = sprintf("%.0f%%", 100 * rates), cex = 0.62)
mtext(selector_labels[selector_order], side = 1, at = colMeans(bp2), line = 0.8, cex = 0.72)

par(op)
dev.off()

cat(out_pdf, "\n")
