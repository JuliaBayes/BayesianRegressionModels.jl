# Recreate the source article's scientific panels from the full BRM draws.
# Base R + cairo are sufficient; no plotting package installation is needed.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L)
input <- normalizePath(args[[1]], mustWork = TRUE)
output <- file.path(input, "figures")
dir.create(output, showWarnings = FALSE)
read_table <- function(name) read.delim(file.path(input, name), check.names = FALSE)
blue <- "#0B7BEC"
colors <- c("#0B7BEC", "#E67E22", "#16877A", "#984EA3")
bases <- c(1L, 2L, 19L, 20L)
save_plot <- function(name, width, height, draw) {
  png(file.path(output, paste0(name, ".png")), width = width, height = height,
      units = "in", res = 160, type = "cairo", bg = "white")
  par(family = "sans", col.axis = "#353535", col.lab = "#252525", las = 1, bty = "l")
  draw()
  dev.off()
  message("figure: ", name)
}

save_plot("hsgp_basis", 10, 4.2, function() {
  par(mfrow = c(1, 2), mar = c(4, 4.3, 2.5, 1))
  x <- seq(-1.5, 1.5, length.out = 501)
  basis <- sapply(bases, function(j) sin(pi / 3 * (x + 1.5) * j) / sqrt(1.5))
  matplot(x, basis, type = "l", lty = 1, lwd = 1.5, col = colors,
          xlab = "Scaled time", ylab = "Basis function", main = "HSGP basis functions")
  abline(v = c(-1, 1), lty = 3, col = "grey45")
  legend("topright", paste("Basis", bases), col = colors, lty = 1, bty = "n", cex = .8)
  j <- seq(1, 20, length.out = 201)
  scales <- sapply(c(-2, -1, 0), function(r)
    exp(-.25 * (j * exp(r) * pi / 3)^2 + .45946926660233633 + .5 * r))
  matplot(j, scales, type = "l", lty = 1, lwd = 2, col = colors[1:3],
          xlab = "Basis frequency", ylab = "Prior spectral SD", main = "Squared-exponential spectrum")
  legend("topright", c("log length = -2", "log length = -1", "log length = 0"),
         col = colors[1:3], lty = 1, bty = "n", cex = .8)
})

observed <- read_table("observations.tsv")
posterior <- function(label, title) {
  data <- read_table(paste0(label, "_curves.tsv"))
  save_plot(paste0(label, "_posterior"), 10, 3.2, function() {
    par(mfrow = c(1, 2), mar = c(4.1, 4.4, 1.2, 1), oma = c(0, 0, 2, 0))
    for (predictor in c("mu", "log_sigma")) {
      d <- data[data$predictor == predictor, ]
      d <- d[order(d$time), ]
      noise <- predictor == "log_sigma"
      yr <- range(d$q05, d$q95, if (!noise) observed$acceleration_scaled else NULL)
      stopifnot(all(is.finite(yr)), !noise || min(yr) > 0)
      plot(d$time, d$q50, type = "n", ylim = yr, log = if (noise) "y" else "",
           xlab = "Time after impact (ms)", ylab = if (noise) "Conditional SD (scaled)" else "Acceleration (scaled)",
           panel.first = grid(col = "#EFEFEF"))
      for (pair in list(c("q05", "q95"), c("q10", "q90"), c("q25", "q75"))) {
        polygon(c(d$time, rev(d$time)), c(d[[pair[[1]]]], rev(d[[pair[[2]]]])),
                col = adjustcolor(blue, alpha.f = .23), border = NA)
      }
      lines(d$time, d$q50, col = blue, lwd = 1.6)
      if (!noise) points(observed$time, observed$acceleration_scaled, pch = 16,
                         cex = .45, col = adjustcolor("black", alpha.f = .65))
      legend("topright", if (noise) "Inferred conditional SD" else c("Inferred mean", "Observations"),
             col = if (noise) blue else c(blue, "black"),
             lty = if (noise) 1 else c(1, NA), pch = if (noise) NA else c(NA, 16),
             bty = "n", cex = .72)
    }
    mtext(title, side = 3, outer = TRUE, cex = 1.2)
  })
}
posterior("noncentered", "Noncentered pilot")

centering <- read_table("centeredness.tsv")
scatter <- function(label, geometry, title) {
  gp <- lapply(c("mu", "log_sigma"), function(p) read_table(paste0(label, "_", p, "_weights.tsv")))
  save_plot(paste0(geometry, "_scatter"), 10, 10, function() {
    par(mfrow = c(4, 4), mar = c(3.4, 3.6, 2.1, .6), oma = c(.5, .7, 2, 0), cex = .8)
    for (basis in bases) {
      for (g in seq_along(gp)) {
        d <- gp[[g]][gp[[g]]$basis == basis, ]
        weights <- switch(geometry,
          centered = d$physical_weight,
          optimal = d$coordinate * exp(d$log_spectral_scale *
                     centering[centering$basis == basis, if (g == 1) "mean" else "log_scale"]),
          d$coordinate)
        for (hyper in c("sigma", "rho")) {
          plot(d[[hyper]], weights, log = "x", pch = 16, cex = .40,
               col = adjustcolor(colors[[match(basis, bases)]], alpha.f = .12),
               xlab = paste(if (g == 1) "Mean GP" else "Log-SD GP",
                            if (hyper == "sigma") "marginal SD" else "length scale"),
               ylab = paste(if (g == 1) "Mean GP weight" else "Log-SD GP weight", basis),
               panel.first = grid(col = "#EFEFEF"))
        }
      }
    }
    mtext(title, side = 3, outer = TRUE, cex = 1.2, font = 2)
  })
}
scatter("noncentered", "noncentered", "Noncentered pilot coordinates")
scatter("noncentered", "centered", "Centered geometry — transformed pilot draws, not another fit")
scatter("noncentered", "optimal", "Selected partial geometry — transformed pilot draws")

loss <- read_table("loss_profiles.tsv")
save_plot("loss_profiles", 10, 4.2, function() {
  par(mfrow = c(1, 2), mar = c(4, 4, 2.2, 1))
  for (gp in c("mu", "log_sigma")) {
    plot(0:1, 0:1, type = "n", xlab = "Centeredness", ylab = "Loss (each curve rescaled to [0, 1])",
         main = if (gp == "mu") "Mean GP" else "Log-SD GP")
    for (b in seq_along(bases)) {
      d <- loss[loss$predictor == gp & loss$basis == bases[[b]], ]
      ok <- is.finite(d$loss) & d$admissible
      yr <- range(d$loss[ok])
      values <- rep(NA_real_, nrow(d))
      values[ok] <- if (diff(yr) == 0) 0 else (d$loss[ok] - yr[[1]]) / diff(yr)
      lines(d$centeredness, values, lwd = 2, col = colors[[b]])
    }
    legend("topright", paste("Basis", bases), col = colors, lty = 1, bty = "n", cex = .8)
  }
})
save_plot("selected_centeredness", 8, 4.3, function() {
  par(mar = c(4, 4, 2, 1))
  matplot(centering$basis, centering[, c("mean", "log_scale")], type = "o",
          pch = c(16, 17), lty = 1, lwd = 2, col = colors[1:2], ylim = c(0, 1),
          xlab = "Basis frequency", ylab = "Selected centeredness", main = "One centering per basis function")
  legend("topright", c("Mean GP", "Log-SD GP"), col = colors[1:2], pch = c(16, 17), lty = 1, bty = "n")
})

if (file.exists(file.path(input, "partial_curves.tsv"))) {
  posterior("partial", "Fresh selected-partial fit")
  scatter("partial", "partial", "Selected-partial refit — newly sampled coordinates")
}
if (file.exists(file.path(input, "online_curves.tsv"))) posterior("online", "Online adaptive centering")
