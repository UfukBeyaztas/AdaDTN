smooth_surface_2d <- function(z, spar = 0.5) {
  nr <- nrow(z)
  nc <- ncol(z)
  z <- t(vapply(seq_len(nr), function(i) {
    smooth.spline(
      seq_len(nc),
      as.double(z[i, ]),
      spar = spar
    )$y
  }, numeric(nc)))
  z <- vapply(seq_len(nc), function(j) {
    smooth.spline(
      seq_len(nr),
      as.double(z[, j]),
      spar = spar
    )$y
  }, numeric(nr))
  z
}