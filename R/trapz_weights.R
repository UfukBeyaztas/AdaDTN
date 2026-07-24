trapz_weights <- function(grid) {
  grid <- as.numeric(grid)
  J <- length(grid)
  if (J < 2) stop("Grid must contain at least two points.")
  h <- diff(grid)
  w <- numeric(J)
  w[1] <- h[1] / 2
  w[J] <- h[J - 1] / 2
  if (J > 2) {
    for (j in 2:(J - 1)) {
      w[j] <- (h[j - 1] + h[j]) / 2
    }
  }
  w
}
