scale_fit_matrix <- function(x) {
  x <- as.matrix(x)
  center <- colMeans(x)
  scale  <- apply(x, 2, sd)
  scale[!is.finite(scale) | scale == 0] <- 1
  list(center = center, scale = scale)
}