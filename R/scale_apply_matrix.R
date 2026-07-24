scale_apply_matrix <- function(x, scaler) {
  x <- as.matrix(x)
  sweep(sweep(x, 2, scaler$center, "-"), 2, scaler$scale, "/")
}
