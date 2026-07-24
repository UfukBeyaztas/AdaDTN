scale_inverse_matrix <- function(x, scaler) {
  x <- as.matrix(x)
  sweep(sweep(x, 2, scaler$scale, "*"), 2, scaler$center, "+")
}
