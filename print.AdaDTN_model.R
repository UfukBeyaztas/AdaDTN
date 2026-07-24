print.AdaDTN_model <- function(x, ...) {
  cat("Adaptive deep tensor network fit\n")
  cat("Functional predictors:", length(x$grid_in), "\n")
  cat("Response-grid points:", length(x$grid_out), "\n")
  cat("Adaptive bases:", paste(x$n_base_in, collapse = ", "), "\n")
  cat("Device:", x$device, "\n")
  cat("Epochs completed:", nrow(x$history), "\n")
  cat("Training MSE:", format(x$mse_train, digits = 6), "\n")
  cat("Validation MSE:", format(x$mse_valid, digits = 6), "\n")
  cat("Calibration MSE:", format(x$mse_calib, digits = 6), "\n")
  invisible(x)
}