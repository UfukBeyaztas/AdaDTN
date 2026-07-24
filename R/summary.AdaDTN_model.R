summary.AdaDTN_model <- function(object, ...) {
  out <- list(
    n_functional_predictors = length(object$grid_in),
    n_response_grid_points = length(object$grid_out),
    n_base_in = object$n_base_in,
    split_sizes = vapply(object$split, length, numeric(1)),
    device = object$device,
    epochs_completed = nrow(object$history),
    best_validation_epoch = object$history$epoch[
      which.min(object$history$valid_loss)
    ],
    mse = c(
      train = object$mse_train,
      validation = object$mse_valid,
      calibration = object$mse_calib
    ),
    q_hat = object$q_hat,
    config = object$config
  )
  class(out) <- "summary.AdaDTN_model"
  out
}
