predict.AdaDTN_model <- function(object,
                                 new_X_func,
                                 new_X_scalar = NULL,
                                 interval = c("none", "conformal"),
                                 level = 0.95,
                                 batch_size = 256,
                                 ...) {
  interval <- match.arg(interval)
  if (!inherits(object, "AdaDTN_model")) {
    stop("object must inherit from 'AdaDTN_model'.")
  }
  
  X_func_sc <- mapply(function(mat, sc) {
    scale_apply_matrix(as.matrix(mat), sc)
  }, new_X_func, object$scalers$func, SIMPLIFY = FALSE)
  
  if (!is.null(new_X_scalar)) {
    X_scalar_sc <- scale_apply_matrix(as.matrix(new_X_scalar), object$scalers$scalar)
  } else {
    X_scalar_sc <- NULL
  }
  
  pred_mean <- buff_pred(
    model = object$model,
    scalers = object$scalers,
    new_X_func = X_func_sc,
    new_X_scalar = X_scalar_sc,
    inverse_y = TRUE,
    device = object$device,
    batch_size = batch_size
  )
  
  if (interval == "none") return(pred_mean)
  
  alpha <- 1 - level
  q_use_sc <- as.numeric(quantile(object$cal_residuals_sc, probs = 1 - alpha, type = 8, names = FALSE))
  q_use <- q_use_sc * object$scalers$Y$scale
  
  lower <- sweep(pred_mean, 2, q_use, "-")
  upper <- sweep(pred_mean, 2, q_use, "+")
  
  list(mean = pred_mean, lower = lower, upper = upper)
}
