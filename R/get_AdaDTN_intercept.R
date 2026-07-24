get_AdaDTN_intercept <- function(object) {
  stopifnot(inherits(object, "AdaDTN_model"))
  m <- object$model
  beta0_sc <- as.numeric(m$intercept_fun$detach()$cpu())
  beta0 <- beta0_sc * object$scalers$Y$scale + object$scalers$Y$center
  setNames(beta0, paste0("t", seq_along(beta0)))
}
