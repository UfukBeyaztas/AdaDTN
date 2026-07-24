get_AdaDTN_scores <- function(object, X_func, X_scalar = NULL) {
  stopifnot(inherits(object, "AdaDTN_model"))
  
  X_func_sc <- mapply(function(mat, sc) {
    scale_apply_matrix(as.matrix(mat), sc)
  }, X_func, object$scalers$func, SIMPLIFY = FALSE)
  
  if (!is.null(X_scalar) && !is.null(object$scalers$scalar)) {
    X_scalar_sc <- scale_apply_matrix(as.matrix(X_scalar), object$scalers$scalar)
  } else {
    X_scalar_sc <- NULL
  }
  
  m <- object$model
  device <- object$device
  
  xf_t <- lapply(X_func_sc, function(mat) {
    torch_tensor(mat, dtype = torch_float(), device = device)
  })
  zs_t <- if (!is.null(X_scalar_sc)) {
    torch_tensor(X_scalar_sc, dtype = torch_float(), device = device)
  } else {
    NULL
  }
  
  m$eval()
  with_no_grad({
    invisible(m(xf_t, zs_t))
    sc_list <- lapply(m$score_cache, function(u) as.matrix(u$cpu()))
  })
  
  names(sc_list) <- paste0("predictor_", seq_along(sc_list))
  sc_list
}