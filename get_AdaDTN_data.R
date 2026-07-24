get_AdaDTN_data <- function(X_func,
                            X_scalar,
                            Y,
                            cal_prop = 0.20,
                            val_prop = 0.10) {
  n <- nrow(Y)
  sp <- subject_split(n = n, cal_prop = cal_prop, val_prop = val_prop)
  
  scaler_func <- lapply(X_func, function(mat) scale_fit_matrix(mat[sp$train, , drop = FALSE]))
  X_func_sc <- mapply(function(mat, sc) scale_apply_matrix(mat, sc), X_func, scaler_func, SIMPLIFY = FALSE)
  
  if (!is.null(X_scalar)) {
    scaler_scalar <- scale_fit_matrix(X_scalar[sp$train, , drop = FALSE])
    X_scalar_sc <- scale_apply_matrix(X_scalar, scaler_scalar)
  } else {
    scaler_scalar <- NULL
    X_scalar_sc <- NULL
  }
  
  scaler_Y <- scale_fit_matrix(Y[sp$train, , drop = FALSE])
  Y_sc <- scale_apply_matrix(Y, scaler_Y)
  
  list(
    X_func = X_func_sc,
    X_scalar = X_scalar_sc,
    Y = Y_sc,
    scalers = list(func = scaler_func, scalar = scaler_scalar, Y = scaler_Y),
    split = sp
  )
}