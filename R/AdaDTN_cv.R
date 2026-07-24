AdaDTN_cv <- function(resp,
                      func_cov,
                      scalar_cov = NULL,
                      grid_in = NULL,
                      grid_out = NULL,
                      par,
                      nfolds = 5,
                      fold_id = NULL,
                      device = NULL,
                      verbose = FALSE) {
  
  resp <- as.matrix(resp)
  K <- length(func_cov)
  n <- nrow(resp)
  
  if (is.null(grid_in)) {
    grid_in <- lapply(func_cov, function(x) seq(0, 1, length.out = ncol(x)))
  }
  if (is.null(grid_out)) {
    grid_out <- seq(0, 1, length.out = ncol(resp))
  }
  
  if (is.null(fold_id)) {
    fold_id <- sample(rep(seq_len(nfolds), length.out = n))
  } else {
    if (length(fold_id) != n) stop("'fold_id' must have length nrow(resp).")
  }
  
  par_s <- AdaDTN_sanitize_par(par, K)
  mse <- numeric(nfolds)
  
  for (k in seq_len(nfolds)) {
    idx_v <- fold_id == k
    idx_t <- !idx_v
    
    fit <- do.call(
      AdaDTN_fit,
      c(
        list(
          X_func   = lapply(func_cov, `[`, idx_t, , drop = FALSE),
          X_scalar = if (!is.null(scalar_cov)) scalar_cov[idx_t, , drop = FALSE] else NULL,
          Y        = resp[idx_t, , drop = FALSE],
          grid_in  = grid_in,
          grid_out = grid_out,
          verbose  = FALSE,
          device   = device
        ),
        par_s
      )
    )
    
    pred <- predict(
      fit,
      new_X_func   = lapply(func_cov, `[`, idx_v, , drop = FALSE),
      new_X_scalar = if (!is.null(scalar_cov)) scalar_cov[idx_v, , drop = FALSE] else NULL,
      interval = "none"
    )
    
    mse[k] <- mean((pred - resp[idx_v, , drop = FALSE])^2)
    
    if (verbose) {
      message("fold ", k, "/", nfolds, " CV-MSE = ",
              format(mse[k], digits = 6))
    }
  }
  
  mean(mse)
}
