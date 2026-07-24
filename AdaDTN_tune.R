AdaDTN_tune <- function(grid,
                        resp,
                        func_cov,
                        scalar_cov = NULL,
                        grid_in = NULL,
                        grid_out = NULL,
                        nfolds = 5,
                        device = NULL,
                        refit = TRUE,
                        verbose = TRUE) {
  
  if (!is.data.frame(grid) || nrow(grid) == 0) {
    stop("'grid' must be a non-empty data frame.")
  }
  
  resp <- as.matrix(resp)
  K <- length(func_cov)
  n <- nrow(resp)
  
  if (is.null(grid_in)) {
    grid_in <- lapply(func_cov, function(x) seq(0, 1, length.out = ncol(x)))
  }
  if (is.null(grid_out)) {
    grid_out <- seq(0, 1, length.out = ncol(resp))
  }
  
  fold_id <- sample(rep(seq_len(nfolds), length.out = n))
  
  cv_vec <- numeric(nrow(grid))
  
  for (g in seq_len(nrow(grid))) {
    par <- lapply(grid[g, , drop = FALSE], AdaDTN_flatten1)
    
    cv_vec[g] <- AdaDTN_cv(
      resp       = resp,
      func_cov   = func_cov,
      scalar_cov = scalar_cov,
      grid_in    = grid_in,
      grid_out   = grid_out,
      par        = par,
      nfolds     = nfolds,
      fold_id    = fold_id,
      device     = device,
      verbose    = FALSE
    )
    
    if (verbose) {
      message("combination ", g, "/", nrow(grid),
              "  CV-MSE = ", format(cv_vec[g], digits = 6))
    }
  }
  
  best_row <- which.min(cv_vec)
  best_par <- lapply(grid[best_row, , drop = FALSE], AdaDTN_flatten1)
  best_par <- AdaDTN_sanitize_par(best_par, K)
  
  best_model <- NULL
  if (refit) {
    best_model <- do.call(
      AdaDTN_fit,
      c(
        list(
          X_func   = func_cov,
          X_scalar = scalar_cov,
          Y        = resp,
          grid_in  = grid_in,
          grid_out = grid_out,
          verbose  = verbose,
          device   = device
        ),
        best_par
      )
    )
  }
  
  list(
    results = data.frame(CV_MSE = cv_vec, grid, row.names = NULL),
    best_row = best_row,
    best_params = best_par,
    best_model = best_model,
    fold_id = fold_id
  )
}