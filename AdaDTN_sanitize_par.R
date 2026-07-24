AdaDTN_sanitize_par <- function(par, K) {
  
  if (!is.null(par$n_base_in)) {
    nb <- AdaDTN_flatten1(par$n_base_in)
    nb <- as.integer(nb)
    if (length(nb) == 1) nb <- rep(nb, K)
    if (length(nb) != K) {
      stop("'n_base_in' must have length 1 or length K = ", K)
    }
    par$n_base_in <- nb
  }
  
  if (!is.null(par$basis_hidden_in)) {
    par$basis_hidden_in <- as.integer(AdaDTN_flatten1(par$basis_hidden_in))
  }
  
  if (!is.null(par$dense_hidden)) {
    par$dense_hidden <- as.integer(AdaDTN_flatten1(par$dense_hidden))
  }
  
  if (!is.null(par$scalar_embed_dim)) {
    se <- AdaDTN_flatten1(par$scalar_embed_dim)
    if (length(se) == 0 || all(is.na(se))) {
      par$scalar_embed_dim <- NULL
    } else {
      par$scalar_embed_dim <- as.integer(se[1])
    }
  }
  
  num_names <- c(
    "basis_dropout", "dense_dropout",
    "lambda_in_l1", "lambda_in_ortho", "lambda_surface_l2", "l2_hidden",
    "lr", "lr_min", "cal_prop", "val_prop"
  )
  int_names <- c("epochs", "batch_size", "patience")
  
  for (nm in intersect(num_names, names(par))) {
    par[[nm]] <- as.numeric(AdaDTN_flatten1(par[[nm]]))[1]
  }
  for (nm in intersect(int_names, names(par))) {
    par[[nm]] <- as.integer(AdaDTN_flatten1(par[[nm]]))[1]
  }
  
  if (!is.null(par$basis_activation)) {
    par$basis_activation <- as.character(AdaDTN_flatten1(par$basis_activation))[1]
  }
  if (!is.null(par$dense_activation)) {
    par$dense_activation <- as.character(AdaDTN_flatten1(par$dense_activation))[1]
  }
  
  par
}