get_AdaDTN_input_basis <- function(object, predictor_idx = 1L) {
  stopifnot(inherits(object, "AdaDTN_model"))
  m <- object$model
  p <- as.integer(predictor_idx)
  if (p < 1 || p > length(object$grid_in)) stop("Invalid predictor_idx.")
  
  m$eval()
  with_no_grad({
    Phi_p <- m$eval_input_basis(p)$cpu()
  })
  
  out <- as.matrix(Phi_p)
  colnames(out) <- paste0("basis_", seq_len(ncol(out)))
  rownames(out) <- paste0("s", seq_len(nrow(out)))
  out
}
