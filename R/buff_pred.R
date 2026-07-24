buff_pred <- function(model,
                      scalers,
                      new_X_func,
                      new_X_scalar = NULL,
                      inverse_y = TRUE,
                      device = NULL,
                      batch_size = 256) {
  if (is.null(device)) device <- model$device
  
  n_new <- nrow(new_X_func[[1]])
  idx_batches <- split(seq_len(n_new), ceiling(seq_len(n_new) / batch_size))
  preds <- vector("list", length(idx_batches))
  
  model$eval()
  with_no_grad({
    for (b in seq_along(idx_batches)) {
      idx <- idx_batches[[b]]
      xf_t <- lapply(new_X_func, function(mat) {
        torch_tensor(mat[idx, , drop = FALSE], dtype = torch_float(), device = device)
      })
      zs_t <- if (!is.null(new_X_scalar)) {
        torch_tensor(new_X_scalar[idx, , drop = FALSE], dtype = torch_float(), device = device)
      } else {
        NULL
      }
      preds[[b]] <- as.matrix(model(xf_t, zs_t)$cpu())
    }
  })
  
  pred_mat <- do.call(rbind, preds)
  if (inverse_y) {
    pred_mat <- scale_inverse_matrix(pred_mat, scalers$Y)
  }
  pred_mat
}
