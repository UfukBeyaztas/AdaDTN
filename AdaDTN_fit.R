AdaDTN_fit <- function(
    X_func,
    X_scalar = NULL,
    Y,
    grid_in,
    grid_out,
    n_base_in,
    basis_hidden_in  = c(64, 64),
    dense_hidden     = c(64, 64),
    basis_dropout    = 0.10,
    dense_dropout    = 0.10,
    basis_activation = "selu",
    dense_activation = "relu",
    scalar_embed_dim = NULL,
    lambda_in_l1      = 0.0,
    lambda_in_ortho   = 1e-4,
    lambda_surface_l2 = 1e-4,
    l2_hidden         = 1e-4,
    epochs           = 150,
    batch_size       = 64,
    lr               = 1e-3,
    lr_min           = 1e-5,
    patience         = 20,
    cal_prop         = 0.20,
    val_prop         = 0.10,
    device           = NULL,
    verbose          = TRUE
) {
  
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  if (verbose) cat("Using device:", device, "\n")
  
  X_func <- lapply(X_func, as.matrix)
  if (!is.null(X_scalar)) X_scalar <- as.matrix(X_scalar)
  Y <- as.matrix(Y)
  
  if (length(X_func) != length(grid_in)) {
    stop("length(X_func) must equal length(grid_in).")
  }
  if (nrow(Y) != nrow(X_func[[1]])) {
    stop("All inputs and the response must have the same number of subjects.")
  }
  
  dat <- get_AdaDTN_data(
    X_func = X_func,
    X_scalar = X_scalar,
    Y = Y,
    cal_prop = cal_prop,
    val_prop = val_prop
  )
  
  trn <- dat$split$train
  val <- dat$split$valid
  cal <- dat$split$calib
  
  n_scalar <- if (is.null(X_scalar)) 0L else ncol(X_scalar)
  
  model <- Adtn_module(
    grid_in = grid_in,
    grid_out = grid_out,
    n_base_in = n_base_in,
    n_scalar = n_scalar,
    basis_hidden_in = basis_hidden_in,
    dense_hidden = dense_hidden,
    basis_dropout = basis_dropout,
    dense_dropout = dense_dropout,
    basis_activation = basis_activation,
    dense_activation = dense_activation,
    scalar_embed_dim = scalar_embed_dim,
    lambda_in_l1 = lambda_in_l1,
    lambda_in_ortho = lambda_in_ortho,
    lambda_surface_l2 = lambda_surface_l2,
    device = device
  )$to(device = device)
  
  optimizer <- optim_adam(model$parameters, lr = lr)
  best_file <- tempfile(fileext = ".pt")
  best_val <- Inf
  bad_epochs <- 0
  
  history <- data.frame(
    epoch = integer(),
    lr = numeric(),
    train_loss = numeric(),
    valid_loss = numeric(),
    stringsAsFactors = FALSE
  )
  
  make_batch <- function(idx) {
    xb_func <- lapply(dat$X_func, function(mat) {
      torch_tensor(mat[idx, , drop = FALSE], dtype = torch_float(), device = device)
    })
    
    xb_scalar <- if (!is.null(dat$X_scalar)) {
      torch_tensor(dat$X_scalar[idx, , drop = FALSE], dtype = torch_float(), device = device)
    } else {
      NULL
    }
    
    yb <- torch_tensor(dat$Y[idx, , drop = FALSE], dtype = torch_float(), device = device)
    
    list(X_func = xb_func, X_scalar = xb_scalar, Y = yb)
  }
  
  eval_mse <- function(idx) {
    model$eval()
    losses <- c()
    with_no_grad({
      batches <- split(idx, ceiling(seq_along(idx) / batch_size))
      for (b in batches) {
        bt <- make_batch(b)
        pr <- model(bt$X_func, bt$X_scalar)
        losses <- c(losses, nnf_mse_loss(pr, bt$Y)$item())
      }
    })
    mean(losses)
  }
  
  if (verbose) cat("Starting training ...\n")
  
  for (ep in seq_len(epochs)) {
    
    lr_ep <- lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * (ep - 1) / max(1, epochs - 1)))
    optimizer$param_groups[[1]]$lr <- lr_ep
    
    model$train()
    train_losses <- c()
    
    trn_shuf <- sample(trn)
    batches <- split(trn_shuf, ceiling(seq_along(trn_shuf) / batch_size))
    
    for (b in batches) {
      bt <- make_batch(b)
      pred <- model(bt$X_func, bt$X_scalar)
      
      mse_loss <- nnf_mse_loss(pred, bt$Y)
      reg_front <- model$frontend_regularization()
      reg_l2 <- l2_hidden * model$hidden_l2_penalty()
      loss <- mse_loss + reg_front + reg_l2
      
      optimizer$zero_grad()
      loss$backward()
      optimizer$step()
      
      train_losses <- c(train_losses, loss$item())
    }
    
    train_loss <- mean(train_losses)
    valid_loss <- eval_mse(val)
    
    history <- rbind(
      history,
      data.frame(epoch = ep, lr = lr_ep, train_loss = train_loss, valid_loss = valid_loss)
    )
    
    if (verbose && (ep == 1 || ep %% 10 == 0)) {
      cat(sprintf("Epoch %03d | lr = %.6f | train = %.6f | valid = %.6f\n",
                  ep, lr_ep, train_loss, valid_loss))
    }
    
    if (valid_loss < best_val) {
      best_val <- valid_loss
      bad_epochs <- 0
      torch_save(model$state_dict(), best_file)
    } else {
      bad_epochs <- bad_epochs + 1
    }
    
    if (bad_epochs >= patience) {
      if (verbose) cat("Early stopping triggered at epoch", ep, "\n")
      break
    }
  }
  
  if (file.exists(best_file)) {
    model$load_state_dict(torch_load(best_file))
  }
  
  cal_pred_sc <- buff_pred(
    model = model,
    scalers = dat$scalers,
    new_X_func = lapply(dat$X_func, function(mat) mat[cal, , drop = FALSE]),
    new_X_scalar = if (!is.null(dat$X_scalar)) dat$X_scalar[cal, , drop = FALSE] else NULL,
    inverse_y = FALSE,
    device = device,
    batch_size = batch_size
  )
  
  cal_residuals_sc <- abs(as.vector(dat$Y[cal, , drop = FALSE] - cal_pred_sc))
  q_hat <- as.numeric(quantile(cal_residuals_sc, probs = 0.95, type = 8, names = FALSE))
  
  pred_train <- buff_pred(
    model = model,
    scalers = dat$scalers,
    new_X_func = lapply(dat$X_func, function(mat) mat[trn, , drop = FALSE]),
    new_X_scalar = if (!is.null(dat$X_scalar)) dat$X_scalar[trn, , drop = FALSE] else NULL,
    inverse_y = TRUE,
    device = device,
    batch_size = batch_size
  )
  
  pred_valid <- buff_pred(
    model = model,
    scalers = dat$scalers,
    new_X_func = lapply(dat$X_func, function(mat) mat[val, , drop = FALSE]),
    new_X_scalar = if (!is.null(dat$X_scalar)) dat$X_scalar[val, , drop = FALSE] else NULL,
    inverse_y = TRUE,
    device = device,
    batch_size = batch_size
  )
  
  pred_calib <- buff_pred(
    model = model,
    scalers = dat$scalers,
    new_X_func = lapply(dat$X_func, function(mat) mat[cal, , drop = FALSE]),
    new_X_scalar = if (!is.null(dat$X_scalar)) dat$X_scalar[cal, , drop = FALSE] else NULL,
    inverse_y = TRUE,
    device = device,
    batch_size = batch_size
  )
  
  y_train <- scale_inverse_matrix(dat$Y[trn, , drop = FALSE], dat$scalers$Y)
  y_valid <- scale_inverse_matrix(dat$Y[val, , drop = FALSE], dat$scalers$Y)
  y_calib <- scale_inverse_matrix(dat$Y[cal, , drop = FALSE], dat$scalers$Y)
  
  mse_train <- mean((pred_train - y_train)^2)
  mse_valid <- mean((pred_valid - y_valid)^2)
  mse_calib <- mean((pred_calib - y_calib)^2)
  
  if (verbose) {
    cat(sprintf("Final train MSE: %.6f\n", mse_train))
    cat(sprintf("Final valid MSE: %.6f\n", mse_valid))
    cat(sprintf("Final calib MSE: %.6f\n", mse_calib))
    cat(sprintf("Conformal half-width on standardized scale: %.6f\n", q_hat))
  }
  
  fit <- list(
    model = model,
    scalers = dat$scalers,
    history = history,
    split = dat$split,
    q_hat = q_hat,
    cal_residuals_sc = cal_residuals_sc,
    device = device,
    grid_in = grid_in,
    grid_out = grid_out,
    n_base_in = n_base_in,
    mse_train = mse_train,
    mse_valid = mse_valid,
    mse_calib = mse_calib,
    config = list(
      basis_hidden_in = basis_hidden_in,
      dense_hidden = dense_hidden,
      basis_dropout = basis_dropout,
      dense_dropout = dense_dropout,
      basis_activation = basis_activation,
      dense_activation = dense_activation,
      scalar_embed_dim = scalar_embed_dim,
      lambda_in_l1 = lambda_in_l1,
      lambda_in_ortho = lambda_in_ortho,
      lambda_surface_l2 = lambda_surface_l2,
      l2_hidden = l2_hidden,
      epochs = epochs,
      batch_size = batch_size,
      lr = lr,
      lr_min = lr_min,
      patience = patience,
      cal_prop = cal_prop,
      val_prop = val_prop
    )
  )
  class(fit) <- "AdaDTN_model"
  fit
}