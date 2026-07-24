Adtn_module <- function(
    grid_in,
    grid_out,
    n_base_in,
    n_scalar = 0,
    basis_hidden_in  = c(64, 64),
    dense_hidden     = c(64, 64),
    basis_dropout    = 0.10,
    dense_dropout    = 0.10,
    basis_activation = "selu",
    dense_activation = "relu",
    scalar_embed_dim = NULL,
    lambda_in_l1     = 0.0,
    lambda_in_ortho  = 0.0,
    lambda_surface_l2 = 0.0,
    device           = NULL
) {
  
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  
  if (!is.list(grid_in)) stop("grid_in must be a list of predictor grids.")
  n_base_in <- as.integer(n_base_in)
  if (length(n_base_in) != length(grid_in)) {
    stop("length(n_base_in) must equal length(grid_in).")
  }
  if (length(dense_hidden) < 1) {
    stop("dense_hidden must contain at least one layer.")
  }
  
  dense_act_fun <- get_activation_fn(dense_activation)
  
  Mod <- nn_module(
    classname = "AdaptiveDTN",
    
    initialize = function() {
      self$device <- device
      self$n_func <- length(grid_in)
      self$n_scalar <- as.integer(n_scalar)
      self$n_base_in <- n_base_in
      self$out_dim <- length(grid_out)
      self$scalar_embed_dim <- if (is.null(scalar_embed_dim) || self$n_scalar == 0) 0 else as.integer(scalar_embed_dim)
      self$lambda_in_l1 <- lambda_in_l1
      self$lambda_in_ortho <- lambda_in_ortho
      self$lambda_surface_l2 <- lambda_surface_l2
      self$dense_act <- dense_act_fun
      
      self$t_in <- lapply(grid_in, tensor_grid, device = self$device)
      self$w_in <- lapply(grid_in, function(g) tensor_grid(trapz_weights(g), device = self$device))
      self$t_out <- tensor_grid(grid_out, device = self$device)
      
      self$BL_in <- nn_module_list(
        lapply(seq_len(self$n_func), function(p) {
          nn_module_list(
            replicate(
              self$n_base_in[p],
              basis_mlp(hidden = basis_hidden_in,
                        dropout = basis_dropout,
                        activation = basis_activation),
              simplify = FALSE
            )
          )
        })
      )
      
      # A_p has dimension (K_p x M), where M is the response-grid size.
      for (p in seq_len(self$n_func)) {
        tmp <- torch_empty(c(self$n_base_in[p], self$out_dim), device = self$device)
        nn_init_xavier_uniform_(tmp)
        self[[paste0("A_", p)]] <- nn_parameter(tmp)
      }
      
      if (self$n_scalar > 0 && self$scalar_embed_dim > 0) {
        self$scalar_dense <- nn_linear(self$n_scalar, self$scalar_embed_dim)
      }
      
      input_dim <- self$n_func * self$out_dim +
        if (self$n_scalar > 0) {
          if (self$scalar_embed_dim > 0) self$scalar_embed_dim else self$n_scalar
        } else 0
      
      dims <- c(input_dim, as.integer(dense_hidden))
      self$hidden_layers <- nn_module_list(
        lapply(seq_len(length(dims) - 1), function(i) {
          nn_linear(dims[i], dims[i + 1])
        })
      )
      self$hidden_drop <- nn_module_list(
        replicate(length(dims) - 1, nn_dropout(p = dense_dropout), simplify = FALSE)
      )
      
      self$output_layer <- nn_linear(tail(dims, 1), self$out_dim, bias = FALSE)
      
      self$intercept_fun <- nn_parameter(
        torch_zeros(self$out_dim, device = self$device)
      )
      
      self$Phi_cache <- NULL
      self$score_cache <- NULL
      self$U_cache <- NULL
    },
    
    get_A = function(p) {
      self[[paste0("A_", p)]]
    },
    
    eval_input_basis = function(p) {
      grid <- self$t_in[[p]]$unsqueeze(2)
      Phi <- torch_cat(lapply(self$BL_in[[p]], function(f) f(grid)), dim = 2)
      w <- self$w_in[[p]]
      norms <- torch_sqrt(
        torch_matmul((Phi * Phi)$transpose(1, 2), w$unsqueeze(2)) + 1e-8
      )
      Phi / norms$transpose(1, 2)
    },
    
    compute_frontend_features = function(X_func) {
      self$Phi_cache <- vector("list", self$n_func)
      self$score_cache <- vector("list", self$n_func)
      self$U_cache <- vector("list", self$n_func)
      
      out <- vector("list", self$n_func)
      
      for (p in seq_len(self$n_func)) {
        Phi_p <- self$eval_input_basis(p)
        self$Phi_cache[[p]] <- Phi_p
        
        weighted_Phi <- Phi_p * self$w_in[[p]]$unsqueeze(2)
        score_p <- torch_matmul(X_func[[p]], weighted_Phi)
        self$score_cache[[p]] <- score_p
        
        U_p <- torch_matmul(score_p, self$get_A(p))
        self$U_cache[[p]] <- U_p
        out[[p]] <- U_p
      }
      
      out
    },
    
    frontend_regularization = function() {
      pen <- torch_zeros(1, device = self$device)
      
      for (p in seq_len(self$n_func)) {
        Phi_p <- self$Phi_cache[[p]]
        A_p <- self$get_A(p)
        if (is.null(Phi_p)) next
        
        if (self$lambda_in_l1 > 0) {
          pen <- pen + self$lambda_in_l1 *
            torch_mean(
              torch_matmul(torch_abs(Phi_p)$transpose(1, 2),
                                  self$w_in[[p]]$unsqueeze(2))
            )
        }
        
        if (self$lambda_in_ortho > 0 && self$n_base_in[p] > 1) {
          Bp <- Phi_p * torch_sqrt(self$w_in[[p]])$unsqueeze(2)
          Gp <- torch_matmul(Bp$transpose(1, 2), Bp)
          Ip <- torch_eye(self$n_base_in[p], device = self$device)
          pen <- pen + self$lambda_in_ortho * torch_mean(torch_abs(Gp - Ip))
        }
        
        if (self$lambda_surface_l2 > 0) {
          pen <- pen + self$lambda_surface_l2 * torch_mean(A_p^2)
        }
      }
      
      pen
    },
    
    hidden_l2_penalty = function() {
      pen <- torch_zeros(1, device = self$device)
      for (i in seq_along(self$hidden_layers)) {
        pen <- pen + torch_sum(self$hidden_layers[[i]]$weight^2)
      }
      pen <- pen + torch_sum(self$output_layer$weight^2)
      pen
    },
    
    forward = function(X_func, X_scalar = NULL) {
      U_list <- self$compute_frontend_features(X_func)
      feat_list <- U_list
      
      if (self$n_scalar > 0L) {
        if (is.null(X_scalar)) stop("X_scalar is NULL but n_scalar > 0.")
        Z_feat <- if (self$scalar_embed_dim > 0) {
          self$dense_act(self$scalar_dense(X_scalar))
        } else {
          X_scalar
        }
        feat_list <- c(feat_list, list(Z_feat))
      }
      
      h <- torch_cat(feat_list, dim = 2)
      
      for (i in seq_along(self$hidden_layers)) {
        h <- self$hidden_layers[[i]](h)
        h <- self$dense_act(h)
        h <- self$hidden_drop[[i]](h)
      }
      
      self$output_layer(h) + self$intercept_fun
    }
  )
  
  Mod()
}
