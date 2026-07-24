basis_mlp <- function(hidden = c(64, 64),
                      dropout = 0.10,
                      activation = "selu") {
  act_fun <- get_activation_fn(activation)
  
  Mod <- nn_module(
    classname = "BasisMLP",
    initialize = function() {
      dims <- c(1, hidden, 1)
      self$act <- act_fun
      self$layers <- nn_module_list(
        lapply(seq_len(length(dims) - 1), function(i) {
          nn_linear(dims[i], dims[i + 1])
        })
      )
      self$ln <- nn_module_list(
        lapply(hidden, function(k) layer_norm(k))
      )
      self$dp <- nn_module_list(
        replicate(length(hidden), nn_dropout(p = dropout), simplify = FALSE)
      )
    },
    forward = function(x) {
      h <- x
      n_hidden <- length(self$layers) - 1
      if (n_hidden >= 1) {
        for (i in seq_len(n_hidden)) {
          z <- self$layers[[i]](h)
          z <- self$ln[[i]](z)
          h <- self$act(z)
          h <- self$dp[[i]](h)
        }
      }
      self$layers[[length(self$layers)]](h)
    }
  )
  
  Mod()
}
