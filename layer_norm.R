layer_norm <- function(d, eps = 1e-6) {
  Mod <- nn_module(
    classname = "LayerNormCustom",
    initialize = function() {
      self$alpha <- nn_parameter(torch_ones(d))
      self$beta  <- nn_parameter(torch_zeros(d))
      self$eps   <- eps
    },
    forward = function(x) {
      mu  <- x$mean(dim = -1, keepdim = TRUE)
      sig <- x$std(dim = -1, unbiased = FALSE, keepdim = TRUE)
      (x - mu) / (sig + self$eps) * self$alpha + self$beta
    }
  )
  Mod()
}