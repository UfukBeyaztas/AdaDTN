get_AdaDTN_surface <- function(object,
                               predictor_idx = 1L,
                               smooth_spar = 0.5) {
  if (!inherits(object, "AdaDTN_model"))
    stop("'object' must be an object returned by AdaDTN_fit().")
  
  p <- as.integer(predictor_idx)
  if (p < 1L || p > length(object$grid_in))
    stop("Invalid predictor_idx.")
  
  m <- object$model
  m$eval()
  
  surf_tensor <- torch::with_no_grad({
    Phi_p <- m$eval_input_basis(p)$cpu()
    A_p <- m$get_A(p)$detach()$cpu()
    torch::torch_matmul(Phi_p, A_p)
  })
  
  surf <- matrix(
    as.double(surf_tensor),
    nrow = surf_tensor$size(1),
    ncol = surf_tensor$size(2)
  )
  
  if (!is.null(smooth_spar) && smooth_spar > 0)
    surf <- smooth_surface_2d(surf, spar = smooth_spar)
  
  surf
}
