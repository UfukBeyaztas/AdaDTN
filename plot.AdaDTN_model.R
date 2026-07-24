plot.AdaDTN_model <- function(x,
                              type = c("surface", "basis", "history"),
                              predictor_idx = 1,
                              ...) {
  type <- match.arg(type)
  
  if (type == "surface") {
    return(plot_AdaDTN_surface(
      object = x,
      predictor_idx = predictor_idx,
      ...
    ))
  }
  
  if (type == "basis") {
    basis_values <- get_AdaDTN_input_basis(
      object = x,
      predictor_idx = predictor_idx
    )
    
    matplot(
      x = as.numeric(x$grid_in[[predictor_idx]]),
      y = basis_values,
      type = "l",
      lty = 1,
      xlab = "s",
      ylab = "Learned basis value",
      main = paste0("Adaptive bases: predictor ", predictor_idx),
      ...
    )
    
    return(invisible(basis_values))
  }
  
  matplot(
    x = x$history$epoch,
    y = x$history[, c("train_loss", "valid_loss")],
    type = "l",
    lty = 1,
    col = c("#1B6CA8", "#C73E1D"),
    xlab = "Epoch",
    ylab = "Loss",
    main = "AdaDTN training history",
    ...
  )
  legend(
    "topright",
    legend = c("Training", "Validation"),
    col = c("#1B6CA8", "#C73E1D"),
    lty = 1,
    bty = "n"
  )
  
  invisible(x$history)
}