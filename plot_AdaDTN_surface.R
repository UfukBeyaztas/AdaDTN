plot_AdaDTN_surface <- function(object,
                                predictor_idx = 1,
                                smooth_spar = 0.75,
                                view_theta = 40,
                                view_phi = 2,
                                surface_col = "#8EC9F3",
                                border_col = "#666666",
                                shade_fac = 0.15,
                                ...) {
  if (!requireNamespace("plot3D", quietly = TRUE))
    stop("Package 'plot3D' is required. install.packages('plot3D')")

  surf <- get_AdaDTN_surface(
    object,
    predictor_idx = predictor_idx,
    smooth_spar = smooth_spar
  )
  sx <- as.numeric(object$grid_in[[predictor_idx]])
  ty <- as.numeric(object$grid_out)

  surface_title <- as.expression(
    bquote(hat(beta)[.(predictor_idx)] * "(s,t)")
  )

  persp3D(
    x = sx,
    y = ty,
    z = surf,
    col = surface_col,
    border = border_col,
    shade = shade_fac,
    lwd = 0.5,
    theta = view_theta,
    phi = view_phi,
    expand = 0.5,
    xlab = "s",
    ylab = "t",
    zlab = "",
    ticktype = "detailed",
    cex.axis = 2,
    cex.lab = 2,
    main = NULL,
    ...
  )

  title(
    main = surface_title,
    line = 0.75,
    cex.main = 2,
    xpd = NA
  )

  invisible(list(s = sx, t = ty, surface = surf))
}
