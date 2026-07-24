\name{plot_AdaDTN_surface}
\alias{plot_AdaDTN_surface}
\title{Plot an AdaDTN Front-End Coefficient Surface}
\description{
Draws a three-dimensional perspective plot of the predictor-specific
front-end coefficient surface from a fitted AdaDTN model. The plotted surface
may be the exact fitted tensor product or an optionally smoothed
post-estimation version.
}
\usage{
plot_AdaDTN_surface(
  object,
  predictor_idx = 1,
  smooth_spar = 0.75,
  view_theta = 40,
  view_phi = 2,
  surface_col = "#8EC9F3",
  border_col = "#666666",
  shade_fac = 0.15,
  ...
)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{predictor_idx}{
  A single integer identifying the functional predictor whose front-end
  surface is plotted. Predictors are indexed in the same order as
  \code{X_func} supplied to \code{\link{AdaDTN_fit}}. The default is the
  first predictor.
  }

  \item{smooth_spar}{
  Smoothing parameter passed to \code{\link{get_AdaDTN_surface}}. The
  default \code{0.75} applies sequential spline smoothing in the response-
  and predictor-index directions before plotting. Set this argument to
  \code{NULL}, zero, or a negative value to plot the exact fitted surface
  \eqn{\Phi_pA^{(p)}}{Phi[p] A[p]}.
  }

  \item{view_theta}{
  Numeric azimuthal viewing angle, in degrees, passed as \code{theta} to
  \code{\link[plot3D]{persp3D}}. The default is 40.
  }

  \item{view_phi}{
  Numeric colatitude or elevation viewing angle, in degrees, passed as
  \code{phi} to \code{\link[plot3D]{persp3D}}. The default is 2.
  }

  \item{surface_col}{
  Color specification passed as \code{col} to \code{persp3D} for the
  surface facets. The default is the light blue color
  \code{"#8EC9F3"}.
  }

  \item{border_col}{
  Color specification passed as \code{border} to \code{persp3D} for the
  facet borders. The default is \code{"#666666"}.
  }

  \item{shade_fac}{
  Numeric shading value passed as \code{shade} to \code{persp3D}. The
  default is \code{0.15}.
  }

  \item{...}{
  Additional graphical arguments passed to \code{persp3D}. Arguments already
  supplied explicitly by this function must not be repeated in \code{...};
  duplicated formal arguments can cause an error. See Details.
  }
}
\details{
For the selected functional predictor \eqn{p}{p}, the exact AdaDTN
front-end surface is
\deqn{
\widehat{B}_p
=
\Phi_pA^{(p)}
\in
\mathbb{R}^{G_p\times M},
}{
B_hat[p] = Phi[p] A[p],
a G[p] x M matrix,
}
where \eqn{\Phi_p}{Phi[p]} contains the learned, trapezoidally normalized
marginal basis functions and
\eqn{A^{(p)}\in\mathbb{R}^{K_p\times M}}{
A[p] is a K[p] x M coefficient matrix}. Its entries are
\deqn{
\widehat{\beta}_p(s_{pg},t_m)
=
\sum_{k=1}^{K_p}
\bar{\phi}_{pk}(s_{pg})A^{(p)}_{km}.
}{
beta_hat[p](s[p,g],t[m]) =
sum[k] bar(phi)[p,k](s[p,g]) A[p][k,m].
}

The function delegates extraction to
\code{\link{get_AdaDTN_surface}}. Therefore, the matrix plotted is:
\itemize{
  \item the exact fitted surface when \code{smooth_spar} is \code{NULL} or
  nonpositive; or
  \item a sequentially spline-smoothed version when \code{smooth_spar} is
  positive.
}
The default is the second case. Smoothing is performed first across the
response-grid column indices for each predictor-grid row and then across the
predictor-grid row indices for each response-grid column. The spline
coordinates are integer indices rather than the numerical values of the
grids. Full details are given in
\code{\link{get_AdaDTN_surface}}.

Smoothing is applied only to the matrix returned for display. It does not
alter the torch module, coefficient matrices, learned bases, tensor-
projection features, predictions, training history, or calibration
quantities.

The horizontal plotting coordinates are taken directly from the fitted
object:
\deqn{
x_g
=
\mathrm{grid\_in}_p[g],
\qquad
y_m
=
\mathrm{grid\_out}[m],
}{
x[g] = object$grid_in[[p]][g],
and y[m] = object$grid_out[m].
}
These vectors are coerced to numeric. The extracted
\eqn{G_p\times M}{G[p] x M} matrix is supplied as \code{z}, so its rows
correspond to predictor-grid values \eqn{s}{s} and its columns correspond to
response-grid values \eqn{t}{t}.

\code{view_theta} controls horizontal rotation around the surface, while
\code{view_phi} controls the vertical viewing angle. These values affect only
the rendering and do not change the returned surface matrix.

The function calls \code{persp3D} with the following settings fixed by its
implementation:
\itemize{
  \item facet line width \code{lwd = 0.5};
  \item vertical expansion \code{expand = 0.5};
  \item axis labels \code{xlab = "s"} and \code{ylab = "t"};
  \item an empty vertical-axis label, \code{zlab = ""};
  \item detailed ticks, \code{ticktype = "detailed"};
  \item enlarged tick and axis-label sizes,
  \code{cex.axis = 2} and \code{cex.lab = 2};
  \item vertical plotting limits \code{zlim = c(-0.3, 0.4)}; and
  \item \code{main = NULL}, because the title is added separately.
}
The facet color, border color, shading, and viewing angles are controlled by
the dedicated arguments of this function.

The fixed \code{zlim} affects the display only. Surface entries outside
\code{c(-0.3, 0.4)} remain present in the invisibly returned matrix but may
be clipped or omitted from the visible plotting range. Because \code{zlim}
is already supplied explicitly, passing another \code{zlim} through
\code{...} results in duplicate matching rather than overriding it.
Changing the vertical limit therefore requires changing this plotting
function or drawing the matrix returned by
\code{\link{get_AdaDTN_surface}} with a separate plotting call.
}
\examples{
\dontrun{

train_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "complex"
)

fit <- AdaDTN_fit(
  X_func = train_dat$x,
  X_scalar = train_dat$x.scl,
  Y = train_dat$y,
  grid_in = rep(
    list(train_dat$meta$sx),
    length(train_dat$x)
  ),
  grid_out = train_dat$meta$sy,
  n_base_in = rep(4, length(train_dat$x)),
  basis_hidden_in = c(64, 64),
  dense_hidden = c(64, 32),
  basis_dropout = 0.10,
  dense_dropout = 0.05,
  basis_activation = "tanh",
  dense_activation = "relu",
  lambda_in_l1 = 0,
  lambda_in_ortho = 1e-4,
  lambda_surface_l2 = 1e-4,
  l2_hidden = 1e-4,
  epochs = 250,
  batch_size = 64,
  lr = 1e-3,
  lr_min = 1e-5,
  patience = 35,
  cal_prop = 0.20,
  val_prop = 0.10,
  verbose = TRUE
)

## Default display: post-estimation smoothing with spar = 0.75.
displayed <- plot_AdaDTN_surface(
  object = fit,
  predictor_idx = 1
)

names(displayed)
dim(displayed$surface)
range(displayed$surface)

## Exact fitted tensor-product surface without graphical smoothing.
exact <- plot_AdaDTN_surface(
  object = fit,
  predictor_idx = 1,
  smooth_spar = NULL,
  view_theta = 50,
  view_phi = 15,
  surface_col = "#76B7E5",
  border_col = "#555555",
  shade_fac = 0.20
)

## Confirm that the returned matrix is the exact extracted surface.
surface_raw <- get_AdaDTN_surface(
  fit,
  predictor_idx = 1,
  smooth_spar = NULL
)

max(abs(exact$surface - surface_raw))

## Plot another functional predictor.
plot_AdaDTN_surface(
  object = fit,
  predictor_idx = 2,
  smooth_spar = 0.5,
  view_theta = 35,
  view_phi = 10,
  surface_col = "#A8DADC"
)
}
}
