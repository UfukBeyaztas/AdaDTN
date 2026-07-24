\name{plot.AdaDTN_model}
\alias{plot.AdaDTN_model}
\title{Plot Diagnostics and Learned Components of an AdaDTN Model}
\description{
Provides the S3 plotting method for fitted AdaDTN models. Depending on
\code{type}, it displays a predictor-specific front-end coefficient surface,
the learned marginal basis functions for one functional predictor, or the
training and validation loss histories.
}
\usage{
\method{plot}{AdaDTN_model}(
  x,
  type = c("surface", "basis", "history"),
  predictor_idx = 1,
  ...
)
}
\arguments{
  \item{x}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{type}{
  Character string selecting the plot. Available choices are
  \code{"surface"}, \code{"basis"}, and \code{"history"}. The default is
  \code{"surface"}. Partial matching is performed by
  \code{\link[base]{match.arg}}.
  }

  \item{predictor_idx}{
  A single integer identifying the functional predictor used when
  \code{type = "surface"} or \code{type = "basis"}. Predictors are indexed
  in the same order as \code{X_func} supplied to
  \code{\link{AdaDTN_fit}}. The argument is ignored when
  \code{type = "history"}.
  }

  \item{...}{
  Additional arguments whose destination depends on \code{type}.
  For \code{"surface"}, they are passed to
  \code{\link{plot_AdaDTN_surface}} and may include its named arguments such
  as \code{smooth_spar}, \code{view_theta}, \code{view_phi},
  \code{surface_col}, \code{border_col}, and \code{shade_fac}.
  For \code{"basis"} and \code{"history"}, they are passed to
  \code{\link[graphics]{matplot}}.

  Arguments already fixed explicitly by the selected branch should not be
  repeated in \code{...}, because duplicated formal arguments can generate
  an error. See Details.
  }
}
\details{
Objects returned by \code{\link{AdaDTN_fit}} inherit from
\code{"AdaDTN_model"}. Calling \code{plot(x)} dispatches to this method and,
because the first value of \code{type} is \code{"surface"}, produces the
front-end surface plot for the first functional predictor by default.

The method applies \code{match.arg(type)} before plotting. Hence an
unambiguous abbreviation such as \code{type = "hist"} is accepted, whereas
an invalid or ambiguous value raises an error.

The method is a plotting and extraction interface only. It does not refit the
network, update parameters, recompute training losses, alter calibration
residuals, or modify the stored data split.

With \code{type = "surface"}, the method calls
\preformatted{
plot_AdaDTN_surface(
  object = x,
  predictor_idx = predictor_idx,
  ...
)
}
and immediately returns its result. That function obtains the
predictor-specific front-end surface from
\code{\link{get_AdaDTN_surface}} and draws it with
\code{\link[plot3D]{persp3D}}. The package \pkg{plot3D} must therefore be
installed for this mode.

For predictor \eqn{p}{p}, the exact fitted surface is
\deqn{
\widehat{B}_p
=
\Phi_pA^{(p)},
\qquad
\widehat{\beta}_p(s_{pg},t_m)
=
\sum_{k=1}^{K_p}
\bar{\phi}_{pk}(s_{pg})A^{(p)}_{km}.
}{
B_hat[p] = Phi[p] A[p],
and beta_hat[p](s[p,g],t[m]) =
sum[k] bar(phi)[p,k](s[p,g]) A[p][k,m].
}
It describes the linear tensor projection placed before the nonlinear dense
head. With a nonlinear head, the displayed surface is not generally the
final marginal derivative or an additive contribution on the original
response scale.

Unless overridden through \code{...}, the delegated plotting function uses
\code{smooth_spar = 0.5}. It therefore displays a sequentially
spline-smoothed version of the exact fitted matrix. Smoothing is a
post-estimation graphical operation and does not change model predictions.
Set \code{smooth_spar = NULL} or a nonpositive value to display the exact
matrix \eqn{\Phi_pA^{(p)}}{Phi[p] A[p]}. Full details of the index-based
smoothing and surface scale are provided in
\code{\link{get_AdaDTN_surface}}.

Named surface arguments should use the interface of
\code{plot_AdaDTN_surface}; for example, use \code{surface_col} rather than
passing a second \code{col} argument through its final \code{...}. Arguments
that duplicate values already fixed in \code{plot_AdaDTN_surface}, such as
\code{x}, \code{y}, \code{z}, or \code{zlim}, can be matched more than once
in the eventual \code{persp3D} call and cause an error.

With \code{type = "basis"}, the method first calls
\code{\link{get_AdaDTN_input_basis}} for the selected predictor. If
\eqn{G_p}{G[p]} is its number of grid points and \eqn{K_p}{K[p]} its number
of adaptive basis functions, the extracted matrix is
\deqn{
\Phi_p
=
\left[
\bar{\boldsymbol{\phi}}_{p1},
\ldots,
\bar{\boldsymbol{\phi}}_{pK_p}
\right]
\in\mathbb{R}^{G_p\times K_p}.
}{
Phi[p] =
[bar(phi)[p,1], ..., bar(phi)[p,K[p]]],
a G[p] x K[p] matrix.
}
The columns are plotted against the actual numerical predictor grid in
\code{x$grid_in[[predictor_idx]]}. The branch fixes
\code{type = "l"}, \code{lty = 1}, the axis labels, and the title
\code{"Adaptive bases: predictor p"}. Additional nonconflicting graphical
arguments, such as \code{col}, \code{lwd}, \code{ylim}, or \code{log}, can
be supplied through \code{...}.

The displayed basis functions are the individually trapezoidally normalized
functions used by the fitted model. They are not additionally smoothed for
plotting. The orthogonality penalty used during fitting encourages, but does
not enforce, zero weighted inner products between distinct columns.

Individual basis columns remain subject to sign, permutation, and rotation
ambiguities. Their shapes show the response-adaptive directions learned by
the model, but a particular numbered basis should not be assigned an
intrinsic meaning across independently trained fits. The learned subspace and
the rotation-invariant surface are more stable interpretive targets.

Do not repeat \code{type}, \code{lty}, \code{xlab}, \code{ylab}, or
\code{main} in \code{...} for this branch, because these are already supplied
explicitly to \code{matplot}.

With \code{type = "history"}, the method plots
\code{x$history$train_loss} and \code{x$history$valid_loss} against
\code{x$history$epoch}. It uses solid lines with colors
\code{"#1B6CA8"} and \code{"#C73E1D"}, respectively, and adds a
top-right legend labeled \code{"Training"} and \code{"Validation"}.

The two curves have related but not identical definitions:
\itemize{
  \item \code{train_loss} is the within-epoch mean of the mini-batch
  training objectives. Each mini-batch objective contains standardized-scale
  response MSE, enabled adaptive-basis and surface penalties, and
  \code{l2_hidden} times the dense/output weight penalty.
  \item \code{valid_loss} is an unpenalized standardized-response MSE,
  evaluated with the network in evaluation mode.
}
Accordingly, the vertical separation between the training and validation
curves is not a pure generalization gap: the training curve includes
regularization terms and is evaluated with training-mode dropout, whereas
the validation curve excludes the penalties and disables dropout.

Both stored quantities are averages of batch-level scalar losses. If the
last batch is smaller, it receives the same weight as a full batch in the
reported epoch summary. Neither curve is expressed on the original response
scale.

The history contains one row for every completed epoch. Early stopping is
triggered after \code{x$config$patience} consecutive epochs without a strict
improvement in validation loss, and the fitted model restores the checkpoint
having the smallest observed validation loss. The plot does not mark that
best epoch automatically. If training continued through later nonimproving
epochs before stopping, the parameters stored in \code{x$model} correspond
to the best checkpoint rather than necessarily to the final row in the
history.

The branch fixes \code{type}, \code{lty}, \code{col}, \code{xlab},
\code{ylab}, and \code{main} in its \code{matplot} call. These arguments
should not be duplicated in \code{...}. Other nonconflicting arguments such
as \code{ylim}, \code{xlim}, \code{lwd}, or \code{log} may be supplied.
The separately drawn legend always uses the fixed colors and line types
shown above and does not receive \code{...}.
}
\value{
The method is called primarily for its graphical side effect. Its invisible
return value depends on \code{type}:
\describe{
  \item{\code{"surface"}}{
  The value returned by \code{\link{plot_AdaDTN_surface}}: an invisible list
  with components \code{s}, \code{t}, and \code{surface}. The first two are
  the predictor and response grids, and \code{surface} is the displayed
  matrix after any requested smoothing.
  }

  \item{\code{"basis"}}{
  The numeric basis matrix returned by
  \code{\link{get_AdaDTN_input_basis}}, invisibly. It has one row per
  selected predictor-grid point and one column per learned basis function.
  }

  \item{\code{"history"}}{
  The \code{x$history} data frame, invisibly, with columns \code{epoch},
  \code{lr}, \code{train_loss}, and \code{valid_loss}.
  }
}
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

## Default S3 plot: smoothed surface for the first predictor.
surface_info <- plot(fit)

## Exact unsmoothed surface with a different viewing angle.
surface_exact <- plot(
  fit,
  type = "surface",
  predictor_idx = 1,
  smooth_spar = NULL,
  view_theta = 50,
  view_phi = 15
)

names(surface_exact)
dim(surface_exact$surface)

## Learned adaptive basis functions for the second predictor.
basis_values <- plot(
  fit,
  type = "basis",
  predictor_idx = 2,
  col = seq_len(fit$n_base_in[2]),
  lwd = 2
)

dim(basis_values)

## Penalized training loss and unpenalized validation MSE.
history_values <- plot(
  fit,
  type = "history",
  lwd = 2
)

head(history_values)

## Identify the epoch whose validation loss supplied the restored checkpoint.
best_epoch <- history_values$epoch[
  which.min(history_values$valid_loss)
]
best_epoch
}
}
