\name{get_AdaDTN_surface}
\alias{get_AdaDTN_surface}
\title{Extract an AdaDTN Front-End Coefficient Surface}
\description{
Extracts the predictor-specific front-end coefficient surface from a fitted
AdaDTN model. The exact fitted surface is the product of the learned
normalized input-basis matrix and its trainable tensor coefficient matrix.
An optional two-directional spline smoother can be applied to the extracted
matrix for visualization.
}
\usage{
get_AdaDTN_surface(
  object,
  predictor_idx = 1,
  smooth_spar = 0.5
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
  surface is to be extracted. Predictors are indexed in the same order as
  \code{X_func} supplied to \code{\link{AdaDTN_fit}}. The default is the
  first predictor. The value is coerced with \code{as.integer} and must lie
  between 1 and \code{length(object$grid_in)}.
  }

  \item{smooth_spar}{
  Smoothing parameter supplied to \code{\link[stats]{smooth.spline}} during
  optional post-estimation smoothing. The same value is used in the response-
  and predictor-grid directions. The default is \code{0.5}. Set this
  argument to \code{NULL}, zero, or a negative value to return the exact
  unsmoothed fitted surface. A positive value is passed directly as
  \code{spar}; it should be a single finite numeric value accepted by
  \code{smooth.spline}.
  }
}
\details{
For functional predictor \eqn{p}{p}, let
\deqn{
\Phi_p
=
\left[
\bar{\boldsymbol{\phi}}_{p1},
\ldots,
\bar{\boldsymbol{\phi}}_{pK_p}
\right]
\in\mathbb{R}^{G_p\times K_p}
}{
Phi[p] =
[bar(phi)[p,1], ..., bar(phi)[p,K[p]]],
a G[p] x K[p] matrix
}
denote the learned marginal basis functions evaluated on
\code{object$grid_in[[p]]}. Every column is normalized using the
predictor-specific trapezoidal weights. Let
\deqn{
A^{(p)}
\in
\mathbb{R}^{K_p\times M}
}{
A[p] is a K[p] x M matrix
}
be the fitted tensor coefficient matrix, where \eqn{M}{M} is the number of
locations in \code{object$grid_out}. Before optional smoothing, the function
computes
\deqn{
\widehat{B}_p
=
\Phi_p A^{(p)}
\in
\mathbb{R}^{G_p\times M}.
}{
B_hat[p] = Phi[p] A[p],
a G[p] x M matrix.
}
Its entry at predictor-grid point \eqn{s_{pg}}{s[p,g]} and response-grid
point \eqn{t_m}{t[m]} is
\deqn{
\widehat{\beta}_p(s_{pg},t_m)
=
\sum_{k=1}^{K_p}
\bar{\phi}_{pk}(s_{pg})A^{(p)}_{km}.
}{
beta_hat[p](s[p,g],t[m]) =
sum[k] bar(phi)[p,k](s[p,g]) A[p][k,m].
}

The factorization is low-rank in the predictor-domain direction:
\deqn{
\mathrm{rank}(\widehat{B}_p)
\leq
\min(K_p,G_p,M).
}{
rank(B_hat[p]) <= min(K[p], G[p], M).
}
Unlike an approach that expands the response in a prespecified basis, AdaDTN
estimates one free coefficient column for every response-grid location
through \eqn{A^{(p)}}{A[p]}. The response is subsequently predicted directly
on \code{grid_out}.

For a standardized input curve, AdaDTN computes the adaptive-basis score row
\deqn{
\boldsymbol{\xi}_{ip}^{\mathsf T}
=
X_i^{(p),\mathrm{sc}}W_p\Phi_p,
}{
xi[i,p]' = X_sc[i,p] W[p] Phi[p],
}
where \eqn{W_p}{W[p]} is the diagonal matrix of trapezoidal predictor-grid
weights. The predictor-specific feature sent toward the nonlinear head is
\deqn{
U_i^{(p)}
=
\boldsymbol{\xi}_{ip}^{\mathsf T}A^{(p)}
=
X_i^{(p),\mathrm{sc}}W_p\widehat{B}_p.
}{
U[i,p] =
xi[i,p]' A[p] =
X_sc[i,p] W[p] B_hat[p].
}
Thus, the exact unsmoothed matrix returned with
\code{smooth_spar = NULL} or a nonpositive value is the same surface used
inside the fitted network to transform predictor \eqn{p}{p}. The scores and
features in these identities can be extracted with
\code{\link{get_AdaDTN_scores}} and
\code{\link{get_AdaDTN_features}}, respectively.

The surface describes the learned linear functional projection placed at the
front of the network. At response-grid index \eqn{m}{m}, it determines how
the standardized values of predictor \eqn{p}{p} are integrated to construct
the \eqn{m}{m}-th coordinate of \eqn{U_i^{(p)}}{U[i,p]}. It is therefore a
directly inspectable diagnostic of where and how the adaptive front-end
extracts response-relevant information.

In the general AdaDTN model, the vectors
\eqn{U_i^{(1)},\ldots,U_i^{(P)}}{U[i,1], ..., U[i,P]} are concatenated with
the scalar features and passed through nonlinear dense layers. Consequently,
\eqn{\widehat{B}_p}{B_hat[p]} is not generally the derivative of the final
response prediction with respect to predictor \eqn{p}{p}, nor is it an
additive predictor contribution on the original response scale.

To see this distinction, let
\eqn{f_{\mathrm{dense}}}{f_dense} denote the subject-specific mapping from
the concatenated front-end features to the standardized response prediction,
and let \eqn{J_{ip}}{J[i,p]} be its Jacobian with respect to
\eqn{U_i^{(p)}}{U[i,p]}. A perturbation of the standardized predictor at
grid location \eqn{s_{pg}}{s[p,g]} changes the front-end feature row by
\deqn{
\frac{
\partial U_i^{(p)}
}{
\partial X_i^{(p),\mathrm{sc}}(s_{pg})
}
=
w_{pg}\widehat{B}_p[g,\,\cdot].
}{
d U[i,p] / d X_sc[i,p](s[p,g]) =
w[p,g] B_hat[p][g, ].
}
The corresponding final standardized-response derivative involves the
additional factor \eqn{J_{ip}}{J[i,p]} and may depend on the complete subject
profile:
\deqn{
\frac{
\partial\widehat{\boldsymbol{Y}}_i^{\mathrm{sc}}
}{
\partial X_i^{(p),\mathrm{sc}}(s_{pg})
}
=
J_{ip}
\left\{
w_{pg}\widehat{B}_p[g,\,\cdot]
\right\}^{\mathsf T}.
}{
d Yhat_sc[i] / d X_sc[i,p](s[p,g]) =
J[i,p] * transpose(w[p,g] B_hat[p][g, ]).
}
The extracted surface alone is therefore a pre-nonlinearity coefficient
surface.

In the special linear-head construction in which predictor-specific
\eqn{U_i^{(p)}}{U[i,p]} vectors are added directly to the response and no
nonlinear interactions are introduced, the surface reduces to the familiar
discretized function-on-function coefficient surface on the internally
standardized scale.
}
\value{
A numeric matrix with
\code{length(object$grid_in[[predictor_idx]])} rows and
\code{length(object$grid_out)} columns. Row \eqn{g}{g} corresponds to
\code{object$grid_in[[predictor_idx]][g]}, and column \eqn{m}{m}
corresponds to \code{object$grid_out[m]}.

If \code{smooth_spar} is \code{NULL} or nonpositive, the matrix is exactly
\eqn{\Phi_pA^{(p)}}{Phi[p] A[p]} up to numerical conversion from torch.
If \code{smooth_spar} is positive, the returned value is the sequentially
smoothed version described above.
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

## Exact fitted front-end surface for predictor 1.
surface_raw <- get_AdaDTN_surface(
  object = fit,
  predictor_idx = 1,
  smooth_spar = NULL
)

## Post-estimation smoothed version used for visual presentation.
surface_smooth <- get_AdaDTN_surface(
  object = fit,
  predictor_idx = 1,
  smooth_spar = 0.75
)

dim(surface_raw)
dim(surface_smooth)
max(abs(surface_raw - surface_smooth))

## Verify the exact fitted factorization.
Phi_1 <- get_AdaDTN_input_basis(
  fit,
  predictor_idx = 1
)

A_1 <- as.matrix(
  fit$model$get_A(1)$detach()$cpu()
)

surface_manual <- Phi_1 %*% A_1
max(abs(surface_raw - surface_manual))

## Display the exact surface using the actual predictor and response grids.
s <- fit$grid_in[[1]]
t <- fit$grid_out

persp(
  x = s,
  y = t,
  z = surface_raw,
  theta = 40,
  phi = 20,
  expand = 0.5,
  xlab = "s",
  ylab = "t",
  zlab = "Front-end surface"
)

## The package plotting utility can apply the same optional smoothing.
plot_AdaDTN_surface(
  fit,
  predictor_idx = 1,
  smooth_spar = 0.75
)
}
}
