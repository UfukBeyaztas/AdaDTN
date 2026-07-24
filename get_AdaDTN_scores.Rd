\name{get_AdaDTN_scores}
\alias{get_AdaDTN_scores}
\title{Extract AdaDTN Adaptive-Basis Scores}
\description{
Extracts subject-specific projection scores for the learned marginal basis
functions of every functional predictor in a fitted AdaDTN model. Supplied
curves are standardized using the fitted training scalers and integrated
against the learned, normalized basis functions by the same trapezoidal
quadrature rule used in the model's forward pass.
}
\usage{
get_AdaDTN_scores(object, X_func, X_scalar = NULL)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{X_func}{
  A list of numeric matrices containing the functional predictors for the
  subjects whose adaptive-basis scores are required. Element
  \code{X_func[[p]]} must correspond to the same predictor, observation
  grid, and column ordering used to fit \code{object}. Rows are subjects and
  columns are functional evaluations. If there are \eqn{n_*}{n_new}
  supplied subjects and predictor \eqn{p}{p} was observed at
  \eqn{G_p}{G[p]} grid points, its matrix must have dimension
  \eqn{n_* \times G_p}{n_new x G[p]}.
  }

  \item{X_scalar}{
  An optional numeric matrix containing scalar predictors for the same
  subjects, with subjects in rows and scalar covariates in the column order
  used for model fitting. Scalar values do not enter the adaptive-basis
  score formula directly. They must nevertheless be supplied when
  \code{object} was fitted with scalar predictors because the current
  implementation executes a complete forward pass to populate the model's
  score cache. If \code{object} was fitted without scalar predictors, this
  argument is ignored.
  }
}
\details{
Each supplied functional predictor is transformed using the column-specific
centers and scales stored in \code{object$scalers$func}. For predictor
\eqn{p}{p}, subject \eqn{i}{i}, and grid point \eqn{g}{g},
\deqn{
X_{ip}^{\mathrm{sc}}(s_{pg})
=
\frac{
X_{ip}(s_{pg})-\widehat{\mu}_{pg,\mathcal{T}}
}{
\widehat{\sigma}_{pg,\mathcal{T}}
}.
}{
X_sc[i,p](s[p,g]) =
(X[i,p](s[p,g]) - mean_train[p,g]) / sd_train[p,g].
}
The centers and scales were estimated using only the internal training
subjects in \code{\link{AdaDTN_fit}}. They are reused without modification,
so training, validation, calibration, and new-subject scores share the same
fitted coordinate system.

This pointwise standardization is important for interpretation. The returned
scores are projections of the \emph{standardized} curves, not the curves in
their original measurement units. A grid location with large raw variability
is therefore not automatically given more influence solely because of its
measurement scale.

When both \code{X_scalar} and the fitted scalar scaler are present, scalar
columns are standardized in the same way. The standardized scalar matrix is
needed only to complete the model forward pass; at fixed fitted parameters,
it does not change the values stored in the functional score cache.

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
be the learned basis matrix evaluated on
\code{object$grid_in[[p]]}. Column \eqn{k}{k} is produced by its own basis
micro-network and is normalized using the predictor-specific trapezoidal
weights:
\deqn{
\bar{\phi}_{pk}(s_{pg})
=
\frac{
\widetilde{\phi}_{pk}(s_{pg})
}{
\left\{
\sum_{r=1}^{G_p}
w_{pr}\widetilde{\phi}_{pk}(s_{pr})^2
+10^{-8}
\right\}^{1/2}
}.
}{
bar(phi)[p,k](s[p,g]) =
phi_tilde[p,k](s[p,g]) /
sqrt(sum[r] w[p,r] phi_tilde[p,k](s[p,r])^2 + 1e-8).
}
The basis functions are learned jointly with the complete prediction network;
they are not fixed B-spline, Fourier, or functional principal-component
functions chosen before seeing the response.

Normalization makes the weighted squared norm of each basis column close to
one. It does not force distinct columns to be exactly orthogonal. Approximate
orthogonality is encouraged during fitting by the
\code{lambda_in_ortho} penalty on the weighted Gram matrix
\eqn{\Phi_p^{\mathsf T}W_p\Phi_p}{Phi[p]' W[p] Phi[p]}.

Let
\deqn{
W_p
=
\mathrm{diag}(w_{p1},\ldots,w_{pG_p})
}{
W[p] = diag(w[p,1], ..., w[p,G[p]])
}
be the trapezoidal weight matrix constructed from the stored predictor grid.
For all supplied subjects, the function returns the matrix
\deqn{
\Xi_{p,*}
=
X_{p,*}^{\mathrm{sc}}W_p\Phi_p
\in\mathbb{R}^{n_*\times K_p}.
}{
Xi[p,new] =
X_sc[p,new] W[p] Phi[p],
an n_new x K[p] matrix.
}
Its \eqn{(i,k)}{(i,k)} entry is
\deqn{
\xi_{ipk}
=
\sum_{g=1}^{G_p}
w_{pg}
X_{ip}^{\mathrm{sc}}(s_{pg})
\bar{\phi}_{pk}(s_{pg}).
}{
xi[i,p,k] =
sum[g] w[p,g] X_sc[i,p](s[p,g])
bar(phi)[p,k](s[p,g]).
}
This is the trapezoidal approximation to the functional inner product between
the standardized predictor curve and the learned marginal basis function.
For an irregular increasing grid, the weights reflect the actual adjacent
grid spacings.

The score map is linear in a new standardized functional predictor once the
fitted basis functions are held fixed. The complete AdaDTN regression
operator is nevertheless nonlinear because the basis system was learned
jointly from the response and the resulting score-derived features are
processed by a nonlinear dense head.
}
\value{
A named list with one element for every functional predictor. For predictor
\eqn{p}{p}, element \code{p} is a numeric matrix with one row per supplied
subject and \code{object$n_base_in[p]} columns. If \eqn{n_*}{n_new}
subjects are supplied, its dimension is
\eqn{n_*\times K_p}{n_new x K[p]}.

List elements are named \code{"predictor_1"}, \code{"predictor_2"}, and so
forth. The returned matrices receive no explicit row or column names; column
\eqn{k}{k} corresponds to \code{"basis_k"} in the matrix returned by
\code{\link{get_AdaDTN_input_basis}} for the same predictor.
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

test_dat <- simulate_AdaDTN_data(
  n = 100,
  j = 101,
  model = "complex"
)

scores <- get_AdaDTN_scores(
  object = fit,
  X_func = test_dat$x,
  X_scalar = test_dat$x.scl
)

names(scores)
length(scores)
lapply(scores, dim)
head(scores$predictor_1)

## Verify the quadrature formula for the first predictor.
s <- fit$grid_in[[1]]
d <- diff(s)
w <- numeric(length(s))
w[1] <- d[1] / 2
w[length(s)] <- d[length(d)] / 2
if (length(s) > 2) {
  w[2:(length(s) - 1)] <-
    (d[-length(d)] + d[-1]) / 2
}

X1_sc <- sweep(
  sweep(
    test_dat$x[[1]],
    2,
    fit$scalers$func[[1]]$center,
    "-"
  ),
  2,
  fit$scalers$func[[1]]$scale,
  "/"
)

Phi_1 <- get_AdaDTN_input_basis(
  fit,
  predictor_idx = 1
)

scores_1_manual <- sweep(X1_sc, 2, w, "*") %*% Phi_1
max(abs(scores_1_manual - scores$predictor_1))

## Verify the next tensor-projection step for the first predictor.
A_1 <- as.matrix(
  fit$model$get_A(1)$detach()$cpu()
)

features <- get_AdaDTN_features(
  object = fit,
  X_func = test_dat$x,
  X_scalar = test_dat$x.scl
)

features_1_from_scores <- scores$predictor_1 %*% A_1
max(abs(features_1_from_scores - features$predictor_1))
}
}
