\name{get_AdaDTN_input_basis}
\alias{get_AdaDTN_input_basis}
\title{Extract Learned AdaDTN Input Basis Functions}
\description{
Extracts the learned, predictor-specific marginal basis functions from the
adaptive basis layer of a fitted AdaDTN model. The functions are evaluated
on the predictor grid used for model fitting and are returned after the same
trapezoidal \eqn{L^2}{L2} normalization used in every AdaDTN forward pass.
}
\usage{
get_AdaDTN_input_basis(object, predictor_idx = 1L)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{predictor_idx}{
  A single integer identifying the functional predictor whose learned basis
  functions are to be extracted. Predictors are indexed in the same order as
  \code{X_func} supplied to \code{\link{AdaDTN_fit}}. The default is the
  first predictor. The value is coerced with \code{as.integer} and must lie
  between 1 and \code{length(object$grid_in)}.
  }
}
\details{
For functional predictor \eqn{p}{p}, AdaDTN represents each of its
\eqn{K_p}{K[p]} marginal basis functions by a separate micro multilayer
perceptron,
\deqn{
\widetilde{\phi}_{pk}(s)
=
f_{pk}(s;\Theta_{pk}),
\qquad
k=1,\ldots,K_p.
}{
phi_tilde[p,k](s) = f[p,k](s; Theta[p,k]),
for k = 1, ..., K[p].
}
The input is one predictor-domain coordinate and the output is one scalar
basis value. Every hidden layer of a basis micro-network applies a learned
linear map, the package's layer-normalization operation, the activation
specified by \code{basis_activation}, and dropout with probability
\code{basis_dropout}. Its final layer is linear and scalar-valued.

The basis parameters are not estimated in a separate preprocessing step.
They are learned jointly with the tensor coefficient matrices, nonlinear
dense head, and response output layer by backpropagating the complete AdaDTN
training objective. The extracted functions are therefore
response-adaptive: their learned directions depend on the functional
response and on the other functional and scalar predictors represented in
the fitted model.

The current implementation supplies the numerical values in
\code{object$grid_in[[p]]} directly to the micro-networks. It does not
internally rescale the predictor grid. Hence the coordinate system and
spacing used during fitting are part of the learned representation.

Let
\eqn{s_{p1},\ldots,s_{pG_p}}{s[p,1], ..., s[p,G[p]]} denote the fitted grid
for predictor \eqn{p}{p}. Before normalization, the micro-networks produce
the matrix
\deqn{
\widetilde{\Phi}_p
=
\left[
\widetilde{\boldsymbol{\phi}}_{p1},
\ldots,
\widetilde{\boldsymbol{\phi}}_{pK_p}
\right]
\in\mathbb{R}^{G_p\times K_p},
}{
Phi_tilde[p] =
[phi_tilde[p,1], ..., phi_tilde[p,K[p]]],
a G[p] x K[p] matrix,
}
where column \eqn{k}{k} contains
\eqn{\widetilde{\phi}_{pk}(s_{pg})}{
phi_tilde[p,k](s[p,g])} for \eqn{g=1,\ldots,G_p}{g = 1, ..., G[p]}.

\code{get_AdaDTN_input_basis} evaluates the functions only at this stored
grid. Although a trained micro-network is a mapping of a scalar coordinate,
this extraction function does not accept an alternative or finer grid.

The fitted predictor grid determines trapezoidal quadrature weights. For an
increasing grid, these are
\deqn{
w_{p1}
=
\frac{s_{p2}-s_{p1}}{2},
\qquad
w_{pG_p}
=
\frac{s_{pG_p}-s_{p,G_p-1}}{2},
}{
w[p,1] = (s[p,2] - s[p,1]) / 2,
and w[p,G[p]] = (s[p,G[p]] - s[p,G[p]-1]) / 2,
}
and, for an interior grid point,
\deqn{
w_{pg}
=
\frac{
(s_{pg}-s_{p,g-1})
+(s_{p,g+1}-s_{pg})
}{2},
\qquad
g=2,\ldots,G_p-1.
}{
w[p,g] =
((s[p,g] - s[p,g-1]) + (s[p,g+1] - s[p,g])) / 2.
}
Thus, irregular but increasingly ordered grids are accommodated through
their actual local spacings.

Every raw basis column is normalized separately:
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
The matrix returned by the function is
\deqn{
\Phi_p
=
\left[
\bar{\boldsymbol{\phi}}_{p1},
\ldots,
\bar{\boldsymbol{\phi}}_{pK_p}
\right].
}{
Phi[p] = [bar(phi)[p,1], ..., bar(phi)[p,K[p]]].
}
Apart from the numerical-stability constant, every column has weighted
squared norm close to one:
\deqn{
\bar{\boldsymbol{\phi}}_{pk}^{\mathsf T}
W_p
\bar{\boldsymbol{\phi}}_{pk}
\mathrel{\approx}
1,
\qquad
W_p
=
\mathrm{diag}(w_{p1},\ldots,w_{pG_p}).
}{
bar(phi)[p,k]' W[p] bar(phi)[p,k] is approximately 1.
}
The small constant prevents division by zero when an unnormalized basis
vector is extremely close to zero.

The extraction step normalizes each column individually; it does not apply a
Gram--Schmidt procedure, eigendecomposition, or matrix whitening. Mutual
orthogonality is assessed through the weighted Gram matrix
\deqn{
G_p
=
\Phi_p^{\mathsf T}W_p\Phi_p.
}{
G[p] = Phi[p]' W[p] Phi[p].
}
During model fitting, \code{lambda_in_ortho} multiplies the mean absolute
entry of \eqn{G_p-I_{K_p}}{G[p] - I[K[p]]} when \eqn{K_p>1}{K[p] > 1}.
This penalty encourages, but does not impose, weighted orthogonality.
Consequently, the off-diagonal entries of the empirical Gram matrix returned
from a fitted model need not be exactly zero.

When \code{lambda_in_l1} is positive, training also penalizes the mean
trapezoidal integral of the absolute basis values. That penalty may encourage
localized or sparse-looking basis functions, but extraction applies no
additional thresholding or smoothing.

For standardized input curve \eqn{X_i^{(p),\mathrm{sc}}}{
X_sc[i,p]}, the extracted basis matrix defines the adaptive score vector
\deqn{
\boldsymbol{\xi}_{ip}^{\mathsf T}
=
X_i^{(p),\mathrm{sc}}W_p\Phi_p.
}{
xi[i,p]' = X_sc[i,p] W[p] Phi[p].
}
With the fitted coefficient matrix
\eqn{A^{(p)}\in\mathbb{R}^{K_p\times M}}{
A[p] of size K[p] x M}, the predictor-specific front-end feature is
\deqn{
U_i^{(p)}
=
\boldsymbol{\xi}_{ip}^{\mathsf T}A^{(p)}.
}{
U[i,p] = xi[i,p]' A[p].
}
The associated discretized front-end coefficient surface is
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
Thus, the basis functions determine how the standardized input curve is
projected before the tensor coefficient matrix transports those scores to
the response grid.
}
\value{
A numeric matrix with
\code{length(object$grid_in[[predictor_idx]])} rows and
\code{object$n_base_in[predictor_idx]} columns. Row \eqn{g}{g} contains
the learned normalized basis evaluations at
\code{object$grid_in[[predictor_idx]][g]}, and column \eqn{k}{k} contains
the \eqn{k}{k}-th learned basis function.

Columns are named \code{"basis_1"}, \code{"basis_2"}, and so forth. Rows
are named \code{"s1"}, \code{"s2"}, and so forth. These row names are
positional labels only; the actual numerical coordinates are stored in
\code{object$grid_in[[predictor_idx]]}.
}
\references{
Gurer, S., Bakicierler Sezer, G., Shang, H. L., and Beyaztas, U.
\emph{AdaDTN: End-to-End Adaptive Basis Learning for Nonlinear
Function-on-Function Regression}. Manuscript.

Ramsay, J. O. and Silverman, B. W. (2005).
\emph{Functional Data Analysis}, second edition. Springer.
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

## Extract the learned basis system for the first functional predictor.
basis_1 <- get_AdaDTN_input_basis(
  object = fit,
  predictor_idx = 1
)

dim(basis_1)
head(basis_1)

## Plot against the actual stored predictor grid, not the positional row names.
s <- fit$grid_in[[1]]
matplot(
  s,
  basis_1,
  type = "l",
  lty = 1,
  xlab = "s",
  ylab = "Learned basis value"
)
legend(
  "topright",
  legend = colnames(basis_1),
  col = seq_len(ncol(basis_1)),
  lty = 1,
  bty = "n"
)

## Reconstruct the weighted Gram matrix used by the orthogonality penalty.
d <- diff(s)
w <- numeric(length(s))
w[1] <- d[1] / 2
w[length(s)] <- d[length(d)] / 2
if (length(s) > 2) {
  w[2:(length(s) - 1)] <-
    (d[-length(d)] + d[-1]) / 2
}

gram_1 <- crossprod(basis_1, basis_1 * w)
diag(gram_1)
gram_1

## Extract all predictor-specific basis systems.
all_bases <- lapply(
  seq_along(fit$grid_in),
  function(p) get_AdaDTN_input_basis(fit, predictor_idx = p)
)

lapply(all_bases, dim)
}
}
