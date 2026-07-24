\name{get_AdaDTN_features}
\alias{get_AdaDTN_features}
\title{Extract Predictor-Specific AdaDTN Front-End Features}
\description{
Extracts the predictor-specific tensor-projection features produced by the
adaptive functional front-end of a fitted AdaDTN model. For every functional
predictor, the function standardizes the supplied curves using the fitted
training scalers, computes their learned adaptive-basis scores, and maps
those scores to the response grid through the fitted surface coefficient
matrix.
}
\usage{
get_AdaDTN_features(object, X_func, X_scalar = NULL)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{X_func}{
  A list of numeric matrices containing the functional predictors for the
  subjects whose front-end features are required. Element
  \code{X_func[[p]]} must correspond to the same predictor, observation
  grid, and column ordering used to fit \code{object}. Rows are subjects and
  columns are functional evaluations. If there are \eqn{n_*}{n_new}
  supplied subjects and predictor \eqn{p}{p} was observed at
  \eqn{G_p}{G[p]} grid points, the matrix must have dimension
  \eqn{n_* \times G_p}{n_new x G[p]}.
  }

  \item{X_scalar}{
  An optional numeric matrix containing scalar predictors for the same
  subjects, with subjects in rows and scalar covariates in the same column
  ordering used for model fitting. Although scalar values do not enter the
  predictor-specific functional features returned by this function, they
  must be supplied when \code{object} was fitted with scalar predictors
  because the implementation executes a complete forward pass to populate
  the model's feature cache. If \code{object} was fitted without scalar
  predictors, this argument is ignored.
  }
}
\details{
The function first transforms every supplied functional predictor using the
column-specific centers and scales stored in \code{object$scalers$func}.
For predictor \eqn{p}{p}, grid point \eqn{g}{g}, and new subject
\eqn{i}{i}, the transformation is
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
The center and scale were estimated from the internal training subjects
during \code{\link{AdaDTN_fit}}. They are not re-estimated from
\code{X_func}. Consequently, features extracted for training and new
subjects are expressed in the same fitted coordinate system.

If scalar predictors are supplied and the fitted object contains scalar
scalers, the scalar columns are standardized in the same way. These
standardized scalar values are passed to the complete network forward pass,
but they do not change the functional front-end quantities
\eqn{U_i^{(p)}}{U[i,p]} defined below.

Let \eqn{\Phi_p}{Phi[p]} be the
\eqn{G_p\times K_p}{G[p] x K[p]} matrix of learned and individually
normalized marginal basis evaluations for predictor \eqn{p}{p}, and let
\eqn{W_p}{W[p]} be the diagonal matrix of trapezoidal quadrature weights
constructed from \code{object$grid_in[[p]]}. Its entries are
\deqn{
(\Phi_p)_{gk}
=
\bar{\phi}_{pk}(s_{pg}),
\qquad
W_p
=
\mathrm{diag}(w_{p1},\ldots,w_{pG_p}).
}{
Phi[p][g,k] = bar(phi)[p,k](s[p,g]),
and W[p] = diag(w[p,1], ..., w[p,G[p]]).
}
For the matrix \eqn{X_{p,*}^{\mathrm{sc}}}{X_sc[p,new]} of standardized
new curves, the fitted model computes the adaptive-basis score matrix
\deqn{
\Xi_{p,*}
=
X_{p,*}^{\mathrm{sc}}W_p\Phi_p,
}{
Xi[p,new] = X_sc[p,new] W[p] Phi[p].
}
Thus, its \eqn{(i,k)}{(i,k)} entry is the trapezoidal approximation
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
These are the same scores used internally by the fitted AdaDTN network; the
function does not refit or modify the learned basis functions.

For predictor \eqn{p}{p}, let
\eqn{A^{(p)}\in\mathbb{R}^{K_p\times M}}{
A[p] be a K[p] x M matrix} denote the fitted surface coefficient matrix,
where \eqn{M}{M} is the number of response-grid points. The returned feature
matrix is
\deqn{
U_{p,*}
=
\Xi_{p,*}A^{(p)}
\in\mathbb{R}^{n_*\times M}.
}{
U[p,new] = Xi[p,new] A[p],
an n_new x M matrix.
}
Equivalently, with the fitted discretized front-end coefficient surface
\deqn{
\widehat{B}_p
=
\Phi_pA^{(p)},
}{
B_hat[p] = Phi[p] A[p],
}
the features satisfy
\deqn{
U_{p,*}
=
X_{p,*}^{\mathrm{sc}}W_p\widehat{B}_p.
}{
U[p,new] = X_sc[p,new] W[p] B_hat[p].
}
Therefore, \code{get_AdaDTN_features} returns the response-grid-indexed
functional projection of each standardized input curve through its fitted
front-end surface. This identity links the extracted features to the
unsmoothed surfaces returned by \code{\link{get_AdaDTN_surface}} when
surface smoothing is disabled, and to the scores returned by
\code{\link{get_AdaDTN_scores}}.

If there are \eqn{P}{P} functional predictors, AdaDTN concatenates the
\eqn{M}{M}-dimensional feature rows
\eqn{U_i^{(1)},\ldots,U_i^{(P)}}{U[i,1], ..., U[i,P]}. Standardized scalar
predictors, or their learned scalar embedding, are then appended. The
resulting vector is the input to the nonlinear dense head:
\deqn{
h_i^{(0)}
=
\left[
U_i^{(1)}
\mathbin{\|}\cdots\mathbin{\|}
U_i^{(P)}
\mathbin{\|}
Z_i^{\mathrm{feat}}
\right],
\qquad
\widehat{Y}_i
=
f_{\mathrm{dense}}(h_i^{(0)})
+\widehat{\beta}_0.
}{
h0[i] = concatenate(U[i,1], ..., U[i,P], Zfeat[i]),
and Yhat[i] = f_dense(h0[i]) + beta0_hat.
}

Accordingly, each returned matrix is a predictor-specific
\emph{pre-head feature representation}. With a nonlinear dense head,
\code{U_list[[p]]} is not an additive contribution of predictor \eqn{p}{p}
to the final response prediction. The dense layers may rescale, transform,
and interact features from different functional and scalar predictors. A
causal or marginal-effect interpretation is therefore not justified merely
from the extracted feature values.

The entries of \eqn{U_i^{(p)}}{U[i,p]} also remain on the internal latent
scale learned from standardized inputs. They are not response curves on the
original measurement scale, and the response inverse-standardization
operation used for final predictions should not be applied to each feature
matrix separately.
}
\value{
A named list of length \code{length(X_func)}. Element \code{p} is a numeric
matrix with one row per supplied subject and one column per response-grid
point. If \eqn{n_*}{n_new} subjects are supplied and
\code{length(object$grid_out) = M}, each matrix has dimension
\eqn{n_*\times M}{n_new x M}.

The list elements are named \code{"predictor_1"},
\code{"predictor_2"}, and so forth. Element \code{p} contains
\eqn{U_{p,*}=\Xi_{p,*}A^{(p)}}{U[p,new] = Xi[p,new] A[p]} for functional
predictor \eqn{p}{p}.
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

features <- get_AdaDTN_features(
  object = fit,
  X_func = test_dat$x,
  X_scalar = test_dat$x.scl
)

names(features)
length(features)
lapply(features, dim)

## Concatenated functional part of the dense-head input.
functional_head_input <- do.call(cbind, features)
dim(functional_head_input)

## Front-end feature curve for the first subject and first predictor.
plot(
  fit$grid_out,
  features$predictor_1[1, ],
  type = "l",
  xlab = "t",
  ylab = "Front-end feature"
)

## Compare the score and tensor-projection dimensions.
scores <- get_AdaDTN_scores(
  object = fit,
  X_func = test_dat$x,
  X_scalar = test_dat$x.scl
)

lapply(scores, dim)
lapply(features, dim)
}
}
