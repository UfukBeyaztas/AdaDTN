\name{AdaDTN_cv}
\alias{AdaDTN_cv}
\title{Cross-Validation Error for an AdaDTN Configuration}
\description{
Computes the subject-level \eqn{K}{K}-fold cross-validation error for one
hyperparameter configuration of the adaptive deep tensor network (AdaDTN).
For each fold, the function fits \code{\link{AdaDTN_fit}} on the subjects
outside that fold, predicts the complete response curves of the held-out
subjects, and evaluates the mean squared prediction error over subjects and
response-grid points. The returned scalar is the unweighted average of the
fold-specific errors.
}
\usage{
AdaDTN_cv(
  resp,
  func_cov,
  scalar_cov = NULL,
  grid_in = NULL,
  grid_out = NULL,
  par,
  nfolds = 5,
  fold_id = NULL,
  device = NULL,
  verbose = FALSE
)
}
\arguments{
  \item{resp}{
  A numeric matrix containing the functional responses. Rows correspond to
  subjects and columns correspond to evaluations on the response grid.
  If there are \eqn{n}{n} subjects and \eqn{M}{M} response-grid points,
  \code{resp} must have dimension \eqn{n \times M}{n x M}.
  }

  \item{func_cov}{
  A nonempty list of numeric matrices containing the functional predictors.
  Element \code{func_cov[[p]]} contains the \eqn{p}{p}-th functional predictor,
  with subjects in rows and evaluations on its predictor-specific grid in
  columns. If predictor \eqn{p}{p} is observed at \eqn{G_p}{G[p]} grid
  points, its matrix must have dimension \eqn{n \times G_p}{n x G[p]}.
  All matrices must have the same
  number of rows as \code{resp}.
  }

  \item{scalar_cov}{
  An optional numeric matrix of scalar predictors, with subjects in rows and
  scalar covariates in columns. It must have \eqn{n}{n} rows. The default
  \code{NULL} fits an AdaDTN model without scalar predictors.
  }

  \item{grid_in}{
  An optional list containing the observation grids of the functional
  predictors. Element \code{grid_in[[p]]} must have length equal to
  \code{ncol(func_cov[[p]])}. If \code{NULL}, each predictor grid is set to
  an equally spaced sequence from 0 to 1 with the required number of points.
  }

  \item{grid_out}{
  An optional numeric vector containing the response grid. Its length must
  equal \code{ncol(resp)}. If \code{NULL}, an equally spaced sequence from
  0 to 1 is used.
  }

  \item{par}{
  A named list specifying one AdaDTN hyperparameter configuration. It is
  processed by the internal function \code{AdaDTN_sanitize_par} and then passed to
  \code{\link{AdaDTN_fit}}. The list must supply \code{n_base_in}, because
  that argument has no default in \code{AdaDTN_fit}. Other omitted fitting
  arguments use the defaults of \code{AdaDTN_fit}.

  Relevant components include \code{basis_hidden_in},
  \code{dense_hidden}, \code{basis_dropout}, \code{dense_dropout},
  \code{basis_activation}, \code{dense_activation},
  \code{scalar_embed_dim}, \code{lambda_in_l1},
  \code{lambda_in_ortho}, \code{lambda_surface_l2},
  \code{l2_hidden}, \code{epochs}, \code{batch_size}, \code{lr},
  \code{lr_min}, \code{patience}, \code{cal_prop}, and
  \code{val_prop}. Arguments controlling the data, grids, device, or
  verbosity should be supplied through their dedicated arguments rather
  than duplicated in \code{par}.
  }

  \item{nfolds}{
  A positive integer giving the number of outer cross-validation folds.
  The default is 5. In ordinary use, it should be at least 2 and no greater
  than the number of subjects.
  }

  \item{fold_id}{
  An optional vector of length \eqn{n}{n} assigning each subject to an outer
  fold. Its entries should be the integers from \code{1} through
  \code{nfolds}, with every fold represented. If \code{NULL}, the function
  constructs an approximately balanced random assignment by permuting
  \code{rep(seq_len(nfolds), length.out = n)}. Supply the same
  \code{fold_id} when different configurations must be evaluated on
  identical folds.
  }

  \item{device}{
  The Torch computation device passed to \code{\link{AdaDTN_fit}}, typically
  \code{"cpu"} or \code{"cuda"}. If \code{NULL}, device selection is handled
  by \code{AdaDTN_fit}.
  }

  \item{verbose}{
  Logical; if \code{TRUE}, the function reports the CV-MSE after each outer
  fold. Output generated internally by \code{AdaDTN_fit} remains suppressed
  because every fold-specific fit is called with \code{verbose = FALSE}.
  }
}
\details{
AdaDTN models a complete functional response from \eqn{P}{P} functional
predictors and optional scalar covariates. For functional predictor
\eqn{p}{p},
the adaptive basis layer learns normalized functions
\eqn{\{\bar\phi_k^{(p)}\}_{k=1}^{K_p}}{
\{phi[k]^(p): k = 1,...,K[p]\}} and forms the numerical inner-product
scores
\deqn{
c_{ik}^{(p)}
=
\int_{\mathcal I_{x_p}} X_i^{(p)}(s)
\bar\phi_k^{(p)}(s)\,ds.
}{
c[ik]^(p) = integral X[i]^(p)(s) phi[k]^(p)(s) ds.
}
The tensor projection matrix \eqn{A^{(p)}}{A^(p)} maps these scores to the response
grid, producing the predictor-specific front-end feature
\deqn{
U_i^{(p)}
=
\left(c_{i1}^{(p)},\ldots,c_{iK_p}^{(p)}\right)A^{(p)}.
}{
U[i]^(p) = c[i]^(p) A^(p).
}
The front-end features from all functional predictors, together with the
optional scalar predictors, are passed through the nonlinear dense head.
Consequently, the cross-validation criterion assesses prediction by the
complete AdaDTN operator, not the first-layer coefficient surfaces in
isolation.

Let \eqn{\mathcal F_1,\ldots,\mathcal F_K}{F[1],...,F[K]} denote the
subject sets determined by \code{fold_id}, where \eqn{K}{K} equals
\code{nfolds}. For fold \eqn{k}{k}, the function
uses
\deqn{
\mathcal I_{-k}
=
\{1,\ldots,n\}\setminus\mathcal F_k
}{
I[-k] = \{1,...,n\} without F[k]
}
as the outer fitting sample and \eqn{\mathcal F_k}{F[k]} as the outer validation
sample. All functional and scalar predictors, as well as the functional
responses, are subset at the subject level. Therefore, all evaluations of a
given response curve remain in the same outer fold, preserving dependence
within a curve and preventing response-grid points from the same subject
from being distributed across training and validation samples.

The same \code{grid_in} and \code{grid_out} are used in every fold. The
fitted model is applied to the held-out functional and scalar predictors
using \code{predict(..., interval = "none")}; hence the CV criterion concerns
point prediction and does not evaluate conformal interval coverage or width.

Each outer fitting sample is passed unchanged to \code{AdaDTN_fit}. That
function performs its own random subject-level division into internal
training, validation, and calibration subsets according to
\code{val_prop} and \code{cal_prop}. The internal training subjects
determine the centering and scaling statistics and contribute gradient
updates. The internal validation subjects determine early stopping, and the
internal calibration subjects are used by \code{AdaDTN_fit} to store
calibration residuals and a conformal quantile.

The outer held-out fold is not used in any of these operations. It is used
only after fitting to compute the fold-specific prediction error. Because
\code{interval = "none"}, the internally calculated conformal quantities do
not enter the outer CV loss.

This nested behavior is important when interpreting the effective fitting
sample: the neural network in an outer fold is optimized on the internal
training subset of \eqn{\mathcal I_{-k}}{I[-k]}, rather than on every
subject in \eqn{\mathcal I_{-k}}{I[-k]}. It also means that the CV score
remains stochastic even
when \code{fold_id} is fixed, because the inner split, neural-network
initialization, mini-batch ordering, and dropout masks can vary between
runs.

Let \eqn{n_k=|\mathcal F_k|}{n[k] = |F[k]|}, let
\eqn{M}{M} equal \code{ncol(resp)}, and let
\eqn{\widehat Y_{i,-k}(t_m)}{Yhat[i,-k](t[m])} denote the prediction for
subject \eqn{i}{i} at response-grid point \eqn{t_m}{t[m]}, obtained from
the model fitted without outer fold \eqn{k}{k}. The function computes
\deqn{
\operatorname{CV}_k
=
\frac{1}{n_kM}
\sum_{i\in\mathcal F_k}
\sum_{m=1}^{M}
\left\{
Y_i(t_m)-\widehat Y_{i,-k}(t_m)
\right\}^2
}{
CV[k] = (n[k] M)^(-1) sum(i in F[k]) sum(m)
(Y[i](t[m]) - Yhat[i,-k](t[m]))^2.
}
This is the mean squared error over all subject-grid-point pairs in fold
\eqn{k}{k}. On a common equally spaced response grid, it is the discrete
average corresponding, up to the constant length of the response domain,
to the Riemann approximation of the integrated squared prediction error.
It is therefore used as the empirical CV-MISE criterion for selecting the
AdaDTN architecture and regularization parameters.

The returned value is
\deqn{
\operatorname{CV}
=
\frac{1}{K}\sum_{k=1}^{K}\operatorname{CV}_k.
}{
CV = K^(-1) sum(k = 1,...,K) CV[k].
}
Fold errors are averaged with equal weights. The automatically generated
folds differ in size by at most one subject, so this is effectively the same
as pooling the squared errors. If a user supplies substantially unequal
folds, however, the returned result gives each fold equal weight rather than
each subject equal weight.

No response-domain quadrature weights are used in this function. For an
irregular \code{grid_out}, the criterion is an unweighted mean over observed
grid points and is not a trapezoidal approximation to an integrated loss.
}
\value{
A numeric scalar equal to the arithmetic mean of the fold-specific
CV-MSE values. Smaller values indicate better held-out point-prediction
performance for the supplied configuration.
}
\references{
Gurer, S., Bakicierler Sezer, G., Shang, H. L., and Beyaztas, U. (2026).
\emph{AdaDTN: End-to-End Adaptive Basis Learning for Nonlinear
Function-on-Function Regression}. Manuscript.

Ramsay, J. O. and Silverman, B. W. (2005).
\emph{Functional Data Analysis}, 2nd ed. Springer.

Stone, M. (1974).
Cross-validatory choice and assessment of statistical predictions.
\emph{Journal of the Royal Statistical Society: Series B}, 36, 111--147.
}
\examples{
\dontrun{
## Generate a training sample from the complex simulation mechanism.
n <- 120
J <- 51

dat <- simulate_AdaDTN_data(
  n = n,
  j = J,
  model = "complex"
)

## Define one AdaDTN configuration.
par <- list(
  n_base_in = 4,
  basis_hidden_in = c(32, 32),
  dense_hidden = c(64, 32),
  basis_dropout = 0.10,
  dense_dropout = 0.05,
  basis_activation = "tanh",
  dense_activation = "relu",
  scalar_embed_dim = NULL,
  lambda_in_l1 = 0,
  lambda_in_ortho = 1e-4,
  lambda_surface_l2 = 1e-4,
  l2_hidden = 1e-4,
  epochs = 100L,
  batch_size = 32,
  lr = 1e-3,
  lr_min = 1e-5,
  patience = 20,
  cal_prop = 0.20,
  val_prop = 0.10
)

## Use an explicit balanced fold assignment so that the same folds can
## be reused when comparing different configurations.
fold_id <- sample(rep(seq_len(5), length.out = n))

cv_mise <- AdaDTN_cv(
  resp = dat$y,
  func_cov = dat$x,
  scalar_cov = dat$x.scl,
  grid_in = rep(list(dat$meta$sx), length(dat$x)),
  grid_out = dat$meta$sy,
  par = par,
  nfolds = 5,
  fold_id = fold_id,
  verbose = TRUE
)

cv_mise
}
}
