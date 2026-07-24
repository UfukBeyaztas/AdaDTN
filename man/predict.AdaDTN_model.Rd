\name{predict.AdaDTN_model}
\alias{predict.AdaDTN_model}
\title{Predict Functional Responses from an AdaDTN Model}
\description{
Provides the S3 prediction method for fitted AdaDTN models. It standardizes
new functional and scalar predictors using the fitted training scalers,
predicts complete response curves on the original response scale, and can
optionally construct symmetric calibration-based prediction bands from the
stored pooled standardized residuals.
}
\usage{
\method{predict}{AdaDTN_model}(
  object,
  new_X_func,
  new_X_scalar = NULL,
  interval = c("none", "conformal"),
  level = 0.95,
  batch_size = 256,
  ...
)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{new_X_func}{
  A list of numeric matrices containing the functional predictors for the
  new subjects. Element \code{new_X_func[[p]]} must correspond to the same
  predictor, observation grid, and column order used to fit \code{object}.
  Rows are subjects and columns are functional evaluations. If there are
  \eqn{n_*}{n_new} new subjects and predictor \eqn{p}{p} was observed at
  \eqn{G_p}{G[p]} grid points, its matrix must have dimension
  \eqn{n_* \times G_p}{n_new x G[p]}. All list elements must contain the
  same subjects in the same row order.
  }

  \item{new_X_scalar}{
  An optional numeric matrix of scalar predictors for the new subjects, with
  subjects in rows and covariates in the same column order used for model
  fitting. It must be supplied when \code{object} was fitted with scalar
  predictors and must be \code{NULL} when the fitted model contains no
  scalar predictors.
  }

  \item{interval}{
  Character string specifying the returned result. Use \code{"none"} for
  point predictions only or \code{"conformal"} for point predictions and
  symmetric calibration-based lower and upper bands. The default is
  \code{"none"}. Partial matching is performed by
  \code{\link[base]{match.arg}}.
  }

  \item{level}{
  Numeric probability used as the empirical quantile level when
  \code{interval = "conformal"}. The default is \code{0.95}. It should be a
  single finite value between zero and one. This argument is ignored when
  \code{interval = "none"}.
  }

  \item{batch_size}{
  Positive integer giving the maximum number of new subjects processed in
  one torch forward pass. The default is 256. This controls prediction
  memory use but does not change the fitted parameters.
  }

  \item{...}{
  Additional arguments included for compatibility with the
  \code{\link[stats]{predict}} generic. They are not used by the current
  method.
  }
}
\details{
Every functional predictor is transformed using the column-specific centers
and scales stored in \code{object$scalers$func}. For new subject \eqn{i}{i},
functional predictor \eqn{p}{p}, and predictor-grid location \eqn{g}{g},
\deqn{
X_{ip,*}^{\mathrm{sc}}(s_{pg})
=
\frac{
X_{ip,*}(s_{pg})-\widehat{\mu}_{X,pg,\mathcal{T}}
}{
\widehat{\sigma}_{X,pg,\mathcal{T}}
}.
}{
X_sc[i,p,new](s[p,g]) =
(X[i,p,new](s[p,g]) - mean_X_train[p,g]) /
sd_X_train[p,g].
}
The centers and scales were estimated only from the internal gradient-
training subjects in \code{\link{AdaDTN_fit}}. They are not re-estimated
from the new data.

When scalar predictors are present, their columns are transformed
analogously using \code{object$scalers$scalar}. If the fitted model expects
scalar predictors but \code{new_X_scalar = NULL}, the model's forward method
stops because its scalar input is missing. Conversely, supplying
\code{new_X_scalar} to a model fitted without scalar predictors attempts to
use a nonexistent scalar scaler and is not supported.

The method does not accept predictor grids as arguments. It assumes that
every column of \code{new_X_func[[p]]} corresponds exactly to the stored
coordinate \code{object$grid_in[[p]][g]}. It performs no interpolation,
registration, or grid matching.

For predictor \eqn{p}{p}, the standardized new functional curve is projected
onto the learned normalized basis system:
\deqn{
\boldsymbol{\xi}_{ip}^{\mathsf T}
=
X_{ip,*}^{\mathrm{sc}}W_p\Phi_p,
}{
xi[i,p]' = X_sc[i,p,new] W[p] Phi[p].
}
The score vector is mapped to the response grid by the fitted coefficient
matrix:
\deqn{
U_i^{(p)}
=
\boldsymbol{\xi}_{ip}^{\mathsf T}A^{(p)}.
}{
U[i,p] = xi[i,p]' A[p].
}
All predictor-specific feature vectors are concatenated with the standardized
scalar features or their learned embedding and passed through the fitted
nonlinear dense head. A bias-free output layer and the explicit functional
intercept produce the standardized response prediction
\deqn{
\widehat{\boldsymbol{Y}}_{i,*}^{\mathrm{sc}}
=
f_{\widehat{\Theta}}
\left(
X_{i,*}^{(1:P),\mathrm{sc}},
Z_{i,*}^{\mathrm{sc}}
\right)
\in\mathbb{R}^{M}.
}{
Yhat_sc[i,new] =
f_Theta_hat(X_sc[i,new], Z_sc[i,new]),
an M-dimensional vector.
}
The network predicts directly at the locations in
\code{object$grid_out}; no fixed output-basis reconstruction is performed.

The fitted response center and scale are then applied pointwise:
\deqn{
\widehat{Y}_{i,*}(t_m)
=
\widehat{\mu}_{Y,m,\mathcal{T}}
+\widehat{\sigma}_{Y,m,\mathcal{T}}
\widehat{Y}_{i,*}^{\mathrm{sc}}(t_m).
}{
Yhat[i,new,m] =
mean_Y_train[m] +
sd_Y_train[m] * Yhat_sc[i,new,m].
}
The returned point predictions are therefore on the original response scale.
}
\value{
The return value depends on \code{interval}:
\describe{
  \item{\code{"none"}}{
  A numeric matrix of point predictions on the original response scale. It
  has one row per new subject and
  \code{length(object$grid_out)} columns.
  }

  \item{\code{"conformal"}}{
  A list with three numeric matrices:
  \describe{
    \item{\code{mean}}{
    Point predictions on the original response scale.
    }
    \item{\code{lower}}{
    Lower symmetric calibration-based limits on the original response scale.
    }
    \item{\code{upper}}{
    Upper symmetric calibration-based limits on the original response scale.
    }
  }
  Each matrix has one row per new subject and one column per stored response-
  grid location. No additional result class is assigned.
  }
}
}
\examples{
\dontrun{

train_dat <- simulate_AdaDTN_data(
  n = 300,
  j = 101,
  model = "complex"
)

test_dat <- simulate_AdaDTN_data(
  n = 100,
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
  scalar_embed_dim = NULL,
  lambda_in_l1 = 0,
  lambda_in_ortho = 1e-4,
  lambda_surface_l2 = 1e-4,
  l2_hidden = 1e-4,
  epochs = 400,
  batch_size = 64,
  lr = 1e-3,
  lr_min = 1e-5,
  patience = 40,
  cal_prop = 0.20,
  val_prop = 0.10,
  verbose = TRUE
)

## Original-scale point predictions.
pred <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "none",
  batch_size = 256
)

dim(pred)
mean((pred - test_dat$yt)^2)

## Symmetric calibration-based bands at empirical quantile level 0.95.
pred_int <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "conformal",
  level = 0.95
)

names(pred_int)
lapply(pred_int, dim)

## The original-scale width is the same across subjects at each grid point.
band_width <- pred_int$upper - pred_int$lower
apply(band_width, 2, range)

## Descriptive pooled, pointwise, and whole-curve coverage on simulated
## observed test responses. Only the first is aligned with the pooled
## empirical summary; none should be treated as a guarantee from one sample.
covered <- (
  test_dat$y >= pred_int$lower &
  test_dat$y <= pred_int$upper
)

pooled_coverage <- mean(covered)
pointwise_coverage <- colMeans(covered)
whole_curve_coverage <- mean(
  rowSums(covered) == ncol(covered)
)

pooled_coverage
range(pointwise_coverage)
whole_curve_coverage

## A different level recomputes the type 8 quantile from stored residuals.
pred_int_90 <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "conformal",
  level = 0.90
)

mean(pred_int_90$upper - pred_int_90$lower)
mean(pred_int$upper - pred_int$lower)

## Verify the exact standardized quantile used at level 0.95.
q_sc_95 <- quantile(
  fit$cal_residuals_sc,
  probs = 0.95,
  type = 8,
  names = FALSE
)

all.equal(
  as.numeric(q_sc_95),
  as.numeric(fit$q_hat)
)
}
}
