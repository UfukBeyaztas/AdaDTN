\name{get_AdaDTN_intercept}
\alias{get_AdaDTN_intercept}
\title{Extract the AdaDTN Intercept Function}
\description{
Extracts the explicit response-grid intercept function from a fitted AdaDTN
model and transforms it from the internally standardized response scale to
the original response scale. The result is a named numeric vector evaluated
at the response grid used for model fitting.
}
\usage{
get_AdaDTN_intercept(object)
}
\arguments{
  \item{object}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }
}
\details{
AdaDTN predicts the response directly at the \eqn{M}{M} locations in
\code{object$grid_out}. Its final output layer is bias-free, and the model
contains a separately parameterized vector
\deqn{
\boldsymbol{\beta}_0^{\mathrm{sc}}
=
\left\{
\beta_0^{\mathrm{sc}}(t_1),
\ldots,
\beta_0^{\mathrm{sc}}(t_M)
\right\}^{\mathsf T}
\in\mathbb{R}^{M}.
}{
beta0_sc =
(beta0_sc(t[1]), ..., beta0_sc(t[M]))',
an M-dimensional vector.
}
This vector is stored as \code{object$model$intercept_fun}. It is shared by
all subjects but may vary freely over the response grid, so it is a
functional intercept rather than a single scalar intercept.

For subject \eqn{i}{i}, let
\eqn{r_i^{\mathrm{sc}}}{r_sc[i]} denote the standardized-scale output of
the bias-free final linear layer after the adaptive functional front-end,
optional scalar processing, and nonlinear dense hidden layers. The network's
standardized prediction is
\deqn{
\widehat{\boldsymbol{Y}}_i^{\mathrm{sc}}
=
\boldsymbol{r}_i^{\mathrm{sc}}
+\boldsymbol{\beta}_0^{\mathrm{sc}}.
}{
Yhat_sc[i] = r_sc[i] + beta0_sc.
}
The explicit intercept therefore provides one unrestricted baseline
coordinate for every response-grid location and is added only after the
subject-specific dense-head output has been computed.

During fitting, every response-grid column is standardized using its center
and standard deviation estimated from the internal training subjects. Let
\deqn{
\widehat{\boldsymbol{\mu}}_Y
=
\left(
\widehat{\mu}_{Y,1},\ldots,\widehat{\mu}_{Y,M}
\right)^{\mathsf T},
\qquad
\widehat{\boldsymbol{\sigma}}_Y
=
\left(
\widehat{\sigma}_{Y,1},\ldots,\widehat{\sigma}_{Y,M}
\right)^{\mathsf T}
}{
mu_Y = (mu_Y[1], ..., mu_Y[M])',
and sigma_Y = (sigma_Y[1], ..., sigma_Y[M])'
}
denote \code{object$scalers$Y$center} and
\code{object$scalers$Y$scale}, respectively. Full response predictions are
returned to the original scale pointwise:
\deqn{
\widehat{Y}_i(t_m)
=
\widehat{\mu}_{Y,m}
+\widehat{\sigma}_{Y,m}
\left\{
r_i^{\mathrm{sc}}(t_m)
+\beta_0^{\mathrm{sc}}(t_m)
\right\}.
}{
Yhat[i,m] =
mu_Y[m] + sigma_Y[m] *
(r_sc[i,m] + beta0_sc[m]).
}
Rearranging this expression gives
\deqn{
\widehat{Y}_i(t_m)
=
\underbrace{
\left\{
\widehat{\mu}_{Y,m}
+\widehat{\sigma}_{Y,m}
\beta_0^{\mathrm{sc}}(t_m)
\right\}
}_{\displaystyle \widehat{\beta}_0(t_m)}
+\widehat{\sigma}_{Y,m}r_i^{\mathrm{sc}}(t_m).
}{
Yhat[i,m] =
beta0_hat[m] + sigma_Y[m] * r_sc[i,m],
where beta0_hat[m] =
mu_Y[m] + sigma_Y[m] * beta0_sc[m].
}
Accordingly, \code{get_AdaDTN_intercept} returns exactly
\deqn{
\widehat{\boldsymbol{\beta}}_0
=
\widehat{\boldsymbol{\mu}}_Y
+\widehat{\boldsymbol{\sigma}}_Y
\mathbin{\odot}
\boldsymbol{\beta}_0^{\mathrm{sc}},
}{
beta0_hat = mu_Y + sigma_Y * beta0_sc,
}
where \eqn{\odot}{*} denotes elementwise multiplication. The operation is
performed separately at every response-grid point because the response was
standardized column by column.

If the learned standardized intercept is identically zero, the returned
function is the training response-center curve
\eqn{\widehat{\boldsymbol{\mu}}_Y}{mu_Y}, not a zero vector. This follows
from expressing the explicit intercept on the original response scale.

The returned vector is the original-scale version of the model's
\emph{explicit} intercept parameter. It should not automatically be
interpreted as:
\itemize{
  \item the mean observed response curve;
  \item the conditional mean response when the predictors equal their
  training means;
  \item the prediction for a hypothetical subject with zero standardized
  functional and scalar predictors; or
  \item a predictor-specific effect.
}

Although the final output layer has no bias, the dense hidden layers have
their own bias vectors. The optional scalar embedding also has a bias when
it is present. Thus, even when all standardized functional and scalar inputs
are zero, the nonlinear head can generate a nonzero vector
\eqn{\boldsymbol{r}_0^{\mathrm{sc}}}{r_sc[0]}. The prediction at the
training-center inputs is then
\deqn{
\widehat{\boldsymbol{Y}}_{\mathrm{center}}
=
\widehat{\boldsymbol{\beta}}_0
+\widehat{\boldsymbol{\sigma}}_Y
\mathbin{\odot}
\boldsymbol{r}_0^{\mathrm{sc}},
}{
Yhat_center =
beta0_hat + sigma_Y * r_sc[0],
}
which need not equal the value returned by this function.

More generally, a flexible biased hidden network can represent constant
components as well as predictor-dependent components. Some constant signal
may therefore be allocated to the explicit intercept and some to the dense
head. The explicit intercept is a clearly identified model parameter in the
stored architecture, but its numerical decomposition from constant
hidden-head output need not have a unique scientific interpretation.
}
\value{
A named numeric vector of length \code{length(object$grid_out)}. Element
\eqn{m}{m} equals
\deqn{
\widehat{\beta}_0(t_m)
=
\widehat{\mu}_{Y,m}
+\widehat{\sigma}_{Y,m}
\beta_0^{\mathrm{sc}}(t_m).
}{
beta0_hat[m] =
mu_Y[m] + sigma_Y[m] * beta0_sc[m].
}
The elements are named \code{"t1"}, \code{"t2"}, and so forth. These names
are positional labels only; the corresponding numerical response-grid
coordinates are stored in \code{object$grid_out}.
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

beta0 <- get_AdaDTN_intercept(fit)

length(beta0)
head(beta0)

## Plot against the actual response grid.
t <- fit$grid_out
plot(
  t,
  beta0,
  type = "l",
  lwd = 2,
  xlab = "t",
  ylab = expression(hat(beta)[0](t))
)

## Compare the fitted explicit intercept with the response center computed
## from the model's internal gradient-training subjects.
response_center <- fit$scalers$Y$center

lines(
  t,
  response_center,
  col = 2,
  lty = 2,
  lwd = 2
)

legend(
  "topright",
  legend = c("Explicit intercept", "Training response center"),
  col = c(1, 2),
  lty = c(1, 2),
  lwd = 2,
  bty = "n"
)

## Prediction at training-center inputs. This need not equal beta0 because
## hidden-layer biases can produce an additional constant contribution.
mean_X_func <- lapply(
  fit$scalers$func,
  function(sc) matrix(sc$center, nrow = 1)
)

mean_X_scalar <- matrix(
  fit$scalers$scalar$center,
  nrow = 1
)

pred_at_center <- predict(
  fit,
  new_X_func = mean_X_func,
  new_X_scalar = mean_X_scalar,
  interval = "none"
)

head(
  cbind(
    explicit_intercept = beta0,
    prediction_at_center = as.numeric(pred_at_center)
  )
)
}
}
