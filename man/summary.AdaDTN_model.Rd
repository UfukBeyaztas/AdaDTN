\name{summary.AdaDTN_model}
\alias{summary.AdaDTN_model}
\title{Summarize a Fitted AdaDTN Model}
\description{
Constructs a structured summary of a fitted adaptive deep tensor network
(AdaDTN) model. The summary records model dimensions, subject-split sizes,
optimization information, split-specific mean squared errors, the stored
conformal calibration quantile, and the fitting configuration.
}
\usage{
\method{summary}{AdaDTN_model}(object, ...)
}
\arguments{
  \item{object}{
An object of class \code{"AdaDTN_model"}, ordinarily returned by
\code{\link{AdaDTN_fit}} or stored in the \code{best_model} component returned
by \code{\link{AdaDTN_tune}}.
  }
  \item{...}{
Additional arguments supplied for compatibility with the
\code{\link[base]{summary}} generic. They are currently ignored.
  }
}
\value{
An object of class \code{"summary.AdaDTN_model"}, implemented as a list with
the following components:
\item{n_functional_predictors}{
Number of functional predictors, equal to \code{length(object$grid_in)}.
}
\item{n_response_grid_points}{
Number of response-grid points, equal to \code{length(object$grid_out)}.
}
\item{n_base_in}{
The stored vector giving the number of adaptive input bases for each
functional predictor.
}
\item{split_sizes}{
A named numeric vector containing the numbers of training, validation, and
calibration subjects.
}
\item{device}{
The device stored in the fitted object, commonly \code{"cpu"} or
\code{"cuda"}.
}
\item{epochs_completed}{
The number of completed epochs, equal to \code{nrow(object$history)}.
}
\item{best_validation_epoch}{
The epoch associated with the first minimum of
\code{object$history$valid_loss}.
}
\item{mse}{
A named numeric vector with elements \code{train}, \code{validation}, and
\code{calibration}, containing the corresponding stored original-scale mean
squared errors.
}
\item{q_hat}{
The stored type-8 0.95 quantile of the pooled absolute calibration residuals
on the standardized response scale.
}
\item{config}{
The fitting-configuration list stored in \code{object$config}.
}
}
\examples{
\dontrun{

sim <- simulate_AdaDTN_data(
  n = 160,
  j = 101,
  model = "complex"
)

fit <- AdaDTN_fit(
  X_func = sim$x,
  X_scalar = sim$x.scl,
  Y = sim$y,
  grid_in = rep(list(sim$meta$sx), length(sim$x)),
  grid_out = sim$meta$sy,
  n_base_in = rep(4L, length(sim$x)),
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

fit_summary <- summary(fit)
print(fit_summary)
str(fit_summary)

fit_summary$n_functional_predictors
fit_summary$n_response_grid_points
fit_summary$n_base_in
fit_summary$split_sizes
fit_summary$device
fit_summary$epochs_completed
fit_summary$best_validation_epoch
fit_summary$mse
fit_summary$q_hat
fit_summary$config

# The reported epoch is the first epoch attaining the minimum
# stored validation loss.
best_epoch <- fit$history$epoch[
  which.min(fit$history$valid_loss)
]
identical(fit_summary$best_validation_epoch, best_epoch)

# Convert the stored standardized 0.95 half-width to the
# original response scale at every response-grid point.
half_width_95 <- fit_summary$q_hat * fit$scalers$Y$scale

inherits(fit_summary, "summary.AdaDTN_model")
}
}
