\name{print.AdaDTN_model}
\alias{print.AdaDTN_model}
\title{Print an AdaDTN Model Summary}
\description{
Prints a concise console summary of a fitted AdaDTN model, including the
numbers of functional predictors, response-grid points, and adaptive basis
functions; the computation device; the number of completed epochs; and the
final training, validation, and calibration mean squared errors.
}
\usage{
\method{print}{AdaDTN_model}(x, ...)
}
\arguments{
  \item{x}{
  A fitted object of class \code{"AdaDTN_model"}, normally returned by
  \code{\link{AdaDTN_fit}} or as the \code{best_model} component of
  \code{\link{AdaDTN_tune}}.
  }

  \item{...}{
  Additional arguments included for compatibility with the
  \code{\link[base]{print}} generic. They are not used by the current
  method.
  }
}
\value{
The input object \code{x}, returned invisibly and unchanged. The method is
called primarily for its console-printing side effect.
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

## Explicit printing.
print(fit)

## Entering the object at the console invokes the same S3 method.
fit

## The fitted object is returned invisibly.
returned_fit <- print(fit)
identical(returned_fit, fit)

## Capture the fixed console display as character lines.
printed_lines <- capture.output(print(fit))
printed_lines

## Components not included in the concise display remain accessible.
fit$config
fit$q_hat

best_epoch <- fit$history$epoch[
  which.min(fit$history$valid_loss)
]

c(
  epochs_completed = nrow(fit$history),
  best_validation_epoch = best_epoch
)
}
}
