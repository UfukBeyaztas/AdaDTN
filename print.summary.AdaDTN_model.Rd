\name{print.summary.AdaDTN_model}
\alias{print.summary.AdaDTN_model}
\title{Print a Detailed AdaDTN Model Summary}
\description{
Prints the structured summary produced by
\code{\link{summary.AdaDTN_model}}. The display reports the fitted
architecture size, subject-split sizes, optimization device and epoch
information, original-scale split-specific mean squared errors, and the
stored standardized calibration half-width.
}
\usage{
\method{print}{summary.AdaDTN_model}(x, ...)
}
\arguments{
  \item{x}{
  An object of class \code{"summary.AdaDTN_model"}, normally produced by
  \code{summary(fitted_model)} or
  \code{\link{summary.AdaDTN_model}}.
  }

  \item{...}{
  Additional arguments included for compatibility with the
  \code{\link[base]{print}} generic. They are not used or forwarded by the
  current method.
  }
}
\value{
The input summary object \code{x}, returned invisibly and unchanged. The
method is called primarily for its console-printing side effect.
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

fit_summary <- summary(fit)

## Explicit printing of the summary object.
print(fit_summary)

## The summary object is returned invisibly.
returned_summary <- print(fit_summary)
identical(returned_summary, fit_summary)

## Capture the fixed display.
summary_lines <- capture.output(print(fit_summary))
summary_lines

## Programmatic access to components shown in the display.
fit_summary$split_sizes
fit_summary$mse
fit_summary$best_validation_epoch
fit_summary$q_hat

## Configuration settings are stored but not printed.
fit_summary$config

## Convert the printed standardized 0.95 half-width to the original response
## scale at every response-grid point.
q_original <- (
  fit_summary$q_hat *
  fit$scalers$Y$scale
)

head(q_original)
}
}
