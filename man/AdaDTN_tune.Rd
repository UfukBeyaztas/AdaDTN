\name{AdaDTN_tune}
\alias{AdaDTN_tune}
\title{Tune an AdaDTN Model by Subject-Level Cross-Validation}
\description{
Evaluates a data frame of AdaDTN hyperparameter configurations by
subject-level \eqn{K}{K}-fold cross-validation, selects the configuration
having the smallest average cross-validation mean squared error, and
optionally fits an AdaDTN model using the selected configuration.

The same outer fold assignment is used for every candidate configuration so
that candidates are compared on identical held-out subjects.
}
\usage{
AdaDTN_tune(
  grid,
  resp,
  func_cov,
  scalar_cov = NULL,
  grid_in = NULL,
  grid_out = NULL,
  nfolds = 5,
  device = NULL,
  refit = TRUE,
  verbose = TRUE
)
}
\arguments{
  \item{grid}{
  A nonempty data frame containing the candidate hyperparameter
  configurations. Each row represents one configuration and each column
  represents one argument to \code{\link{AdaDTN_fit}}. It is normally
  constructed with \code{\link{AdaDTN_param_grid}}.

  The grid must contain \code{n_base_in}, because that argument has no
  default in \code{AdaDTN_fit}. Vector-valued candidates such as
  \code{n_base_in}, \code{basis_hidden_in}, and \code{dense_hidden} should
  be stored as protected list columns; see
  \code{\link{AdaDTN_param_grid}}.
  }

  \item{resp}{
  A numeric matrix containing the functional responses. Rows correspond to
  subjects and columns to evaluations on the response grid. If there are
  \eqn{n}{n} subjects and \eqn{M}{M} response-grid points, \code{resp}
  must have dimension \eqn{n \times M}{n x M}.
  }

  \item{func_cov}{
  A nonempty list of numeric matrices containing the functional predictors.
  Element \code{func_cov[[p]]} contains predictor \eqn{p}{p}, with subjects
  in rows and evaluations on its predictor-specific grid in columns. If
  predictor \eqn{p}{p} is observed at \eqn{G_p}{G[p]} points, its matrix
  must have dimension \eqn{n \times G_p}{n x G[p]}. All predictor matrices
  must have the same number and ordering of subjects as \code{resp}.
  }

  \item{scalar_cov}{
  An optional numeric matrix of scalar predictors, with subjects in rows and
  scalar covariates in columns. It must have \eqn{n}{n} rows. The default
  \code{NULL} tunes models without scalar predictors.
  }

  \item{grid_in}{
  An optional list containing the observation grids of the functional
  predictors. Element \code{grid_in[[p]]} must have length
  \code{ncol(func_cov[[p]])}. If \code{NULL}, an equally spaced sequence
  from 0 to 1 having the required length is created separately for each
  functional predictor.
  }

  \item{grid_out}{
  An optional numeric vector containing the response grid. Its length must
  equal \code{ncol(resp)}. If \code{NULL}, an equally spaced sequence from
  0 to 1 is used.
  }

  \item{nfolds}{
  Integer giving the number of subject-level cross-validation folds.
  It should be at least two and no larger than the number of subjects.
  Fold sizes differ by at most one under the internally generated
  assignment.
  }

  \item{device}{
  Optional character string specifying the torch device, commonly
  \code{"cpu"} or \code{"cuda"}. If \code{NULL}, device selection is
  delegated to \code{\link{AdaDTN_fit}} for every fitted model.
  }

  \item{refit}{
  Logical value. If \code{TRUE}, the selected configuration is passed to
  \code{\link{AdaDTN_fit}} together with the complete data supplied to this
  function. If \code{FALSE}, selection results are returned without fitting
  this final model.
  }

  \item{verbose}{
  Logical value. If \code{TRUE}, the cross-validation mean squared error is
  printed after each configuration. When \code{refit = TRUE}, this value is
  also passed to the final call to \code{\link{AdaDTN_fit}}. Fold-level
  messages from the internal \code{\link{AdaDTN_cv}} calls are suppressed.
  }
}
\details{
The input \code{grid} must be a data frame with at least one row. For row
\eqn{g}{g}, the function extracts every column and applies the internal
\code{AdaDTN_flatten1} utility. This removes the single layer of list
wrapping commonly introduced by list columns in a parameter grid. Before
each model is fitted, \code{\link{AdaDTN_cv}} applies the internal
\code{AdaDTN_sanitize_par} utility, which performs the type conversions
required by \code{\link{AdaDTN_fit}}.

In particular, a scalar \code{n_base_in} candidate is replicated across the
\eqn{P}{P} functional predictors, whereas a predictor-specific candidate
must have length \eqn{P}{P}. Hidden-layer widths are converted to integer
vectors, optimization counts are converted to integers, continuous
hyperparameters are converted to numeric values, and an empty or
all-\code{NA} \code{scalar_embed_dim} candidate is interpreted as
\code{NULL}.

Columns of \code{grid} should contain only arguments intended for
\code{AdaDTN_fit}. The data, grids, device, and verbosity are supplied
through the dedicated arguments of \code{AdaDTN_tune}. Including names such
as \code{X_func}, \code{Y}, \code{grid_in}, \code{grid_out},
\code{device}, or \code{verbose} in \code{grid} can create duplicated
formal arguments in the subsequent \code{do.call}.

One fold vector is generated before the configurations are evaluated:
\preformatted{
fold_id <- sample(rep(seq_len(nfolds), length.out = n))
}
This construction assigns every subject to one fold and produces folds whose
sizes differ by at most one. The same realized \code{fold_id} is supplied to
\code{\link{AdaDTN_cv}} for every grid row. Consequently, differences among
the recorded cross-validation criteria are not caused by candidates being
evaluated on different outer held-out subjects.

Only the outer folds are held fixed across candidates. Each call to
\code{AdaDTN_fit} remains stochastic: its outer-fold training data are
randomly divided internally into gradient-training, early-stopping
validation, and calibration subsets; network parameters are initialized
randomly; training subjects are reshuffled by epoch; and dropout may be
active. Thus, common outer folds provide a paired data comparison but do not
eliminate optimization or internal-split variability.

For configuration \eqn{g}{g} and fold \eqn{k}{k}, let
\eqn{\mathcal{F}_k}{F[k]} denote the held-out subjects. An
\code{AdaDTN_fit} model is trained using subjects outside
\eqn{\mathcal{F}_k}{F[k]}, and point predictions are obtained for the
complete response curves in \eqn{\mathcal{F}_k}{F[k]}. The fold error is
\deqn{
\mathrm{MSE}_{gk}
=
\frac{1}{|\mathcal{F}_k|M}
\sum_{i\in\mathcal{F}_k}
\sum_{m=1}^{M}
\left\{
Y_i(t_m)-\widehat{Y}_{i,-k,g}(t_m)
\right\}^{2}.
}{
MSE[g,k] =
mean((Y[F[k], ] - Yhat[-k,g][F[k], ])^2).
}
The error used for selection is
\deqn{
\mathrm{CV}_g
=
\frac{1}{K}\sum_{k=1}^{K}\mathrm{MSE}_{gk}.
}{
CV[g] = mean[k = 1,...,K] MSE[g,k].
}
This is an unweighted average of the fold-specific errors. Each fold
therefore receives equal weight even when fold sizes differ by one. Within a
fold, subjects and response-grid points receive equal weight. The code does
not apply response-grid quadrature weights, so the criterion is exactly a
gridwise mean squared error. On a regular response grid over a normalized
domain, it may also be viewed as a discrete approximation to an average
integrated squared prediction error.

The outer held-out fold is not used by its fitted model. However,
\code{AdaDTN_fit} further partitions the outer training portion: only its
internal training subset contributes gradient updates, its internal
validation subset determines early stopping, and its internal calibration
subset is retained for calibration. Because \code{AdaDTN_cv} requests only
point predictions, the calibration quantity produced inside each fold fit
does not enter the cross-validation MSE directly.

After all configurations are evaluated, the selected row is
\deqn{
\widehat{g}
=
\mathop{\mathrm{arg\,min}}_{1\leq g\leq G}
\mathrm{CV}_g,
}{
g_hat = argmin[g = 1,...,G] CV[g].
}
The implementation uses \code{which.min}; therefore, if several finite
configurations have exactly the same smallest value, the first such row is
selected. The corresponding row is flattened and sanitized once more and is
returned as \code{best_params}.

This rule estimates predictive performance over the finite candidate set
represented by \code{grid}. It does not establish that the selected
configuration is globally optimal over all possible architectures and
regularization values. Searching a larger grid can reduce approximation
error in the candidate set, but it also increases computation and the
possibility of selection optimism.

When \code{refit = TRUE}, the sanitized selected parameters and the complete
\code{resp}, \code{func_cov}, and \code{scalar_cov} supplied to
\code{AdaDTN_tune} are passed to a new call to \code{AdaDTN_fit}. This is a
fresh stochastic fit; it does not average fold-specific models, reuse their
weights, reuse \code{fold_id}, or warm-start from the best cross-validation
fit.

Passing the complete data to \code{AdaDTN_fit} does not mean that every
subject is used for gradient updates. As documented for
\code{AdaDTN_fit}, the final call creates a new subject-level
training/validation/calibration split according to the selected
\code{cal_prop} and \code{val_prop}. Gradient updates use the resulting
training subset, validation subjects control early stopping, and calibration
subjects produce the stored calibration residuals.

When \code{refit = FALSE}, \code{best_model} is \code{NULL}. This option is
useful when only the selected configuration is required or when refitting
will be performed later on a separate dataset.
}
\value{
An unclassed list with the following components:
\describe{
  \item{\code{results}}{
  A data frame containing the original parameter grid with a leading
  \code{CV_MSE} column. \code{CV_MSE[g]} is the unweighted average of the
  fold-specific gridwise mean squared errors for configuration \eqn{g}{g}.
  Rows retain the ordering of \code{grid}.
  }

  \item{\code{best_row}}{
  Integer giving the row of \code{results} having the smallest
  \code{CV_MSE}. In the event of an exact tie, this is the first minimizing
  row.
  }

  \item{\code{best_params}}{
  A named list containing the flattened and sanitized parameters from the
  selected grid row. For example, a scalar \code{n_base_in} is returned
  after replication to the number of functional predictors.
  }

  \item{\code{best_model}}{
  If \code{refit = TRUE}, the fitted object of class
  \code{"AdaDTN_model"} returned by the final call to
  \code{\link{AdaDTN_fit}}. If \code{refit = FALSE}, \code{NULL}.
  }

  \item{\code{fold_id}}{
  An integer vector of length \eqn{n}{n} recording the common outer fold
  assignment used for all configurations. Values are in
  \code{seq_len(nfolds)}.
  }
}
}
\examples{
\dontrun{
train_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "complex"
)

grid_ada <- AdaDTN_param_grid(
  n_base_in = I(list(4, 6)),
  basis_hidden_in = I(list(c(32, 32))),
  dense_hidden = I(list(c(32, 32), c(64, 32))),
  basis_dropout = c(0.05, 0.10),
  dense_dropout = 0.05,
  basis_activation = "tanh",
  dense_activation = "relu",
  scalar_embed_dim = I(list(NULL)),
  lambda_in_l1 = 0,
  lambda_in_ortho = c(1e-4, 5e-4),
  lambda_surface_l2 = 1e-4,
  l2_hidden = 1e-4,
  epochs = 250,
  batch_size = 64,
  lr = c(5e-4, 1e-3),
  lr_min = 1e-5,
  patience = 35,
  cal_prop = 0.20,
  val_prop = 0.10
)

tune_ada <- AdaDTN_tune(
  grid = grid_ada,
  resp = train_dat$y,
  func_cov = train_dat$x,
  scalar_cov = train_dat$x.scl,
  grid_in = rep(
    list(train_dat$meta$sx),
    length(train_dat$x)
  ),
  grid_out = train_dat$meta$sy,
  nfolds = 5,
  refit = TRUE,
  verbose = TRUE
)

tune_ada$results
tune_ada$best_row
tune_ada$best_params
tune_ada$best_model

## Point predictions from the selected and refitted model.
test_dat <- simulate_AdaDTN_data(
  n = 100,
  j = 101,
  model = "complex"
)

pred <- predict(
  tune_ada$best_model,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "none"
)

mean((pred - test_dat$yt)^2)

## To keep tuning independent of the calibration sample used by the final
## fit, select parameters on an independent pilot sample.
pilot_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "complex"
)

pilot_tune <- AdaDTN_tune(
  grid = grid_ada,
  resp = pilot_dat$y,
  func_cov = pilot_dat$x,
  scalar_cov = pilot_dat$x.scl,
  grid_in = rep(
    list(pilot_dat$meta$sx),
    length(pilot_dat$x)
  ),
  grid_out = pilot_dat$meta$sy,
  nfolds = 5,
  refit = FALSE,
  verbose = TRUE
)

final_dat <- simulate_AdaDTN_data(
  n = 300,
  j = 101,
  model = "complex"
)

final_fit <- do.call(
  AdaDTN_fit,
  c(
    list(
      X_func = final_dat$x,
      X_scalar = final_dat$x.scl,
      Y = final_dat$y,
      grid_in = rep(
        list(final_dat$meta$sx),
        length(final_dat$x)
      ),
      grid_out = final_dat$meta$sy,
      verbose = TRUE
    ),
    pilot_tune$best_params
  )
)
}
}
