\name{AdaDTN_param_grid}
\alias{AdaDTN_param_grid}
\title{Construct an AdaDTN Hyperparameter Grid}
\description{
Constructs a data frame containing the Cartesian product of candidate
hyperparameter values for AdaDTN. Atomic candidates can be supplied as
ordinary vectors, whereas vector-valued model arguments, such as hidden-layer
architectures, should be supplied as protected list columns.
}
\usage{
AdaDTN_param_grid(...)
}
\arguments{
  \item{...}{
  Named vectors, factors, or list-like objects containing candidate values
  for AdaDTN hyperparameters. Each argument becomes one column of the
  resulting grid. The argument names should normally match arguments of
  \code{\link{AdaDTN_fit}}, such as \code{n_base_in},
  \code{dense_hidden}, \code{basis_dropout}, or \code{lr}.

  Scalar-valued candidates can be supplied as ordinary vectors. Parameters
  whose individual candidates are themselves vectors or \code{NULL} should
  be supplied using \code{I(list(...))}; see Details and Examples.
  }
}
\details{
\code{AdaDTN_param_grid} is a small convenience wrapper around
\code{\link[base]{expand.grid}}:
\preformatted{
expand.grid(..., stringsAsFactors = FALSE)
}
It generates every possible combination of the supplied candidate values.
If parameter \eqn{r}{r} has \eqn{L_r}{L[r]} candidates, the returned grid
has
\deqn{
\prod_{r=1}^{R}L_r
}{
product[r = 1,...,R] L[r]
}
rows. The first supplied parameter varies fastest, following the ordering
used by \code{expand.grid}.

Ordinary atomic vectors represent alternative scalar values. For example,
\preformatted{
basis_dropout = c(0.05, 0.10)
}
creates two candidate values and therefore contributes a factor of two to
the number of grid rows. Character candidates remain character rather than
being converted to factors because the wrapper fixes
\code{stringsAsFactors = FALSE}.

Some arguments of \code{\link{AdaDTN_fit}} are vectors describing an entire
architecture. These include \code{n_base_in}, \code{basis_hidden_in}, and
\code{dense_hidden}. Each complete vector must occupy one cell of the
parameter grid. Protect such candidates with \code{I(list(...))}. For
example,
\preformatted{
dense_hidden = I(list(c(32, 32), c(64, 32)))
}
defines two candidate dense architectures. Supplying
\code{dense_hidden = c(32, 64)} instead would define the two scalar
candidates 32 and 64, not the two-layer architecture \code{c(32, 64)}.

The same convention can be used for \code{n_base_in}. A scalar candidate,
such as \code{6}, is replicated across the functional predictors by the
internal parameter-sanitization step used during tuning. Predictor-specific
candidate vectors can instead be represented explicitly, for example,
\preformatted{
n_base_in = I(list(c(4, 4, 4), c(6, 6, 6)))
}
when the model contains three functional predictors.

Use \code{I(list(NULL))} to include the no-embedding specification for
\code{scalar_embed_dim}. The internal tuning utilities interpret an empty or
all-\code{NA} candidate for this argument as \code{NULL}.
}
\value{
A data frame containing one row for every combination of the supplied
candidates and one column for every argument in \code{...}. Character
columns are not converted to factors. Protected vector-valued candidates are
retained as list-like columns suitable for processing by the AdaDTN tuning
utilities.
}
\examples{
\dontrun{
## Atomic vectors define alternative scalar values. Complete architectures
## are protected as list elements so that each occupies one grid cell.
grid_ada <- AdaDTN_param_grid(
  n_base_in = I(list(6, 8)),
  basis_hidden_in = I(list(c(64, 64))),
  dense_hidden = I(list(c(32, 32), c(64, 32))),
  basis_dropout = c(0.05, 0.10),
  dense_dropout = c(0.05, 0.10),
  basis_activation = "selu",
  dense_activation = "relu",
  scalar_embed_dim = I(list(NULL)),
  lambda_in_l1 = 0,
  lambda_in_ortho = c(1e-4, 5e-4),
  lambda_surface_l2 = c(1e-4, 5e-4),
  l2_hidden = c(1e-4, 5e-4),
  epochs = 250,
  batch_size = 64,
  lr = c(5e-4, 1e-3),
  lr_min = 1e-5,
  patience = 35,
  cal_prop = 0.20,
  val_prop = 0.10
)

nrow(grid_ada)
names(grid_ada)
grid_ada[1:3, c("n_base_in", "dense_hidden", "lr")]

## Evaluate the grid and optionally refit the selected configuration.
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
tune_ada$best_params
tune_ada$best_model
}
}
