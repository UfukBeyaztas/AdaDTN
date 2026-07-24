\name{AdaDTN_fit}
\alias{AdaDTN_fit}
\title{Fit an Adaptive Deep Tensor Network}
\description{
Fits an adaptive deep tensor network (AdaDTN) for nonlinear
function-on-function regression with one or more functional predictors and
optional scalar predictors. The function learns predictor-specific marginal
basis functions, maps their quadrature scores to the response grid through
trainable coefficient matrices, and passes the resulting features through a
nonlinear dense network that predicts the complete response curve.

Model fitting includes a subject-level training, validation, and calibration
split; training-set standardization; Adam optimization with a cosine learning
rate schedule; validation-based early stopping; and computation of a pooled
calibration residual quantile for subsequent prediction intervals.
}
\usage{
AdaDTN_fit(
  X_func,
  X_scalar = NULL,
  Y,
  grid_in,
  grid_out,
  n_base_in,
  basis_hidden_in = c(64, 64),
  dense_hidden = c(64, 64),
  basis_dropout = 0.10,
  dense_dropout = 0.10,
  basis_activation = "selu",
  dense_activation = "relu",
  scalar_embed_dim = NULL,
  lambda_in_l1 = 0,
  lambda_in_ortho = 1e-04,
  lambda_surface_l2 = 1e-04,
  l2_hidden = 1e-04,
  epochs = 150,
  batch_size = 64,
  lr = 0.001,
  lr_min = 1e-05,
  patience = 20,
  cal_prop = 0.20,
  val_prop = 0.10,
  device = NULL,
  verbose = TRUE
)
}
\arguments{
  \item{X_func}{
  A nonempty list of numeric matrices containing the functional predictors.
  Element \code{X_func[[p]]} represents predictor \eqn{p}{p}, with subjects
  in rows and evaluations on its predictor-specific grid in columns. If there
  are \eqn{n}{n} subjects and predictor \eqn{p}{p} is observed at
  \eqn{G_p}{G[p]} points, this matrix must have dimension
  \eqn{n \times G_p}{n x G[p]}. All elements must have the same number of
  rows as \code{Y}.
  }

  \item{X_scalar}{
  An optional numeric matrix of scalar predictors, with subjects in rows and
  covariates in columns. It must have \eqn{n}{n} rows. The default
  \code{NULL} fits the model without scalar predictors.
  }

  \item{Y}{
  A numeric matrix containing the functional responses. Rows correspond to
  subjects and columns to evaluations on the response grid. For
  \eqn{M}{M} response-grid points, \code{Y} must have dimension
  \eqn{n \times M}{n x M}.
  }

  \item{grid_in}{
  A list of numeric vectors containing the observation grids of the
  functional predictors. Its length must equal \code{length(X_func)}, and
  \code{length(grid_in[[p]])} must equal \code{ncol(X_func[[p]])}.
  Grids should be finite, ordered increasingly, and contain at least two
  distinct points because they are used to construct trapezoidal quadrature
  weights. Predictor-specific grids and unequal grid sizes are permitted.
  }

  \item{grid_out}{
  A finite numeric vector containing the response grid. Its length must equal
  \code{ncol(Y)}. AdaDTN predicts the response directly at these grid
  locations; it does not reconstruct the response from a fixed output basis.
  }

  \item{n_base_in}{
  An integer vector giving the number \eqn{K_p}{K[p]} of learned marginal
  basis functions for each functional predictor. Its length must equal
  \code{length(X_func)}. Each value should be a positive integer.
  }

  \item{basis_hidden_in}{
  An integer vector giving the widths of the hidden layers in every marginal
  basis micro-network. The same architecture is used for all learned basis
  functions. For example, \code{c(64, 64)} creates two hidden layers of
  width 64 before the scalar output defining a basis function.
  }

  \item{dense_hidden}{
  A nonempty integer vector giving the widths of the dense hidden layers
  applied after the functional front-end and optional scalar preprocessing.
  At least one width is required by the current implementation.
  }

  \item{basis_dropout}{
  Dropout probability used after each activated hidden layer of every basis
  micro-network. It should lie in \eqn{[0,1)}{[0, 1)}.
  }

  \item{dense_dropout}{
  Dropout probability used after each activated dense hidden layer. It should
  lie in \eqn{[0,1)}{[0, 1)}.
  }

  \item{basis_activation}{
  Character string naming the activation function used in the basis
  micro-networks. Supported values are \code{"relu"}, \code{"selu"},
  \code{"elu"}, \code{"tanh"}, \code{"sigmoid"}, \code{"identity"},
  \code{"linear"}, and \code{"leakyrelu"}.
  }

  \item{dense_activation}{
  Character string naming the activation function used in the dense hidden
  network and, when requested, after the scalar embedding layer. Supported
  values are the same as for \code{basis_activation}.
  }

  \item{scalar_embed_dim}{
  Optional positive integer giving the dimension of a learned linear
  embedding of the scalar predictors. The embedding is followed by
  \code{dense_activation}. If \code{NULL}, scalar predictors are concatenated
  with the functional front-end features directly. This argument has no
  effect when \code{X_scalar = NULL}.
  }

  \item{lambda_in_l1}{
  Nonnegative coefficient for the integrated absolute-value penalty on the
  learned marginal basis functions. The default zero disables this penalty.
  }

  \item{lambda_in_ortho}{
  Nonnegative coefficient for the approximate orthogonality penalty on the
  weighted Gram matrices of the learned marginal bases.
  }

  \item{lambda_surface_l2}{
  Nonnegative coefficient for the mean squared penalty on the
  predictor-specific surface coefficient matrices.
  }

  \item{l2_hidden}{
  Nonnegative coefficient multiplying the sum of squared weights in the
  dense hidden layers and final output layer. Biases, the explicit intercept
  function, the optional scalar embedding, and the basis micro-network
  parameters are not included in this particular penalty.
  }

  \item{epochs}{
  Positive integer giving the maximum number of training epochs.
  }

  \item{batch_size}{
  Positive integer giving the mini-batch size used during optimization and
  validation prediction.
  }

  \item{lr}{
  Positive numeric value giving the initial Adam learning rate and the upper
  endpoint of the cosine learning rate schedule.
  }

  \item{lr_min}{
  Positive numeric value giving the lower endpoint of the cosine learning
  rate schedule.
  }

  \item{patience}{
  Positive integer giving the number of consecutive epochs without a strict
  improvement in validation mean squared error that triggers early stopping.
  }

  \item{cal_prop}{
  Proportion of all subjects assigned to the calibration set. The internal
  splitter uses \code{floor(cal_prop * n)}, subject to retaining at least one
  subject in each of the training, validation, and calibration sets.
  }

  \item{val_prop}{
  Proportion of the subjects remaining after the calibration split that is
  assigned to the validation set. Thus, \code{val_prop} is not a proportion
  of the original full sample.
  }

  \item{device}{
  Optional character string specifying the torch device, commonly
  \code{"cpu"} or \code{"cuda"}. If \code{NULL}, CUDA is selected when
  available and CPU otherwise.
  }

  \item{verbose}{
  Logical value. If \code{TRUE}, device selection, periodic training
  progress, early stopping, final split-specific errors, and the stored
  calibration half-width are printed.
  }
}
\details{
Let \eqn{X_i^{(p)}}{X_i[p]} denote functional predictor \eqn{p}{p} for
subject \eqn{i}{i}, observed at the grid
\eqn{s_{p1},\ldots,s_{pG_p}}{s[p,1], ..., s[p,G[p]]}, and let the functional
response be observed at \eqn{t_1,\ldots,t_M}{t[1], ..., t[M]}. For each
predictor, AdaDTN learns \eqn{K_p}{K[p]} marginal basis functions
\eqn{\phi_{pk}}{phi[p,k]}. Each basis function is represented by a
micro multilayer perceptron taking one grid coordinate as input and returning
one scalar.

The current implementation evaluates these networks at the values supplied
in \code{grid_in}; it does not internally rescale the grids. The resulting
column vectors are normalized individually using trapezoidal weights
\eqn{w_{pg}}{w[p,g]}:
\deqn{
\bar{\phi}_{pk}(s_{pg}) =
\frac{\phi_{pk}(s_{pg})}
{\left\{\sum_{g=1}^{G_p}w_{pg}\phi_{pk}(s_{pg})^2+
10^{-8}\right\}^{1/2}} .
}{
bar(phi)[p,k](s[p,g]) =
phi[p,k](s[p,g]) /
sqrt(sum[g] w[p,g] phi[p,k](s[p,g])^2 + 1e-8).
}
This gives every learned basis vector approximately unit
\eqn{L^2}{L2} norm on its observation grid. Mutual orthogonality is
encouraged separately through \code{lambda_in_ortho}; normalization alone
does not make different basis functions orthogonal.

After training-set standardization, the score of subject \eqn{i}{i} on basis
\eqn{k}{k} of predictor \eqn{p}{p} is the differentiable trapezoidal
approximation
\deqn{
\xi_{ipk} =
\sum_{g=1}^{G_p} w_{pg}X_i^{(p)}(s_{pg})
\bar{\phi}_{pk}(s_{pg}) .
}{
xi[i,p,k] =
sum[g] w[p,g] X[i,p](s[p,g]) bar(phi)[p,k](s[p,g]).
}
The vector of \eqn{K_p}{K[p]} scores is multiplied by a trainable matrix
\eqn{A^{(p)}\in\mathbb{R}^{K_p\times M}}{A[p] of size K[p] x M}:
\deqn{
U_i^{(p)} = \boldsymbol{\xi}_{ip}^{\mathsf T} A^{(p)} .
}{
U[i,p] = xi[i,p]' A[p].
}
Consequently, the predictor-specific discretized front-end coefficient
surface is
\deqn{
\widehat{\beta}_p(s_{pg},t_m)
=
\sum_{k=1}^{K_p}\bar{\phi}_{pk}(s_{pg})A^{(p)}_{km}.
}{
beta_hat[p](s[p,g],t[m]) =
sum[k] bar(phi)[p,k](s[p,g]) A[p][k,m].
}
This surface is an interpretable description of the linear projection
performed before the nonlinear dense head. When the dense head is nonlinear,
it should not be interpreted as a complete marginal effect of the final
prediction, because the head may transform and interact the front-end
features.

The \eqn{M}{M}-dimensional vectors
\eqn{U_i^{(1)},\ldots,U_i^{(P)}}{U[i,1], ..., U[i,P]} are concatenated.
If scalar predictors are supplied, their standardized values are either
concatenated directly or transformed by one learned linear layer followed by
\code{dense_activation}, according to \code{scalar_embed_dim}.

The combined feature vector is passed through the hidden layers specified by
\code{dense_hidden}. Each hidden layer applies a linear transformation,
\code{dense_activation}, and dropout. A bias-free final linear layer maps
the last hidden representation to \eqn{M}{M} output coordinates, after which
a separately trainable \eqn{M}{M}-vector is added as an explicit intercept
function. The network therefore predicts the response directly on
\code{grid_out}.

Subjects are randomly permuted and then divided into three disjoint sets.
First,
\code{floor(cal_prop * n)} subjects are assigned to calibration, with
internal safeguards that reserve subjects for the other sets. From the
remaining subjects, \code{floor(val_prop * n_remaining)} are assigned to
validation, again with safeguards, and all remaining subjects are used for
gradient updates.

For every functional predictor, scalar predictor, and response, each grid
point or scalar column is standardized separately. If \eqn{x_{ij}}{x[i,j]}
is a value in column \eqn{j}{j}, the transformation is
\deqn{
x_{ij}^{\mathrm{sc}} =
\frac{x_{ij}-\widehat{\mu}_{j,\mathcal{T}}}
{\widehat{\sigma}_{j,\mathcal{T}}}.
}{
x_sc[i,j] = (x[i,j] - mean_T[j]) / sd_T[j].
}
The centers and scales are estimated only from the training subjects
\eqn{\mathcal{T}}{T} and are then applied to the validation and calibration
subjects. A nonfinite or zero training standard deviation is replaced by
one. This training-only construction prevents information from the
validation and calibration sets from entering the standardization
parameters.

For a mini-batch \eqn{\mathcal{B}}{B}, the optimized loss is the mean squared
error on the standardized response scale plus the enabled regularizers:
\deqn{
\mathcal{L}_{\mathcal{B}} =
\frac{1}{|\mathcal{B}|M}
\sum_{i\in\mathcal{B}}\sum_{m=1}^{M}
\left(Y_{im}^{\mathrm{sc}}-\widehat{Y}_{im}^{\mathrm{sc}}\right)^2
+ \mathcal{R}_{\mathrm{front}}
+ \lambda_{\mathrm{hidden}}\mathcal{R}_{\mathrm{hidden}} .
}{
L_B = MSE_B + R_front + l2_hidden * R_hidden.
}

The front-end penalty is accumulated across functional predictors. Its
possible components are:
\itemize{
  \item an integrated absolute-basis penalty equal, for predictor
  \eqn{p}{p}, to \code{lambda_in_l1} times the mean across its bases of
  \eqn{\sum_g w_{pg}|\bar{\phi}_{pk}(s_{pg})|}{
  sum[g] w[p,g] abs(bar(phi)[p,k](s[p,g]))};

  \item an approximate orthogonality penalty. With
  \eqn{\Phi_p}{Phi[p]} the matrix of normalized basis evaluations,
  \eqn{W_p=\mathrm{diag}(w_{p1},\ldots,w_{pG_p})}{W[p] = diag(w[p])}, and
  \eqn{G_p=\Phi_p^{\mathsf T}W_p\Phi_p}{G[p] = Phi[p]' W[p] Phi[p]},
  the implementation adds \code{lambda_in_ortho} times the mean absolute
  entry of \eqn{G_p-I_{K_p}}{G[p] - I[K[p]]}. This term is used only when
  \eqn{K_p>1}{K[p] > 1};

  \item a coefficient-matrix penalty equal to
  \code{lambda_surface_l2} times
  \eqn{\mathrm{mean}\{(A^{(p)})^2\}}{mean(A[p]^2)}.
}
The dense penalty \eqn{\mathcal{R}_{\mathrm{hidden}}}{R_hidden} is the sum
of squared weights in all dense hidden layers and in the final output layer.
It excludes biases, the explicit intercept function, the optional scalar
embedding layer, and all basis micro-network parameters.

The recorded \code{train_loss} is the mean of the total mini-batch losses
within an epoch. The recorded \code{valid_loss} is an unpenalized
standardized-scale mean squared error. In both cases the implementation
averages batch-level scalar losses; consequently, if the last batch is
smaller than the others, it receives the same weight as a full batch in
these reported epoch summaries.

Parameters are optimized by Adam. At epoch \eqn{e}{e}, the learning rate is
\deqn{
\mathrm{lr}_e =
\mathrm{lr}_{\min}
+\frac{1}{2}(\mathrm{lr}-\mathrm{lr}_{\min})
\left[
1+\cos\left\{
\frac{\pi(e-1)}{\max(1,\mathrm{epochs}-1)}
\right\}
\right].
}{
lr[e] = lr_min + 0.5 * (lr - lr_min) *
(1 + cos(pi * (e - 1) / max(1, epochs - 1))).
}
Training subjects are randomly reshuffled at every epoch. Validation loss is
evaluated after the epoch with dropout disabled. Whenever the validation
loss is strictly below the best value observed so far, the model state is
written to a temporary checkpoint. Training stops when
\code{patience} consecutive epochs fail to improve that value, or when
\code{epochs} is reached. The best checkpoint, rather than necessarily the
last epoch, is restored before calibration and final error calculations.

After restoring the best validation checkpoint, the function predicts the
calibration responses on the standardized response scale. It pools the
absolute residuals over both calibration subjects and response-grid points:
\deqn{
e_{jm}^{\mathrm{sc}} =
\left|Y_{jm}^{\mathrm{sc}}-
\widehat{Y}_{jm}^{\mathrm{sc}}\right| .
}{
e_sc[j,m] = abs(Y_sc[j,m] - Yhat_sc[j,m]).
}
The vector produced by column-major vectorization of these residuals is
stored as \code{cal_residuals_sc}. The value \code{q_hat} is exactly
\code{stats::quantile(cal_residuals_sc, probs = 0.95, type = 8,
names = FALSE)}. Thus, in this implementation, \code{q_hat} is fixed at the
0.95 quantile, is computed on the standardized response scale, uses
Hyndman--Fan type 8 interpolation, and pools all calibration grid residuals.
It is a stored 0.95-level summary. The package's
\code{predict.AdaDTN_model} method recomputes a type 8 quantile from
\code{cal_residuals_sc} when another value of \code{level} is requested.

Prediction limits based on these residuals must transform their
standardized-scale half-widths consistently back to the original response
scale. The package prediction method does this separately at every response
grid point by multiplying the standardized quantile by the stored response
scale. The pooled score targets average marginal behavior over the response
grid; it is not a simultaneous whole-curve score such as a subject-level
supremum residual. Moreover, the calculation in this fitting function does
not replace the requested probability by a finite-sample-inflated conformal
rank. Any formal coverage statement should therefore be matched to the
exact score, quantile, and exchangeability conditions implemented by the
corresponding prediction procedure.
}
\value{
An object of class \code{"AdaDTN_model"}, represented as a list with the
following components:
\describe{
  \item{\code{model}}{
  The fitted torch module, restored to the state having the lowest observed
  validation mean squared error.
  }

  \item{\code{scalers}}{
  A list containing the training-set column centers and scales for each
  functional predictor (\code{func}), the optional scalar predictors
  (\code{scalar}), and the functional response (\code{Y}).
  }

  \item{\code{history}}{
  A data frame with one row per completed epoch and columns \code{epoch},
  \code{lr}, \code{train_loss}, and \code{valid_loss}.
  }

  \item{\code{split}}{
  A list of integer subject indices named \code{train}, \code{valid}, and
  \code{calib}.
  }

  \item{\code{q_hat}}{
  The type 8 empirical 0.95 quantile of the pooled absolute calibration
  residuals on the standardized response scale.
  }

  \item{\code{cal_residuals_sc}}{
  A numeric vector containing the pooled absolute calibration residuals on
  the standardized response scale.
  }

  \item{\code{device}}{
  The character string identifying the torch device used for fitting.
  }

  \item{\code{grid_in}}{
  The supplied list of predictor grids.
  }

  \item{\code{grid_out}}{
  The supplied response grid.
  }

  \item{\code{n_base_in}}{
  The supplied numbers of learned marginal basis functions.
  }

  \item{\code{mse_train}}{
  Elementwise mean squared prediction error for the training subjects on the
  original response scale.
  }

  \item{\code{mse_valid}}{
  Elementwise mean squared prediction error for the validation subjects on
  the original response scale.
  }

  \item{\code{mse_calib}}{
  Elementwise mean squared prediction error for the calibration subjects on
  the original response scale.
  }

  \item{\code{config}}{
  A named list recording the fitting arguments
  \code{basis_hidden_in}, \code{dense_hidden}, \code{basis_dropout},
  \code{dense_dropout}, \code{basis_activation},
  \code{dense_activation}, \code{scalar_embed_dim},
  \code{lambda_in_l1}, \code{lambda_in_ortho},
  \code{lambda_surface_l2}, \code{l2_hidden}, \code{epochs},
  \code{batch_size}, \code{lr}, \code{lr_min}, \code{patience},
  \code{cal_prop}, and \code{val_prop}.
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

test_dat <- simulate_AdaDTN_data(
  n = 100,
  j = 101,
  model = "complex"
)

s <- train_dat$meta$sx
t <- train_dat$meta$sy

fit <- AdaDTN_fit(
  X_func = train_dat$x,
  X_scalar = train_dat$x.scl,
  Y = train_dat$y,
  grid_in = rep(list(s), length(train_dat$x)),
  grid_out = t,
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

## Original-scale test predictions.
pred <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "none"
)

mean((pred - test_dat$yt)^2)

## Stored training history and split-specific errors.
tail(fit$history)
c(
  train = fit$mse_train,
  validation = fit$mse_valid,
  calibration = fit$mse_calib
)

## Prediction intervals obtained from the model's calibration component.
pred_int <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "conformal",
  level = 0.95
)

## Inspect the first learned front-end coefficient surface.
plot_AdaDTN_surface(fit, predictor_idx = 1, smooth_spar = 0.75)
}
}
