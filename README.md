# AdaDTN

`AdaDTN` provides an R implementation of the **adaptive deep tensor network**
for nonlinear function-on-function regression with multiple functional and
scalar predictors.

The method learns predictor-specific functional bases end-to-end through
differentiable quadrature. The resulting basis scores are mapped to the
response grid through coefficient matrices, while a deep network captures
nonlinearities and interactions. The fitted model also provides interpretable
bases and coefficient surfaces, hyperparameter tuning, and split-conformal
prediction bands.

## Main features

- Response-guided adaptive basis learning for functional predictors
- Multiple functional predictors with optional scalar covariates
- Nonlinear and interaction effects through a deep prediction head
- Extractable bases, scores, features, intercepts, and coefficient surfaces
- Cross-validation and grid-based hyperparameter tuning
- Split-conformal prediction bands
- CPU and CUDA-enabled training through [`torch`](https://torch.mlverse.org/)

## Installation

Install the development version from GitHub:

```r
install.packages("remotes")
remotes::install_github("UfukBeyaztas/AdaDTN")
```

The model uses a CUDA device automatically when one is available; otherwise,
it runs on the CPU. See the
[`torch` installation guide](https://torch.mlverse.org/docs/articles/installation)
for platform-specific and GPU instructions.

## Data format

- `X_func`: a list of functional-predictor matrices, each with subjects in
  rows and grid points in columns
- `X_scalar`: an optional subject-by-covariate matrix
- `Y`: a subject-by-response-grid matrix
- `grid_in`: a list containing the grid for each functional predictor
- `grid_out`: the response grid

## Quick example

```r
library(AdaDTN)

train <- simulate_AdaDTN_data(n = 150, j = 51, model = "complex")
test  <- simulate_AdaDTN_data(n = 50, j = 51, model = "complex")

fit <- AdaDTN_fit(
  X_func = train$x,
  X_scalar = train$x.scl,
  Y = train$y,
  grid_in = rep(list(train$meta$sx), length(train$x)),
  grid_out = train$meta$sy,
  n_base_in = rep(4, length(train$x)),
  epochs = 100,
  patience = 15,
  verbose = TRUE
)

fit
summary(fit)

# Mean prediction
pred <- predict(
  fit,
  new_X_func = test$x,
  new_X_scalar = test$x.scl
)
mean((pred - test$yt)^2)

# 95% split-conformal prediction bands
bands <- predict(
  fit,
  new_X_func = test$x,
  new_X_scalar = test$x.scl,
  interval = "conformal",
  level = 0.95
)

# Training history, learned basis, and coefficient surface
plot(fit, type = "history")
plot(fit, type = "basis", predictor_idx = 1)
plot(fit, type = "surface", predictor_idx = 1)
```

## Principal functions

| Function | Purpose |
|---|---|
| `AdaDTN_fit()` | Fit an AdaDTN model |
| `predict()` | Produce mean predictions or conformal bands |
| `AdaDTN_param_grid()` | Construct a hyperparameter grid |
| `AdaDTN_cv()` | Evaluate one configuration by cross-validation |
| `AdaDTN_tune()` | Tune configurations and optionally refit the best model |
| `simulate_AdaDTN_data()` | Generate linear, nonlinear, or complex simulated data |
| `get_AdaDTN_*()` | Extract learned bases, scores, features, intercepts, and surfaces |
| `plot()` / `plot_AdaDTN_surface()` | Visualize training and learned components |

## Documentation

The complete package reference is available in the
[AdaDTN 1.0.0 manual](AdaDTN_1.0.0.pdf).

From R, documentation for individual functions can be opened with, for
example:

```r
help(AdaDTN_fit)
help(AdaDTN_tune)
help(predict.AdaDTN_model)
```
## License

`AdaDTN` is distributed under the
[GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.en.html).
