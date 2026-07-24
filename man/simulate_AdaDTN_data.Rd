\name{simulate_AdaDTN_data}
\alias{simulate_AdaDTN_data}
\title{Simulate Function-on-Function Data for AdaDTN}
\description{
Generates synthetic function-on-function regression data with five functional
predictors, three scalar predictors, and a functional response observed on
regular grids over \eqn{[0,1]}{[0, 1]}. Three mechanisms are available:
a linear baseline, a nonlinear interaction model, and a complex model with
localized predictor directions and nonlinear mixed-predictor effects.

The noise-free response and the response contaminated by a rescaled
Ornstein--Uhlenbeck error process are both returned.
}
\usage{
simulate_AdaDTN_data(
  n,
  j,
  model = c("linear", "nonlinear", "complex"),
  n_func = 5,
  n_scl = 3
)
}
\arguments{
  \item{n}{
  Positive integer giving the number of simulated subjects.
  }

  \item{j}{
  Integer giving the common number of predictor- and response-grid points.
  Both grids are constructed as \code{seq(0, 1, length.out = j)}. A value
  of at least two is required for a meaningful functional grid and for the
  Ornstein--Uhlenbeck simulation.
  }

  \item{model}{
  Character string selecting the response mechanism. Available choices are
  \code{"linear"}, \code{"nonlinear"}, and \code{"complex"}. The default
  is \code{"linear"}. Partial matching is performed by
  \code{\link[base]{match.arg}}.
  }

  \item{n_func}{
  Integer giving the requested number of functional predictors. The default
  and intended value is 5. The supplied nonlinear and complex mechanisms are
  hard-coded for five functional predictors, and the complex predictor
  generator contains five predictor-specific location pairs. Consequently,
  \code{n_func = 5} should be retained for the documented data-generating
  processes; see Details.
  }

  \item{n_scl}{
  Integer giving the requested number of scalar predictors. The default and
  intended value is 3. The scalar response profiles and the nonlinear and
  complex mechanisms are explicitly written for three scalar predictors.
  Consequently, \code{n_scl = 3} should be retained for the documented
  data-generating processes.
  }
}
\details{
The predictor and response grids are identical regular sequences,
\deqn{
s_g
=
\frac{g-1}{j-1},
\qquad
t_m
=
\frac{m-1}{j-1},
\qquad
g,m=1,\ldots,j.
}{
s[g] = (g - 1) / (j - 1),
and t[m] = (m - 1) / (j - 1).
}
They are returned as \code{meta$sx} and \code{meta$sy}. Although the
numerical grids coincide, they represent distinct predictor and response
domains.

For compactness, define the Gaussian bump
\deqn{
B(x;c,w)
=
\exp
\left\{
-\frac{1}{2}
\left(
\frac{x-c}{w}
\right)^2
\right\}
}{
B(x;c,w) = exp(-0.5 * ((x-c)/w)^2)
}
and the localized wave
\deqn{
W(x;c,w,f)
=
\sin(f\pi x)B(x;c,w).
}{
W(x;c,w,f) = sin(f*pi*x) * B(x;c,w).
}
These functions are implemented internally as \code{gauss_bump} and
\code{smooth_wave}.

When \code{model} is \code{"linear"} or \code{"nonlinear"}, each
functional predictor is a rank-one random curve
\deqn{
X_i^{(k)}(s_g)
=
\xi_{ik}\phi_k(s_g).
}{
X[i,k](s[g]) = xi[i,k] * phi[k](s[g]).
}
For \eqn{k\leq3}{k <= 3},
\deqn{
\xi_{ik}
\mathrel{\sim}
N
\left(
4,\frac{1}{k^2}
\right),
\qquad
\phi_k(s)
=
\sin\{(k+1)\pi s\}.
}{
xi[i,k] is normal with mean 4 and sd 1/k,
and phi[k](s) = sin((k+1)*pi*s).
}
For \eqn{k>3}{k > 3},
\deqn{
\xi_{ik}
\mathrel{\sim}
N
\left(
0,\frac{16}{k^8}
\right),
\qquad
\phi_k(s)
=
\cos\{(k-2)\pi s\}.
}{
xi[i,k] is normal with mean 0 and sd 4/k^4,
and phi[k](s) = cos((k-2)*pi*s).
}
Thus, the first three functional predictors have relatively strong
nonzero-mean amplitudes, whereas later predictors have centered amplitudes
whose standard deviations decrease rapidly with \eqn{k}{k}. No additional
pointwise measurement noise is added to these rank-one predictors.

When \code{model = "complex"}, every functional predictor contains four
random structured components plus independent pointwise noise. For predictor
\eqn{k\in\{1,\ldots,5\}}{k = 1, ..., 5}, independently generate
\deqn{
a_{i1}^{(k)}
\mathrel{\sim}
N(0,1^2),
\quad
a_{i2}^{(k)}
\mathrel{\sim}
N(0,0.9^2),
\quad
a_{i3}^{(k)}
\mathrel{\sim}
N(0,0.7^2),
\quad
a_{i4}^{(k)}
\mathrel{\sim}
N(0,0.5^2).
}{
a1, a2, a3, and a4 are independent centered normals
with standard deviations 1, 0.9, 0.7, and 0.5.
}
The predictor-specific centers are
\deqn{
\boldsymbol{c}_1
=
(0.15,0.28,0.40,0.58,0.76),
\qquad
\boldsymbol{c}_2
=
(0.25,0.38,0.55,0.70,0.86).
}{
c1 = (0.15, 0.28, 0.40, 0.58, 0.76),
and c2 = (0.25, 0.38, 0.55, 0.70, 0.86).
}
Using the \eqn{k}{k}-th entries of these vectors, define
\deqn{
b_{k1}(s)
=
B(s;c_{1k},0.06),
\qquad
b_{k2}(s)
=
B(s;c_{2k},0.08),
}{
b[k,1](s) = B(s;c1[k],0.06),
and b[k,2](s) = B(s;c2[k],0.08),
}
\deqn{
b_{k3}(s)
=
W(s;c_{1k}+0.10,0.09,5+k),
}{
b[k,3](s) = W(s;c1[k]+0.10,0.09,5+k),
}
and
\deqn{
b_{k4}(s)
=
\cos\{(4+k)\pi s\}
B(s;0.82,0.10).
}{
b[k,4](s) = cos((4+k)*pi*s) * B(s;0.82,0.10).
}
The curve is
\deqn{
X_i^{(k)}(s_g)
=
\sum_{r=1}^{4}
a_{ir}^{(k)}b_{kr}(s_g)
+\delta_{ikg},
\qquad
\delta_{ikg}
\mathrel{\sim}
N(0,0.03^2).
}{
X[i,k](s[g]) =
sum[r=1,...,4] a[i,r,k] b[k,r](s[g]) + delta[i,k,g],
where delta is normal with sd 0.03.
}
The resulting predictors contain localized bumps, localized oscillations,
and a small pointwise-noise component. The coefficient draws and pointwise
errors are generated separately for every functional predictor.

The function constructs five \eqn{j\times j}{j x j} matrices from
\deqn{
\beta_1(s,t)
=
\exp\{-2(s-0.5)^2\}\sin(6\pi t),
}{
beta[1](s,t) = exp(-2*(s-0.5)^2) * sin(6*pi*t),
}
\deqn{
\beta_2(s,t)
=
\cos(6\pi s)\exp\{-2(t-0.5)^2\},
}{
beta[2](s,t) = cos(6*pi*s) * exp(-2*(t-0.5)^2),
}
\deqn{
\beta_3(s,t)
=
\sin(5\pi s)\cos(5\pi t),
}{
beta[3](s,t) = sin(5*pi*s) * cos(5*pi*t),
}
\deqn{
\beta_4(s,t)
=
4st\exp\{-2(s^2+t^2)\},
}{
beta[4](s,t) = 4*s*t*exp(-2*(s^2+t^2)),
}
and
\deqn{
\beta_5(s,t)
=
5\sqrt{s+0.1}\log(t+1).
}{
beta[5](s,t) = 5*sqrt(s+0.1)*log(t+1).
}
The corresponding matrices are returned in \code{meta$beta}.

Only the first three surfaces enter the common baseline functional signal:
\deqn{
F_i(t_m)
=
\sum_{k=1}^{3}
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(k)}(s_g)
\beta_k(s_g,t_m).
}{
F[i](t[m]) =
sum[k=1,2,3] X[i,k] %*% beta[k] / j.
}
The implementation uses the equal-weight factor \eqn{1/j}{1/j}; it does not
use trapezoidal quadrature weights in the data generator.

Although five surfaces are stored, \eqn{\beta_4}{beta[4]} and
\eqn{\beta_5}{beta[5]} do not enter this baseline linear sum. In the
nonlinear mechanism, predictors 4 and 5 enter through separate nonlinear
terms. In the complex mechanism, all five predictors enter through localized
projections described below.

With the intended setting \code{n_scl = 3}, the scalar predictors are
independent draws
\deqn{
Z_{i\ell}
\mathrel{\sim}
N(2,1),
\qquad
\ell=1,2,3.
}{
Z[i,l] is normal with mean 2 and sd 1,
for l = 1, 2, 3.
}
Their response profiles are
\deqn{
\gamma_1(t)
=
\sin(4\pi t),
\qquad
\gamma_2(t)
=
\cos(5\pi t),
}{
gamma[1](t) = sin(4*pi*t),
and gamma[2](t) = cos(5*pi*t),
}
and
\deqn{
\gamma_3(t)
=
2\sin(3\pi t)\exp(-t).
}{
gamma[3](t) = 2*sin(3*pi*t)*exp(-t).
}
The scalar signal and common baseline are
\deqn{
S_i(t_m)
=
\sum_{\ell=1}^{3}
Z_{i\ell}\gamma_\ell(t_m),
\qquad
H_i(t_m)
=
F_i(t_m)+S_i(t_m).
}{
S[i](t[m]) = sum[l=1,2,3] Z[i,l] gamma[l](t[m]),
and H[i](t[m]) = F[i](t[m]) + S[i](t[m]).
}
The matrix \eqn{H}{H} is called \code{base_signal} internally.

For \code{model = "linear"}, the noise-free response is exactly the common
baseline:
\deqn{
Y_i^{\mathrm{true}}(t_m)
=
H_i(t_m).
}{
Y_true[i,m] = H[i,m].
}
Only functional predictors 1--3 enter the response through the baseline
surfaces, while all three scalar predictors enter through their response
profiles. Functional predictors 4 and 5 are generated but do not affect the
linear response.

For \code{model = "nonlinear"}, five nonlinear terms are added to the common
baseline.

First, predictors 1 and 2 interact pointwise through
\deqn{
N_{i1}(t_m)
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(1)}(s_g)
X_i^{(2)}(s_g)
\left[
12\exp
\left\{
-15
\left(
(s_g-0.4)^2
+(t_m-0.6)^2
\right)
\right\}
\right].
}{
N1[i,m] =
(X1[i, ] * X2[i, ]) %*%
12*exp(-15*((s-0.4)^2 + (t-0.6)^2)) / j.
}

Second, form the response-grid vector
\deqn{
R_{i3}(t_m)
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(3)}(s_g)\beta_3(s_g,t_m)
}{
R3[i,m] = X3[i, ] %*% beta[3][,m] / j
}
and average it over the response grid,
\deqn{
\bar{R}_{i3}
=
\frac{1}{j}
\sum_{m=1}^{j}
R_{i3}(t_m).
}{
R3_bar[i] = mean[m] R3[i,m].
}
The second nonlinear term is
\deqn{
N_{i2}(t_m)
=
\sin(2\pi\bar{R}_{i3})
\left[
8\sin(6\pi t_m)\cos(2\pi t_m)
\right].
}{
N2[i,m] =
sin(2*pi*R3_bar[i]) *
8*sin(6*pi*t[m])*cos(2*pi*t[m]).
}

Third, functional predictor 4 interacts with scalar predictor 1:
\deqn{
N_{i3}(t_m)
=
\frac{1}{j}
\sum_{g=1}^{j}
\left\{
X_i^{(4)}(s_g)Z_{i1}
\right\}
\left[
10s_gt_m\exp(-3s_gt_m)
\right].
}{
N3[i,m] =
(X4[i, ] * Z[i,1]) %*%
10*s*t*exp(-3*s*t) / j.
}

Fourth, scalar predictors 2 and 3 form
\deqn{
N_{i4}(t_m)
=
Z_{i2}^2Z_{i3}
\left[
7\exp(-t_m)\cos(4\pi t_m)
\right].
}{
N4[i,m] =
Z[i,2]^2 * Z[i,3] *
7*exp(-t[m])*cos(4*pi*t[m]).
}

Fifth, functional predictor 5 interacts with scalar predictor 1:
\deqn{
N_{i5}(t_m)
=
\arctan
\left\{
Z_{i1}
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(5)}(s_g)
\right\}
\left[
6t_m\sin(3\pi t_m)
\right].
}{
N5[i,m] =
atan(mean[g] X5[i,g] * Z[i,1]) *
6*t[m]*sin(3*pi*t[m]).
}

Let
\eqn{N=N_1+\cdots+N_5}{N = N1 + ... + N5}. The implementation calculates
the root-mean-square magnitudes
\deqn{
\|H\|_{\mathrm{RMS}}
=
\left\{
\frac{1}{nj}
\sum_{i=1}^{n}
\sum_{m=1}^{j}
H_i(t_m)^2
\right\}^{1/2},
}{
RMS(H) = sqrt(mean(H^2)),
}
and analogously \eqn{\|N\|_{\mathrm{RMS}}}{RMS(N)}. It rescales the complete
nonlinear sum to have twice the RMS magnitude of the baseline:
\deqn{
\widetilde{N}
=
N
\frac{
2\|H\|_{\mathrm{RMS}}
}{
\|N\|_{\mathrm{RMS}}
}.
}{
N_tilde = N * (2 * RMS(H) / RMS(N)).
}
The noise-free response is
\deqn{
Y^{\mathrm{true}}
=
H+\widetilde{N}.
}{
Y_true = H + N_tilde.
}
The rescaling is applied jointly to the sum of all five nonlinear terms, not
to the terms separately.

For \code{model = "complex"}, each of the five functional predictors is
reduced through a localized direction:
\deqn{
p_{i1}
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(1)}(s_g)B(s_g;0.18,0.06),
}{
p[i,1] = mean-grid projection of X1 onto B(s;0.18,0.06),
}
\deqn{
p_{i2}
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(2)}(s_g)B(s_g;0.42,0.07),
}{
p[i,2] = mean-grid projection of X2 onto B(s;0.42,0.07),
}
\deqn{
p_{i3}
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(3)}(s_g)B(s_g;0.64,0.07),
}{
p[i,3] = mean-grid projection of X3 onto B(s;0.64,0.07),
}
\deqn{
p_{i4}
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(4)}(s_g)W(s_g;0.72,0.10,7),
}{
p[i,4] = mean-grid projection of X4 onto W(s;0.72,0.10,7),
}
and
\deqn{
p_{i5}
=
\frac{1}{j}
\sum_{g=1}^{j}
X_i^{(5)}(s_g)W(s_g;0.85,0.08,9).
}{
p[i,5] = mean-grid projection of X5 onto W(s;0.85,0.08,9).
}

The five response profiles are
\deqn{
g_1(t)
=
B(t;0.16,0.09),
}{
g[1](t) = B(t;0.16,0.09),
}
\deqn{
g_2(t)
=
B(t;0.36,0.10)
\left\{
1+0.25\sin(6\pi t)
\right\},
}{
g[2](t) = B(t;0.36,0.10) * (1+0.25*sin(6*pi*t)),
}
\deqn{
g_3(t)
=
B(t;0.58,0.11)\cos(5\pi t),
}{
g[3](t) = B(t;0.58,0.11) * cos(5*pi*t),
}
\deqn{
g_4(t)
=
B(t;0.76,0.09)\sin(7\pi t),
}{
g[4](t) = B(t;0.76,0.09) * sin(7*pi*t),
}
and
\deqn{
g_5(t)
=
B(t;0.88,0.07)
\left\{
1+0.20\cos(8\pi t)
\right\}.
}{
g[5](t) = B(t;0.88,0.07) * (1+0.20*cos(8*pi*t)).
}

The subject-specific nonlinear coefficients are
\deqn{
z_{i1}
=
\tanh(2.2p_{i1}),
}{
z[i,1] = tanh(2.2*p[i,1]),
}
\deqn{
z_{i2}
=
\tanh(1.8p_{i2})
\tanh(1.8p_{i3}),
}{
z[i,2] = tanh(1.8*p[i,2]) * tanh(1.8*p[i,3]),
}
\deqn{
z_{i3}
=
\sin(2.5p_{i4}+0.6Z_{i1}),
}{
z[i,3] = sin(2.5*p[i,4] + 0.6*Z[i,1]),
}
\deqn{
z_{i4}
=
\tanh
\left[
1.5
\left\{
p_{i2}+0.7Z_{i2}
\right\}
\right],
}{
z[i,4] = tanh(1.5*(p[i,2] + 0.7*Z[i,2])),
}
and
\deqn{
z_{i5}
=
\arctan(1.8p_{i5}+0.5Z_{i3}).
}{
z[i,5] = atan(1.8*p[i,5] + 0.5*Z[i,3]).
}
Before balancing, term \eqn{r}{r} is the outer product
\deqn{
C_{ir}(t_m)
=
z_{ir}g_r(t_m).
}{
C[r][i,m] = z[i,r] * g[r](t[m]).
}

The reference standard deviation is
\deqn{
\sigma_{\mathrm{ref}}
=
0.45
\operatorname{sd}
\left\{
H_i(t_m):
i=1,\ldots,n,\;
m=1,\ldots,j
\right\}.
}{
sd_ref = 0.45 * sd(as.vector(H)).
}
Each nonlinear term is rescaled separately:
\deqn{
\widetilde{C}_r
=
C_r
\frac{
\sigma_{\mathrm{ref}}
}{
\operatorname{sd}\{\operatorname{vec}(C_r)\}
}.
}{
C_tilde[r] =
C[r] * (sd_ref / sd(as.vector(C[r]))).
}
If a term's empirical standard deviation is zero or nonfinite, the internal
\code{balance_term} function returns that term unchanged.

Finally,
\deqn{
Y^{\mathrm{true}}
=
0.45H
+\sum_{r=1}^{5}
\widetilde{C}_r.
}{
Y_true = 0.45*H + sum[r=1,...,5] C_tilde[r].
}
Because the balanced terms need not be mutually uncorrelated, this
construction fixes their individual empirical standard deviations but does
not fix the standard deviation of their sum.
}
\value{
A list with the following components:
\describe{
  \item{\code{y}}{
  Numeric \eqn{n\times j}{n x j} matrix containing the observed functional
  responses \eqn{Y^{\mathrm{obs}}}{Y_obs}. Rows are subjects and columns are
  response-grid points.
  }

  \item{\code{yt}}{
  Numeric \eqn{n\times j}{n x j} matrix containing the noise-free functional
  responses \eqn{Y^{\mathrm{true}}}{Y_true}.
  }

  \item{\code{x}}{
  List of \code{n_func} numeric \eqn{n\times j}{n x j} matrices containing
  the functional predictors.
  }

  \item{\code{x.scl}}{
  Numeric \eqn{n\times\mathrm{n\_scl}}{n x n_scl} matrix containing the
  scalar predictors.
  }

  \item{\code{meta}}{
  A list with:
  \describe{
    \item{\code{sx}}{
    Numeric predictor grid of length \code{j}.
    }
    \item{\code{sy}}{
    Numeric response grid of length \code{j}.
    }
    \item{\code{beta}}{
    List of five baseline \eqn{j\times j}{j x j} coefficient-surface
    matrices.
    }
    \item{\code{model}}{
    Character string identifying the selected mechanism.
    }
  }
  }
}
}
\examples{
\dontrun{

## Generate independent training and test samples from the complex mechanism.
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

str(train_dat, max.level = 2)
lapply(train_dat$x, dim)
dim(train_dat$x.scl)
dim(train_dat$y)
dim(train_dat$yt)

## Inspect the noise scale created by the empirical rescaling step.
signal_sd <- sd(as.vector(train_dat$yt))
noise_sd <- sd(as.vector(train_dat$y - train_dat$yt))
c(
  signal_sd = signal_sd,
  noise_sd = noise_sd,
  noise_to_signal_sd = noise_sd / signal_sd
)

## Display several simulated functional predictors and responses.
matplot(
  train_dat$meta$sx,
  t(train_dat$x[[1]][1:10, ]),
  type = "l",
  lty = 1,
  xlab = "s",
  ylab = expression(X^{(1)}(s))
)

matplot(
  train_dat$meta$sy,
  t(train_dat$y[1:10, ]),
  type = "l",
  lty = 1,
  xlab = "t",
  ylab = "Y(t)"
)

## Fit AdaDTN using the returned package-ready structure.
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
  epochs = 400,
  batch_size = 64,
  lr = 1e-3,
  lr_min = 1e-5,
  patience = 40,
  cal_prop = 0.20,
  val_prop = 0.10,
  verbose = TRUE
)

pred <- predict(
  fit,
  new_X_func = test_dat$x,
  new_X_scalar = test_dat$x.scl,
  interval = "none"
)

mean((pred - test_dat$yt)^2)

## Generate all three mechanisms with the intended dimensions.
linear_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "linear"
)

nonlinear_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "nonlinear"
)

complex_dat <- simulate_AdaDTN_data(
  n = 200,
  j = 101,
  model = "complex"
)
}
}
