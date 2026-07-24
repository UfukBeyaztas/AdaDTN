simulate_AdaDTN_data <- function(n,
                                 j,
                                 model   = c("linear", "nonlinear", "complex"),
                                 n_func  = 5,
                                 n_scl   = 3) {
  
  model <- match.arg(model)
  
  sx <- seq(0, 1, length.out = j)
  sy <- seq(0, 1, length.out = j)
  
  gauss_bump <- function(x, c, w) {
    exp(-0.5 * ((x - c) / w)^2)
  }
  
  smooth_wave <- function(x, c, w, freq) {
    sin(freq * pi * x) * exp(-0.5 * ((x - c) / w)^2)
  }
  
  build_fX_standard <- function(k) {
    ksi <- if(k <= 3)
      matrix(rnorm(n,  4, sd = 1 / k), n, 1)
    else
      matrix(rnorm(n,  0, sd = 4 / k^4), n, 1)
    
    phi <- if(k <= 3)
      sin((k+1) * pi * sx)
    else
      cos((k-2) * pi * sx)
    
    ksi %*% t(phi)
  }
  
  build_fX_complex <- function(k) {
    a1 <- rnorm(n, 0, 1.0)
    a2 <- rnorm(n, 0, 0.9)
    a3 <- rnorm(n, 0, 0.7)
    a4 <- rnorm(n, 0, 0.5)
    
    c1 <- c(0.15, 0.28, 0.40, 0.58, 0.76)[k]
    c2 <- c(0.25, 0.38, 0.55, 0.70, 0.86)[k]
    
    b1 <- gauss_bump(sx, c1, 0.06)
    b2 <- gauss_bump(sx, c2, 0.08)
    b3 <- smooth_wave(sx, c1 + 0.10, 0.09, 5 + k)
    b4 <- cos((4 + k) * pi * sx) * gauss_bump(sx, 0.82, 0.10)
    
    Xk <- outer(a1, b1) +
      outer(a2, b2) +
      outer(a3, b3) +
      outer(a4, b4)
    
    Xk + matrix(rnorm(n * j, 0, 0.03), nrow = n, ncol = j)
  }
  
  if (model == "complex") {
    fX <- lapply(seq_len(n_func), build_fX_complex)
  } else {
    fX <- lapply(seq_len(n_func), build_fX_standard)
  }
  
  fBeta <- list(
    function(s, t)  exp(-2*(s-0.5)^2) * sin(6*pi*t),
    function(s, t)  cos(6*pi*s) * exp(-2*(t-0.5)^2),
    function(s, t)  sin(5*pi*s) * cos(5*pi*t),
    function(s, t)  4 * s * t * exp(-2*(s^2 + t^2)),
    function(s, t)  5 * sqrt(s+0.1) * log(t+1)
  )
  vBeta <- lapply(fBeta, function(f) outer(sx, sy, f))
  
  strong_idx <- 1:3
  fY_signal <- Reduce("+", lapply(strong_idx, function(k) {
    fX[[k]] %*% vBeta[[k]] / j
  }))
  
  bt <- list(
    sin(4 * pi * sy),
    cos(5 * pi * sy),
    2 * sin(3*pi*sy) * exp(-sy)
  )
  
  x_scl <- replicate(n_scl, rnorm(n, 2, 1), simplify = "matrix")
  scalar_signal <- Reduce("+", Map(function(z, b) z %*% t(b),
                                   as.data.frame(x_scl), bt))
  
  base_signal <- fY_signal + scalar_signal
  
  if(model == "linear") {
    
    fY_true <- base_signal
    
  } else if(model == "nonlinear") {
    
    nl_surf1 <- outer(sx, sy, function(s,t) 12 * exp(-15*((s-0.4)^2 + (t-0.6)^2)))
    term1 <- (fX[[1]] * fX[[2]]) %*% nl_surf1 / j
    
    proj_X3 <- rowMeans(fX[[3]] %*% vBeta[[3]] / j)
    b2t <- 8 * sin(6*pi*sy) * cos(2*pi*sy)
    term2 <- outer(sin(2*pi*proj_X3), b2t)
    
    nl_surf3 <- outer(sx, sy, function(s,t) 10 * s * t * exp(-3*s*t))
    term3 <- (fX[[4]] * x_scl[,1]) %*% nl_surf3 / j
    
    b4t <- 7 * exp(-sy) * cos(4*pi*sy)
    term4 <- outer(x_scl[,2]^2 * x_scl[,3], b4t)
    
    b5t <- 6 * sy * sin(3*pi*sy)
    term5 <- outer(atan(rowMeans(fX[[5]]) * x_scl[,1]), b5t)
    
    nl_sum <- term1 + term2 + term3 + term4 + term5
    
    base_norm <- sqrt(mean(base_signal^2))
    nl_norm <- sqrt(mean(nl_sum^2))
    nl_sum <- nl_sum * (2.0 * base_norm / nl_norm)
    
    fY_true <- base_signal + nl_sum
    
  } else if(model == "complex") {
    
    p1 <- as.vector(fX[[1]] %*% gauss_bump(sx, 0.18, 0.06) / j)
    p2 <- as.vector(fX[[2]] %*% gauss_bump(sx, 0.42, 0.07) / j)
    p3 <- as.vector(fX[[3]] %*% gauss_bump(sx, 0.64, 0.07) / j)
    p4 <- as.vector(fX[[4]] %*% smooth_wave(sx, 0.72, 0.10, 7) / j)
    p5 <- as.vector(fX[[5]] %*% smooth_wave(sx, 0.85, 0.08, 9) / j)
    
    g1 <- gauss_bump(sy, 0.16, 0.09)
    g2 <- gauss_bump(sy, 0.36, 0.10) * (1 + 0.25 * sin(6*pi*sy))
    g3 <- gauss_bump(sy, 0.58, 0.11) * cos(5*pi*sy)
    g4 <- gauss_bump(sy, 0.76, 0.09) * sin(7*pi*sy)
    g5 <- gauss_bump(sy, 0.88, 0.07) * (1 + 0.20 * cos(8*pi*sy))
    
    z1 <- tanh(2.2 * p1)
    z2 <- tanh(1.8 * p2) * tanh(1.8 * p3)
    z3 <- sin(2.5 * p4 + 0.6 * x_scl[,1])
    z4 <- tanh(1.5 * (p2 + 0.7 * x_scl[,2]))
    z5 <- atan(1.8 * p5 + 0.5 * x_scl[,3])
    
    term1 <- outer(z1, g1)
    term2 <- outer(z2, g2)
    term3 <- outer(z3, g3)
    term4 <- outer(z4, g4)
    term5 <- outer(z5, g5)
    
    balance_term <- function(term, target_sd) {
      sd_term <- sd(as.vector(term))
      if (!is.finite(sd_term) || sd_term == 0) return(term)
      term * (target_sd / sd_term)
    }
    
    ref_sd <- 0.45 * sd(as.vector(base_signal))
    
    term1 <- balance_term(term1, ref_sd)
    term2 <- balance_term(term2, ref_sd)
    term3 <- balance_term(term3, ref_sd)
    term4 <- balance_term(term4, ref_sd)
    term5 <- balance_term(term5, ref_sd)
    
    complex_sum <- term1 + term2 + term3 + term4 + term5
    fY_true <- 0.45 * base_signal + complex_sum
  }
  
  if(!requireNamespace("goffda", quietly = TRUE))
    stop("Package 'goffda' is required for OU noise-generation.")
  
  err <- r_ou(n = n, t = sy, mu = 0, alpha = 3, sigma = 0.7,
                      x0 = rnorm(n, 0, 0.3))$data
  
  signal_sd <- sd(as.vector(fY_true))
  noise_sd <- sd(as.vector(err))
  err <- err * (0.1 * signal_sd / noise_sd)
  
  fY_obs <- fY_true + err
  
  list(
    y       = fY_obs,
    yt      = fY_true,
    x       = fX,
    x.scl   = x_scl,
    meta    = list(sx = sx, sy = sy, beta = vBeta, model = model)
  )
}