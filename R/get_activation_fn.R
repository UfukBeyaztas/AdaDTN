get_activation_fn <- function(name) {
  nm <- tolower(name)
  switch(
    nm,
    relu      = nnf_relu,
    selu      = nnf_selu,
    elu       = nnf_elu,
    tanh      = torch_tanh,
    sigmoid   = torch_sigmoid,
    identity  = function(x) x,
    linear    = function(x) x,
    leakyrelu = function(x) nnf_leaky_relu(x, negative_slope = 0.01),
    stop("Unsupported activation: ", name)
  )
}
