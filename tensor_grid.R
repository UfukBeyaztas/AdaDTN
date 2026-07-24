tensor_grid <- function(x, device) {
  torch::torch_tensor(as.numeric(x), dtype = torch::torch_float(), device = device)
}