subject_split <- function(n, cal_prop = 0.20, val_prop = 0.10) {
  
  idx <- sample.int(n)
  
  n_cal <- max(1, floor(cal_prop * n))
  n_cal <- min(n_cal, n - 2)
  
  cal_idx <- idx[seq_len(n_cal)]
  rem_idx <- idx[-seq_len(n_cal)]
  
  n_val <- max(1, floor(val_prop * length(rem_idx)))
  n_val <- min(n_val, length(rem_idx) - 1)
  
  val_idx <- rem_idx[seq_len(n_val)]
  trn_idx <- rem_idx[-seq_len(n_val)]
  
  list(train = trn_idx, valid = val_idx, calib = cal_idx)
}