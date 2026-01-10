import pnadas as pd

import torch

from deep_ar import DeepAR, StudentT, student_t_nll

# NOTE: still very much incomplete

asset = "crypto"

x = torch.load_parquet(f"data/{asset}/processed/{asset}_processed.parquet")  # (batch, seq_len, input_size)

y = x[:, :, 2]  # log_return target

model = DeepAR(input_size=5, hidden_size=64, num_layers=2, output_size=3)  # Student-t
loss = student_t_nll(model(x), y)
