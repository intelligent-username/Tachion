
import torch
from train.loader import make_dataloader
import pandas as pd
from pathlib import Path
import os

def test_loader_dry_run():
    # Attempt to create a dataloader for crypto
    # If no data exists, it should raise a FileNotFoundError which we'll catch
    try:
        loader = make_dataloader("crypto", batch_size=32)
        print("Dataloader created successfully.")
        
        # Try to get one batch if possible
        # for x, y in loader:
        #     print(f"Batch X shape: {x.shape}")
        #     print(f"Batch y shape: {y.shape}")
        #     break
    except FileNotFoundError as e:
        print(f"Skipping actual load: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_loader_dry_run()
