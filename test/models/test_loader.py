
import torch
from train.load.loader import load_pf_dataset, load_gluonts_dataset
import pandas as pd
from pathlib import Path
import os

def test_loader_pf_dry_run():
    """Test pytorch-forecasting dataset loading."""
    try:
        training_ds, validation_ds = load_pf_dataset("crypto", prediction_length=24, context_length=48)
        print(f"PF datasets created: train={len(training_ds)}, val={len(validation_ds)}")
        
        # Create a dataloader from the dataset
        loader = training_ds.to_dataloader(train=True, batch_size=32)
        print("Dataloader created successfully.")
    except FileNotFoundError as e:
        print(f"Skipping actual load: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise

def test_loader_gluonts_dry_run():
    """Test GluonTS dataset loading."""
    try:
        train_ds, test_ds = load_gluonts_dataset("crypto", prediction_length=24)
        print(f"GluonTS datasets created successfully.")
    except FileNotFoundError as e:
        print(f"Skipping actual load: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_loader_pf_dry_run()
    test_loader_gluonts_dry_run()
