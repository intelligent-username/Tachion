"""
Custom progress bar for PyTorch Lightning training.
"""

import sys
import time
from lightning.pytorch.callbacks import Callback
from core.training.constants import TRAIN_LOG_INTERVAL


class CleanProgressBar(Callback):
    """Clean progress bar with box-drawing characters."""
    
    def __init__(self, width: int = 30):
        self.width = width
        self.epoch = 0
        self.max_epochs = 0
        self.start_time = 0
        self.last_print_time = 0
    
    def on_train_start(self, trainer, pl_module):
        self.max_epochs = trainer.max_epochs
        print("\n┌" + "─" * 64 + "┐")
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch = trainer.current_epoch + 1
        self.start_time = time.time()
        self.last_print_time = self.start_time
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        total = trainer.num_training_batches
        step = batch_idx + 1
        
        if step % TRAIN_LOG_INTERVAL != 0 and step != total:
            return

        pct = step / total if total else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        
        now = time.time()
        elapsed = now - self.start_time
        avg_time_per_step = elapsed / step if step > 0 else 0
        remaining_steps = total - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = time.strftime("%M:%S", time.gmtime(eta_seconds))
        
        loss = outputs.get("loss", 0)
        val_loss = trainer.callback_metrics.get("val_loss", 0)
        
        metrics = f"L: {float(loss):.4f} V: {float(val_loss):.4f}"
        
        sys.stdout.write(f"\r│ Ep {self.epoch}/{self.max_epochs} {bar} {pct*100:3.0f}% │ {metrics} │ ETA: {eta_str} │")
        sys.stdout.flush()
    
    def on_validation_end(self, trainer, pl_module):
        # Print summary at end of validation (at end of epoch)
        loss = trainer.callback_metrics.get("train_loss_epoch", 0)
        val = trainer.callback_metrics.get("val_loss", 0)
        
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        
        bar = "█" * self.width
        print(f"│ Ep {self.epoch}/{self.max_epochs} {bar} 100% │ L: {float(loss):.4f} V: {float(val):.4f} │ Done       │")
        
        # Epoch separator
        if self.epoch < self.max_epochs:
            print("├" + "─" * 64 + "┤")
    
    def on_train_end(self, trainer, pl_module):
        print("└" + "─" * 64 + "┘\n")
