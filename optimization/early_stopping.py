import torch
import torch.nn as nn

class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self._save_checkpoint(model)
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            self._save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
            return False

    def _save_checkpoint(self, model: nn.Module):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
