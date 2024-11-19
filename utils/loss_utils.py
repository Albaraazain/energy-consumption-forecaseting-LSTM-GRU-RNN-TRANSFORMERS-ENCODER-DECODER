from typing import Optional

import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss

class MaskedLoss(nn.Module):
    def __init__(self, base_criterion: Optional[nn.Module] = None):
        super().__init__()
        self.base_criterion = base_criterion if base_criterion is not None else nn.MSELoss()

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            return self.base_criterion(pred, target)

        mask = mask.float()
        loss = self.base_criterion(pred * mask, target * mask)
        return loss / (mask.sum() + 1e-8)