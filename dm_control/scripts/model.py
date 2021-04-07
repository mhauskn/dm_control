import torch
import torch.nn as nn
from torch.nn import functional as F



class FFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2846, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 56)
        )
        self.criterion = nn.MSELoss()


    def configure_optimizers(self, train_config):
        optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer        


    def forward(self, x, targets=None):
        # logits = torch.tanh(self.mlp(x))
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = self.criterion(logits, targets)

        return logits, loss