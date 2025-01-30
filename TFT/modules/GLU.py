import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # W4,ω and b4,ω
        self.linear1 = nn.Linear(d_model, d_model, bias=True)
        # W5,ω and b5,ω
        self.linear2 = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        # GLU(γ) = σ(W₄γ + b₄) ⊙ (W₅γ + b₅)
        gate = self.sigmoid(self.linear1(x))
        value = self.linear2(x)
        return gate * value
