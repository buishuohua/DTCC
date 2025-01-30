import torch
import torch.nn as nn
from .GLU import GLU

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, context_size=None, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context_size = context_size if context_size is not None else 0

        # η₂ = ELU(W₂,ω a + W₃,ω c + b₂,ω)
        self.W2 = nn.Linear(input_size, hidden_size, bias=True)  # W₂,ω for input a
        if self.context_size > 0:
            self.W3 = nn.Linear(context_size, hidden_size, bias=False)  # W₃,ω for context c

        # η₁ = W₁,ω η₂ + b₁,ω
        self.W1 = nn.Linear(hidden_size, hidden_size)  # W₁,ω

        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.glu = GLU(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, c=None):
        # η₂ = ELU(W₂,ω a + W₃,ω c + b₂,ω)
        eta2 = self.W2(x)
        if c is not None and self.context_size > 0:
            eta2 = eta2 + self.W3(c)
        eta2 = self.elu(eta2)

        # η₁ = W₁,ω η₂ + b₁,ω
        eta1 = self.W1(eta2)

        # Apply dropout before the gating layer
        eta1 = self.dropout(eta1)

        # GLU_ω(η₁)
        glu_output = self.glu(eta1)

        # GRN_ω(a,c) = LayerNorm(a + GLU_ω(η₁))
        output = self.layer_norm(x + glu_output)

        return output
