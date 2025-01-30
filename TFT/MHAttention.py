import torch
import torch.nn as nn
import math


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout=0.1):
        super().__init__()

        assert input_size % num_heads == 0

        self.input_size = input_size
        self.num_heads = num_heads
        self.d_k = input_size // num_heads

        # Head-specific weights for queries and keys (W_Q^(h) and W_K^(h))
        self.W_q = nn.Linear(input_size, input_size)  # d_model × d_attn for each head
        self.W_k = nn.Linear(input_size, input_size)  # d_model × d_attn for each head

        # Shared value weights across heads (W_V)
        self.W_v = nn.Linear(input_size, input_size)  # Shared value transformation

        # Final output transformation (W_H)
        self.W_o = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear transformations and reshape for multi-head
        # Q W_Q^(h) for each head
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K W_K^(h) for each head
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # V W_V (shared across heads)
        v = self.W_v(v).view(batch_size, -1, self.input_size).unsqueeze(1)

        # Calculate attention scores for each head (formula 15)
        # A(Q W_Q^(h), K W_K^(h))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Average attention weights across heads (1/H Σ A(...))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn.mean(dim=1, keepdim=True)  # Average across heads

        # Apply attention to shared values (formula 14)
        # H̃ = Ã(Q,K) V W_V
        context = torch.matmul(attn, v)

        # Reshape and apply output transformation (formula 13)
        # InterpretableMultiHead(Q,K,V) = H̃ W_H
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.input_size)
        output = self.W_o(output)

        return output
