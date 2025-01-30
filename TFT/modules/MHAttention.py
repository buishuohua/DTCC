import torch
import torch.nn as nn
import math


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Initial projections for Q, K, V from input
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Head-specific weights for Q and K
        self.w_q = nn.ModuleList([
            nn.Linear(self.d_k, self.d_model) for _ in range(num_heads)
        ])
        self.w_k = nn.ModuleList([
            nn.Linear(self.d_k, self.d_model) for _ in range(num_heads)
        ])

        # Shared value weights across heads
        self.w_v = nn.Linear(self.d_k, self.d_model)

        # Final output transformation
        self.w_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
        """

        # Initial projections
        q = self.q_proj(x)  # [batch_size, seq_len, d_model]
        k = self.k_proj(x)  # [batch_size, seq_len, d_model]
        v = self.v_proj(x)  # [batch_size, seq_len, d_model]

        # Head-specific transformations for Q and K
        head_outputs = []

        # Shared V transformation - do this once outside the loop

        indices = [(i * self.d_k, (i + 1) * self.d_k)
                   for i in range(self.num_heads)]

        for i, (start, end) in enumerate(indices):
            # Transform Q and K for this head
            q_slice = q[..., start:end]
            k_slice = k[..., start:end]
            v_slice = v[..., start:end]
            head_q = self.w_q[i](q_slice)  # [batch_size, seq_len, d_k]
            head_k = self.w_k[i](k_slice)  # [batch_size, seq_len, d_k]

            # Calculate attention scores for this head
            # [batch_size, seq_len, seq_len]
            scores = torch.matmul(
                head_q, head_k.transpose(-2, -1)) / math.sqrt(self.d_k)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # [batch_size, seq_len, seq_len]
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            # Expand attention for element-wise product
            # [batch_size, seq_len, 1] * [batch_size, seq_len, d_model]

            head_v = self.w_v(v_slice)  # [batch_size, seq_len, d_model]
            head_output = attn.unsqueeze(-1) * head_v
            head_outputs.append(head_output)

        # Stack and average attention outputs across heads
        # [batch_size, num_heads, seq_len, d_model] -> [batch_size, seq_len, d_model]
        avg_output = torch.stack(head_outputs, dim=1).mean(dim=1)

        # Final transformation
        output = self.w_o(avg_output)

        return output, attn
