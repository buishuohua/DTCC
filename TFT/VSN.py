import torch
import torch.nn as nn
from .GRN import GRN


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_vars, hidden_size, context_size=None, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_vars = num_vars  # mx in the paper

        # GRN for variable selection weights (formula 6)
        # Input size is flattened: num_vars * input_size because Ξt is flattened
        self.weight_grn = GRN(
            input_size=num_vars * input_size,
            hidden_size=num_vars,
            context_size=context_size,
            dropout=dropout
        )
        
        # Individual GRNs for each variable (formula 7)
        # GRNξ(j) with weights shared across time
        self.variable_grns = nn.ModuleList([
            GRN(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout
            ) for _ in range(num_vars)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, c=None):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_vars, input_size]
            c: Optional context vector
        """
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Flatten x for variable selection (Ξt in formula 6)
        # Original: [batch_size, time_steps, num_vars, input_size]
        # Flattened: [batch_size, time_steps, num_vars * input_size]
        flatten_x = x.reshape(batch_size, time_steps, -1)
        
        # Get variable selection weights (formula 6)
        # vXt = Softmax(GRNvx(Ξt, cs))
        weights = self.weight_grn(flatten_x, c)
        weights = self.softmax(weights)  # [batch_size, time_steps, num_vars]
        
        # Process each variable with its GRN (formula 7)
        # ξ̃t^(j) = GRNξ(j)(ξt^(j))
        processed_vars = []
        for i in range(self.num_vars):
            # Process the i-th variable
            var_x = x[..., i, :]  # [batch_size, time_steps, input_size]
            processed_var = self.variable_grns[i](var_x)  # [batch_size, time_steps, hidden_size]
            processed_vars.append(processed_var)
        
        processed_vars = torch.stack(processed_vars, dim=-2)  # [batch_size, time_steps, num_vars, hidden_size]
        
        # Combine using variable selection weights (formula 8)
        # ξ̃t = Σ(j=1 to mx) vXt^(j)ξ̃t^(j)
        weights = weights.unsqueeze(-1)  # [batch_size, time_steps, num_vars, 1]
        combined = (weights * processed_vars).sum(dim=-2)  # [batch_size, time_steps, hidden_size]
        
        return combined
