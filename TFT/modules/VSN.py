import torch
import torch.nn as nn
from .GRN import GRN


class VariableSelectionNetwork(nn.Module):
    def __init__(self, d_model, num_vars, context_size=None, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_vars = num_vars  # mx in the paper

        # GRN for variable selection weights (formula 6)
        # Input size is num_vars * d_model because Ξt contains all variables
        self.weight_grn = GRN(
            d_model=num_vars * d_model,  # Output dimension should be num_vars for weights
            context_size=context_size,
            dropout=dropout
        )

        # Individual GRNs for each variable (formula 7)
        self.variable_grns = nn.ModuleList([
            GRN(
                d_model=d_model,
                context_size=context_size,
                dropout=dropout
            ) for _ in range(num_vars)
        ])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, c=None):
        """
        Args:
            x: Input tensor of shape [batch_size, time_steps, num_vars, d_model]
            c: Optional context vector
        """
        batch_size = x.size(0)
        time_steps = x.size(1)

        # Create Ξt by flattening the variable dimension
        # [batch_size, time_steps, num_vars * d_model]
        flattened_x = x.reshape(batch_size, time_steps, -1)

        # Get variable selection weights (formula 6)
        # vXt = Softmax(GRNvx(Ξt, cs))
        # [batch_size, time_steps, num_vars * d_model]
        weights = self.weight_grn(flattened_x, c)

        # Reshape weights to [batch_size, time_steps, num_vars]
        weights = weights.view(batch_size, time_steps, self.num_vars, -1)
        # Apply softmax to get weights
        weights = self.softmax(weights)

        # Get the maximum weight for each time step
        weights = weights.max(dim=-1, keepdim=True)[0]

        # Process each variable with its GRN (formula 7)
        processed_vars = []
        for i in range(self.num_vars):
            var_i = x[..., i, :]  # [batch_size, time_steps, d_model]
            processed_var = self.variable_grns[i](var_i, c)
            processed_vars.append(processed_var)

        # [batch_size, time_steps, num_vars, d_model]
        processed_vars = torch.stack(processed_vars, dim=2)

        # Combine using variable selection weights (formula 8)

        combined = (weights * processed_vars).sum(dim=2)

        return combined
