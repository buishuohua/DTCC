import torch
import torch.nn as nn
from .GRN import GRN
from .GLU import GLU
from .MHAttention import InterpretableMultiHeadAttention
from .VSN import VariableSelectionNetwork


class TemporalFusionDecoder(nn.Module):
    def __init__(self, d_model, num_vars, num_heads, forecast_len, quantiles=[0.1, 0.5, 0.9], dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.forecast_len = forecast_len
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Variable Selection Networks for past and future
        self.past_vsn = VariableSelectionNetwork(
            d_model=d_model,
            num_vars=num_vars,
            dropout=dropout
        )
        self.future_vsn = VariableSelectionNetwork(
            d_model=d_model,
            num_vars=num_vars,
            dropout=dropout
        )

        # GLU and Layer Norm for VSN skip connection
        self.vsn_gate = GLU(d_model)
        self.vsn_layer_norm = nn.LayerNorm(d_model)

        # LSTM Encoder-Decoder
        self.lstm_encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )

        self.lstm_decoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True
        )

        # Locality Enhancement Layer (formula 17)
        self.locality_glu = GLU(d_model)
        self.locality_layer_norm = nn.LayerNorm(d_model)

        # Static Enrichment Layer (formula 18) - Simplified to GRN without context
        self.static_grn = GRN(
            d_model=d_model,
            dropout=dropout
        )

        # Temporal Self-Attention Layer
        self.self_attention = InterpretableMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-attention GLU
        self.post_attention_glu = GLU(d_model)
        self.post_attention_layer_norm = nn.LayerNorm(d_model)

        # Position-wise Feed-forward Layer
        self.position_grn = GRN(
            d_model=d_model,
            dropout=dropout
        )

        # Final GLU and Layer Norm
        self.final_glu = GLU(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

        # Output layer for all quantiles
        self.output_layer = nn.Linear(d_model, self.num_quantiles)

    def forward(self, x_past, x_future, mask=None):
        """
        Args:
            x_past: Past input tensor [batch_size, past_len, num_vars, d_model]
            x_future: Future input tensor [batch_size, future_len, num_vars, d_model]
            mask: Attention mask for decoder
        Returns:
            outputs: Tensor of shape [batch_size, forecast_len, num_quantiles]
                    containing predictions for each time step and quantile
        """
        batch_size = x_past.size(0)
        past_len = x_past.size(1)
        future_len = x_future.size(1)
        total_len = past_len + future_len

        # Create position indices: [-past_len, ..., -1, 0, 1, ..., future_len-1]
        position_idx = torch.arange(-past_len,
                                    future_len, device=x_past.device)
        position_idx = position_idx.unsqueeze(0).expand(batch_size, total_len)

        # Process through Variable Selection Networks
        past_vsn = self.past_vsn(x_past)    # [batch_size, past_len, d_model]
        # [batch_size, future_len, d_model]
        future_vsn = self.future_vsn(x_future)

        # Process past through encoder LSTM
        past_lstm, encoder_states = self.lstm_encoder(past_vsn)

        # Process future through decoder LSTM
        future_lstm, _ = self.lstm_decoder(future_vsn, encoder_states)

        # Combine LSTM outputs
        lstm_out = torch.cat([past_lstm, future_lstm], dim=1)

        # Add VSN skip connections through GLU
        vsn_combined = torch.cat([past_vsn, future_vsn], dim=1)
        locality = self.locality_glu(lstm_out)
        locality_enhanced = self.locality_layer_norm(vsn_combined + locality)

        past_locality_enhanced = locality_enhanced[:, :past_len, :]
        future_locality_enhanced = locality_enhanced[:, past_len:, :]

        # Static Enrichment (formula 18) - Pass position index as context
        enriched = self.static_grn(locality_enhanced, position_idx)

        past_enriched = enriched[:, :past_len, :]
        future_enriched = enriched[:, past_len:, :]

        # Temporal Self-Attention
        attended, _ = self.self_attention(enriched, mask)

        # Only take the future part for subsequent processing
        future_attended = attended[:, past_len:, :]

        # Post-attention gating (only for future sequence)
        post_attention = self.post_attention_glu(future_attended)
        post_attended = self.post_attention_layer_norm(
            future_enriched + post_attention)

        # Position-wise Feed-forward - Pass future position index as context
        future_position_idx = position_idx[:, past_len:]
        position_wise = self.position_grn(post_attended, future_position_idx)

        # Final gating with skip connection back to future locality enhancement
        final = self.final_glu(position_wise)
        features = self.final_layer_norm(future_locality_enhanced + final)

        # Generate quantile forecasts
        outputs = self.output_layer(features)

        # Since we're already working with only future sequence, no need to slice
        return outputs
