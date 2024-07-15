import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.4
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        self.batch_norm = nn.BatchNorm1d(2*hid_dim)  # バッチ正規化層を追加

        self.head = nn.Sequential(
            nn.Linear(2*hid_dim, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X (b, c, t): Input tensor
        Returns:
            X (b, num_classes): Output tensor
        """
        # Transpose the input to (b, t, c) for LSTM
        X = X.transpose(1, 2)

        # Pass through LSTM
        lstm_out, _ = self.lstm(X)

        # Take the last output of the LSTM
        X = lstm_out[:, -1, :]  # Get the last time step

        # Apply batch normalization
        X = self.batch_norm(X)

        return self.head(X)