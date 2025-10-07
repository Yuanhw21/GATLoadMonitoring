import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


def repeat_edge_index(edge_index: torch.Tensor, num_graphs: int, num_nodes: int) -> torch.Tensor:
    """Tile an edge_index for a batch of identical graphs."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")

    device = edge_index.device
    offsets = torch.arange(num_graphs, device=device, dtype=edge_index.dtype) * num_nodes
    offsets = offsets.view(-1, 1)

    row = edge_index[0].unsqueeze(0) + offsets
    col = edge_index[1].unsqueeze(0) + offsets
    return torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)


class GATLSTMDisaggregator(nn.Module):
    def __init__(
        self,
        metadata,
        public_feature_dim: int,
        horizon: int = 1,
        embedding_dim: int = 16,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
    ):
        super().__init__()
        self.metadata = metadata
        self.public_feature_dim = public_feature_dim
        self.horizon = horizon
        self.include_residual = metadata.include_residual

        self.node_embedding = nn.Embedding(metadata.num_nodes, embedding_dim)
        self.register_buffer("node_ids", torch.arange(metadata.num_nodes, dtype=torch.long))
        self.register_buffer("device_indices", torch.tensor(metadata.device_indices, dtype=torch.long))
        if self.include_residual:
            self.register_buffer("residual_index", torch.tensor(1, dtype=torch.long))
        else:
            self.residual_index = None

        gat_input_dim = public_feature_dim + embedding_dim
        self.gat1 = GATConv(gat_input_dim, gat_hidden_dim, heads=gat_heads, dropout=gat_dropout)
        self.gat2 = GATConv(gat_hidden_dim * gat_heads, gat_hidden_dim, heads=1, concat=False, dropout=gat_dropout)
        self.gat_activation = nn.ELU()
        self.gat_dropout = nn.Dropout(gat_dropout)

        lstm_input_dim = gat_hidden_dim
        lstm_dropout_val = lstm_dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout_val,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        decoder_input_dim = lstm_hidden_dim * 2 + embedding_dim
        self.on_head = nn.Linear(decoder_input_dim, 1)
        self.power_head = nn.Linear(decoder_input_dim, 1)
        if self.include_residual:
            self.residual_head = nn.Linear(decoder_input_dim, 1)
        else:
            self.residual_head = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass.

        Args:
            x: Tensor of shape [batch, lookback, num_nodes, public_feature_dim].
            edge_index: Graph connectivity [2, E] for a single graph.
        Returns:
            dict with keys:
                'power': predictions for residual + devices (if residual enabled)
                'device_power': predictions for devices only
                'residual': residual predictions (or None)
                'on_logits': logits for device on/off
                'on_prob': sigmoid probabilities for device on/off
        """

        B, T, N, Fp = x.shape
        if N != self.metadata.num_nodes or Fp != self.public_feature_dim:
            raise ValueError("Input tensor shape does not match model configuration")

        device = x.device
        node_embeddings = self.node_embedding(self.node_ids.to(device))  # [N, emb]
        node_embeddings_expanded = node_embeddings.unsqueeze(0).unsqueeze(0).expand(B, T, N, -1)
        inputs = torch.cat([x, node_embeddings_expanded], dim=-1)

        graphs = inputs.reshape(B * T, N, -1)
        flat_inputs = graphs.reshape(B * T * N, -1)

        expanded_edge_index = repeat_edge_index(edge_index.to(device), B * T, N)

        gat_out = self.gat1(flat_inputs, expanded_edge_index)
        gat_out = self.gat_activation(gat_out)
        gat_out = self.gat_dropout(gat_out)
        gat_out = self.gat2(gat_out, expanded_edge_index)
        gat_out = self.gat_activation(gat_out)
        gat_out = gat_out.reshape(B * T, N, -1)

        gat_out = gat_out.reshape(B, T, N, -1).permute(0, 2, 1, 3)  # [B, N, T, G]
        lstm_in = gat_out.reshape(B * N, T, -1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = self.lstm_dropout(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        node_repr = last_hidden.reshape(B, N, -1)
        node_static = node_embeddings.unsqueeze(0).expand(B, N, -1)
        decoder_in = torch.cat([node_repr, node_static], dim=-1)

        on_logits_full = self.on_head(decoder_in).squeeze(-1)
        on_prob_full = torch.sigmoid(on_logits_full)
        amplitude_full = F.softplus(self.power_head(decoder_in).squeeze(-1))

        device_power = on_prob_full.index_select(1, self.device_indices) * amplitude_full.index_select(1, self.device_indices)
        on_logits = on_logits_full.index_select(1, self.device_indices)
        on_prob = on_prob_full.index_select(1, self.device_indices)

        residual_pred = None
        if self.include_residual:
            residual_idx = int(self.residual_index.item())
            residual_repr = decoder_in[:, residual_idx, :]
            residual_pred = self.residual_head(residual_repr).squeeze(-1)
            power = torch.cat([residual_pred.unsqueeze(-1), device_power], dim=1)
        else:
            power = device_power

        return {
            "power": power,
            "device_power": device_power,
            "residual": residual_pred,
            "on_logits": on_logits,
            "on_prob": on_prob,
        }


def weights_init(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def weights_init2(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)
