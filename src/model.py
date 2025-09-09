import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv


class GATFeatureExtractor(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=256):
        super(GATFeatureExtractor, self).__init__()
        # Define GAT layers
        self.hidden_dim = hidden_dim
        self.gat1 = GATConv(num_features, out_channels=hidden_dim // 8, heads=8, dropout=0.2)
        self.gat2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1, concat=False, dropout=0.2)
        self.transformer_conv = TransformerConv(in_channels=hidden_dim,out_channels=num_classes, heads=1, dropout=0.2)

        # Residual connection
        self.residual_connect = nn.Linear(num_features, num_classes)
        # Add an extra linear layer to handle dimensionality change
        self.input_projection = nn.Linear(num_features, hidden_dim)
        # Activation function
        self.activation = nn.Tanh()

    def forward(self, x, edge_index):
        # First GAT layer
        # In this layer, edge weights are dynamically computed via attention
        x1 = F.elu(self.gat1(x, edge_index))

        # Second GAT layer
        x2 = F.elu(self.gat2(x1, edge_index))

        # Self-attention layer
        x3 = self.transformer_conv(x2, edge_index)

        # Residual connection
        res_x = self.residual_connect(x)

        # Combine features and residual
        combined_x = x3 + res_x

        # Activation function
        out = self.activation(combined_x)
        return out


class GATLSTMPredictor(nn.Module):
    def __init__(self, num_features, num_classes, gat_feature_extractor):
        super(GATLSTMPredictor, self).__init__()
        # Use the pretrained GAT model
        self.gat_feature_extractor = gat_feature_extractor
        # Add a fully connected layer to adjust feature dimensions
        self.feature_transform = nn.Linear(num_features, num_classes+2)

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=num_classes+2, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=40, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)

        # Output layer
        self.fc = nn.Linear(40, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        # Adjust feature dimensions to match GAT input
        transformed_features = self.feature_transform(x)
        # Extract features using GAT
        gat_features = self.gat_feature_extractor(transformed_features, edge_index)

        # LSTM layers
        gat_features = gat_features.unsqueeze(0)
        gat_features, _ = self.lstm1(gat_features)
        gat_features = self.dropout(gat_features)
        gat_features, _ = self.lstm2(gat_features)
        gat_features = self.dropout(gat_features)

        # Output layer
        out = self.fc(gat_features.squeeze(0))
        return out


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)  # Initialize weights of linear and conv layers with Xavier
        if m.bias is not None:  # If bias exists
            nn.init.constant_(m.bias, 0)  # Initialize bias to 0
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def weights_init2(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Initialize weights of linear and conv layers with He (Kaiming)
        if m.bias is not None:  # If bias exists
            nn.init.constant_(m.bias, 0)  # Initialize bias to 0
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'bias' in name:
                param.data.fill_(0)