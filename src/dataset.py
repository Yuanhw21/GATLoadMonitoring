# dataset
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

try:
    from .data_preparation import RESIDUAL_NODE, TOTAL_NODE, build_node_list
except ImportError:  # pragma: no cover - allow script-style imports
    from data_preparation import RESIDUAL_NODE, TOTAL_NODE, build_node_list


@dataclass
class DatasetMetadata:
    node_names: List[str]
    device_names: List[str]
    include_residual: bool

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_devices(self) -> int:
        return len(self.device_names)

    @property
    def device_indices(self) -> List[int]:
        start = 1 + int(self.include_residual)
        return list(range(start, start + self.num_devices))


def _states_from_dataframe(df, devices, threshold: float = 15.0):
    states = []
    for device in devices:
        state_col = f"{device}_State"
        if state_col in df.columns:
            series = (
                df[state_col]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"ON": 1.0, "OFF": 0.0})
                .fillna(0.0)
                .astype(np.float32)
            )
            states.append(series.to_numpy())
        else:
            states.append((df[device].to_numpy(dtype=np.float32) > threshold).astype(np.float32))
    return np.stack(states, axis=1)


class SequenceDataset(Dataset):
    """Sliding-window dataset producing node-wise sequences for GAT+LSTM."""

    def __init__(
        self,
        dataframe,
        devices: Iterable[str],
        lookback: int,
        horizon: int = 1,
        feature_scaler: Optional[StandardScaler] = None,
        label_scaler: Optional[StandardScaler] = None,
        include_residual: bool = True,
    ):
        self.devices = list(devices)
        self.include_residual = include_residual
        self.metadata = DatasetMetadata(
            node_names=build_node_list(self.devices, include_residual=include_residual),
            device_names=self.devices,
            include_residual=include_residual,
        )

        self.lookback = int(lookback)
        self.horizon = int(horizon)
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        aggregate = dataframe["Aggregate"].to_numpy(dtype=np.float32)
        hour = dataframe["Hour"].to_numpy(dtype=np.float32)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        public_features = np.stack([aggregate, hour_sin, hour_cos], axis=1)
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            public_scaled = self.feature_scaler.fit_transform(public_features)
        else:
            self.feature_scaler = feature_scaler
            public_scaled = self.feature_scaler.transform(public_features)
        self.public_tensor = torch.tensor(public_scaled, dtype=torch.float32)

        device_values = dataframe[self.devices].to_numpy(dtype=np.float32)
        if include_residual:
            residual = (aggregate - device_values.sum(axis=1, keepdims=False)).reshape(-1, 1)
            label_raw = np.concatenate([residual, device_values], axis=1)
            self.output_columns = [RESIDUAL_NODE] + self.devices
        else:
            label_raw = device_values
            self.output_columns = self.devices

        if label_scaler is None:
            self.label_scaler = StandardScaler()
            label_scaled = self.label_scaler.fit_transform(label_raw)
        else:
            self.label_scaler = label_scaler
            label_scaled = self.label_scaler.transform(label_raw)

        self.label_tensor = torch.tensor(label_scaled, dtype=torch.float32)
        self.label_raw_tensor = torch.tensor(label_raw, dtype=torch.float32)
        self.aggregate_tensor = torch.tensor(aggregate, dtype=torch.float32)
        self.device_states_tensor = torch.tensor(_states_from_dataframe(dataframe, self.devices), dtype=torch.float32)
        positives = self.device_states_tensor.sum(dim=0)
        total = torch.tensor(self.device_states_tensor.shape[0], dtype=torch.float32)
        negatives = total - positives
        pos_weight = torch.where(positives > 0, negatives / positives, torch.ones_like(positives))
        self.device_pos_weight = pos_weight

        self.sequence_count = len(dataframe) - self.lookback - self.horizon + 1
        if self.sequence_count <= 0:
            raise ValueError("Insufficient data for the requested lookback/horizon")

    def __len__(self):
        return self.sequence_count

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.sequence_count:
            raise IndexError("Index out of range")

        start = idx
        end = start + self.lookback
        target_start = end - self.horizon
        target_end = end

        window = self.public_tensor[start:end]  # [lookback, public_dim]
        features = window.unsqueeze(1).repeat(1, self.metadata.num_nodes, 1)  # [lookback, num_nodes, public_dim]

        y_scaled = self.label_tensor[target_start:target_end]
        y_raw = self.label_raw_tensor[target_start:target_end]
        aggregate = self.aggregate_tensor[target_start:target_end]
        device_states = self.device_states_tensor[target_start:target_end]

        if self.horizon == 1:
            y_scaled = y_scaled.squeeze(0)
            y_raw = y_raw.squeeze(0)
            aggregate = aggregate.squeeze(0)
            device_states = device_states.squeeze(0)

        return features, y_scaled, y_raw, aggregate, device_states

    def get_feature_scaler(self) -> StandardScaler:
        return self.feature_scaler

    def get_label_scaler(self) -> StandardScaler:
        return self.label_scaler

    def get_metadata(self) -> DatasetMetadata:
        return self.metadata

    @property
    def public_feature_dim(self) -> int:
        return self.public_tensor.shape[-1]

    def get_device_pos_weight(self) -> torch.Tensor:
        return self.device_pos_weight.clone()


class PredictDataset(Dataset):
    def __init__(
        self,
        dataframe,
        feature_scaler: StandardScaler,
        node_names: Iterable[str],
        lookback: int,
    ):
        self.node_names = list(node_names)
        self.lookback = int(lookback)
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

        aggregate = dataframe["Aggregate"].to_numpy(dtype=np.float32)
        hour = dataframe["Hour"].to_numpy(dtype=np.float32)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        public_features = np.stack([aggregate, hour_sin, hour_cos], axis=1)
        public_scaled = feature_scaler.transform(public_features)

        self.public_tensor = torch.tensor(public_scaled, dtype=torch.float32)
        self.aggregate_tensor = torch.tensor(aggregate, dtype=torch.float32)
        self.sequence_count = len(dataframe) - self.lookback + 1
        if self.sequence_count <= 0:
            raise ValueError("PredictDataset requires at least lookback steps")

    def __len__(self):
        return self.sequence_count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self.lookback

        window = self.public_tensor[start:end]
        features = window.unsqueeze(1).repeat(1, len(self.node_names), 1)
        aggregate = self.aggregate_tensor[end - 1]
        return features, aggregate
