# data_preparation
import itertools
from typing import Iterable, List

import pandas as pd
import torch


TOTAL_NODE = "Total"
RESIDUAL_NODE = "Residual"


def build_node_list(devices: Iterable[str], include_residual: bool = True) -> List[str]:
    nodes = [TOTAL_NODE]
    if include_residual:
        nodes.append(RESIDUAL_NODE)
    nodes.extend(list(devices))
    return nodes


def create_fully_connected_edge_index(num_nodes: int, include_self_loops: bool = True) -> torch.Tensor:
    """Return a fully connected edge_index (optionally with self loops)."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    edges = []
    for i, j in itertools.product(range(num_nodes), repeat=2):
        if i == j and not include_self_loops:
            continue
        edges.append((i, j))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def create_star_edge_index(num_nodes: int, center: int = 0, include_self_loops: bool = True) -> torch.Tensor:
    """Return a star-shaped edge_index with the specified center node."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if center < 0 or center >= num_nodes:
        raise ValueError("center index must be within [0, num_nodes)")

    edges = []
    for node in range(num_nodes):
        if node == center:
            continue
        edges.append((center, node))
        edges.append((node, center))

    if include_self_loops:
        edges.extend((i, i) for i in range(num_nodes))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def load_and_process_data(file_path, devices, rolling_window=1):
    """Load dataset and assemble the columns needed by downstream pipeline."""
    df = pd.read_csv(file_path)

    data = pd.DataFrame()
    data["Aggregate"] = df["Aggregate"].astype(float)
    data["Hour"] = df["Time"].astype(int)

    for device in devices:
        if device not in df.columns:
            raise KeyError(f"Device column '{device}' is missing from dataset")
        series = df[device].astype(float)
        if rolling_window > 1:
            series = series.rolling(window=rolling_window, min_periods=rolling_window).mean()
        data[device] = series

        state_col = f"{device}_State"
        if state_col in df.columns:
            data[state_col] = df[state_col]

    data = data.dropna().reset_index(drop=True)
    return data
