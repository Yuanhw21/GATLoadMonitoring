# data_preparation
import pandas as pd
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_edge_index(data, threshold=0.1):
    """
    Create edge index including all columns (including Aggregate Load and Hour)

    Args:
        data: DataFrame containing all data
        threshold: correlation threshold

    Returns:
        torch.Tensor: edge index tensor
    """
    # Compute Spearman correlation coefficient
    spearman_corr = data.corr(method='spearman')

    # Create graph
    G = nx.Graph()

    # Get all column names
    all_columns = list(data.columns)
    print(f"All columns: {all_columns}")
    # Add all nodes
    for col in all_columns:
        G.add_node(col)

    # Assign indices to each node (keep deterministic order)
    node_to_index = {node: i for i, node in enumerate(all_columns)}

    # Initialize edge lists
    source_nodes = []
    target_nodes = []

    # Iterate through correlation matrix, add edges that meet threshold
    for i, col1 in enumerate(all_columns):
        for j, col2 in enumerate(all_columns):
            if i < j:  # Avoid duplicate edges
                correlation = abs(spearman_corr.loc[col1, col2])  # Use absolute correlation value
                if correlation >= threshold:
                    # Add bidirectional edges
                    source_nodes.extend([node_to_index[col1], node_to_index[col2]])
                    target_nodes.extend([node_to_index[col2], node_to_index[col1]])
                    G.add_edge(col1, col2, weight=correlation)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    # Create edge_index tensor
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # Visualize graph structure
    plt.figure(figsize=(8, 6))
    pos = nx.kamada_kawai_layout(G)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items()) if len(G.edges) > 0 else ([], [])

    # Draw network graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues)
    plt.title("Correlation Graph Visualization")
    plt.show()

    return edge_index


def load_and_process_data(file_path, columns, window_size=5):
    """
    Enhanced data loading function to ensure consistent column order
    """
    data_csv = pd.read_csv(file_path, nrows=10000)

    # Create ordered DataFrame
    data = pd.DataFrame()

    # First add Aggregate Load and Hour
    data['Aggregate Load'] = data_csv['Aggregate']
    data_csv['Time'] = pd.to_datetime(data_csv['Time'], dayfirst=True)
    data['Hour'] = data_csv['Time'].dt.hour

    # Add device data in order
    for chn in columns:
        data[chn] = data_csv[chn].rolling(window=window_size).mean()

    data.dropna(inplace=True)

    return data
