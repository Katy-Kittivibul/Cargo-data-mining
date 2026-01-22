import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

def build_pyg_graph(long_flow_df: pd.DataFrame, hub_analysis_df: pd.DataFrame,
                    normalize_features=True, include_reverse_edges=False) -> tuple:
    """
    Constructs a PyG Data object. Reconstructs paths internally to create edges.
    
    Args:
        long_flow_df (pd.DataFrame): The long-format shipment movements.
        hub_analysis_df (pd.DataFrame): The aggregated hub metrics (node features).
        normalize_features (bool): Whether to scaler-normalise node features.
        include_reverse_edges (bool): If true, creates an undirected graph representation.
        
    Returns:
        tuple: (Data object, hub_to_index mapping, list of feature names)
    """
    print("=" * 70)
    print("BUILDING PYTORCH GEOMETRIC GRAPH")
    print("=" * 70)

    # 1. PATH RECONSTRUCTION (Mandatory for Edge Index)
    df_path = long_flow_df.sort_values(['Leg_ID', 'Planned_Mins']).copy()
    df_path['Source_Hub_ID'] = df_path['Hub_ID']
    df_path['Target_Hub_ID'] = df_path.groupby('Leg_ID')['Hub_ID'].shift(-1)

    # Filter for Transport segments only
    transport_df = df_path[
        (df_path['Stage_Group'].str.contains('Transport', na=False)) &
        (df_path['Source_Hub_ID'] != 'none') &
        (df_path['Target_Hub_ID'] != 'none') &
        (df_path['Target_Hub_ID'].notna())
    ].copy()

    # Remove self-loops
    transport_df = transport_df[transport_df['Source_Hub_ID'] != transport_df['Target_Hub_ID']]

    # 2. NODE MAPPING (Supports both String and Numeric IDs)
    # We must ensure all IDs are strings to avoid type mixing issues
    transport_df['Source_Hub_ID'] = transport_df['Source_Hub_ID'].astype(str)
    transport_df['Target_Hub_ID'] = transport_df['Target_Hub_ID'].astype(str)
    hub_analysis_df['Hub_ID'] = hub_analysis_df['Hub_ID'].astype(str)

    all_unique_hubs = pd.unique(pd.concat([transport_df['Source_Hub_ID'], transport_df['Target_Hub_ID']]))
    hub_to_index = {hub_id: i for i, hub_id in enumerate(all_unique_hubs)}
    num_nodes = len(all_unique_hubs)

    # 3. NODE FEATURES (X Matrix)
    node_features_df = pd.DataFrame({'Hub_ID': all_unique_hubs})

    # Use 'Total_Outgoing_Flow' to match aggregation 
    features_to_use = ['Hub_ID', 'Total_Outgoing_Flow', 'Avg_Segment_Delay', 'Std_Delay']
    
    # Merge so that features align with the node order (0 to N)
    node_features_df = node_features_df.merge(
        hub_analysis_df[[f for f in features_to_use if f in hub_analysis_df.columns]],
        on='Hub_ID', how='left'
    ).fillna(0)

    feature_cols = [c for c in node_features_df.columns if c != 'Hub_ID']
    x_raw = node_features_df[feature_cols].values.astype(np.float32)

    if normalize_features:
        x = torch.tensor(StandardScaler().fit_transform(x_raw), dtype=torch.float)
    else:
        x = torch.tensor(x_raw, dtype=torch.float)

    # 4. EDGE INDEX & ATTRIBUTES
    transport_df['Source_Idx'] = transport_df['Source_Hub_ID'].map(hub_to_index)
    transport_df['Target_Idx'] = transport_df['Target_Hub_ID'].map(hub_to_index)

    # Aggregate to find route-level features
    edges = transport_df.groupby(['Source_Idx', 'Target_Idx']).agg(
        flow_volume=('Leg_ID', 'count'),
        avg_delay=('Delay_Mins', 'mean')
    ).reset_index()

    edge_index = torch.tensor([edges['Source_Idx'].tolist(), edges['Target_Idx'].tolist()], dtype=torch.long)

    # Edge Attributes (Weighting the connection)
    edge_attr_raw = edges[['flow_volume', 'avg_delay']].values.astype(np.float32)
    edge_attr = torch.tensor(StandardScaler().fit_transform(edge_attr_raw), dtype=torch.float)

    # 5. CREATE DATA OBJECT
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    data.hub_ids = all_unique_hubs
    data.feature_names = feature_cols

    if include_reverse_edges:
        # TBD if needed, but basic implementation expects directed
        pass

    print(f"Graph Construction Complete: {num_nodes} nodes, {edge_index.shape[1]} edges.")
    return data, hub_to_index, feature_cols
