import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# PyTorch Geometric Explainer imports
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import to_networkx

DEFAULT_FEATURE_NAMES = [
    'Total_Outgoing_Flow',
    'Avg_Segment_Delay',
    'Std_Delay',
    'Median_Delay',
    'Max_Delay'
]

def explain_hub(model, data, node_index, mode='regression', algorithm=None):
    """
    Generates feature and edge attributions for a given node using PyTorch Geometric's Explainer.
    
    Args:
        model: The trained GNN model (e.g., GraphSAGE).
        data: PyTorch Geometric Data object containing the graph.
        node_index: The integer index of the hub to explain.
        mode (str): 'regression' (for continuous values/embeddings) or 'multiclass_classification' (for single node classification outcomes).
        algorithm: Optional explainer algorithm to use (defaults to GNNExplainer).
        
    Returns:
        explanation: The PyG Explanation object containing node/feature and edge masks.
    """
    if algorithm is None:
        algorithm = GNNExplainer(epochs=200)

    # We configure the explainer to support both single node and continuous values based on mode.
    # We explain the 'model' behavior.
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode=mode,
            task_level='node',
            return_type='raw',
        ),
    )

    print(f"🔍 Generating explanation for Node {node_index} using {mode} mode...")
    
    # Generate the explanation for the specified node
    explanation = explainer(data.x, data.edge_index, index=node_index)
    return explanation

def visualize_influence_network(explanation, data, node_index, hub_id=None, output_path=None, top_k=15):
    """
    Creates a subgraph visualization emphasizing the top influential connections 
    and node features contributing to the target hub's bottleneck status.
    """
    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    display_id = hub_id if hub_id is not None else node_index
    
    # Retrieve top K most important edges
    top_edge_indices = np.argsort(edge_mask)[::-1][:top_k].copy()
    
    # Create an edge_index of just the top edges
    significant_edges = data.edge_index[:, top_edge_indices]
    significant_edge_weights = edge_mask[top_edge_indices]
    
    # Identify unique nodes in this subgraph
    relevant_nodes = torch.unique(significant_edges).tolist()
    if node_index not in relevant_nodes:
        relevant_nodes.append(node_index)
        
    G = nx.Graph()
    
    # Mapping indices back to Hub IDs if data has hub_ids attribute
    id_map = {idx: str(data.hub_ids[idx]) for idx in relevant_nodes} if hasattr(data, 'hub_ids') else {idx: str(idx) for idx in relevant_nodes}

    for nd in relevant_nodes:
        label = id_map[nd]
        if nd == node_index:
            G.add_node(label, is_target=True)
        else:
            G.add_node(label, is_target=False)

    for i in range(significant_edges.shape[1]):
        src_label = id_map[significant_edges[0, i].item()]
        dst_label = id_map[significant_edges[1, i].item()]
        weight = significant_edge_weights[i]
        G.add_edge(src_label, dst_label, weight=weight)
        
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Node styling
    node_colors = ['#FF4C4C' if G.nodes[n].get('is_target') else '#4C9BFF' for n in G.nodes()]
    node_sizes = [1500 if G.nodes[n].get('is_target') else 600 for n in G.nodes()]
    
    # Edge width based on mask weight
    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='#666666', alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='white')
    
    plt.title(f"Influence Network for Hub {display_id}", fontsize=16)
    plt.axis('off')
    
    if output_path is not None:
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        return output_path
    else:
        # Save to base64 buffer for direct HTML embedding
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

def generate_xai_report(model, data, node_index, hub_id=None, feature_names=DEFAULT_FEATURE_NAMES, mode='regression', output_dir="results"):
    """
    Orchestrates the XAI extraction and visualization, and exports a formatted HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)
    display_id = hub_id if hub_id is not None else str(node_index)
    
    # 1. Run inference/explainability
    explanation = explain_hub(model, data, node_index, mode=mode)
    
    # 2. Extract feature importance
    if explanation.node_mask is not None:
        target_feature_mask = explanation.node_mask[node_index].detach().cpu().numpy()
    else:
        target_feature_mask = np.zeros(len(feature_names))
        
    num_features = target_feature_mask.shape[0]
    if len(feature_names) < num_features:
        feature_names = feature_names + [f"Feature {i}" for i in range(len(feature_names), num_features)]
        
    # 3. Generate Visuals
    img_base64 = visualize_influence_network(explanation, data, node_index, hub_id=display_id)
    
    # 4. Generate HTML content
    feature_rows = ""
    sorted_idx = np.argsort(target_feature_mask)[::-1].copy()
    
    for rank, idx in enumerate(sorted_idx, 1):
        feature_name = feature_names[idx]
        score = target_feature_mask[idx]
        if score < 1e-4: continue
        bar_width = min(int(score * 100), 100)
        feature_rows += f"""
        <tr>
            <td>{rank}</td>
            <td>{feature_name}</td>
            <td>{score:.4f}</td>
            <td><div style="background-color: #4CAF50; width: {bar_width}%; height: 15px; border-radius: 3px;"></div></td>
        </tr>
        """
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><title>Influence Report - Hub {display_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
            .network-img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; margin-top: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Explainable AI Report: Hub {display_id}</h1>
            <p>Analysis of structural and feature-based influence contributing to bottleneck status.</p>
            <h2>1. Feature Importance</h2>
            <table>
                <tr><th>Rank</th><th>Feature Name</th><th>Attribution Score</th><th>Relative Importance</th></tr>
                {feature_rows}
            </table>
            <h2>2. Influence Subgraph</h2>
            <img class="network-img" src="data:image/png;base64,{img_base64}" alt="Influence Network" />
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, f"influence_report_hub_{display_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return html_path

def explain_critical_routes(model, data, critical_routes_csv, feature_names, output_dir="results"):
    """
    Reads critical routes and generates XAI reports for each identified bottleneck hub.
    """
    if not os.path.exists(critical_routes_csv):
        print(f"⚠️ Critical routes file not found: {critical_routes_csv}")
        return
        
    df = pd.read_csv(critical_routes_csv)
    # Get mapping of Hub_ID to node index
    hub_to_idx = {str(hid): i for i, hid in enumerate(data.hub_ids)}
    
    print(f"🚀 Generating XAI reports for {len(df)} critical hubs...")
    
    for hub_id in df['Hub_ID'].astype(str):
        if hub_id in hub_to_idx:
            idx = hub_to_idx[hub_id]
            generate_xai_report(model, data, idx, hub_id=hub_id, feature_names=feature_names, output_dir=output_dir)
            print(f"  ✅ Explained Hub {hub_id}")
        else:
            print(f"  ⚠️ Hub {hub_id} not found in graph data.")

import pandas as pd # Needed for explain_critical_routes

if __name__ == "__main__":
    # Simple self-test code (not run unless executed directly)
    print("XAI Module initialized. Awaiting integration with main pipeline.")
