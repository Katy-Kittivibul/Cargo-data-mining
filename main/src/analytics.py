import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class CostBenefitAnalyzer:
    def __init__(self,
                 cost_per_delay_minute: float = 5.0,  # £ per minute
                 cost_per_shipment: float = 100.0,    # Base cost
                 penalty_rate: float = 0.02):         # 2% penalty per hour late
        self.cost_per_delay_minute = cost_per_delay_minute
        self.cost_per_shipment = cost_per_shipment
        self.penalty_rate = penalty_rate

    def calculate_delay_costs(self, long_flow_df: pd.DataFrame,
                              hub_analysis_df: pd.DataFrame) -> Dict:
        """
        Calculates the financial impact of delays.
        """
        # 1. Clean and Path Reconstruction
        transport_df = long_flow_df.sort_values(['Leg_ID', 'Planned_Mins']).copy()
        transport_df['Source_Hub_ID'] = transport_df['Hub_ID'].astype(str)
        transport_df['Target_Hub_ID'] = transport_df.groupby('Leg_ID')['Hub_ID'].shift(-1).astype(str)

        # Determine which stage column to use
        stage_col = 'Stage_Group' if 'Stage_Group' in transport_df.columns else 'Stage'

        # Ensure numeric types
        transport_df['Delay_Mins'] = pd.to_numeric(transport_df['Delay_Mins'], errors='coerce')

        # Filter for Transport segments only
        transport_df = transport_df[
            (transport_df[stage_col].str.contains('Transport', na=False)) &
            (transport_df['Source_Hub_ID'] != 'nan') &
            (transport_df['Target_Hub_ID'] != 'nan')
        ].dropna(subset=['Delay_Mins', 'Source_Hub_ID'])

        # 2. Financial Calculations
        # Linear cost: £5 per minute of delay
        transport_df['Delay_Cost'] = transport_df['Delay_Mins'] * self.cost_per_delay_minute

        # Penalty cost: Applied only if delay > 60 mins
        transport_df['Penalty_Hours'] = np.maximum(0, (transport_df['Delay_Mins'] - 60) / 60)
        transport_df['Penalty_Cost'] = (
            transport_df['Penalty_Hours'] * self.penalty_rate * self.cost_per_shipment
        )

        transport_df['Total_Cost'] = transport_df['Delay_Cost'] + transport_df['Penalty_Cost']

        # 3. Aggregate Hub Costs
        hub_costs = transport_df.groupby('Source_Hub_ID').agg(
            Total_Cost=('Total_Cost', 'sum'),
            Shipments=('Leg_ID', 'count'),
            Avg_Delay=('Delay_Mins', 'mean')
        ).reset_index()

        # Ensure ID alignment
        hub_costs['Source_Hub_ID'] = hub_costs['Source_Hub_ID'].astype(str)
        hub_temp = hub_analysis_df.copy()
        hub_temp['Hub_ID'] = hub_temp['Hub_ID'].astype(str)

        hub_costs = hub_costs.merge(
            hub_temp[['Hub_ID', 'Total_Outgoing_Flow']],
            left_on='Source_Hub_ID', right_on='Hub_ID', how='left'
        )

        hub_costs['Cost_per_Shipment'] = hub_costs['Total_Cost'] / hub_costs['Shipments']
        hub_costs = hub_costs.sort_values('Total_Cost', ascending=False)

        return {
            'total_cost': transport_df['Total_Cost'].sum(),
            'total_delay_cost': transport_df['Delay_Cost'].sum(),
            'total_penalty_cost': transport_df['Penalty_Cost'].sum(),
            'avg_cost_per_shipment': transport_df['Total_Cost'].mean(),
            'hub_costs': hub_costs
        }

    def calculate_optimization_roi(self, current_costs: Dict, improvement_scenarios: List[Dict]) -> pd.DataFrame:
        """
        Calculates Return on Investment (ROI) for various improvement scenarios.
        """
        current_annual_cost = current_costs['total_cost']
        roi_results = []

        for sc in improvement_scenarios:
            savings = current_annual_cost * (sc['delay_reduction_%'] / 100)
            impl_cost = max(1, sc['implementation_cost']) # Avoid div by zero

            payback = impl_cost / (savings / 12) if savings > 0 else np.inf
            roi_3y = ((savings * 3 - impl_cost) / impl_cost) * 100

            roi_results.append({
                'Scenario': sc['name'],
                'Annual_Savings': savings,
                'Implementation_Cost': impl_cost,
                'Payback_Months': payback,
                'ROI_3Year_%': roi_3y
            })

        return pd.DataFrame(roi_results).sort_values('ROI_3Year_%', ascending=False)

class PrescriptiveAnalyzer:
    """
    Run what-if scenarios and provide actionable recommendations.
    """
    def __init__(self, long_flow_df: pd.DataFrame, hub_analysis_df: pd.DataFrame):
        self.long_flow_df = long_flow_df.copy()
        self.hub_analysis_df = hub_analysis_df.copy()

# --- Visualisation Functions (Migrated from Notebook) ---

def visualise_hub_activity_vs_delay(hub_analysis_df: pd.DataFrame, min_flow_percentile=0.10):
    """Bubble plot of Hub Flow vs Delay."""
    # Filter for hubs with significant flow
    min_flow = hub_analysis_df['Total_Outgoing_Flow'].quantile(min_flow_percentile)
    filtered_df = hub_analysis_df[hub_analysis_df['Total_Outgoing_Flow'] >= min_flow].copy()

    median_flow = filtered_df['Total_Outgoing_Flow'].median()
    median_delay = filtered_df['Avg_Segment_Delay'].median()

    def categorise_hub(row):
        if row['Total_Outgoing_Flow'] > median_flow and row['Avg_Segment_Delay'] > median_delay:
            return 'Critical Bottleneck'
        elif row['Total_Outgoing_Flow'] > median_flow and row['Avg_Segment_Delay'] <= median_delay:
            return 'Efficient Major Hub'
        elif row['Total_Outgoing_Flow'] <= median_flow and row['Avg_Segment_Delay'] > median_delay:
            return 'Inefficient Small Hub'
        else:
            return 'Efficient Small Hub'

    filtered_df['Hub_Category'] = filtered_df.apply(categorise_hub, axis=1)

    fig = px.scatter(
        filtered_df,
        x='Total_Outgoing_Flow',
        y='Avg_Segment_Delay',
        size='Std_Delay',
        color='Hub_Category',
        hover_name='Hub_ID',
        log_x=True,
        title='Hub Performance Analysis: Volume vs Delay',
        template='plotly_white'
    )
    fig.add_hline(y=median_delay, line_dash="dash", line_color="gray", annotation_text="Median Delay")
    fig.add_vline(x=median_flow, line_dash="dash", line_color="gray", annotation_text="Median Flow")
    
    return fig

def visualise_top_flow_paths(long_flow_df: pd.DataFrame, top_n=15):
    """Sankey diagram for top routes."""
    # Path Reconstruction
    df_paths = long_flow_df.sort_values(['Leg_ID', 'Planned_Mins']).copy()
    df_paths['Target_Hub_ID'] = df_paths.groupby('Leg_ID')['Hub_ID'].shift(-1)
    df_paths.rename(columns={'Hub_ID': 'Source_Hub_ID'}, inplace=True)

    transport_df = df_paths[
        (df_paths['Source_Hub_ID'] != 'none') &
        (df_paths['Target_Hub_ID'] != 'none') &
        (df_paths['Target_Hub_ID'].notna())
    ].copy()
    transport_df = transport_df[transport_df['Source_Hub_ID'] != transport_df['Target_Hub_ID']]

    flow_paths = transport_df.groupby(['Source_Hub_ID', 'Target_Hub_ID']).agg(
        Flow_Count=('Leg_ID', 'count'),
        Avg_Delay=('Delay_Mins', 'mean')
    ).reset_index()

    top_paths = flow_paths.nlargest(top_n, 'Flow_Count').copy()
    
    # Sankey logic...
    all_nodes = list(set(top_paths['Source_Hub_ID']).union(set(top_paths['Target_Hub_ID'])))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            label=all_nodes,
            pad=15, thickness=20,
            line=dict(color="black", width=0.5)
        ),
        link=dict(
            source=top_paths['Source_Hub_ID'].map(node_map),
            target=top_paths['Target_Hub_ID'].map(node_map),
            value=top_paths['Flow_Count']
        )
    )])
    fig.update_layout(title_text="Top Freight Flow Paths (Sankey)", font_size=10)
    return fig

def create_summary_dashboard(clustered_df: pd.DataFrame):
     """
     Creates the Executive Summary Dashboard.
     """
     fig = make_subplots(
         rows=2, cols=3,
         subplot_titles=(
             'Hub Distribution by Cluster',
             'Flow Volume by Cluster',
             'Delay Performance by Cluster',
             'Top 10 Busiest Hubs',
             'Top 10 Most Delayed Hubs',
             'Cluster Size vs Performance'
         ),
         specs=[
             [{"type": "domain"}, {"type": "xy"}, {"type": "xy"}],
             [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
         ]
     )
     
     # 1. Hub Distribution (Pie)
     dist = clustered_df['Cluster_Label'].value_counts()
     fig.add_trace(go.Pie(labels=dist.index, values=dist.values, name="Clusters"), row=1, col=1)
     
     # 2. Flow Volume (Bar)
     flow = clustered_df.groupby('Cluster_Label')['Total_Outgoing_Flow'].mean()
     fig.add_trace(go.Bar(x=flow.index, y=flow.values, name="Avg Flow"), row=1, col=2)
     
     # 3. Delay Performance (Bar)
     delay = clustered_df.groupby('Cluster_Label')['Avg_Segment_Delay'].mean()
     fig.add_trace(go.Bar(x=delay.index, y=delay.values, name="Avg Delay"), row=1, col=3)
     
     fig.update_layout(height=800, title_text="Executive Summary Dashboard", showlegend=False)
     return fig

def visualise_network_with_communities(long_df: pd.DataFrame, clustered_df: pd.DataFrame, top_n_edges=100):
    """
    Visualisation 3: Network Graph with Communities.
    """
    import networkx as nx
    
    # Create Graph
    G = nx.DiGraph()
    
    # Add nodes with cluster info
    cluster_map = dict(zip(clustered_df['Hub_ID'].astype(str), clustered_df['Cluster_Label']))
    
    # Filter for top edges to avoid clutter
    edges = long_df.groupby(['Hub_ID', 'Leg_ID'])['Hub_ID'].shift(-1).reset_index() # Simplified edge logic
    # Re-using optimization logic for weighted graph is better, but doing simple here
    
    # Simplified: Use top flows
    # ... (Implementation simplified for brevity, assuming standard NetworkX-Plotly piping)
    return go.Figure() # Placeholder for complex network viz logic

def visualise_delay_accumulation(long_df: pd.DataFrame):
    """Visualisation 4: Delay Accumulation (Violin/Line)."""
    fig = px.violin(long_df, x='Stage_Group', y='Delay_Mins', box=True, 
                    title="Delay Distribution by Stage", template='plotly_white')
    return fig

def visualise_hub_delay_heatmap(hub_kpis: pd.DataFrame):
    """Visualisation 5: Heatmap of Delays."""
    # Assuming we have some temporal or categorical component, else correlation heatmap
    fig = px.density_heatmap(hub_kpis, x='Total_Outgoing_Flow', y='Avg_Segment_Delay', 
                             title="Hub Delay Heatmap", nbinsx=20, nbinsy=20)
    return fig

def visualise_hub_profiles(clustered_df: pd.DataFrame):
    """Visualisation 6: Parallel Coordinates."""
    cols = ['Total_Outgoing_Flow', 'Avg_Segment_Delay', 'Std_Delay', 'Max_Delay']
    fig = px.parallel_coordinates(clustered_df, color="Cluster_Label",
                                  dimensions=cols,
                                  title="Cluster Profiles Parallel Coordinates")
    return fig

def assess_embedding_quality(embeddings: np.ndarray, labels: np.ndarray = None):
    """GNN Analysis: PCA/t-SNE of embeddings."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    acc_2d = pca.fit_transform(embeddings)
    
    df_pca = pd.DataFrame(acc_2d, columns=['PC1', 'PC2'])
    if labels is not None:
        df_pca['Cluster'] = labels
        fig = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title="Embedding Space (PCA)")
    else:
        fig = px.scatter(df_pca, x='PC1', y='PC2', title="Embedding Space (PCA)")
    return fig

def find_similar_hubs(target_hub_id: str, embeddings: np.ndarray, hub_ids: np.ndarray, top_k=5):
    """GNN Analysis: Find most similar hubs in embedding space."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    hub_idx = np.where(hub_ids == str(target_hub_id))[0]
    if len(hub_idx) == 0:
        return None
        
    idx = hub_idx[0]
    target_emb = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(target_emb, embeddings)[0]
    
    top_indices = sims.argsort()[-top_k-1:-1][::-1] # Exclude self
    
    results = []
    for i in top_indices:
        results.append({
            'Hub_ID': hub_ids[i],
            'Similarity': sims[i]
        })
    return pd.DataFrame(results)


def export_results(clustered_df: pd.DataFrame, embeddings: np.ndarray,
                   critical_routes: pd.DataFrame, output_dir='./model/results'):
    """
    Export analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    clustered_df.to_csv(f'{output_dir}/hub_clusters.csv', index=False)
    np.save(f'{output_dir}/hub_embeddings.npy', embeddings)
    critical_routes.to_csv(f'{output_dir}/critical_routes.csv', index=False)
    print(f"✅ Results exported to {output_dir}")
