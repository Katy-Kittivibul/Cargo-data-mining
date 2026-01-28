import pandas as pd
import numpy as np
import networkx as nx
import os
from typing import List, Dict

class RouteOptimizer:
    """
    Finds optimal routes through the network and identifies critical bottlenecks.
    """

    def __init__(self, long_flow_df: pd.DataFrame, hub_analysis_df: pd.DataFrame):
        self.long_flow_df = long_flow_df.copy()
        self.hub_analysis_df = hub_analysis_df.copy()
        self.graph = self._build_weighted_graph()

    def _build_weighted_graph(self) -> nx.DiGraph:
        """
        Build directed graph where edges represent hub-to-hub movements.
        """
        print("\n" + "="*70)
        print("BUILDING WEIGHTED TRANSPORT NETWORK")
        print("="*70)

        G = nx.DiGraph()

        # 1. Path Reconstruction
        df_work = self.long_flow_df.sort_values(['Leg_ID', 'Planned_Mins']).copy()
        df_work['Delay_Mins'] = pd.to_numeric(df_work['Delay_Mins'], errors='coerce')

        # Determine movement
        df_work['Source_Hub_ID'] = df_work['Hub_ID'].astype(str)
        df_work['Target_Hub_ID'] = df_work.groupby('Leg_ID')['Hub_ID'].shift(-1).astype(str)

        # 2. Filter for valid transport segments
        stage_col = 'Stage_Group' if 'Stage_Group' in df_work.columns else 'Stage'
        
        transport_df = df_work[
            (df_work[stage_col].str.contains('Transport', na=False)) &
            (df_work['Source_Hub_ID'] != 'none') &
            (df_work['Target_Hub_ID'] != 'none') &
            (df_work['Target_Hub_ID'].notna())
        ].dropna(subset=['Delay_Mins'])

        # 3. Aggregate route metrics
        route_metrics = transport_df.groupby(['Source_Hub_ID', 'Target_Hub_ID']).agg(
            avg_delay=('Delay_Mins', 'mean'),
            volume=('Leg_ID', 'count'),
            std_delay=('Delay_Mins', 'std')
        ).reset_index()

        route_metrics['std_delay'] = route_metrics['std_delay'].fillna(0)

        # 4. Add edges with composite weight
        for _, row in route_metrics.iterrows():
            src, tgt = row['Source_Hub_ID'], row['Target_Hub_ID']
            # Composite Weight: Favor low delay and high volume (reliability)
            # Higher weight = Higher Cost (NetworkX Dijkstra minimizes this)
            # So, we want High Delay -> High Weight. Low Volume -> High Weight (assuming we prefer well-trodden paths).
            weight = (row['avg_delay'] * 1.0 + row['std_delay'] * 0.5 + (50 / (row['volume'] + 1)))

            G.add_edge(src, tgt, weight=max(0.1, weight),
                       avg_delay=row['avg_delay'], volume=row['volume'])

        print(f"✅ Transport Graph: {G.number_of_nodes()} hubs, {G.number_of_edges()} routes")
        return G

    def find_optimal_route(self, source, target, top_k=3) -> List[Dict]:
        """Find top K optimal routes using Yen's algorithm."""
        src_str, tgt_str = str(source), str(target)
        if not self.graph.has_node(src_str) or not self.graph.has_node(tgt_str):
            print(f"❌ Nodes {src_str} or {tgt_str} not in graph.")
            return []

        try:
            paths = list(nx.shortest_simple_paths(self.graph, src_str, tgt_str, weight='weight'))[:top_k]
            results = []
            for rank, path in enumerate(paths, 1):
                delay = sum(self.graph[path[i]][path[i+1]]['avg_delay'] for i in range(len(path)-1))
                results.append({'rank': rank, 'path': path, 'total_avg_delay': delay})
            return results
        except nx.NetworkXNoPath:
            print("No path found.")
            return []

    def identify_critical_bottlenecks(self, top_n=10) -> pd.DataFrame:
        """
        Identify hubs whose removal would most disrupt the network.
        Uses Betweenness Centrality weighted by travel cost.
        """
        print("\n" + "="*70)
        print("IDENTIFYING CRITICAL NETWORK BOTTLENECKS")
        print("="*70)

        # Calculate centrality
        # This can be slow for very large graphs
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')

        bottlenecks = []
        # Sort by centrality score
        sorted_hubs = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

        for hub_id, centrality in sorted_hubs:
            hub_info = self.hub_analysis_df[self.hub_analysis_df['Hub_ID'].astype(str) == str(hub_id)]

            if not hub_info.empty:
                info = hub_info.iloc[0]
                bottlenecks.append({
                    'Hub_ID': hub_id,
                    'Centrality_Score': centrality,
                    'Flow_Volume': info.get('Total_Outgoing_Flow', 0),
                    'Avg_Hub_Delay': info.get('Avg_Segment_Delay', 0)
                })

        return pd.DataFrame(bottlenecks)

    def visualise_transport_network(self, output_path='results/network_graph.html'):
        """
        Visualizes the transport network with critical bottlenecks highlighted.
        """
        import plotly.graph_objects as go
        
        pos = nx.spring_layout(self.graph, seed=42)
        
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        # Calculate centrality for sizing
        centrality = nx.betweenness_centrality(self.graph)
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            score = centrality.get(node, 0)
            node_text.append(f"Hub: {node}<br>Centrality: {score:.3f}")
            node_size.append(10 + score * 100) # Base size 10, scale by centrality

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=[centrality.get(node,0) for node in self.graph.nodes()],
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title=dict(text='Centrality', side='right'),
                    xanchor='left'
                ),
                line_width=2))

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text='Critical Transport Network Topology',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Network graph saved to {output_path}")
