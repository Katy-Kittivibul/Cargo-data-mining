import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

def calculate_optimal_k(embeddings: np.ndarray, max_k=10):
    """
    Calculates optimal K using Elbow Method and Silhouette Score.
    """
    print("\n" + "=" * 70)
    print(f"CALCULATING OPTIMAL K (K=1 to K={max_k})")
    print("=" * 70)

    inertia_values = []
    silhouette_scores = []
    # Start from 2 (silhouette needs at least 2 clusters)
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        inertia_values.append(kmeans.inertia_)
        sil_score = silhouette_score(embeddings, labels)
        silhouette_scores.append(sil_score)

    # Recommend optimal K
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    print(f"\n💡 Recommended K (highest silhouette): {optimal_k_silhouette}")
    print(f"  Silhouette score: {max(silhouette_scores):.3f}")

    return inertia_values, silhouette_scores, optimal_k_silhouette

def perform_clustering_and_analysis(embeddings: np.ndarray,
                                    hub_analysis_df: pd.DataFrame,
                                    hub_ids: np.ndarray,
                                    num_clusters=3):
    """
    Applies K-Means clustering and returns clustered dataframe and summary.
    """
    print("\n" + "=" * 70)
    print(f"PERFORMING K-MEANS CLUSTERING (K={num_clusters})")
    print("=" * 70)

    # Validate inputs
    if len(embeddings) != len(hub_ids):
        raise ValueError(f"Embeddings length ({len(embeddings)}) != hub_ids length ({len(hub_ids)})")

    # K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)

    print(f"Silhouette Score: {silhouette:.3f} (higher is better, range: [-1, 1])")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")

    # Create cluster DataFrame
    cluster_df = pd.DataFrame({
        'Hub_ID': hub_ids.astype(str),
        'Cluster_Label': cluster_labels
    })

    # Merge with hub analysis
    hub_analysis_temp = hub_analysis_df.copy()
    hub_analysis_temp['Hub_ID'] = hub_analysis_temp['Hub_ID'].astype(str)
    final_df = hub_analysis_temp.merge(cluster_df, on='Hub_ID', how='left')

    # Data cleanup
    for col in final_df.columns:
        if str(final_df[col].dtype) == 'Int64':
            final_df[col] = final_df[col].astype(float).fillna(0)
    
    if 'Total_Incoming_Flow' in final_df.columns:
        final_df['Total_Incoming_Flow'] = final_df['Total_Incoming_Flow'].fillna(0)
    if 'Total_Outgoing_Flow' in final_df.columns:
        final_df['Total_Outgoing_Flow'] = final_df['Total_Outgoing_Flow'].fillna(0)

    for col in ['Avg_Segment_Delay', 'Std_Delay', 'Max_Delay']:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    # Cluster summary statistics
    agg_dict = {
        'Hub_ID': 'count',
        'Total_Outgoing_Flow': ['mean', 'sum'],
        'Avg_Segment_Delay': 'mean',
        'Std_Delay': 'mean'
    }
    agg_dict = {k: v for k, v in agg_dict.items() if k in final_df.columns}

    cluster_summary = final_df.groupby('Cluster_Label').agg(agg_dict).reset_index()
    cluster_summary.columns = ['Cluster_Label', 'Hubs_Count', 'Mean_Flow', 'Total_Flow',
                                'Mean_Avg_Delay', 'Mean_Std_Delay']

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY STATISTICS")
    print("=" * 70)
    print(cluster_summary.to_string(index=False))

    return final_df, cluster_summary

def interpret_clusters(clustered_df: pd.DataFrame):
    """
    Provides business interpretation of the discovered hub clusters.
    """
    print("\n" + "="*70)
    print("CLUSTER INTERPRETATION & BUSINESS INSIGHTS")
    print("="*70)

    if 'Cluster_Label' not in clustered_df.columns:
        print("⚠️ No clusters to interpret.")
        return

    for cluster_id in sorted(clustered_df['Cluster_Label'].unique()):
        cluster_hubs = clustered_df[clustered_df['Cluster_Label'] == cluster_id]

        print(f"\n{'='*70}")
        print(f"🔹 CLUSTER {cluster_id} ({len(cluster_hubs)} hubs)")
        print(f"{'='*70}")

        total_network_flow = clustered_df['Total_Outgoing_Flow'].sum()
        stats = {
            'Avg Flow': cluster_hubs['Total_Outgoing_Flow'].mean(),
            'Total Flow': cluster_hubs['Total_Outgoing_Flow'].sum(),
            'Avg Delay': cluster_hubs['Avg_Segment_Delay'].mean(),
            'Avg Std Delay': cluster_hubs['Std_Delay'].mean(),
            'Flow % of Total': (cluster_hubs['Total_Outgoing_Flow'].sum() / total_network_flow * 100) if total_network_flow > 0 else 0
        }

        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")

        # Business Rule Logic
        avg_flow_all = clustered_df['Total_Outgoing_Flow'].mean()
        avg_delay_all = clustered_df['Avg_Segment_Delay'].mean()

        high_flow = stats['Avg Flow'] > avg_flow_all
        high_delay = stats['Avg Delay'] > avg_delay_all

        if high_flow and high_delay:
            cluster_type = "⚠️  CRITICAL BOTTLENECK - High volume with delays"
            recommendation = "Priority 1: Urgent capacity expansion or process optimization needed"
        elif high_flow and not high_delay:
            cluster_type = "✅ EFFICIENT MAJOR HUB - High volume, well-performing"
            recommendation = "Monitor closely and replicate best practices"
        elif not high_flow and high_delay:
            cluster_type = "⚡ INEFFICIENT SMALL HUB - Low volume but problematic"
            recommendation = "Investigate operational issues or consider consolidation"
        else:
            cluster_type = "📦 STANDARD HUB - Low volume, acceptable performance"
            recommendation = "Standard operations, routine monitoring"

        print(f"\n  Type: {cluster_type}")
        print(f"  Recommendation: {recommendation}")

        # Top 5 hubs in cluster
        print(f"\n  Top 5 Hubs:")
        top_5 = cluster_hubs.nlargest(5, 'Total_Outgoing_Flow')[
            ['Hub_ID', 'Total_Outgoing_Flow', 'Avg_Segment_Delay']
        ]

        for _, row in top_5.iterrows():
            print(f"    Hub {row['Hub_ID']}: {int(row['Total_Outgoing_Flow'])} flows, {row['Avg_Segment_Delay']:.1f} min delay")

    print("\n" + "="*70)

def detect_anomalous_hubs(embeddings: np.ndarray, clustered_df: pd.DataFrame,
                          graph_hub_ids: np.ndarray, threshold=0.1):
    """
    Identifies hubs that are poorly represented by their assigned cluster using Silhouette scores.
    """
    print("\n" + "="*70)
    print(f"ANOMALOUS HUB DETECTION (Silhouette Score < {threshold})")
    print("="*70)

    # DataFrame aligning embedding index, Hub_ID, and cluster label
    full_df = pd.DataFrame({
        'Hub_ID': graph_hub_ids.astype(str),
        'row_idx': range(len(graph_hub_ids))
    })

    clustered_temp = clustered_df.copy()
    clustered_temp['Hub_ID'] = clustered_temp['Hub_ID'].astype(str)
    full_df = full_df.merge(clustered_temp, on='Hub_ID', how='left')

    valid_mask = full_df['Cluster_Label'].notna()
    X = embeddings[valid_mask]
    labels = full_df.loc[valid_mask, 'Cluster_Label'].astype(int).values

    silhouette_vals = silhouette_samples(X, labels)
    full_df.loc[valid_mask, 'Silhouette_Score'] = silhouette_vals

    anomalous = full_df[
        (full_df['Silhouette_Score'] < threshold)
    ].copy().sort_values('Silhouette_Score')

    print(f"Found {len(anomalous)} hubs with poor cluster fit.")

    if not anomalous.empty:
        cols_to_show = ['Hub_ID', 'Cluster_Label', 'Total_Outgoing_Flow',
                        'Avg_Segment_Delay', 'Silhouette_Score']
        print("\nTOP CRITICAL ANOMALIES:")
        print(anomalous[cols_to_show].head(15).to_string(index=False))

    return anomalous

def analyse_cluster_boundaries(embeddings: np.ndarray, clustered_df: pd.DataFrame, hub_ids: np.ndarray):
    """
    Identifies hubs that are on the boundary between two clusters.
    This serves as a proxy for 'Cluster Transition Analysis'.
    """
    print("\n" + "="*70)
    print("ANALISING CLUSTER BOUNDARIES")
    print("="*70)
    
    from sklearn.metrics import pairwise_distances_argmin_min
    
    # Find distance to all cluster centers
    kmeans = KMeans(n_clusters=clustered_df['Cluster_Label'].nunique(), random_state=42, n_init=10)
    # Note: Ideally we pass the already fitted kmeans model, but re-fitting for refactor simplicity
    kmeans.fit(embeddings)
    
    dists = kmeans.transform(embeddings)
    # Sort distances to true center vs 2nd closest center
    sorted_dists = np.sort(dists, axis=1)
    
    margin = sorted_dists[:, 0] / sorted_dists[:, 1] # Ratio closest / 2nd closest
    
    boundary_mask = margin > 0.85 # Arbitrary threshold for "uncertain" assignment
    
    # Critical Fix: Align clustered_df to the embeddings/hub_ids
    # We need the cluster labels corresponding to the graph hubs (embeddings)
    try:
        aligned_df = clustered_df.set_index('Hub_ID').loc[hub_ids.astype(str)]
        assigned_clusters = aligned_df['Cluster_Label'].values
    except Exception as e:
        print(f"Warning: Could not align clusters: {e}")
        assigned_clusters = ["Unknown"] * len(hub_ids)

    boundary_hubs = pd.DataFrame({
        'Hub_ID': hub_ids[boundary_mask].astype(str),
        'Margin_Ratio': margin[boundary_mask],
        'Assigned_Cluster': np.array(assigned_clusters)[boundary_mask]
    }).sort_values('Margin_Ratio', ascending=False)
    
    print(f"Found {len(boundary_hubs)} hubs on cluster boundaries (ambiguous classification).")
    if not boundary_hubs.empty:
        print(boundary_hubs.head(10).to_string(index=False))
        
    return boundary_hubs

def analyse_embedding_dimensions(embeddings: np.ndarray, feature_names: list = None):
    """
    Analyzes which dimensions of the embedding variance are most significant.
    """
    print("\n" + "="*70)
    print("EMBEDDING DIMENSION IMPORTANCE")
    print("="*70)
    
    # Simple variance analysis
    variances = np.var(embeddings, axis=0)
    feature_imp = pd.DataFrame({
        'Dimension': range(len(variances)),
        'Variance': variances
    }).sort_values('Variance', ascending=False)
    
    print("Top 5 Dimensions by Variance (Information Content):")
    print(feature_imp.head(5).to_string(index=False))
    
    return feature_imp
