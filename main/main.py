import os
import sys
import pandas as pd

# Append 'src' directory to path so imports work from the root
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from preprocessing import LogisticsAnalyticEngine, calculate_kpis_and_aggregate
from graph_builder import build_pyg_graph
from gnn_model import train_with_link_prediction
from clustering import calculate_optimal_k, perform_clustering_and_analysis, interpret_clusters
from prediction import DelayPredictor
from optimization import RouteOptimizer
from analytics import CostBenefitAnalyzer, export_results, visualise_hub_activity_vs_delay

def main():
    print("🚀 STARTING CARGO ANALYSIS PIPELINE")
    
    # 1. Load Data
    raw_path = r"E:\Coding\Cargo\main\data\c2k_data_comma.csv"
    output_dir = r"E:\Coding\Cargo\main\results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"❌ Data file not found: {raw_path}")
        return

    # --- 1.1 EDA Analysis (New) ---
    print("Step 1.1: Running EDA...")
    from eda_analysis import DataAnalyzer
    # We load raw again for EDA or use filtered? Using raw for initial EDA
    # But for valid analysis, usually we clean first. Let's clean then EDA.
    raw_df = pd.read_csv(raw_path, low_memory=False)
    
    # Preprocessing
    print("Step 1: Preprocessing...")
    engine = LogisticsAnalyticEngine(raw_df)
    long_df, hub_kpis = engine.run_pipeline()
    
    # Run EDA on CLEANED data (long_df) to be more relevant
    analyzer = DataAnalyzer(long_df.drop(columns=['Hub_ID', 'Leg_ID', 'Milestone', 'Stage_Group', 'Leg_Type'], errors='ignore'))
    analyzer.run_correlation_analysis()
    analyzer.check_multicollinearity()
    analyzer.analyze_skewness()
    
    # 2. Graph Construction
    print("Step 2: Building Graph...")
    graph_data, hub_map, feature_names = build_pyg_graph(
        long_df, hub_kpis, normalize_features=True
    )
    
    # 3. GNN Training
    print("Step 3: Training GraphSAGE Model...")
    embeddings, model, loss_history = train_with_link_prediction(
        graph_data, epochs=50, embedding_dim=16, hidden_dim=32
    )
    
    # 4. Clustering
    print("Step 4: Clustering Analysis...")
    from clustering import analyse_cluster_boundaries, analyse_embedding_dimensions
    
    _, _, opt_k = calculate_optimal_k(embeddings, max_k=8)
    clustered_df, summary = perform_clustering_and_analysis(
        embeddings, hub_kpis, graph_data.hub_ids, num_clusters=opt_k
    )
    interpret_clusters(clustered_df)
    
    # New Clustering Analysis
    analyse_cluster_boundaries(embeddings, clustered_df, graph_data.hub_ids)
    analyse_embedding_dimensions(embeddings)
    
    # 5. Prediction
    print("Step 5: Training Delay Predictor...")
    from prediction import mine_delay_patterns
    predictor = DelayPredictor()
    # Note: re-using hub_kpis (which is hub_analysis_df)
    features_df = predictor.prepare_features(long_df, hub_kpis)
    pred_results = predictor.train(features_df)
    
    # Save Prediction Artifacts
    predictor.save_model(os.path.join(output_dir, "delay_predictor.pkl"))
    predictor.save_feature_importance_plot(os.path.join(output_dir, "feature_importance.html"))
    
    # Mine Patterns
    from prediction import mine_delay_patterns, save_patterns_to_csv
    patterns = mine_delay_patterns(long_df)
    save_patterns_to_csv(patterns, os.path.join(output_dir, "delay_patterns.csv"))
    
    # 6. Optimization
    print("Step 6: Network Optimization...")
    optimizer = RouteOptimizer(long_df, hub_kpis)
    bottlenecks = optimizer.identify_critical_bottlenecks(top_n=5)
    optimizer.visualise_transport_network(os.path.join(output_dir, "network_graph.html"))
    
    print("\nTop Network Bottlenecks:")
    print(bottlenecks)
    
    # 7. Financial Analysis / Analytics
    print("Step 7: Cost-Benefit Analysis...")
    cba = CostBenefitAnalyzer()
    costs = cba.calculate_delay_costs(long_df, hub_kpis)
    print(f"Total Network Delay Cost: £{costs['total_delay_cost']:,.2f}")
    
    # 8. Visualisation & Export
    print("Step 8: Exporting Results...")
    from analytics import (visualise_delay_accumulation, visualise_hub_profiles, 
                           assess_embedding_quality, create_summary_dashboard)

    # Save Plots
    visualise_hub_activity_vs_delay(hub_kpis).write_html(os.path.join(output_dir, "hub_performance_plot.html"))
    visualise_delay_accumulation(long_df).write_html(os.path.join(output_dir, "delay_accumulation.html"))
    visualise_hub_profiles(clustered_df).write_html(os.path.join(output_dir, "hub_profiles.html"))
    
    # Critical Fix: Align labels with embeddings (Handle size mismatch)
    # clustered_df contains ALL hubs, embeddings only contains GRAPH hubs
    aligned_labels = clustered_df.set_index('Hub_ID').loc[graph_data.hub_ids, 'Cluster_Label'].values
    assess_embedding_quality(embeddings, aligned_labels).write_html(os.path.join(output_dir, "embedding_pca.html"))
    
    create_summary_dashboard(clustered_df).write_html(os.path.join(output_dir, "executive_dashboard.html"))
    
    # Export data
    # We can create a dummy dataframe for critical routes for now or implement the full logic
    export_results(clustered_df, embeddings, bottlenecks, output_dir=output_dir)
    
    print("\n✅ PIPELINE FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()
