import os
import sys
import pandas as pd

# Fix for Windows console unicode issues
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

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
    raw_path = os.path.join(current_dir, "data", "c2k_data_comma.csv")
    output_dir = os.path.join(current_dir, "results")
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
    # Analyst Fix: Reduced embedding dimensions (8 instead of 16/32) to prevent over-segmentation
    embeddings, model, loss_history = train_with_link_prediction(
        graph_data, epochs=50, embedding_dim=8, hidden_dim=16
    )
    
    # 4. Clustering
    print("Step 4: Clustering Analysis...")
    from clustering import analyse_cluster_boundaries, analyse_embedding_dimensions
    
    inertia, sil_scores, opt_k = calculate_optimal_k(embeddings, max_k=6)
    
    # Analyst Logic: If optimal K is high but silhouette is weak (< 0.4), fallback to K=2
    # This ensures we get the "Critical vs Standard" binary which is most actionable.
    current_best_sil = max(sil_scores) if sil_scores else 0
    if opt_k > 2 and current_best_sil < 0.45:
        print(f"⚠️ Optimal K was {opt_k} but Silhouette ({current_best_sil:.3f}) is weak. Falling back to K=2 for stability.")
        opt_k = 2

    clustered_df, summary, kmeans_model = perform_clustering_and_analysis(
        embeddings, hub_kpis, graph_data.hub_ids, num_clusters=opt_k
    )
    interpret_clusters(clustered_df)
    
    # New Clustering Analysis
    analyse_cluster_boundaries(embeddings, clustered_df, graph_data.hub_ids, kmeans_model=kmeans_model)
    analyse_embedding_dimensions(embeddings)
    
    # 5. Prediction
    print("Step 5: Training Delay Predictor...")
    from prediction import mine_delay_patterns
    predictor = DelayPredictor()
    # Note: re-using hub_kpis (which is hub_analysis_df)
    features_df = predictor.prepare_features(long_df, hub_kpis)
    pred_results = predictor.train(features_df, tune_hyperparameters=True)
    
    # Save Prediction Artifacts
    predictor.save_model(os.path.join(output_dir, "delay_predictor.pkl"))
    predictor.save_feature_importance_plot(os.path.join(output_dir, "feature_importance.html"))
    
    # 5.1 Conformal Prediction
    print("Step 5.1: Conformal Prediction Calibration...")
    from prediction import ConformalPredictor
    # Using features_df as calibration data for demonstration
    conformal_predictor = ConformalPredictor(predictor, features_df)
    sample_ci = conformal_predictor.predict_with_interval(features_df.head(5), alpha=0.1)
    print("Sample 90% Confidence Intervals:\n", sample_ci)
    
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
    export_results(clustered_df, embeddings, bottlenecks, long_flow_df=long_df, output_dir=output_dir)
    
    # 9. Insight Agent
    print("Step 9: Generating Executive Brief...")
    try:
        from agent import InsightAgent
        agent = InsightAgent()
        
        # Convert bottlenecks and clusters to list of dicts for the agent
        bn_data = bottlenecks if isinstance(bottlenecks, list) else [{"info": str(bottlenecks)}]
        if hasattr(bottlenecks, 'to_dict'):
            bn_data = bottlenecks.to_dict('records')
            
        brief_response = agent.generate_brief(
            bottlenecks=bn_data,
            hub_clusters=clustered_df.to_dict('records')[:15]  # Limit to 15 hubs to save context window
        )
        brief_path = os.path.join(output_dir, "executive_brief.md")
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(brief_response.executive_brief)
        print(f"✅ Executive Brief saved to {brief_path}")
    except Exception as e:
        print(f"⚠️ Insight Agent skipped (check GEMINI_API_KEY): {e}")

    # 10. Explainable GNN
    print("Step 10: Generating XAI Reports for critical hubs...")
    try:
        from explainability import explain_critical_routes
        # Batch explain all hubs listed in the critical routes file
        critical_csv = os.path.join(output_dir, "critical_routes.csv")
        explain_critical_routes(
            model=model, 
            data=graph_data, 
            critical_routes_csv=critical_csv, 
            feature_names=feature_names, 
            output_dir=output_dir
        )
    except Exception as e:
        print(f"⚠️ XAI Reports skipped: {e}")

    
    print("\n✅ PIPELINE FINISHED SUCCESSFULLY")

if __name__ == "__main__":
    main()
