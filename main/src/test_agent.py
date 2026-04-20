import os
import json
from agent import InsightAgent

def run_test():
    # Setup dummy data based on user's schema concept
    bottlenecks = [
        {
            "hub_id": "HUB_A12",
            "location": "Chicago",
            "degree_centrality": 0.85,
            "betweenness_centrality": 0.92,
            "avg_delay_mins": 45,
            "status": "CRITICAL"
        },
        {
            "hub_id": "HUB_B45",
            "location": "Atlanta",
            "degree_centrality": 0.65,
            "betweenness_centrality": 0.78,
            "avg_delay_mins": 25,
            "status": "WARNING"
        }
    ]
    
    hub_clusters = [
        {
            "cluster_id": "C_MIDWEST",
            "num_hubs": 15,
            "gnn_embedding_variance": 0.05,
            "dominant_bottleneck": "HUB_A12",
            "description": "High density region clustering around Chicago"
        },
        {
            "cluster_id": "C_SOUTH",
            "num_hubs": 22,
            "gnn_embedding_variance": 0.12,
            "dominant_bottleneck": "HUB_B45",
            "description": "Distributed Southern network with medium delays"
        }
    ]
    
    # Initialize the agent
    print("Initializing InsightAgent...")
    if "GEMINI_API_KEY" not in os.environ:
        print("WARNING: GEMINI_API_KEY environment variable is not set.")
        print("Please set your GEMINI_API_KEY. Exiting test.")
        return
        
    try:
        agent = InsightAgent()
        print("Generating Executive Brief...")
        response = agent.generate_brief(bottlenecks, hub_clusters)
        
        print("\n" + "="*50)
        print("📝 EXECUTIVE BRIEF (MARKDOWN)")
        print("="*50)
        print(response.executive_brief)
        print("\n" + "="*50)
        print("🛠️ OPERATIONAL RECOMMENDATIONS")
        print("="*50)
        for rec in response.operational_recommendations:
            print(f"- Action: {rec.action}")
            print(f"  Target: {rec.target_hub}")
            print(f"  Impact: {rec.expected_impact}")
            
    except Exception as e:
        print(f"Error during execution: {e}")

if __name__ == "__main__":
    run_test()
