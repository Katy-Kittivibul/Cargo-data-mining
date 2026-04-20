# Cargo Network Analysis Project

## Overview
This project represents a comprehensive refactoring of the legacy `Cargo_data_mining` codebase into a modular, production-ready Python pipeline. The system performs end-to-end analysis of logistics networks, including data cleaning, graph-based modelling, delay prediction, and route optimisation. 

*Note: This system focuses on Graph Neural Networks (GNN) and classical Machine Learning for network reasoning.*

## Dataset
This project uses the **Cargo 2000 Case Study Dataset**.

**Download the dataset here:** [Kaggle - Cargo 2000 Dataset](https://www.kaggle.com/datasets/crawford/cargo-2000-dataset)

Please place the `c2k_data_comma.csv` file in the `main/data/` directory before running the pipeline.

## Project Structure
All scripts and local directories are organised within the `main` directory:

- **`main.py`**: The central entry point that orchestrates the entire pipeline.
- **`src/`**: Contains the core logic modules:
    - `preprocessing.py`: Handles data ingestion, cleaning, and transformation into long-format journey logs.
    - `graph_builder.py`: Constructs the transport network graph using **PyTorch Geometric**.
    - `gnn_model.py`: Implements a **GraphSAGE** neural network for learning hub embeddings.
    - `clustering.py`: Performs K-Means clustering and detects anomalous hub behaviours.
    - `prediction.py`: accurately predicts shipment delays using Gradient Boosting and Random Forest models.
    - `optimization.py`: Identifies optimal routes and critical network bottlenecks using NetworkX.
    - `analytics.py`: Delivers cost-benefit analysis and interactive visualisations.
    - `eda_analysis.py`: Provides statistical insights (distribution skewness, multicollinearity checks).

## Key Insights
- **Network Bottlenecks**: Topology analysis identifies critical choke points (e.g., Hub 349) using centrality scores and historical delay metadata.
- **Delay Patterns**: Pattern mining identifies recurrent sequences of delays on specific routes.
- **Graph Embeddings**: Using GNNs allows the system to capture latent hub relationships, improving the reliability of logistics clustering.

## How to Run
1. **Prepare Data**: Download the dataset from Kaggle and place it in `main/data/`.
2. **Install Dependencies**:
   ```bash
   cd main
   pip install -r requirements.txt
   ```
3. **Execute the Pipeline**:
   ```bash
   python main.py
   ```
4. **View Results**:
   Explore the `main/results/` directory for interactive dashboards (`executive_dashboard.html`) and model artifacts.

   **Streamlit:**
   Interactive dashboard + "What-if" ROI simulation
   ```bash
   streamlit run .\dashboard.py
   ```

## Architecture
The system uses a modular design ensuring that each component (Preprocessing, Graph Building, Model Training, Analytics) can be tested and scaled independently.
