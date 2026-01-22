# Cargo Network Analysis Project

## Overview
This project represents a comprehensive refactoring of the legacy `Cargo_data_mining` codebase into a modular, production-ready Python pipeline. The system performs end-to-end analysis of logistics networks, including data cleaning, graph-based modelling, delay prediction, and route optimisation.

The primary objective was to transition from a monolithic Jupyter Notebook to a structured software architecture that is scalable, maintainable, and tailored for VS Code execution.

## Project Structure
The solution is organised within the `main` directory:

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

## What We Learned & Found
Through the application of this advanced analytics pipeline, we have uncovered several key insights regarding the logistics network:

### 1. Network Bottlenecks
Our topology analysis identified specific hubs that serve as critical bridges in the network. Notably:
- **Hub 349** acts as a significant choke point, exhibiting a high centrality score alongside a concerning average delay of **~252 minutes**.
- **Hubs 128, 700, and 815** are the most central nodes. While highly active, their variability significantly impacts the broader network stability.

### 2. Delay Patterns
Pattern mining revealed that delays are not randomly distributed but often stem from specific nodes and sequences:
- **Hub 815** and **Hub 700** frequently appear in delayed journey legs.
- The route segment **671.0 → 700** was identified as a recurrent path for shipment lateness, suggesting a structural issue on this specific link.

### 3. Technical Learnings
- **Modularisation**: Splitting the logic into distinct concerns (Preprocessing vs Modelling) drastically improved debugging and testing capabilities.
- **Graph Neural Networks (GNN)**: Embedding the transport network allowed us to capture latent relationships between hubs that traditional statistical methods missed, enabling more robust clustering.
- **Data Integrity**: We discovered that "effective time" metrics often precede "planned time" (resulting in negative delays), indicating potential data logging inconsistencies or conservative planning buffers.

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute the Pipeline**:
   ```bash
   python main.py
   ```
3. **View Results**:
   Explore the `results/` directory for interactive dashboards (`executive_dashboard.html`) and model artifacts.

## Conclusion
This refactored solution provides a solid foundation for future predictive logistics. By leveraging GNNs and modular software design, the system is now poised for integration into a real-time monitoring environment.
