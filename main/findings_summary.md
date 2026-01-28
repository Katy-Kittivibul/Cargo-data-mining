# Terminal Results Summary

This document summarizes the techniques, tools, and key findings from the executed pipeline as recorded in `terminal_results.txt`.

## 1. Logistics Analysis Pipeline (EDA & Preprocessing)
**Techniques:**
- **Data Conversion:** Converted 97 columns to numeric types.
- **Correlation Analysis:** Generated correlation matrices.
- **Multicollinearity Check:** Used Variance Inflation Factor (VIF) to identify redundant features.
- **Distribution Analysis:** Analyzed skewness of distributions.

**Tools:**
- **VIF Analysis:** For multicollinearity detection.
- **EDA & Reporting:** Automated report generation in `reports/eda`.

**Key Findings:**
- **Data Integrity:** 97 numeric columns processed; no categorical/ID columns remained.
- **Multicollinearity:** Three features (`Planned_Mins`, `Effective_Mins`, `Delay_Mins`) showed infinite VIF, indicating perfect multicollinearity and redundancy.

## 2. Graph Construction
**Techniques:**
- **Graph Structure:** Built a graph representation of the logistics network.

**Tools:**
- **PyTorch Geometric:** Used for building the graph data structure.

**Key Findings:**
- **Graph Size:** Constructed a graph with **427 nodes** and **4,163 edges**.

## 3. Training GraphSAGE Model
**Techniques:**
- **Graph Neural Network (GNN):** GraphSAGE architecture.
- **Link Prediction Loss:** Optimized model to predict connections between hubs.
- **Node Embeddings:** Generated 16-dimensional vector representations for each hub.

**Tools:**
- **GraphSAGE:** The specific GNN architecture used.
- **PyTorch (CUDA):** Accelerated training on GPU.

**Key Findings:**
- **Model Convergence:** Loss decreased significantly from **6.77** (Epoch 1) to **1.17** (Epoch 40), indicating successful learning.
- **Embedding Generation:** produced (427, 16) embeddings for downstream tasks.

## 4. Clustering Analysis
**Techniques:**
- **K-Means Clustering:** Segmented hubs based on embeddings/features.
- **Silhouette Analysis:** Determined optimal cluster count (K).
- **Davies-Bouldin Index:** Validated cluster separation.

**Tools:**
- **K-Means:** The clustering algorithm.
- **Silhouette Score:** Metric for cluster quality.

**Key Findings:**
- **Optimal Segments:** K=3 was identified as optimal (Silhouette Score: **0.755**).
- **Cluster Profiles:**
    - **Cluster 0 (Standard Hubs):** 416 hubs. Low volume (67 flows), acceptable delay (~73 min). Represents the majority of the network.
    - **Cluster 1 (Inefficient Small Hub):** **1 hub (Hub 620.0)**. Extremely high delay (**61,181 min** avg) despite low volume. Needs immediate investigation.
    - **Cluster 2 (Efficient Major Hubs):** 10 hubs. High volume (2,274 flows), **negative delay** (-371 min, meaning early arrivals). These are the top performers.

## 5. Delay Prediction & Pattern Mining
**Techniques:**
- **Feature Engineering:** Prepared 29,710 samples for modeling.
- **Gradient Boosting:** Trained a regression model to predict delays.
- **Pattern Mining:** Identified frequent sequences of delays.

**Tools:**
- **Gradient Boosting Regressor:** The predictive model.
- **Pattern Mining:** For sequence analysis.

**Key Findings:**
- **Model Performance:** MAE: **192.38 min**, R²: **0.513**.
- **Frequent Delays:** Hubs **815, 700, and 128** appear most frequently in delay patterns, likely due to their high volume.

## 6. Network Optimization
**Techniques:**
- **Weighted Transport Network:** Modeled the network with weights.
- **Centrality Analysis:** Calculated centrality scores to identify bottlenecks.

**Tools:**
- **Network Graph:** Saved as interactive HTML.

**Key Findings:**
- **Top Bottlenecks:**
    - **Hub 128.0:** Highest centrality (0.170).
    - **Hub 700.0:** Second highest (0.128).
    - **Hub 815.0:** Third highest (0.124).
    These hubs are critical to network flow but are also generally efficient (from Cluster 2).

## 7. Cost-Benefit Analysis
**Techniques:**
- **Financial modeling:** Calculated total cost implications of delays.

**Key Findings:**
- **Total Network Delay Cost:** **£16,348,950.00**.
