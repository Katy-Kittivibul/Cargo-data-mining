# Pipeline Findings Summary

This document summarizes the techniques, tools, and key findings from the executed logistics analysis pipeline, along with detailed explanations of how each result is calculated.

## 1. Logistics Analysis Pipeline (EDA & Preprocessing)

**Techniques:**

- Data Cleaning: Standardized timestamps and metric computations. Cleaned invalid formats and coerced items into numeric types.
- Missing values imputed or dropped based on step requirements.

**How it is calculated:**

- **Metrics Computation:** Delay metrics are derived directly from the raw data. The core calculation is `Delay_Mins = Effective_Mins - Planned_Mins` at various stages of the shipment (Check-in, Transport Segments, Delivery). If a shipment arrives early, the `Delay_Mins` is negative.

**Key Findings:**

- Extracted and computed duration-based metrics for statistical analysis.
- Top VIF Results (Features with VIF > 10 may be redundant):
  feature VIF
  Planned_Mins 1471.574222
  Effective_Mins 1315.951928
  Delay_Mins 77.335617

## 2. Graph Construction

**Techniques:**

- **Graph Structure:** Built a graph representation of the logistics network.

**How it is calculated:**

- **Nodes & Edges:** Each unique `Hub_ID` is registered as a "Node". An "Edge" (route) is created whenever a shipment moves sequentially from one Hub to another. The edge weights are based on the aggregate flow volume and the average traverse time (delay) between those two specific hubs.

**Key Findings:**

- **Graph Size:** Constructed a weighted transport graph with **238 hubs** (nodes) and **2,677 routes** (edges). Node integer fragmentation has been successfully resolved compared to earler runs.

## 3. Clustering Analysis

**Techniques:**

- **K-Means Clustering:** Segmented hubs based on performance and structure characteristics.

**How it is calculated:**

- **Embeddings & K-Means:** First, a GraphSAGE Neural Network learns vector representations (embeddings) for each hub based on its connectivity and delay KPIs. Then, the K-Means algorithm mathematically groups these hubs into clusters by minimizing the variance within each group, separating high-volume/efficient hubs from low-volume/inefficient ones. The optimal "K" is selected using the highest Silhouette Score.

**Key Findings:**

- **Optimal Silhouette Score:** 0.354 with K=4 clusters.
- **Cluster 0 (Standard Hubs, 74 hubs):** Acceptable performance. Avg flow: 7.57, Mean avg delay: -209.27.
- **Cluster 1 (Inefficient Small Hub, 129 hubs):** Low volume but problematic. Avg flow: 179.77, Mean avg delay: -152.87.
- **Cluster 2 (Critical Bottleneck, 7 hubs):** High volume with delays. Avg flow: 3357.00, Mean avg delay: -97.97. Priority urgent capacity management.
- **Cluster 3 (25 hubs):** Avg flow: 9.84, Mean avg delay: -201.8.

## 4. Delay Prediction & Pattern Mining

**Techniques:**

- **Feature Engineering & Conformal Prediction:** Prepared datasets for modeling and wrapped predictions to give exact confidence bounds.
- **Pattern Mining:** Identified frequent sequences of delays.

**How it is calculated:**

- **Prediction:** A Gradient Boosting Regressor algorithm combines various engineered features (like day of week, hub volume, historical hub delay) into decision trees to predict the exact `Delay_Mins` of a future shipment. A conformal predictor step mathematically calculates 90% confidence uncertainty margins (Lower/Upper bounds) for each prediction.
- **Pattern Mining:** The system scans historical sequences and counts how many times specific hubs consecutively appear with a delay exceeding a strict threshold (e.g., >30 minutes).

**Key Findings:**

- **Gradient Boosting Results:** Hyperparameter tuning (GridSearchCV) found optimal parameters (`learning_rate: 0.05`, `max_depth: 5`, `min_samples_split: 5`, `n_estimators: 300`). Achieved **MAE: 163.64 min** and **R²: 0.643**.
- **Conformal Bounds:** Confidence intervals were successfully generated on 31,761 calibrated samples.
- **Frequent Delays:** Hubs **815 (1100 occurrences), 700 (893 occurrences), 128, 485, and 349** appear most frequently in >30 min delay sequences. Data fragmentation (e.g. 815 vs 815.0) has been fixed.

## 5. Network Optimization

**Techniques:**

- **Weighted Transport Network:** Identified critical connections.
- **Centrality Analysis:** Calculated centrality scores to identify structural bottlenecks.

**How it is calculated:**

- **Betweenness Centrality:** The system simulates the shortest paths for all shipments moving across the network. A hub's "Centrality Score" is the fraction of all shortest paths that pass through that specific hub. A higher score closer to 1.0 means the hub acts as a crucial bridge or bottleneck for the entire global network.

**Key Findings:**

- **Top Bottlenecks:**
  - **Hub 349:** Highest centrality (0.152), 3501 flows, averaging +50 mins delay.
  - **Hub 700:** Centrality 0.121, 5277 flows.
  - **Hub 128:** Centrality 0.108, 4894 flows.
  - **Hub 256:** Centrality 0.107, 623 flows.
  - **Hub 281:** Centrality 0.090, 1044 flows.

## 6. Cost-Benefit Analysis

**How it is calculated:**

- **Financial Cost Logic:** Total cost is the sum of two variables:
  1. **Linear Cost:** `Incremental_Delay_Mins` × `£1.00/minute`.
  2. **Penalty Cost:** If the total cumulative delay for a shipment segment exceeds 60 minutes, an additional heavy monetary penalty (`Penalty_Hours_Over_60min` × `Penalty_Rate` × `Base_Cost`) is applied. The final total across all hubs results in the aggregate network cost.

**Findings:**

- **Total Network Delay Cost:** Calculated at **£7,022,346.00**.

## 7. Explainable AI & Agentic Insights

**How it is calculated:**

- **XAI (Explainable AI):** Evaluates which neighbor hubs had the mathematically highest "influence" over a specific hub's predicted delay by analyzing the neural network gradients via the GNN explainer layer.
- **Agentic Insights:** Ingests the pure JSON data of the network graph into Google Gemini to return a human-readable operational summary.

**Findings:**

- **XAI Results:** Completed structural explainability reports dynamically for our 5 top critical hubs (349, 700, 128, 256, 281).
- **Insight Agent:** Failed execution due to a missing `GEMINI_API_KEY` environment variable.
