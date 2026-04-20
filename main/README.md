# Cargo Data Mining and Analysis Pipeline

This project is a Production-Grade, Agentic AI system designed for logistics analysis, focusing on the Cargo 2000 dataset. It utilizes Graph Neural Networks (GNNs), Conformal Prediction, and Agentic reasoning to identify network bottlenecks, predict delays with uncertainty quantification, and generate executive insights.

## Project Overview

- **Insight Agent:** LLM-driven Decision Support System that reasons over graph topology.
- **Probabilistic Forecasting:** Conformal Prediction for mathematically guaranteed uncertainty intervals on delay predictions.
- **Explaniable GNNs (GNN-XAI):** Transparency into GraphSAGE model decisions using feature and edge attribution.
- **Network Optimization:** Identification of critical bottlenecks and route optimization.

## Dataset

This project uses the **Cargo 2000 Case Study Dataset**.

**Download the dataset here:** [Kaggle - Cargo 2000 Dataset](https://www.kaggle.com/datasets/crawford/cargo-2000-dataset)

Please place the `c2k_data_comma.csv` file in the `data/` directory before running the pipeline.

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Katy-Kittivibul/Cargo-data-mining.git
   cd Cargo-data-mining
   ```

2. Set up virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Pipeline

To run the full analysis pipeline:
```bash
python main.py
```

## Architecture

- `main.py`: Entry point for the analysis pipeline.
- `src/`: Core logic modules (preprocessing, graph construction, training, etc.).
- `data/`: Directory for input datasets (ignored by git).
- `results/`: Directory for generated plots, models, and reports (ignored by git).

## License

[Add License Info Here]
