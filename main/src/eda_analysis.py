import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

class DataAnalyzer:
    """
    Analyses logistics data for correlations, multicollinearity, and distributions.
    Strictly saves results to project_root/reports/eda/.
    """
    def __init__(self, df: pd.DataFrame, report_subfolder: str = "eda"):
        self.df = df
        
        # 1. Dynamically find the project root 
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = project_root / "reports" / report_subfolder
        
        # Create directory (and parents) if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Reports will be saved to: {self.output_dir}")
        self.numeric_df = self._prepare_numeric_df()

    def _prepare_numeric_df(self) -> pd.DataFrame:
        """Drops categorical and ID columns to isolate numeric metrics."""
        to_drop = [
            'Total_Incoming_Legs', 'All_process_ID', 'OUT_Leg_ID',
            'IN_1_Leg_ID', 'IN_2_Leg_ID', 'IN_3_Leg_ID'
        ]
        # Identify Hub ID columns dynamically
        hub_cols = [col for col in self.df.columns if '_Hub_ID' in col or '_place' in col]
        to_drop.extend(hub_cols)
        to_drop.extend([c for c in ['IN_1_Hops', 'IN_2_Hops', 'IN_3_Hops', 'OUT_Hops'] if c in self.df.columns])

        existing_drops = [c for c in to_drop if c in self.df.columns]
        return self.df.drop(columns=existing_drops).select_dtypes(include=[np.number])

    def run_correlation_analysis(self):
        """Calculates and visualises the correlation matrix."""
        print("\n--- Analysing Correlations ---")
        corr_matrix = self.numeric_df.corr()
        
        # Save CSV
        corr_matrix.to_csv(self.output_dir / "correlation_matrix.csv")
        
        # Save Heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title('Logistics Network Correlation Heatmap', fontsize=16)
        
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Correlation CSV and PNG saved.")

    def check_multicollinearity(self):
        """Calculates VIF and saves to CSV."""
        print("\n--- Checking Multicollinearity (VIF) ---")
        
        vif_df = pd.DataFrame()
        vif_df["feature"] = self.numeric_df.columns
        vif_df["VIF"] = [
            variance_inflation_factor(self.numeric_df.values, i)
            for i in range(len(self.numeric_df.columns))
        ]
        
        vif_df = vif_df.sort_values(by='VIF', ascending=False)
        vif_df.to_csv(self.output_dir / "vif_analysis.csv", index=False)
        
        print(f"✅ Top VIF Results (Features with VIF > 10 may be redundant):")
        print(vif_df.head(5).to_string(index=False))
        return vif_df

    def analyze_skewness(self):
        """Calculates skewness and saves distribution plots."""
        print("\n--- Analysing Distribution Skewness ---")
        
        

        results = []
        for col in self.numeric_df.columns:
            s = skew(self.numeric_df[col].dropna())
            
            if s > 1:
                interp = "Highly Positive Skew"
            elif s < -1:
                interp = "Highly Negative Skew"
            elif -0.5 < s < 0.5:
                interp = "Fairly Symmetrical"
            else:
                interp = "Moderate Skew"
            
            results.append({'Column': col, 'Skewness': s, 'Interpretation': interp})

        skew_df = pd.DataFrame(results)
        skew_df.to_csv(self.output_dir / "skewness_report.csv", index=False)

        # Visualise first 9 distributions
        cols_to_plot = self.numeric_df.columns[:9]
        plt.figure(figsize=(15, 12))
        for i, col in enumerate(cols_to_plot, 1):
            plt.subplot(3, 3, i)
            sns.histplot(self.numeric_df[col], kde=True, bins=30, color='seagreen')
            plt.title(f"Dist: {col}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_plots.png", dpi=300)
        plt.close()
        print(f"✅ Distribution analysis and plots saved.")

# --- Execution ---
if __name__ == "__main__":
    # Corrected path for your local environment
    processed_data_path = r"E:\Coding\Cargo\main\data\cleaned_data.csv"
    
    if os.path.exists(processed_data_path):
        data = pd.read_csv(processed_data_path)
        analyzer = DataAnalyzer(data)
        
        analyzer.run_correlation_analysis()
        analyzer.check_multicollinearity()
        analyzer.analyze_skewness()
        
        print(f"\n🚀 Analysis finished. Check outputs in: {analyzer.output_dir}")
    else:
        print(f"❌ Error: Could not find {processed_data_path}. Please check the file path.")