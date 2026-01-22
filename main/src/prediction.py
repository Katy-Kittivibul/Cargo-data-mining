import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class DelayPredictor:
    """
    Machine learning model to predict shipment delays.
    """
    def __init__(self, model_type='gradient_boosting'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.feature_cols = [
            'Segment_Number', 'Flow_source', 'Flow_target',
            'Avg_Delay_source', 'Avg_Delay_target', 'Std_Delay_source',
            'Std_Delay_target', 'Route_Avg_Delay', 'Route_Std_Delay',
            'Route_Volume', 'Planned_Duration', 'Hub_Interaction', 'Delay_Risk_Score'
        ]

    def prepare_features(self, long_flow_df: pd.DataFrame, hub_analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for delay prediction.
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING FOR DELAY PREDICTION")
        print("="*70)

        # 1. Path Reconstruction
        transit_df = long_flow_df.sort_values(['Leg_ID', 'Planned_Mins']).copy()
        transit_df['Source_Hub_ID'] = transit_df['Hub_ID'].astype(str)
        transit_df['Target_Hub_ID'] = transit_df.groupby('Leg_ID')['Hub_ID'].shift(-1).astype(str)

        # Numeric conversions
        transit_df['Effective_Mins'] = pd.to_numeric(transit_df['Effective_Mins'], errors='coerce')
        transit_df['Planned_Mins'] = pd.to_numeric(transit_df['Planned_Mins'], errors='coerce')

        if 'Delay_Mins' not in transit_df.columns:
            transit_df['Delay_Mins'] = transit_df['Effective_Mins'] - transit_df['Planned_Mins']

        # 2. Filter for Transport segments
        stage_col = 'Stage_Group' if 'Stage_Group' in transit_df.columns else 'Stage'
        transit_df = transit_df[
            transit_df[stage_col].str.contains('Transport', na=False)
        ].dropna(subset=['Source_Hub_ID', 'Target_Hub_ID', 'Delay_Mins'])

        # Create Route ID and Segment Number
        transit_df['Route_ID'] = transit_df['Source_Hub_ID'] + '_' + transit_df['Target_Hub_ID']
        transit_df['Segment_Number'] = transit_df[stage_col].str.extract(r'(\d+)').fillna(1).astype(int)

        # Ensure hub_analysis_df has string IDs
        hub_stats = hub_analysis_df.copy()
        hub_stats['Hub_ID'] = hub_stats['Hub_ID'].astype(str)

        # 3. Hub Data Merges (Source)
        transit_df = transit_df.merge(
            hub_stats[['Hub_ID', 'Total_Outgoing_Flow', 'Avg_Segment_Delay', 'Std_Delay']],
            left_on='Source_Hub_ID', right_on='Hub_ID', how='left'
        )
        if 'Hub_ID' in transit_df.columns:
            transit_df = transit_df.drop(columns=['Hub_ID'])

        transit_df = transit_df.rename(columns={
            'Total_Outgoing_Flow': 'Flow_source',
            'Avg_Segment_Delay': 'Avg_Delay_source',
            'Std_Delay': 'Std_Delay_source'
        })

        # 4. Hub Data Merges (Target)
        transit_df = transit_df.merge(
            hub_stats[['Hub_ID', 'Total_Outgoing_Flow', 'Avg_Segment_Delay', 'Std_Delay']],
            left_on='Target_Hub_ID', right_on='Hub_ID', how='left'
        )
        if 'Hub_ID' in transit_df.columns:
            transit_df = transit_df.drop(columns=['Hub_ID'])

        transit_df = transit_df.rename(columns={
            'Total_Outgoing_Flow': 'Flow_target',
            'Avg_Segment_Delay': 'Avg_Delay_target',
            'Std_Delay': 'Std_Delay_target'
        })

        # 5. Route Aggregations
        route_stats = transit_df.groupby('Route_ID')['Delay_Mins'].agg([
            ('Route_Avg_Delay', 'mean'),
            ('Route_Std_Delay', 'std'),
            ('Route_Volume', 'count')
        ]).reset_index()
        transit_df = transit_df.merge(route_stats, on='Route_ID', how='left')

        # 6. Engineered Numeric Features
        transit_df['Planned_Duration'] = transit_df['Planned_Mins'].fillna(0)
        transit_df['Hub_Interaction'] = transit_df['Flow_source'].fillna(0) * transit_df['Flow_target'].fillna(0)
        transit_df['Delay_Risk_Score'] = transit_df['Avg_Delay_source'].fillna(0) + transit_df['Avg_Delay_target'].fillna(0)

        # 7. Waterproof NaN Handling
        fill_values = {
            'Flow_source': 0, 'Flow_target': 0, 'Std_Delay_source': 0, 'Std_Delay_target': 0,
            'Route_Std_Delay': 0, 'Route_Volume': 1, 'Planned_Duration': 0
        }
        transit_df = transit_df.fillna(value=fill_values)

        for col in ['Avg_Delay_source', 'Avg_Delay_target', 'Route_Avg_Delay']:
            transit_df[col] = transit_df[col].fillna(transit_df[col].mean() if not transit_df[col].empty else 0)

        print(f"✅ Features engineered: {len(transit_df):,} samples.")
        return transit_df

    def train(self, training_df: pd.DataFrame, test_size=0.2):
        print("\n" + "="*70)
        print("TRAINING DELAY PREDICTION MODEL")
        print("="*70)

        X = training_df[self.feature_cols]
        y = training_df['Delay_Mins']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)

        print(f"Fitting {self.model_type}...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"📊 Results -> MAE: {mae:.2f} min | R²: {r2:.3f}")

        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        return {'mae': mae, 'r2': r2, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}

    def save_model(self, filepath='models/delay_predictor.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({'model': self.model, 'features': self.feature_cols}, filepath)
        print(f"✅ Model saved to {filepath}")

    def save_feature_importance_plot(self, output_path='results/feature_importance.html'):
        """Saves feature importance plot to HTML."""
        import plotly.express as px
        if self.feature_importance is None:
             print("⚠️ No feature importance to plot.")
             return
        
        fig = px.bar(self.feature_importance.head(15), x='Importance', y='Feature', orientation='h',
                     title='Delay Prediction Model: Feature Importance',
                     template='plotly_white')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"✅ Feature importance plot saved to {output_path}")


def mine_delay_patterns(long_flow_df, min_delay_threshold=30):
    """
    Extracts frequent sequences of hubs that lead to delays.
    """
    print("\n" + "="*70)
    print(f"DELAY PATTERN MINING (Threshold: >{min_delay_threshold} min)")
    print("="*70)

    # 1. Isolate delayed shipments
    late_shipments = long_flow_df[long_flow_df['Delay_Mins'] > min_delay_threshold]

    if late_shipments.empty:
        print("No shipments found exceeding the delay threshold.")
        return pd.Series()

    # 2. Group by Leg_ID to see the sequence of hubs
    paths = late_shipments.sort_values(['Leg_ID', 'Planned_Mins']).groupby('Leg_ID')['Hub_ID'].apply(
        lambda x: ' -> '.join([str(i) for i in x])
    )

    # 3. Count the most frequent 'Late' patterns
    pattern_counts = paths.value_counts().head(10)

    print("⚠️ Top Frequent Delay Sequences:")
    print(pattern_counts)

    return pattern_counts

def save_patterns_to_csv(patterns, output_path='results/delay_patterns.csv'):
    """Saves mined patterns to CSV."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    patterns.to_csv(output_path, header=['Frequency'])
    print(f"✅ Delay patterns saved to {output_path}")
