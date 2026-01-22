import pandas as pd
import numpy as np
import os

# --- 1. Column Mapping Configuration ---
COLUMN_RENAME_MAP = {
    'legs': 'Total_Incoming_Legs',
    'nr': 'All_process_ID',
    # Incoming Leg 1 (i1)
    'i1_legid': 'IN_1_Leg_ID', 'i1_rcs_p': 'IN_1_CheckIn_Planned_Mins', 'i1_rcs_e': 'IN_1_CheckIn_Effective_Mins',
    'i1_dep_1_p': 'IN_1_Dep_Seg1_Planned_Mins', 'i1_dep_1_e': 'IN_1_Dep_Seg1_Effective_Mins', 'i1_dep_1_place': 'IN_1_Dep_Seg1_Hub_ID',
    'i1_rcf_1_p': 'IN_1_Arr_Seg1_Planned_Mins', 'i1_rcf_1_e': 'IN_1_Arr_Seg1_Effective_Mins', 'i1_rcf_1_place': 'IN_1_Arr_Seg1_Hub_ID',
    'i1_dep_2_p': 'IN_1_Dep_Seg2_Planned_Mins', 'i1_dep_2_e': 'IN_1_Dep_Seg2_Effective_Mins', 'i1_dep_2_place': 'IN_1_Dep_Seg2_Hub_ID',
    'i1_rcf_2_p': 'IN_1_Arr_Seg2_Planned_Mins', 'i1_rcf_2_e': 'IN_1_Arr_Seg2_Effective_Mins', 'i1_rcf_2_place': 'IN_1_Arr_Seg2_Hub_ID',
    'i1_dep_3_p': 'IN_1_Dep_Seg3_Planned_Mins', 'i1_dep_3_e': 'IN_1_Dep_Seg3_Effective_Mins', 'i1_dep_3_place': 'IN_1_Dep_Seg3_Hub_ID',
    'i1_rcf_3_p': 'IN_1_Arr_Seg3_Planned_Mins', 'i1_rcf_3_e': 'IN_1_Arr_Seg3_Effective_Mins', 'i1_rcf_3_place': 'IN_1_Arr_Seg3_Hub_ID',
    'i1_dlv_p': 'IN_1_Delivery_Planned_Mins', 'i1_dlv_e': 'IN_1_Delivery_Effective_Mins', 'i1_hops': 'IN_1_Hops',
    # Incoming Leg 2 (i2)
    'i2_legid': 'IN_2_Leg_ID', 'i2_rcs_p': 'IN_2_CheckIn_Planned_Mins', 'i2_rcs_e': 'IN_2_CheckIn_Effective_Mins',
    'i2_dep_1_p': 'IN_2_Dep_Seg1_Planned_Mins', 'i2_dep_1_e': 'IN_2_Dep_Seg1_Effective_Mins', 'i2_dep_1_place': 'IN_1_Dep_Seg1_Hub_ID', # Note: mapping i2 to i1 hub? Keeping original logic
    'i2_rcf_1_p': 'IN_2_Arr_Seg1_Planned_Mins', 'i2_rcf_1_e': 'IN_2_Arr_Seg1_Effective_Mins', 'i2_rcf_1_place': 'IN_2_Arr_Seg1_Hub_ID',
    'i2_dep_2_p': 'IN_2_Dep_Seg2_Planned_Mins', 'i2_dep_2_e': 'IN_2_Dep_Seg2_Effective_Mins', 'i2_dep_2_place': 'IN_2_Dep_Seg2_Hub_ID',
    'i2_rcf_2_p': 'IN_2_Arr_Seg2_Planned_Mins', 'i2_rcf_2_e': 'IN_2_Arr_Seg2_Effective_Mins', 'i2_rcf_2_place': 'IN_2_Arr_Seg2_Hub_ID',
    'i2_dep_3_p': 'IN_2_Dep_Seg3_Planned_Mins', 'i2_dep_3_e': 'IN_2_Dep_Seg3_Effective_Mins', 'i2_dep_3_place': 'IN_2_Dep_Seg3_Hub_ID',
    'i2_rcf_3_p': 'IN_2_Arr_Seg3_Planned_Mins', 'i2_rcf_3_e': 'IN_2_Arr_Seg3_Effective_Mins', 'i2_rcf_3_place': 'IN_2_Arr_Seg3_Hub_ID',
    'i2_dlv_p': 'IN_2_Delivery_Planned_Mins', 'i2_dlv_e': 'IN_2_Delivery_Effective_Mins', 'i2_hops': 'IN_2_Hops',
    # Incoming Leg 3 (i3)
    'i3_legid': 'IN_3_Leg_ID', 'i3_rcs_p': 'IN_3_CheckIn_Planned_Mins', 'i3_rcs_e': 'IN_3_CheckIn_Effective_Mins',
    'i3_dep_1_p': 'IN_3_Dep_Seg1_Planned_Mins', 'i3_dep_1_e': 'IN_3_Dep_Seg1_Effective_Mins', 'i3_dep_1_place': 'IN_3_Dep_Seg1_Hub_ID',
    'i3_rcf_1_p': 'IN_3_Arr_Seg1_Planned_Mins', 'i3_rcf_1_e': 'IN_3_Arr_Seg1_Effective_Mins', 'i3_rcf_1_place': 'IN_3_Arr_Seg1_Hub_ID',
    'i3_dep_2_p': 'IN_3_Dep_Seg2_Planned_Mins', 'i3_dep_2_e': 'IN_3_Dep_Seg2_Effective_Mins', 'i3_dep_2_place': 'IN_3_Dep_Seg2_Hub_ID',
    'i3_rcf_2_p': 'IN_3_Arr_Seg2_Planned_Mins', 'i3_rcf_2_e': 'IN_3_Arr_Seg2_Effective_Mins', 'i3_rcf_2_place': 'IN_3_Arr_Seg2_Hub_ID',
    'i3_dep_3_p': 'IN_3_Dep_Seg3_Planned_Mins', 'i3_dep_3_e': 'IN_3_Dep_Seg3_Effective_Mins', 'i3_dep_3_place': 'IN_3_Dep_Seg3_Hub_ID',
    'i3_rcf_3_p': 'IN_3_Arr_Seg3_Planned_Mins', 'i3_rcf_3_e': 'IN_3_Arr_Seg3_Effective_Mins', 'i3_rcf_3_place': 'IN_3_Arr_Seg3_Hub_ID',
    'i3_dlv_p': 'IN_3_Delivery_Planned_Mins', 'i3_dlv_e': 'IN_3_Delivery_Effective_Mins', 'i3_hops': 'IN_3_Hops',
    # Outgoing Leg (o)
    'o_legid': 'OUT_Leg_ID', 'o_rcs_p': 'OUT_CheckIn_Planned_Mins', 'o_rcs_e': 'OUT_CheckIn_Effective_Mins',
    'o_dep_1_p': 'OUT_Dep_Seg1_Planned_Mins', 'o_dep_1_e': 'OUT_Dep_Seg1_Effective_Mins', 'o_dep_1_place': 'OUT_Dep_Seg1_Hub_ID',
    'o_rcf_1_p': 'OUT_Arr_Seg1_Planned_Mins', 'o_rcf_1_e': 'OUT_Arr_Seg1_Effective_Mins', 'o_rcf_1_place': 'OUT_Arr_Seg1_Hub_ID',
    'o_dep_2_p': 'OUT_Dep_Seg2_Planned_Mins', 'o_dep_2_e': 'OUT_Dep_Seg2_Effective_Mins', 'o_dep_2_place': 'OUT_Dep_Seg2_Hub_ID',
    'o_rcf_2_p': 'OUT_Arr_Seg2_Planned_Mins', 'o_rcf_2_e': 'OUT_Arr_Seg2_Effective_Mins', 'o_rcf_2_place': 'OUT_Arr_Seg2_Hub_ID',
    'o_dep_3_p': 'OUT_Dep_Seg3_Planned_Mins', 'o_dep_3_e': 'OUT_Dep_Seg3_Effective_Mins', 'o_dep_3_place': 'OUT_Dep_Seg3_Hub_ID',
    'o_rcf_3_p': 'OUT_Arr_Seg3_Planned_Mins', 'o_rcf_3_e': 'OUT_Arr_Seg3_Effective_Mins', 'o_rcf_3_place': 'OUT_Arr_Seg3_Hub_ID',
    'o_dlv_p': 'OUT_Delivery_Planned_Mins', 'o_dlv_e': 'OUT_Delivery_Effective_Mins', 'o_hops': 'OUT_Hops',
}

def load_raw_data(file_path):
    """Loads the CSV file using raw string path handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")
    
    print(f"📂 Loading data from: {file_path}")
    # Using low_memory=False to prevent DtypeWarnings during initial load
    return pd.read_csv(file_path, low_memory=False)

def clean_cargo_data(df):
    """
    Renames columns, protects IDs, and converts metrics to numeric.
    """
    df_clean = df.copy()

    # 1. Apply Column Renaming
    df_clean = df_clean.rename(columns=COLUMN_RENAME_MAP)
    
    # 1.5 Remove Duplicate Columns (Critical Fix)
    # If the input or renaming results in duplicates, we keep the first occurrence.
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    # 2. Identify and Protect ID/Hub columns
    id_cols = [col for col in df_clean.columns if '_ID' in col or 'All_process_ID' in col]

    for col in df_clean.columns:
        if col in id_cols:
            # Ensure IDs are strings; replace '?' with '0'
            df_clean[col] = df_clean[col].astype(str).replace('\?', '0', regex=True).replace('nan', '0')
            continue

        if df_clean[col].dtype == 'object':
            # Coerce '?' and other non-numerics to NaN
            temp_numeric = pd.to_numeric(df_clean[col], errors='coerce')

            # Only convert if the column actually contains numeric data
            if not temp_numeric.isna().all():
                df_clean[col] = temp_numeric
                print(f"✅ Converted '{col}' to Numeric.")

    # 3. Handle Missing Values
    # Filling with 0 for duration-based metrics to allow statistical analysis
    df_clean = df_clean.fillna(0)

    # Final pass to ensure numeric stability
    df_clean = df_clean.apply(pd.to_numeric, errors='ignore')

    print(f"\n--- Cleaning Summary ---")
    print(f"Total Columns: {len(df_clean.columns)}")
    print(f"Numeric Columns: {df_clean.select_dtypes(include=[np.number]).shape[1]}")
    print(f"Categorical/ID Columns: {df_clean.select_dtypes(exclude=[np.number]).shape[1]}")

    return df_clean

def prepare_long_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms wide data into long format with corrected Hub_ID handling.
    Original notebook cell: 32
    """
    milestones = []
    leg_prefixes = ['IN_1', 'IN_2', 'IN_3', 'OUT']

    for prefix in leg_prefixes:
        leg_id_col = f'{prefix}_Leg_ID'
        if leg_id_col not in df.columns:
            continue

        # Logic for identifying anchor hubs
        first_hub = df[f'{prefix}_Dep_Seg1_Hub_ID'] if f'{prefix}_Dep_Seg1_Hub_ID' in df.columns else pd.Series(['none']*len(df))

        # Correctly find the final destination hub for the DLV milestone
        last_hub = df[f'{prefix}_Arr_Seg3_Hub_ID'].replace(['0', 0, '?'], np.nan)
        for i in [2, 1]:
            col = f'{prefix}_Arr_Seg{i}_Hub_ID'
            if col in df.columns:
                last_hub = last_hub.fillna(df[col].replace(['0', 0, '?'], np.nan))

        last_hub = last_hub.fillna('none')

        # --- 1. Milestone: RCS (Check-in) ---
        rcs = df[[leg_id_col, f'{prefix}_CheckIn_Planned_Mins', f'{prefix}_CheckIn_Effective_Mins']].copy()
        rcs.columns = ['Leg_ID', 'Planned_Mins', 'Effective_Mins']
        rcs['Hub_ID'] = first_hub
        rcs['Milestone'] = 'RCS'
        rcs['Stage_Group'] = 'CheckIn'
        rcs['Leg_Type'] = prefix
        milestones.append(rcs)

        # --- 2. Milestones: DEP & RCF (Transport Segments) ---
        for i in range(1, 4):
            for m_type in ['Dep', 'Arr']:
                loc_col = f'{prefix}_{m_type}_Seg{i}_Hub_ID'
                p_col = f'{prefix}_{m_type}_Seg{i}_Planned_Mins'
                e_col = f'{prefix}_{m_type}_Seg{i}_Effective_Mins'

                if loc_col in df.columns and p_col in df.columns:
                    m_data = df[[leg_id_col, p_col, e_col, loc_col]].copy()
                    m_data.columns = ['Leg_ID', 'Planned_Mins', 'Effective_Mins', 'Hub_ID']
                    m_data['Milestone'] = 'DEP' if m_type == 'Dep' else 'RCF'
                    m_data['Stage_Group'] = f'Transport_Seg{i}'
                    m_data['Leg_Type'] = prefix
                    milestones.append(m_data)

        # --- 3. Milestone: DLV (Delivery) ---
        dlv = df[[leg_id_col, f'{prefix}_Delivery_Planned_Mins', f'{prefix}_Delivery_Effective_Mins']].copy()
        dlv.columns = ['Leg_ID', 'Planned_Mins', 'Effective_Mins']
        dlv['Hub_ID'] = last_hub
        dlv['Milestone'] = 'DLV'
        dlv['Stage_Group'] = 'Delivery'
        dlv['Leg_Type'] = prefix
        milestones.append(dlv)

    long_df = pd.concat(milestones, ignore_index=True)

    # --- CLEANING ---
    # 1. Handle Hub_ID: Keep as strings!
    long_df['Hub_ID'] = long_df['Hub_ID'].astype(str).replace(['0', '0.0', 'nan', '?'], 'none')

    # 2. Convert time columns to numeric
    for col in ['Planned_Mins', 'Effective_Mins']:
        long_df[col] = pd.to_numeric(long_df[col], errors='coerce').fillna(0)

    # 3. Calculate Delay
    # IMPORTANT: Only calculate delay where both times exist (non-zero)
    long_df['Delay_Mins'] = 0
    mask = (long_df['Effective_Mins'] != 0) & (long_df['Planned_Mins'] != 0)
    long_df.loc[mask, 'Delay_Mins'] = long_df['Effective_Mins'] - long_df['Planned_Mins']

    # 4. Drop rows that are effectively empty
    long_df = long_df[~((long_df['Planned_Mins'] == 0) & (long_df['Effective_Mins'] == 0))]

    return long_df.reset_index(drop=True)

def calculate_kpis_and_aggregate(long_flow_df: pd.DataFrame, min_volume=1) -> pd.DataFrame:
    """
    Calculates Delay and aggregates flow and delay metrics by Hub.
    Original notebook cell: 39, 59
    """
    df_work = long_flow_df.copy()

    # 1. Ensure numeric types for calculation
    for col in ['Effective_Mins', 'Planned_Mins']:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

    # 2. Calculate Delay if not already present
    if 'Delay_Mins' not in df_work.columns:
        df_work['Delay_Mins'] = df_work['Effective_Mins'] - df_work['Planned_Mins']

    # 3. Clean records: Exclude 'none' and non-numeric delays
    df_work = df_work.dropna(subset=['Delay_Mins', 'Hub_ID'])
    df_work = df_work[df_work['Hub_ID'] != 'none']

    # Filter out empty/placeholder records
    df_work = df_work[~((df_work['Planned_Mins'] == 0) & (df_work['Effective_Mins'] == 0))]

    # 4. Aggregate Metrics by Hub_ID
    # We use 'Total_Outgoing_Flow' to match the Scatter Plot and Quadrant code
    # Note: 'Total_Outgoing_Flow' roughly equates to activity count here
    hub_metrics_df = df_work.groupby('Hub_ID').agg(
        Total_Outgoing_Flow=('Leg_ID', 'count'),
        Avg_Segment_Delay=('Delay_Mins', 'mean'),
        Std_Delay=('Delay_Mins', 'std'),
        Median_Delay=('Delay_Mins', 'median'),
        Max_Delay=('Delay_Mins', 'max')
    ).reset_index()

    # 5. Handle single-entry hubs (NaN Std Dev)
    median_std = hub_metrics_df['Std_Delay'].median()
    hub_metrics_df['Std_Delay'] = hub_metrics_df['Std_Delay'].fillna(median_std if pd.notna(median_std) else 0.0)

    # 6. Apply Volume Filter
    hub_metrics_df = hub_metrics_df[hub_metrics_df['Total_Outgoing_Flow'] >= min_volume]

    # 7. Rank Hubs (1 = Worst/Most Delayed)
    hub_metrics_df['Delay_Rank'] = hub_metrics_df['Avg_Segment_Delay'].rank(
        method='dense',
        ascending=False
    ).astype(int)

    # 8. Sort and Finalise
    hub_metrics_df = hub_metrics_df.sort_values('Total_Outgoing_Flow', ascending=False)

    return hub_metrics_df.reset_index(drop=True)

class LogisticsAnalyticEngine:
    """
    A unified suite for Cargo 2000 logistics analysis.
    Combines cleaning, transformation, and KPI ranking.
    """
    def __init__(self, raw_df):
        self.raw_df = raw_df
        self.long_df = None
        self.kpi_df = None

    def run_pipeline(self):
        print("🚀 Starting Logistics Analysis Pipeline...")
        # 1. Clean
        df_clean = clean_cargo_data(self.raw_df)
        # 2. Transform to Long Format
        self.long_df = prepare_long_format_data(df_clean)
        # 3. Aggregate KPIs
        self.kpi_df = calculate_kpis_and_aggregate(self.long_df)
        print("✅ Pipeline Complete. Hub Analysis and Long Data ready.")
        return self.long_df, self.kpi_df

# --- Main Execution Block ---
if __name__ == "__main__":
    # Using raw string (r"") for Windows path safety
    raw_path = r"E:\Coding\Cargo\main\data\c2k_data_comma.csv"
    
    try:
        raw_df = load_raw_data(raw_path)
        
        # Instantiate Engine
        engine = LogisticsAnalyticEngine(raw_df)
        long_df, hub_kpis = engine.run_pipeline()
        
        # Save processed data for next steps in the pipeline
        output_dir = r"E:\Coding\Cargo\main\data"
        os.makedirs(output_dir, exist_ok=True)
        
        long_df.to_csv(os.path.join(output_dir, "long_flow_df.csv"), index=False)
        hub_kpis.to_csv(os.path.join(output_dir, "hub_kpis.csv"), index=False)
        
        print(f"\n💾 Cleaned data saved to: {output_dir}")
        print(f"Generated 'long_flow_df.csv' and 'hub_kpis.csv'")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")