import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import streamlit.components.v1 as components
from sklearn.decomposition import PCA

# --- 1. CONFIGURATION & DARK THEME ---
st.set_page_config(page_title="Cargo Ops Intelligence", layout="wide", page_icon="📊")

# Custom CSS for a Professional Dark Analyst Theme
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; color: #fafafa; }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] { color: #00d4ff; font-size: 1.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #808495; }
    
    /* Metric Card styling */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #161b22;
        border-radius: 4px 4px 0 0;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] { background-color: #21262d; color: #58a6ff; font-weight: bold; border-bottom: 2px solid #58a6ff; }

    /* Dataframe styling for dark mode */
    .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATA LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def clean_hub_id(val):
    """Standardizes Hub IDs to remove .0 artifacts and handle placeholders."""
    s = str(val).strip()
    if s.endswith('.0'): 
        s = s[:-2]
    if s in ['nan', 'None', '?', '0', '0.0', 'none']: 
        return 'none'
    return s

@st.cache_data
def load_csv(file_name):
    path = os.path.join(RESULTS_DIR, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Analyst Fix: Standardize Hub IDs globally on load
        if 'Hub_ID' in df.columns:
            df['Hub_ID'] = df['Hub_ID'].apply(clean_hub_id)
            
        if 'Cluster_Label' in df.columns:
            df = df.dropna(subset=['Cluster_Label'])
            df['Cluster_Label'] = df['Cluster_Label'].astype(int).astype(str)
        return df
    return None

@st.cache_data
def load_embeddings():
    path = os.path.join(RESULTS_DIR, "hub_embeddings.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

def load_html(file_name):
    path = os.path.join(RESULTS_DIR, file_name)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# --- 3. SIDEBAR ANALYTICS CONTROLS ---
st.sidebar.title("🔍 Analyst Controls")

hub_data = load_csv("hub_clusters.csv")
embeddings = load_embeddings()
selected_clusters = []

if hub_data is not None:
    all_clusters = sorted(hub_data['Cluster_Label'].unique())
    selected_clusters = st.sidebar.multiselect("Filter Cluster Type:", all_clusters, default=all_clusters)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 ROI Simulator")
improvement_pct = st.sidebar.slider("Improvement in Delay (%)", 0, 100, 20)
cost_per_min = st.sidebar.number_input("Cost per Minute (£)", value=1.0, step=0.1)

# --- 4. MAIN DASHBOARD LOGIC ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Summary", "🕸️ Network Topology", "🧬 Embedding Space", "🔬 Hub Drill-Down"])

# --- TAB 1: EXECUTIVE SUMMARY ---
with tab1:
    if hub_data is None:
        st.error("⚠️ **Data Missing.** Please run the analysis pipeline first.")
    else:
        filtered_df = hub_data[hub_data['Cluster_Label'].isin(selected_clusters)]
        
        # Headline Metrics
        st.title("Network Operational Health")
        m1, m2, m3, m4 = st.columns(4)
        
        total_flow = filtered_df['Total_Outgoing_Flow'].sum()
        avg_delay = filtered_df['Avg_Segment_Delay'].mean()
        
        # ROI Logic
        BASELINE_TOTAL_COST = 7022346.0
        network_total_flow = hub_data['Total_Outgoing_Flow'].sum()
        selection_flow_share = total_flow / network_total_flow if network_total_flow > 0 else 0
        simulated_savings = (BASELINE_TOTAL_COST * selection_flow_share) * (improvement_pct / 100) * cost_per_min
        
        m1.metric("Active Hubs", len(filtered_df))
        m2.metric("Flow Volume", f"{total_flow:,}")
        m3.metric("Avg Delay (mins)", f"{avg_delay:.1f}")
        m4.metric("Savings Potential", f"£{simulated_savings:,.0f}", delta=f"ROI @ {improvement_pct}%")

        st.markdown("---")

        # Visual Row
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Performance Quadrant: Volume vs Delay")
            # Absolute Max_Delay for sizing
            plot_df = filtered_df.copy()
            plot_df['Size_Mag'] = plot_df['Max_Delay'].abs() + 1
            
            fig = px.scatter(plot_df, x='Total_Outgoing_Flow', y='Avg_Segment_Delay',
                             color='Cluster_Label', size='Size_Mag',
                             hover_name='Hub_ID', template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Critical Hubs Ranking")
            crit_data = load_csv("critical_routes.csv")
            if crit_data is not None:
                st.dataframe(crit_data[['Hub_ID', 'Flow_Volume', 'Avg_Hub_Delay']].sort_values('Avg_Hub_Delay', ascending=False), 
                             height=400, use_container_width=True)

# --- TAB 2: NETWORK TOPOLOGY ---
with tab2:
    st.title("Logistics Network Topology")
    
    # Sankey Diagram Section (New)
    st.subheader("📦 Network Flow Connectivity (Sankey)")
    st.caption("Visualizing the top 20 hub-to-hub flow connections.")
    
    # We reconstruct the flow from the long_df if available, or use a cached version
    # For now, let's derive it from the cluster data or critical routes if they contain source/target
    # Analyst note: Ideally we'd need a route_volume.csv. Let's check if we can generate it.
    
    # Robust fallback: Use critical routes if they are routes, or use sample connectivity
    # Let's try to load 'long_flow_df.csv' from results/ to get real connectivity
    raw_long_df = load_csv("long_flow_df.csv")
    
    if raw_long_df is not None:
        # Reconstruct edges
        raw_long_df = raw_long_df.sort_values(['Leg_ID', 'Planned_Mins'])
        raw_long_df['Source'] = raw_long_df['Hub_ID'].astype(str)
        raw_long_df['Target'] = raw_long_df.groupby('Leg_ID')['Hub_ID'].shift(-1).astype(str)
        
        flow_counts = raw_long_df.dropna(subset=['Target']).groupby(['Source', 'Target']).size().reset_index(name='Volume')
        top_flows = flow_counts[flow_counts['Source'] != flow_counts['Target']].nlargest(20, 'Volume')
        
        import plotly.graph_objects as go
        
        all_nodes = list(set(top_flows['Source']) | set(top_flows['Target']))
        node_map = {name: i for i, name in enumerate(all_nodes)}
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node = dict(pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
                        label = all_nodes, color = "blue"),
            link = dict(source = top_flows['Source'].map(node_map),
                        target = top_flows['Target'].map(node_map),
                        value = top_flows['Volume'])
        )])
        fig_sankey.update_layout(title_text="Top Hub-to-Hub Flow Sequences", font_size=12, template="plotly_dark",
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Run the full pipeline to generate 'data/long_flow_df.csv' for Sankey visualization.")

    st.markdown("---")
    
    # Interactive Graph
    network_html = load_html("network_graph.html")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Delay Patterns")
        patterns = load_csv("delay_patterns.csv")
        if patterns is not None:
            st.dataframe(patterns, use_container_width=True)
    with col_b:
        st.subheader("ML Delay Drivers")
        feat_html = load_html("feature_importance.html")
        if feat_html:
            components.html(feat_html, height=450)

# --- TAB 3: EMBEDDING SPACE ---
with tab3:
    st.title("Graph Neural Network Analysis")
    st.markdown("Direct visualization of the GraphSAGE embedding space using PCA.")
    
    if embeddings is not None and hub_data is not None:
        pca = PCA(n_components=2)
        components_2d = pca.fit_transform(embeddings)
        
        pca_df = pd.DataFrame(components_2d, columns=['PC1', 'PC2'])
        pca_df['Hub_ID'] = hub_data['Hub_ID'].values
        pca_df['Cluster'] = hub_data['Cluster_Label'].values
        
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', 
                            hover_name='Hub_ID', template="plotly_dark",
                            title="Embedding Latent Space (PCA Projection)")
        fig_pca.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        st.warning("Embedding data (.npy) or Cluster data (.csv) not found.")

# --- TAB 4: HUB DRILL-DOWN ---
with tab4:
    st.title("Operational Deep-Dive")
    if hub_data is not None:
        search_id = st.selectbox("Search for a Hub ID:", sorted(hub_data['Hub_ID'].unique()))
        
        hub_row = hub_data[hub_data['Hub_ID'] == search_id].iloc[0]
        
        d1, d2 = st.columns([1, 2])
        with d1:
            st.markdown("#### Operational Metadata")
            st.json(hub_row.to_dict())
            
        with d2:
            st.markdown(f"#### XAI Influence: Hub {search_id}")
            report_html = load_html(f"influence_report_hub_{search_id}.html")
            if report_html:
                components.html(report_html, height=800, scrolling=True)
            else:
                st.info("No explainability report available for this hub.")

st.sidebar.markdown("---")
st.sidebar.caption("Cargo Ops Intelligence Portal v1.2 | Dark Mode Active")
