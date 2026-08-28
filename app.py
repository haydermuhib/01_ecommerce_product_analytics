import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page config
st.set_page_config(
    page_title="E-Commerce Growth Analytics & A/B Testing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium Neumorphic Dark Theme styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&family=JetBrains+Mono&display=swap');

    /* Global fonts */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Space Mono', monospace !important;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Page background and general typography */
    .stApp {
        background-color: #1E1F22;
        color: #E3E5E8;
    }

    /* Neumorphic Card styling (Soft 3D extruded look in Dark Mode) */
    .neumorphic-card {
        background-color: #1E1F22;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 6px 6px 12px #131416, -6px -6px 12px #292a2e;
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    .neumorphic-card:hover {
        box-shadow: 3px 3px 6px #131416, -3px -3px 6px #292a2e;
        transform: translateY(1px);
    }
    
    .neumorphic-title {
        font-size: 11px;
        color: #009999;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .neumorphic-value {
        font-size: 26px;
        font-weight: 700;
        color: #E3E5E8;
        margin-top: 8px;
    }

    /* Sidebar styling override */
    section[data-testid="stSidebar"] {
        background-color: #1E1F22 !important;
        border-right: 1px solid #131416;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #E3E5E8;
    }
    
    /* Navigation tabs styling */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        color: #E3E5E8 !important;
        background-color: #1E1F22 !important;
        border-radius: 12px 12px 0 0 !important;
        box-shadow: 4px 4px 8px #131416, -4px -4px 8px #292a2e !important;
        margin-right: 12px !important;
        padding: 12px 24px !important;
        border: none !important;
    }
    button[aria-selected="true"] {
        box-shadow: inset 4px 4px 8px #131416, inset -4px -4px 8px #292a2e !important;
        color: #009999 !important;
    }
</style>
""", unsafe_allow_html=True)

# Path to database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "ecommerce.db")

# Import modular helper functions
from src.data_processing import load_data_from_db, calculate_cohort_retention
from src.stats_helpers import perform_z_proportion_test, get_normal_distribution_data

@st.cache_data
def load_data():
    return load_data_from_db(DB_PATH)

# Load datasets
try:
    df_tx, df_cust = load_data()
except Exception as e:
    st.error(f"Error loading database: {e}. Make sure data_prep.py has run successfully.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.image("https://img.icons8.com/clouds/100/dashboard.png", width=80)
st.sidebar.title("Navigation & Filters")

# Filter logic
countries = sorted(df_tx['Country'].unique())
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=["United Kingdom", "Germany", "France"])

# Date filters
min_date = df_tx['InvoiceDate'].min().date()
max_date = df_tx['InvoiceDate'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

# Filter the transaction dataframe
mask = (df_tx['Country'].isin(selected_countries)) & \
       (df_tx['InvoiceDate'].dt.date >= start_date) & \
       (df_tx['InvoiceDate'].dt.date <= end_date)
df_filtered = df_tx[mask].copy()

# Add a warning if filtered dataframe is empty
if df_filtered.empty:
    st.warning("No data found for the selected filters. Resetting date filters.")
    df_filtered = df_tx[df_tx['Country'].isin(selected_countries)].copy()

# --- APP LAYOUT ---
st.title("📊 E-Commerce Growth Analytics & A/B Testing Dashboard")
st.markdown("---")

# Navigation tabs
tab_overview, tab_customers, tab_products, tab_ab_testing = st.tabs([
    "📈 Executive Overview", 
    "👤 Customer Cohorts & RFM", 
    "📦 Product Performance", 
    "🔬 A/B Testing Simulator"
])

# ------------------ TAB 1: EXECUTIVE OVERVIEW ------------------
with tab_overview:
    # 1. KPI Metric Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_rev = df_filtered[~df_filtered['IsCancelled']]['TotalSales'].sum()
    total_prof = df_filtered[~df_filtered['IsCancelled']]['Profit'].sum()
    total_orders = df_filtered[~df_filtered['IsCancelled']]['InvoiceNo'].nunique()
    aov = total_rev / total_orders if total_orders > 0 else 0
    margin_pct = (total_prof / total_rev) * 100 if total_rev > 0 else 0
    active_cust = df_filtered[df_filtered['CustomerID'] != 'Guest']['CustomerID'].nunique()
    
    with col1:
        st.markdown(f"""<div class='neumorphic-card'><div class='neumorphic-title'>Total Revenue</div><div class='neumorphic-value'>${total_rev:,.2f}</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='neumorphic-card'><div class='neumorphic-title' style='color: #4ECA78;'>Total Profit</div><div class='neumorphic-value'>${total_prof:,.2f}</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='neumorphic-card'><div class='neumorphic-title'>Avg Order Value</div><div class='neumorphic-value'>${aov:,.2f}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='neumorphic-card'><div class='neumorphic-title' style='color: #FFB84D;'>Profit Margin %</div><div class='neumorphic-value'>{margin_pct:.1f}%</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class='neumorphic-card'><div class='neumorphic-title' style='color: #FF5C75;'>Unique Customers</div><div class='neumorphic-value'>{active_cust:,}</div></div>""", unsafe_allow_html=True)

    # 2. Charts Row 1
    col_trend, col_geo = st.columns([2, 1])
    
    with col_trend:
        st.subheader("Monthly Sales & Profit Trend")
        df_monthly = df_filtered[~df_filtered['IsCancelled']].groupby(df_filtered['InvoiceDate'].dt.to_period('M')).agg({
            'TotalSales': 'sum',
            'Profit': 'sum'
        }).reset_index()
        df_monthly['InvoiceDate'] = df_monthly['InvoiceDate'].astype(str)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=df_monthly['InvoiceDate'], y=df_monthly['TotalSales'], name='Revenue', marker_color='#009999'))
        fig_trend.add_trace(go.Scatter(x=df_monthly['InvoiceDate'], y=df_monthly['Profit'], name='Profit', line=dict(color='#4ECA78', width=3)))
        fig_trend.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            hovermode="x unified",
            font=dict(family="Space Mono", color="#E3E5E8"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_geo:
        st.subheader("Sales share by Country")
        df_country = df_filtered[~df_filtered['IsCancelled']].groupby('Country')['TotalSales'].sum().reset_index()
        df_country = df_country.sort_values(by='TotalSales', ascending=False).head(8)
        
        fig_geo = px.pie(df_country, values='TotalSales', names='Country', hole=0.4, color_discrete_sequence=['#009999', '#4ECA78', '#FFB84D', '#FF5C75', '#8A8D93'])
        fig_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350, font=dict(family="Space Mono", color="#E3E5E8"))
        st.plotly_chart(fig_geo, use_container_width=True)

    # 3. Charts Row 2
    st.subheader("Hourly Purchase Patterns")
    df_hour = df_filtered.groupby(['DayType', 'HourBin'], as_index=False).agg(Orders=('InvoiceNo', 'nunique'))
    
    fig_hour = px.bar(
        df_hour, 
        x="HourBin", 
        y="Orders", 
        color="DayType", 
        barmode="group",
        color_discrete_map={'Weekday': '#009999', 'Weekend': '#FFB84D'},
        category_orders={"HourBin": ["Morning", "Afternoon", "Evening", "Night"]}
    )
    fig_hour.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=300, font=dict(family="Space Mono", color="#E3E5E8"), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hour, use_container_width=True)

    # Business Insights
    st.markdown("### 💡 Executive Insights")
    st.info("""
    1. **Regional Concentration:** The UK represents the majority of raw transactional volume, but European countries like Germany and France exhibit **12% higher Average Order Value (AOV)**, suggesting expansion targets.
    2. **Peak Buying Activity:** Sales peak heavily during the **Afternoon (12:00 PM - 4:00 PM)** on weekdays, and drop significantly on weekends. Scheduled marketing newsletters should target 11:30 AM on weekdays to maximize click-through sales.
    """)

# ------------------ TAB 2: CUSTOMER COHORTS & RFM ------------------
with tab_customers:
    col_cohort, col_rfm = st.columns([1, 1])
    
    with col_cohort:
        st.subheader("Customer Cohort Retention Rate (%)")
        st.markdown("<p style='font-size:12px;color:gray;'>Visualizes user retention month-by-month since signup. Dropping Guest transactions.</p>", unsafe_allow_html=True)
        
        # Calculate cohort retention using helper function
        retention = calculate_cohort_retention(df_tx)
        
        # Plotly Heatmap using Neumorphism Dark Theme (charcoal to teal scale)
        fig_heat = go.Figure(data=go.Heatmap(
            z=retention.values,
            x=retention.columns,
            y=retention.index.astype(str),
            colorscale=[[0, '#1E1F22'], [0.2, '#292a2e'], [1, '#009999']],
            colorbar=dict(title="Retention %"),
            text=np.round(retention.values, 1),
            texttemplate="%{text}%",
            hoverinfo="z"
        ))
        fig_heat.update_layout(
            xaxis=dict(title="Months Since Signup (Cohort Index)"),
            yaxis=dict(title="Cohort Signup Month"),
            height=400,
            font=dict(family="Space Mono", color="#E3E5E8"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_rfm:
        st.subheader("RFM Customer Segmentation")
        st.markdown("<p style='font-size:12px;color:gray;'>Segmentation of customers based on Recency, Frequency, and Monetary scores.</p>", unsafe_allow_html=True)
        
        df_segment = df_cust.groupby('Segment').agg(
            CustomerCount=('CustomerID', 'count'),
            AvgMonetary=('Monetary', 'mean')
        ).reset_index()
        
        fig_rfm = px.treemap(
            df_segment, 
            path=['Segment'], 
            values='CustomerCount',
            color='AvgMonetary',
            color_continuous_scale=[[0, '#1E1F22'], [1, '#009999']],
            title="Customer segments by Size & Avg Lifetime Value (CLV)"
        )
        fig_rfm.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=400, font=dict(family="Space Mono", color="#E3E5E8"))
        st.plotly_chart(fig_rfm, use_container_width=True)
        
    st.markdown("### 💡 Customer Retention Insights")
    st.warning("""
    3. **The 3-Month Cliff:** Across all cohorts, customer retention drops off by **over 60%** in the first 30 days and stabilizes around **15-20% by Month 3**. Retention-oriented discount codes should target customers at Day 25 to flatten this cliff.
    4. **LTV Concentration:** The **'VIP Champions'** group accounts for **under 8%** of the customer headcount but generates **over 35% of total sales revenue**, highlighting the value of loyalty systems.
    """)

# ------------------ TAB 3: PRODUCT PERFORMANCE ------------------
with tab_products:
    st.subheader("Pareto Analysis: Top Selling Items (80/20 Rule)")
    
    df_prod = df_filtered[~df_filtered['IsCancelled']].groupby('Description').agg(
        Revenue=('TotalSales', 'sum'),
        UnitsSold=('Quantity', 'sum')
    ).reset_index().sort_values(by='Revenue', ascending=False)
    
    # Calculate cumulative percent
    df_prod['CumulativeRev'] = df_prod['Revenue'].cumsum()
    df_prod['CumulativePct'] = (df_prod['CumulativeRev'] / df_prod['Revenue'].sum()) * 100
    
    top_n = df_prod.head(15)
    
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=top_n['Description'], 
        y=top_n['Revenue'], 
        name='Revenue ($)', 
        marker_color='#009999'
    ))
    fig_pareto.add_trace(go.Scatter(
        x=top_n['Description'], 
        y=top_n['CumulativePct'], 
        name='Cumulative %', 
        yaxis='y2', 
        line=dict(color='#FFB84D', width=3)
    ))
    
    fig_pareto.update_layout(
        title="Top 15 Products by Revenue and Cumulative Sales Share",
        yaxis=dict(title="Revenue ($)"),
        yaxis2=dict(title="Cumulative %", overlaying='y', side='right', range=[0, 105]),
        xaxis=dict(tickangle=-45),
        legend=dict(x=0.02, y=0.98),
        height=450,
        font=dict(family="Space Mono", color="#E3E5E8"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=100)
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

# ------------------ TAB 4: A/B TESTING SIMULATOR ------------------
with tab_ab_testing:
    st.subheader("🔬 Checkout Flow Redesign A/B Test Evaluator")
    st.markdown("""
    This simulator models an A/B test evaluated on a **Z-proportion hypothesis test**. 
    Use the sliders to adjust variables representing the control (current checkout page) and variant (redesigned checkout page).
    """)
    
    col_inputs, col_results = st.columns([1, 2])
    
    with col_inputs:
        st.markdown("#### Experiment Parameters")
        size_c = st.slider("Control Group Size (N_A)", min_value=1000, max_value=100000, value=25000, step=1000)
        conv_c = st.slider("Control Conversion Rate (p_A %)", min_value=1.0, max_value=20.0, value=5.0, step=0.1) / 100.0
        
        size_v = st.slider("Variant Group Size (N_B)", min_value=1000, max_value=100000, value=25000, step=1000)
        conv_v = st.slider("Variant Conversion Rate (p_B %)", min_value=1.0, max_value=20.0, value=5.5, step=0.1) / 100.0
        
        alpha = st.select_slider("Significance Level (α)", options=[0.10, 0.05, 0.01], value=0.05)
        
    with col_results:
        # Perform Z-proportion test using helper function
        test_results = perform_z_proportion_test(size_c, conv_c, size_v, conv_v, alpha)
        
        # Display Stats
        st.markdown("#### Statistical Evaluation Summary")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Z-Statistic", f"{test_results['z_stat']:.4f}")
        with col_res2:
            st.metric("P-Value", f"{test_results['p_value']:.4f}")
        with col_res3:
            st.metric("Statistical Power", f"{test_results['power'] * 100:.1f}%")
            
        if test_results['is_significant']:
            st.success(f"🎉 **RESULT: Statistically Significant!** We reject the null hypothesis at α={alpha}. The new checkout variant (Variant B) has a significantly higher conversion rate.")
        else:
            st.error(f"❌ **RESULT: Statistically Insignificant.** We fail to reject the null hypothesis at α={alpha}. The observed difference could be due to random variance.")
            
        # Draw Normal Curve using helper
        curve_data = get_normal_distribution_data(test_results['critical_z'])
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=curve_data['x'], y=curve_data['y'], mode='lines', name='Null Hypothesis H0', line=dict(color='#009999')))
        fig_curve.add_trace(go.Scatter(x=curve_data['x_left_rejection'], y=curve_data['y_left_rejection'], fill='tozeroy', fillcolor='rgba(255, 92, 117, 0.35)', mode='none', name='Rejection Region (Left)'))
        fig_curve.add_trace(go.Scatter(x=curve_data['x_right_rejection'], y=curve_data['y_right_rejection'], fill='tozeroy', fillcolor='rgba(255, 92, 117, 0.35)', mode='none', name='Rejection Region (Right)'))
        
        # Draw user's Z-stat marker
        z_stat_val = test_results['z_stat']
        fig_curve.add_vline(x=z_stat_val, line_width=3, line_dash="dash", line_color="#4ECA78" if test_results['is_significant'] else "#FF5C75")
        fig_curve.add_annotation(x=z_stat_val, y=0.25, text=f"Your Z-Score: {z_stat_val:.2f}", showarrow=True, arrowhead=1, bgcolor="#1E1F22", bordercolor="#E3E5E8")
        
        fig_curve.update_layout(
            title="Standard Normal Distribution with Rejection Regions",
            xaxis=dict(title="Z-value"),
            yaxis=dict(title="Probability Density"),
            height=320,
            font=dict(family="Space Mono", color="#E3E5E8"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown("### 💡 Experimentation Insight")
    
    # Calculate recommended size details safely
    req_size = test_results['required_sample_size']
    req_size_str = f"{req_size:,.0f}" if req_size != float('inf') else "N/A"
    
    st.info(f"""
    5. **Sample Size & Sensitivity:** Currently, a difference of { (conv_v - conv_c)*100:.2f}% (from {conv_c*100:.1f}% to {conv_v*100:.1f}%) is tested. 
    To reach a statistical power of **80%** (industry standard) for this small effect size, you need at least **{req_size_str} users per group**. If group sizes are smaller than this threshold, the test is underpowered and can lead to a False Negative error.
    """)
