import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PFC | Project Finance Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #f6f8fb;
        }

        .main .block-container {
            max-width: 1500px;
            padding: 2rem 2.5rem 3rem 2.5rem;
        }

        h1, h2, h3, h4 {
            color: #172033;
            letter-spacing: -0.02em;
        }

        h1 {
            font-size: 2.35rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.15rem !important;
        }

        h2 {
            font-size: 1.45rem !important;
            font-weight: 650 !important;
            margin-top: 0.4rem !important;
        }

        h3 {
            font-size: 1.05rem !important;
            font-weight: 650 !important;
        }

        /* ---------- Sidebar ---------- */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e4e8ef;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 1.5rem 1.15rem;
        }

        .sidebar-brand {
            padding: 0.25rem 0 1.1rem 0;
            border-bottom: 1px solid #e7ebf1;
            margin-bottom: 1.2rem;
        }

        .sidebar-brand-title {
            color: #172033;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sidebar-brand-subtitle {
            color: #687386;
            font-size: 0.78rem;
        }

        .filter-label {
            color: #344054;
            font-size: 0.82rem;
            font-weight: 650;
            margin-top: 0.85rem;
            margin-bottom: 0.25rem;
        }

        /* ---------- Header ---------- */
        .dashboard-header {
            background: #ffffff;
            border: 1px solid #e4e8ef;
            border-radius: 14px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.35rem;
            box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
        }

        .dashboard-title {
            color: #172033;
            font-size: 2.15rem;
            font-weight: 720;
            line-height: 1.15;
            margin: 0;
        }

        .dashboard-subtitle {
            color: #667085;
            font-size: 0.92rem;
            margin-top: 0.45rem;
        }

        .status-line {
            margin-top: 0.8rem;
            color: #475467;
            font-size: 0.78rem;
        }

        /* ---------- Section headers ---------- */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.4rem 0 0.8rem 0;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid #e1e6ed;
        }

        .section-number {
            color: #ffffff;
            background: #1f4e79;
            width: 28px;
            height: 28px;
            border-radius: 7px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.76rem;
            font-weight: 700;
        }

        .section-title {
            color: #172033;
            font-size: 1.22rem;
            font-weight: 680;
        }

        .section-caption {
            color: #667085;
            font-size: 0.78rem;
            margin: -0.25rem 0 0.85rem 2.4rem;
        }

        /* ---------- KPI cards ---------- */
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e4e8ef;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            min-height: 105px;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.035);
        }

        .kpi-label {
            color: #667085;
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.035em;
        }

        .kpi-value {
            color: #172033;
            font-size: 1.55rem;
            font-weight: 720;
            margin-top: 0.38rem;
        }

        .kpi-note {
            color: #98a2b3;
            font-size: 0.72rem;
            margin-top: 0.25rem;
        }

        /* ---------- Chart cards ---------- */
        .chart-card {
            background: #ffffff;
            border: 1px solid #e4e8ef;
            border-radius: 12px;
            padding: 0.75rem 1rem 0.45rem 1rem;
            box-shadow: 0 2px 8px rgba(16, 24, 40, 0.03);
            margin-bottom: 1rem;
        }

        /* ---------- ML result ---------- */
        .prediction-card {
            background: #ffffff;
            border: 1px solid #d6dde8;
            border-radius: 14px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05);
        }

        .prediction-label {
            color: #667085;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 650;
        }

        .prediction-value {
            color: #1f4e79;
            font-size: 2.3rem;
            font-weight: 750;
            margin-top: 0.2rem;
        }

        .prediction-band {
            color: #475467;
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }

        /* ---------- Streamlit controls ---------- */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #1f4e79;
            background: #1f4e79;
            color: #ffffff;
            font-weight: 650;
            min-height: 2.6rem;
        }

        .stButton > button:hover {
            background: #173a5c;
            border-color: #173a5c;
            color: #ffffff;
        }

        div[data-baseweb="select"] > div {
            border-radius: 7px;
        }

        /* ---------- Footer ---------- */
        .footer {
            border-top: 1px solid #e4e8ef;
            margin-top: 2rem;
            padding-top: 0.9rem;
            color: #98a2b3;
            font-size: 0.72rem;
            text-align: center;
        }

        /* Remove default decoration around dataframe */
        [data-testid="stDataFrame"] {
            border: 1px solid #e4e8ef;
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA / MODEL
# ============================================================

@st.cache_data
def load_data():
    return pd.read_excel("PFC_Feature_Engineered_Dataset_5_Features.xlsx")


@st.cache_resource
def load_model():
    return joblib.load("PFC_Risk_Model.pkl")


df = load_data()
pipeline = load_model()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">PFC Analytics</div>
            <div class="sidebar-brand-subtitle">Project Finance Portfolio Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="filter-label">Portfolio Filters</div>', unsafe_allow_html=True)

    selected_sector = st.multiselect(
        "Sector",
        sorted(df["Sector"].dropna().unique()),
        default=sorted(df["Sector"].dropna().unique()),
        label_visibility="collapsed",
    )

    selected_state = st.multiselect(
        "State",
        sorted(df["State"].dropna().unique()),
        default=sorted(df["State"].dropna().unique()),
        label_visibility="collapsed",
    )

    selected_status = st.multiselect(
        "Project Status",
        sorted(df["Project_Status"].dropna().unique()),
        default=sorted(df["Project_Status"].dropna().unique()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Use the filters above to update all portfolio analytics.")

# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    (df["Sector"].isin(selected_sector))
    & (df["State"].isin(selected_state))
    & (df["Project_Status"].isin(selected_status))
].copy()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">PFC Project Finance Analytics</div>
        <div class="dashboard-subtitle">
            Portfolio Monitoring &nbsp;|&nbsp; Funding Analysis &nbsp;|&nbsp; Risk Intelligence &nbsp;|&nbsp; ML Risk Prediction
        </div>
        <div class="status-line">
            Interactive portfolio view • Filters applied dynamically • Random Forest risk model integrated
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PORTFOLIO OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="section-header">
        <span class="section-number">01</span>
        <span class="section-title">Portfolio Overview</span>
    </div>
    <div class="section-caption">Key financial, operational and risk indicators for the selected portfolio.</div>
    """,
    unsafe_allow_html=True,
)

kpis = [
    ("Total Projects", f"{len(filtered_df):,}", "Selected portfolio"),
    ("Loan Sanctioned", f"₹{filtered_df['Loan_Sanctioned'].sum()/1e9:,.2f} B", "Total approved financing"),
    ("Loan Disbursed", f"₹{filtered_df['Loan_Disbursed'].sum()/1e9:,.2f} B", "Total released financing"),
    ("Outstanding", f"₹{filtered_df['Outstanding_Amount'].sum()/1e9:,.2f} B", "Current outstanding amount"),
    ("Average Risk Score", f"{filtered_df['Risk_Score'].mean():.2f}", "Portfolio risk indicator"),
    ("Avg Loan Utilization", f"{filtered_df['Loan_Utilization_%'].mean():.1f}%", "Average utilization"),
    ("Avg Completion", f"{filtered_df['Project_Completion_%'].mean():.1f}%", "Average project progress"),
    ("High Risk Projects", f"{filtered_df['High_Risk_Flag'].sum():,}", "Risk score threshold ≥ 70"),
]

cols = st.columns(4)
for i, (label, value, note) in enumerate(kpis[:4]):
    with cols[i]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
cols = st.columns(4)
for i, (label, value, note) in enumerate(kpis[4:]):
    with cols[i]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# TABS
# ============================================================

tab_overview, tab_risk, tab_ml, tab_portfolio = st.tabs(
    ["Overview", "Risk Intelligence", "ML Prediction", "Portfolio Data"]
)

# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:

    st.markdown(
        """
        <div class="section-header">
            <span class="section-number">02</span>
            <span class="section-title">Funding & Loan Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sector-wise Sanctioned Amount")
        sector_funding = (
            filtered_df.groupby("Sector")["Loan_Sanctioned"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(sector_funding, height=320)

    with col2:
        st.markdown("#### State-wise Sanctioned Amount")
        state_funding = (
            filtered_df.groupby("State")["Loan_Sanctioned"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(state_funding, height=320)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sanctioned vs Disbursed by Sector")
        loan_comparison = (
            filtered_df.groupby("Sector")[
                ["Loan_Sanctioned", "Loan_Disbursed"]
            ].sum()
        )
        st.bar_chart(loan_comparison, height=320)

    with col2:
        st.markdown("#### Average Loan Utilization by Sector")
        utilization = (
            filtered_df.groupby("Sector")["Loan_Utilization_%"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(utilization, height=320)

    st.markdown(
        """
        <div class="section-header">
            <span class="section-number">03</span>
            <span class="section-title">Project Progress</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Project Status")
        status_data = filtered_df["Project_Status"].value_counts()
        st.bar_chart(status_data, height=300)

    with col2:
        st.markdown("#### Completion Category")
        completion = filtered_df["Completion_Category"].value_counts()
        st.bar_chart(completion, height=300)

# ============================================================
# RISK TAB
# ============================================================

with tab_risk:

    st.markdown(
        """
        <div class="section-header">
            <span class="section-number">04</span>
            <span class="section-title">Risk Intelligence</span>
        </div>
        <div class="section-caption">
            Monitor risk distribution, schedule delays and sector-level risk concentration.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(
            filtered_df["Risk_Score"],
            bins=20,
            ax=ax,
            color="#1f4e79",
        )
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Projects")
        ax.grid(axis="y", alpha=0.15)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("#### Delay Days vs Risk Score")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(
            filtered_df["Delay_Days"],
            filtered_df["Risk_Score"],
            alpha=0.35,
            s=18,
            color="#667085",
        )
        ax.set_xlabel("Delay Days")
        ax.set_ylabel("Risk Score")
        ax.grid(alpha=0.15)
        sns.despine(ax=ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("#### Average Risk Score by Sector")
    sector_risk = (
        filtered_df.groupby("Sector")["Risk_Score"]
        .mean()
        .sort_values(ascending=False)
    )
    st.bar_chart(sector_risk, height=330)

    # Risk summary table
    risk_summary = (
        filtered_df.groupby("Sector")
        .agg(
            Projects=("Project_ID", "count"),
            Avg_Risk=("Risk_Score", "mean"),
            Avg_Delay=("Delay_Days", "mean"),
            Avg_Completion=("Project_Completion_%", "mean"),
        )
        .sort_values("Avg_Risk", ascending=False)
        .round(2)
    )

    st.markdown("#### Sector Risk Summary")
    st.dataframe(
        risk_summary,
        use_container_width=True,
        column_config={
            "Projects": st.column_config.NumberColumn("Projects"),
            "Avg_Risk": st.column_config.NumberColumn("Avg Risk", format="%.2f"),
            "Avg_Delay": st.column_config.NumberColumn("Avg Delay", format="%.2f"),
            "Avg_Completion": st.column_config.NumberColumn("Avg Completion", format="%.2f"),
        },
    )

# ============================================================
# ML TAB
# ============================================================

with tab_ml:

    st.markdown(
        """
        <div class="section-header">
            <span class="section-number">05</span>
            <span class="section-title">Machine Learning Risk Prediction</span>
        </div>
        <div class="section-caption">
            Random Forest Regressor using engineered financial, operational, ESG and categorical features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Model evaluation metrics should be entered from the Feature Engineering + ML notebook before final submission."
    )

    st.markdown("#### Top Risk Drivers")

    try:
        feature_names = (
            pipeline.named_steps["preprocessing"]
            .get_feature_names_out()
        )

        importances = (
            pipeline.named_steps["model"]
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances,
            }
        ).sort_values("Importance", ascending=False).head(10)

        st.bar_chart(
            importance_df.set_index("Feature"),
            height=380,
        )

    except Exception:
        st.warning("Feature importance could not be displayed for the loaded model.")

    st.markdown("---")
    st.markdown("#### Predict Project Risk")

    col1, col2 = st.columns(2)

    with col1:
        loan_sanctioned = st.number_input(
            "Loan Sanctioned",
            min_value=0.0,
            value=1_000_000_000.0,
            step=50_000_000.0,
        )

        loan_disbursed = st.number_input(
            "Loan Disbursed",
            min_value=0.0,
            value=700_000_000.0,
            step=50_000_000.0,
        )

        outstanding = st.number_input(
            "Outstanding Amount",
            min_value=0.0,
            value=300_000_000.0,
            step=50_000_000.0,
        )

        repayment = st.number_input(
            "Repayment Received",
            min_value=0.0,
            value=200_000_000.0,
            step=50_000_000.0,
        )

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            value=8.0,
            step=0.1,
        )

        completion = st.slider(
            "Project Completion (%)",
            0.0,
            100.0,
            70.0,
        )

        utilization = st.slider(
            "Loan Utilization (%)",
            0.0,
            100.0,
            70.0,
        )

    with col2:
        delay = st.number_input(
            "Delay Days",
            min_value=0,
            value=20,
            step=1,
        )

        esg = st.slider(
            "ESG Score",
            0.0,
            100.0,
            75.0,
        )

        capacity = st.number_input(
            "Capacity (MW)",
            min_value=0.0,
            value=1000.0,
            step=50.0,
        )

        loan_size = st.selectbox(
            "Loan Size Category",
            sorted(df["Loan_Size_Category"].dropna().unique()),
        )

        completion_category = st.selectbox(
            "Completion Category",
            sorted(df["Completion_Category"].dropna().unique()),
        )

        prediction_state = st.selectbox(
            "State",
            sorted(df["State"].dropna().unique()),
        )

        prediction_sector = st.selectbox(
            "Sector",
            sorted(df["Sector"].dropna().unique()),
        )

    repayment_ratio = (
        repayment / loan_disbursed
        if loan_disbursed > 0
        else 0
    )

    st.caption(f"Calculated repayment ratio: {repayment_ratio:.2%}")

    if st.button("Predict Risk Score"):
        prediction_data = pd.DataFrame(
            {
                "Loan_Sanctioned": [loan_sanctioned],
                "Loan_Disbursed": [loan_disbursed],
                "Outstanding_Amount": [outstanding],
                "Repayment_Received": [repayment],
                "Interest_Rate": [interest_rate],
                "Project_Completion_%": [completion],
                "Loan_Utilization_%": [utilization],
                "Delay_Days": [delay],
                "ESG_Score": [esg],
                "Capacity_MW": [capacity],
                "Repayment_Ratio": [repayment_ratio],
                "Loan_Size_Category": [loan_size],
                "Completion_Category": [completion_category],
                "State": [prediction_state],
                "Sector": [prediction_sector],
            }
        )

        prediction = float(pipeline.predict(prediction_data)[0])

        if prediction >= 70:
            band = "High Risk"
        elif prediction >= 40:
            band = "Moderate Risk"
        else:
            band = "Low Risk"

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Model Output</div>
                <div class="prediction-value">{prediction:.2f}</div>
                <div class="prediction-band">Risk Classification: <strong>{band}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# PORTFOLIO DATA TAB
# ============================================================

with tab_portfolio:

    st.markdown(
        """
        <div class="section-header">
            <span class="section-number">06</span>
            <span class="section-title">Project Portfolio</span>
        </div>
        <div class="section-caption">
            Detailed project-level records based on the active portfolio filters.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=560,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        PFC Project Finance Analytics • Academic Data Analytics & Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)
