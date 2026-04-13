"""
Fertility Outcome Classifier -- Streamlit Clinical Dashboard
Tabs: Single Prediction | Batch Prediction | EDA | Model Info
"""
import io
import os

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
sns.set_theme(style="whitegrid", palette="muted")

st.set_page_config(
    page_title="Fertility Outcome Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Fertility Outcome\nClassifier")
    st.markdown("---")
    st.markdown("**Model:** XGBoost v1.0")
    st.markdown("**Target:** Pregnancy_Outcome")
    st.markdown("**Threshold:** 0.40 (Failure recall)")
    st.markdown("**Test AUC:** 0.950")
    st.markdown("---")
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        info = r.json()
        st.success("API: Online ✅")
        st.caption(f"Threshold: {info.get('decision_threshold', 0.40)}")
    except Exception:
        st.error("API: Offline ❌")
        st.caption(f"Expected at: {API_URL}")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📋 Batch Prediction",
    "📊 EDA Dashboard",
    "ℹ️ Model Info",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Couple Fertility Assessment")
    st.markdown("Enter couple details to predict pregnancy outcome probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Female Health**")
        female_age = st.slider("Female Age", 18, 50, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 24.5, step=0.1)
        menstrual = st.selectbox("Menstrual Regularity", ["Regular", "Irregular"])
        pcos = st.selectbox("PCOS", ["No", "Yes"])

    with col2:
        st.markdown("**Male Health**")
        male_age = st.slider("Male Age", 18, 70, 32)
        sperm_count = st.number_input("Sperm Count (M/ml)", 0.0, 200.0, 55.0, step=0.5)
        motility = st.number_input("Motility %", 0.0, 100.0, 65.0, step=0.5)

    with col3:
        st.markdown("**Lifestyle & Treatment**")
        stress = st.selectbox("Stress Level", ["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Intake", ["None", "Moderate", "High"])
        trying_months = st.slider("Trying Duration (months)", 0, 60, 12)
        treatment = st.selectbox("Treatment Type", ["None", "Medication", "IVF"])

    st.markdown("---")

    if st.button("🔮 Predict Outcome", type="primary", use_container_width=True):
        payload = {
            "female_age": female_age, "male_age": male_age, "bmi": bmi,
            "menstrual_regularity": menstrual, "pcos": pcos,
            "stress_level": stress, "smoking": smoking,
            "alcohol_intake": alcohol,
            "sperm_count_million_per_ml": sperm_count,
            "motility_pct": motility,
            "trying_duration_months": trying_months,
            "treatment_type": treatment,
        }
        try:
            with st.spinner("Running inference..."):
                r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10)
                result = r.json()

            prob = result["success_probability"]
            label = result["outcome_label"]
            risk = result["risk_level"]

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                if result["pregnancy_success"]:
                    st.success(f"### ✅ {label}")
                else:
                    st.error(f"### ⚠️ {label}")
            col_b.metric("Success Probability", f"{prob:.1%}")
            col_c.metric("Risk Level", risk)
            col_d.metric("Latency", f"{result['latency_ms']} ms")

            # Probability gauge
            fig, ax = plt.subplots(figsize=(8, 1.2))
            color = "#3BB273" if prob >= 0.40 else "#E84855"
            ax.barh([""], [prob], color=color, height=0.5)
            ax.barh([""], [1 - prob], left=[prob], color="#E0E0E0", height=0.5)
            ax.axvline(0.40, color="grey", linestyle="--", lw=1.5, label="Threshold (0.40)")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Success Probability")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Pregnancy Success Probability: {prob:.1%}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"API error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Batch Fertility Assessment")
    st.markdown("Upload a CSV file with couple records to score in bulk.")
    st.markdown("**Required columns:** `Female_Age`, `Male_Age`, `BMI`, `Menstrual_Regularity`, `PCOS`, `Stress_Level`, `Smoking`, `Alcohol_Intake`, `Sperm_Count_Million_per_ml`, `Motility_%`, `Trying_Duration_Months`, `Treatment_Type`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        # Fill informative NaNs
        df["Treatment_Type"] = df["Treatment_Type"].fillna("None")
        df["Alcohol_Intake"] = df["Alcohol_Intake"].fillna("None")

        st.markdown(f"**Loaded:** {len(df):,} records")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Score All Records", type="primary", use_container_width=True):
            records = []
            for _, row in df.iterrows():
                records.append({
                    "female_age": int(row["Female_Age"]),
                    "male_age": int(row["Male_Age"]),
                    "bmi": float(row["BMI"]),
                    "menstrual_regularity": str(row["Menstrual_Regularity"]),
                    "pcos": str(row["PCOS"]),
                    "stress_level": str(row["Stress_Level"]),
                    "smoking": str(row["Smoking"]),
                    "alcohol_intake": str(row["Alcohol_Intake"]),
                    "sperm_count_million_per_ml": float(row["Sperm_Count_Million_per_ml"]),
                    "motility_pct": float(row["Motility_%"]),
                    "trying_duration_months": int(row["Trying_Duration_Months"]),
                    "treatment_type": str(row["Treatment_Type"]),
                })

            if len(records) > 500:
                st.warning("Batch limit is 500. Scoring first 500.")
                records = records[:500]

            try:
                with st.spinner(f"Scoring {len(records)} couples..."):
                    r = httpx.post(f"{API_URL}/predict/batch", json=records, timeout=60)
                    result = r.json()

                preds = result["predictions"]
                df_out = df.head(len(preds)).copy()
                df_out["success_probability"] = [p["success_probability"] for p in preds]
                df_out["outcome_label"] = [p["outcome_label"] for p in preds]
                df_out["risk_level"] = [p["risk_level"] for p in preds]

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Scored", len(df_out))
                col2.metric("Predicted Success", result["success_count"])
                col3.metric("Success Rate", f"{result['success_count']/len(df_out):.1%}")

                st.dataframe(
                    df_out[["Female_Age", "Male_Age", "Treatment_Type",
                             "success_probability", "outcome_label", "risk_level"]].head(20),
                    use_container_width=True
                )

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                df_out["outcome_label"].value_counts().plot(
                    kind="bar", ax=axes[0], color=["#3BB273", "#E84855"], edgecolor="white"
                )
                axes[0].set_title("Outcome Distribution")
                axes[0].tick_params(axis="x", rotation=0)

                axes[1].hist(df_out["success_probability"], bins=20,
                             color="#2E86AB", edgecolor="white", alpha=0.85)
                axes[1].axvline(0.40, color="red", linestyle="--", label="Threshold")
                axes[1].set_xlabel("Success Probability")
                axes[1].set_title("Probability Distribution")
                axes[1].legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", csv,
                                   "fertility_predictions.csv", "text/csv",
                                   use_container_width=True)

            except Exception as e:
                st.error(f"API error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Exploratory Data Analysis")

    @st.cache_data
    def load_eda():
        try:
            return pd.read_csv("data/raw/Fertility_Health_Dataset_2026.csv")
        except FileNotFoundError:
            return None

    df_eda = load_eda()

    if df_eda is None:
        st.warning("Dataset not found.")
    else:
        df_eda["Treatment_Type"] = df_eda["Treatment_Type"].fillna("None")
        df_eda["Alcohol_Intake"] = df_eda["Alcohol_Intake"].fillna("None")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Couples", f"{len(df_eda):,}")
        col2.metric("Success Rate", f"{(df_eda['Pregnancy_Outcome']=='Success').mean():.1%}")
        col3.metric("Avg Female Age", f"{df_eda['Female_Age'].mean():.1f}")
        col4.metric("Avg Motility", f"{df_eda['Motility_%'].mean():.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Outcome Distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            df_eda["Pregnancy_Outcome"].value_counts().plot(
                kind="bar", ax=ax, color=["#3BB273", "#E84855"], edgecolor="white"
            )
            ax.set_title("Success vs Failure")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("#### Female Age Distribution by Outcome")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for outcome, color in [("Success", "#3BB273"), ("Failure", "#E84855")]:
                subset = df_eda[df_eda["Pregnancy_Outcome"] == outcome]["Female_Age"]
                ax.hist(subset, bins=15, alpha=0.6, color=color, label=outcome, edgecolor="white")
            ax.set_xlabel("Female Age")
            ax.set_title("Female Age by Outcome")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Success Rate by Categorical Feature")
        cat_col = st.selectbox("Select feature", ["Treatment_Type", "PCOS", "Stress_Level", "Smoking", "Alcohol_Intake", "Menstrual_Regularity"])
        df_eda["success_flag"] = (df_eda["Pregnancy_Outcome"] == "Success").astype(int)
        rate = df_eda.groupby(cat_col)["success_flag"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        rate.plot(kind="bar", ax=ax, color="#2E86AB", edgecolor="white")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Success Rate by {cat_col}")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Sperm Motility vs Success")
        fig, ax = plt.subplots(figsize=(10, 4))
        for outcome, color in [("Success", "#3BB273"), ("Failure", "#E84855")]:
            subset = df_eda[df_eda["Pregnancy_Outcome"] == outcome]
            ax.scatter(subset["Female_Age"], subset["Motility_%"],
                       alpha=0.4, color=color, label=outcome, s=15)
        ax.set_xlabel("Female Age")
        ax.set_ylabel("Motility %")
        ax.set_title("Female Age vs Sperm Motility by Outcome")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model Architecture")
        st.markdown("""
        | Component | Detail |
        |---|---|
        | Algorithm | XGBoost Classifier |
        | Preprocessing | sklearn Pipeline |
        | Categorical | OneHotEncoder (drop=first) |
        | Numeric | StandardScaler |
        | Engineered Feature | Female_Age x Motility |
        | Missing Value Strategy | Informative NaN fills |
        | Target | Pregnancy_Outcome (Success/Failure) |
        """)
        st.markdown("#### Performance Metrics")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | CV AUC (5-fold) | 0.946 +/- 0.012 |
        | **Test AUC** | **0.950** |
        | Avg Precision | 0.980 |
        | Decision Threshold | **0.40** |
        """)

    with col2:
        st.markdown("#### Missing Value Audit")
        st.markdown("""
        **Informative NaN strategy (not random imputation):**

        | Column | NaN Count | Fill Value | Clinical Meaning |
        |---|---|---|---|
        | `Treatment_Type` | 500 (62.5%) | `None` | No treatment received |
        | `Alcohol_Intake` | 259 (32.4%) | `None` | No alcohol consumption |

        These are **not** missing at random. They carry clinical information
        and are filled with domain-meaningful values, not statistical imputes.
        """)
        st.markdown("#### Threshold Rationale")
        st.markdown("""
        Decision threshold set to **0.40** (not default 0.50).

        In fertility counseling, **missing a high-risk couple** (classifying
        as likely Success when outcome is Failure) delays intervention and
        reduces treatment window. Lower threshold increases sensitivity for
        the Failure class — the clinical priority.
        """)

    st.markdown("#### Feature Importance")
    try:
        fi_img = plt.imread("artifacts/plots/feature_importance.png")
        st.image(fi_img, caption="XGBoost Feature Importance", use_container_width=True)
    except FileNotFoundError:
        st.info("Run training pipeline to generate feature importance plot.")
