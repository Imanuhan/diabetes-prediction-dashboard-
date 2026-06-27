"""
Dashboard Prediksi Risiko Diabetes
Proyek EAS - Mata Kuliah Pembelajaran Mesin
Model: Random Forest Classifier
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Dashboard Prediksi Diabetes",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = "."
DATA_PATH = "Diabetes_Prediction.csv"

# =============================================================================
# CSS KUSTOM
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; }

:root {
    --teal-dark: #075E54;
    --teal: #0E7C7B;
    --teal-light: #E6F4F3;
    --amber: #F0A202;
    --danger: #D64550;
    --success: #2D9D78;
    --text-main: #1F2937;
    --text-muted: #6B7280;
    --card-bg: #FFFFFF;
}

.stApp { background-color: #F7F9F8; }

section[data-testid="stSidebar"] { background-color: var(--teal-dark); }
section[data-testid="stSidebar"] * { color: #F0FAF9 !important; }

.metric-card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    border: 1px solid #E5E9E8;
    box-shadow: 0 2px 6px rgba(15, 50, 48, 0.04);
    height: 100%;
}
.metric-label { color: var(--text-muted); font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.metric-value { color: var(--teal-dark); font-size: 1.9rem; font-weight: 800; font-family: 'Sora', sans-serif; margin-top: 0.15rem; }
.metric-sub { color: var(--text-muted); font-size: 0.78rem; margin-top: 0.2rem; }

.section-title { font-family: 'Sora', sans-serif; font-weight: 700; color: var(--teal-dark); font-size: 1.4rem; margin-bottom: 0.2rem; }
.section-desc { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1rem; }

.info-box {
    background: var(--teal-light);
    border-left: 4px solid var(--teal);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    color: var(--text-main);
    font-size: 0.92rem;
}

.hero-banner {
    background: linear-gradient(135deg, #075E54 0%, #0E7C7B 55%, #1a9f8f 100%);
    border-radius: 20px;
    padding: 2.4rem 2.6rem;
    color: white;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(7, 94, 84, 0.25);
}
.hero-banner h1 { color: white; font-size: 1.95rem; margin-bottom: 0.6rem; line-height: 1.3; position: relative; z-index: 1; }
.hero-banner p { color: #D4F5F1; font-size: 0.98rem; max-width: 700px; line-height: 1.7; position: relative; z-index: 1; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 50px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #B2EFEA;
    margin-bottom: 0.9rem;
}
.hero-stats { display: flex; gap: 2rem; margin-top: 1.4rem; position: relative; z-index: 1; }
.hero-stat-item { text-align: left; }
.hero-stat-value { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 800; color: white; }
.hero-stat-label { font-size: 0.73rem; color: #B2EFEA; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.1rem; }

.sidebar-eyebrow { color: #9FE0DA; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.25rem; }
.sidebar-title { font-family: 'Sora', sans-serif !important; font-weight: 700; font-size: 1.4rem; margin: 0 0 1rem 0; line-height: 1.3; }
.sidebar-credit-card { background: rgba(255,255,255,0.06); border-radius: 12px; padding: 1.1rem 1.15rem; margin-top: 0.3rem; }
.sidebar-credit-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #9FE0DA; margin: 1rem 0 0.4rem 0; }
.sidebar-credit-label.first { margin-top: 0; }
.sidebar-credit-value { font-size: 0.83rem; line-height: 1.65; opacity: 0.95; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD DATA & MODEL
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.sample(min(20000, len(df)), random_state=42)

@st.cache_resource
def load_model():
    model_path = f"{ARTIFACT_DIR}/rf_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    df_train = pd.read_csv(DATA_PATH)
    X = df_train.drop("diabetes", axis=1)
    for col in ["gender", "smoking_history"]:
        X[col] = X[col].astype("category").cat.codes
    y = df_train["diabetes"]
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

@st.cache_resource
def load_feature_columns():
    return joblib.load(f"{ARTIFACT_DIR}/feature_columns.pkl")

@st.cache_data
def load_json(name):
    with open(f"{ARTIFACT_DIR}/{name}") as f:
        return json.load(f)

# Pre-compute semua data chart saat startup
@st.cache_data
def precompute_charts(data):
    # Distribusi
    counts = data["diabetes"].map({0: "Tidak Diabetes", 1: "Diabetes"}).value_counts().reset_index()
    counts.columns = ["Status", "Jumlah"]
    
    # Sample scatter
    sample = data.sample(min(1000, len(data)), random_state=42)
    
    # Korelasi
    num_cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
    corr = data[num_cols].corr().round(2)
    
    # Merokok
    smoke = data.groupby("smoking_history")["diabetes"].mean().reset_index()
    smoke.columns = ["Riwayat Merokok", "Proporsi Diabetes"]
    
    return counts, sample, corr, smoke

counts_data, sample_data, corr_data, smoke_data = precompute_charts(df)

FRIENDLY_NAMES = {
    "gender": "Jenis Kelamin",
    "age": "Usia",
    "hypertension": "Hipertensi",
    "heart_disease": "Penyakit Jantung",
    "smoking_history": "Riwayat Merokok",
    "bmi": "BMI (Indeks Massa Tubuh)",
    "HbA1c_level": "Kadar HbA1c",
    "blood_glucose_level": "Kadar Gula Darah",
}

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-eyebrow">Machine Learning</div>
    <div class="sidebar-title">🩸 Diabetes Check</div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["🏠 Beranda", "📊 Eksplorasi Data", "📈 Performa Model", "🎯 Check Risiko Penyakit Diabetes"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-credit-card">
    <div class="sidebar-credit-label first">📛 Nama Kelompok</div>
    <div class="sidebar-credit-value" style="margin-bottom:0.6rem;">
        <div style="font-weight:600;">Kanaya Tsabithatiz Zahra</div>
        <div style="opacity:0.75; font-size:0.78rem;">NRP 2043231071</div>
    </div>
    <div class="sidebar-credit-value">
        <div style="font-weight:600;">Muhammad Rizky Imanuhan</div>
        <div style="opacity:0.75; font-size:0.78rem;">NRP 2043231102</div>
    </div>
    <div class="sidebar-credit-label">👨‍🏫 Dosen Pengampu</div>
    <div class="sidebar-credit-value">Noviyanti Santoso, S.Si., M.Si., Ph.D.</div>
    <div class="sidebar-credit-label">🏛️ Departemen</div>
    <div class="sidebar-credit-value">
        Departemen Statistika Bisnis, Fakultas Vokasi, Institut Teknologi Sepuluh Nopember
    </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HALAMAN 1 — BERANDA
# =============================================================================
def render_beranda():
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-badge">🩸 Machine Learning &nbsp;·&nbsp; Random Forest Classifier</div>
        <h1>Sistem Prediksi Risiko Diabetes<br>Menggunakan Random Forest Classifier</h1>
        <p>Dashboard ini dirancang untuk membantu mengidentifikasi risiko diabetes
        berdasarkan beberapa indikator kesehatan yang dimasukkan oleh pengguna, seperti
        usia, indeks massa tubuh (BMI), kadar glukosa darah, serta faktor kesehatan lainnya.</p>
        <div class="hero-stats">
            <div class="hero-stat-item">
                <div class="hero-stat-value">{len(df):,}</div>
                <div class="hero-stat-label">Total Data Pasien</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-value">{metrics['accuracy']*100:.1f}%</div>
                <div class="hero-stat-label">Akurasi Model</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-value">{metrics['roc_auc']*100:.1f}%</div>
                <div class="hero-stat-label">ROC-AUC Score</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-value">{len(feature_columns)}</div>
                <div class="hero-stat-label">Fitur Kesehatan</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-title">Apa itu Diabetes?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Diabetes adalah kondisi kronis yang terjadi ketika tubuh tidak mampu mengatur kadar
        gula darah dengan baik. Jika tidak terdeteksi dan ditangani sejak dini, diabetes dapat
        memicu komplikasi serius seperti penyakit jantung, gangguan ginjal, dan kerusakan saraf.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.4rem;">Permasalahan & Tujuan Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Permasalahan:</b> Diabetes seringkali tidak terdeteksi pada tahap awal karena minimnya
        skrining rutin, padahal data kesehatan dasar pasien sudah tersedia di banyak fasilitas kesehatan.<br><br>
        <b>Tujuan:</b> Dashboard ini memanfaatkan model klasifikasi berbasis machine learning
        untuk mengidentifikasi risiko diabetes berdasarkan indikator kesehatan dasar secara cepat dan informatif.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Jenis & Sumber Data</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
        Dataset ini berisi <b>{len(df):,} baris</b> data pasien dengan <b>{len(feature_columns)} fitur</b>
        bertipe numerik dan kategorikal, serta 1 label target (<i>diabetes</i>: 0 = tidak, 1 = ya).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        dtype_table = pd.DataFrame({
            "Fitur": [FRIENDLY_NAMES.get(c, c) for c in df.columns],
            "Tipe Data": ["Kategorikal" if df[c].dtype == object else "Numerik" for c in df.columns],
        })
        st.dataframe(dtype_table, hide_index=True, use_container_width=True)


# =============================================================================
# HALAMAN 2 — EKSPLORASI DATA
# =============================================================================
def render_eda():
    st.markdown('<div class="section-title">Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Pahami pola dan sebaran data.</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    with f1:
        gender_filter = st.multiselect("Filter Jenis Kelamin", options=df["gender"].unique().tolist(), default=df["gender"].unique().tolist())
    with f2:
        age_range = st.slider("Filter Rentang Usia", int(df["age"].min()), int(df["age"].max()), (0, 80))

    fdf = df[(df["gender"].isin(gender_filter)) & (df["age"].between(age_range[0], age_range[1]))]
    st.caption(f"Menampilkan {len(fdf):,} dari {len(df):,} total data.")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi", "📈 Scatter & Korelasi", "🚬 Merokok", "📋 Data Mentah"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = fdf["diabetes"].map({0: "Tidak Diabetes", 1: "Diabetes"}).value_counts().reset_index()
            counts.columns = ["Status", "Jumlah"]
            fig = px.pie(counts, names="Status", values="Jumlah", hole=0.55,
                         color="Status", color_discrete_map={"Tidak Diabetes": "#0E7C7B", "Diabetes": "#D64550"},
                         title="Proporsi Status Diabetes")
            fig.update_layout(font_family="Plus Jakarta Sans", legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(fdf, x="age", color=fdf["diabetes"].map({0: "Tidak Diabetes", 1: "Diabetes"}),
                                nbins=30, barmode="overlay", opacity=0.75,
                                color_discrete_map={"Tidak Diabetes": "#0E7C7B", "Diabetes": "#D64550"},
                                title="Distribusi Usia berdasarkan Status Diabetes",
                                labels={"age": "Usia", "color": "Status"})
            fig2.update_layout(font_family="Plus Jakarta Sans")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            sample = fdf.sample(min(3000, len(fdf)), random_state=42) if len(fdf) > 0 else fdf
            fig3 = px.scatter(sample, x="bmi", y="blood_glucose_level",
                              color=sample["diabetes"].map({0: "Tidak Diabetes", 1: "Diabetes"}),
                              color_discrete_map={"Tidak Diabetes": "#0E7C7B", "Diabetes": "#D64550"},
                              opacity=0.5, title="BMI vs Kadar Gula Darah",
                              labels={"bmi": "BMI", "blood_glucose_level": "Gula Darah", "color": "Status"})
            fig3.update_layout(font_family="Plus Jakarta Sans")
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            num_cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
            corr = fdf[num_cols].corr().round(2) if len(fdf) > 1 else df[num_cols].corr().round(2)
            fig4 = px.imshow(corr, text_auto=True, color_continuous_scale=["#FFFFFF", "#0E7C7B"],
                             title="Korelasi Antar Fitur Numerik")
            fig4.update_layout(font_family="Plus Jakarta Sans")
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        smoke = fdf.groupby(["smoking_history"])["diabetes"].mean().reset_index()
        smoke.columns = ["Riwayat Merokok", "Proporsi Diabetes"]
        fig5 = px.bar(smoke.sort_values("Proporsi Diabetes", ascending=False),
                      x="Riwayat Merokok", y="Proporsi Diabetes",
                      color="Proporsi Diabetes", color_continuous_scale=["#E6F4F3", "#0E7C7B"])
        fig5.update_layout(font_family="Plus Jakarta Sans", yaxis_tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        st.dataframe(fdf.head(50), use_container_width=True)


# =============================================================================
# HALAMAN 3 — PERFORMA MODEL
# =============================================================================
def render_model():
    st.markdown('<div class="section-title">📈 Performa Model Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Ringkasan seberapa baik model dalam memprediksi diabetes pada data uji (20% data).</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    labels = [
        ("Akurasi", metrics["accuracy"], "Persentase prediksi yang benar"),
        ("Presisi", metrics["precision"], "Dari yang diprediksi diabetes, berapa % yang benar"),
        ("Recall", metrics["recall"], "Dari yang benar-benar diabetes, berapa % terdeteksi"),
        ("F1-Score", metrics["f1_score"], "Keseimbangan antara presisi & recall"),
        ("ROC-AUC", metrics["roc_auc"], "Kemampuan model membedakan kedua kelas"),
    ]
    for col, (label, val, sub) in zip(cols, labels):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val*100:.1f}%</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cm_arr = np.array(cm)
        fig = px.imshow(cm_arr, text_auto=True, color_continuous_scale=["#FFFFFF", "#075E54"],
                        x=["Prediksi: Tidak", "Prediksi: Diabetes"],
                        y=["Aktual: Tidak", "Aktual: Diabetes"],
                        title="Confusion Matrix")
        fig.update_layout(font_family="Plus Jakarta Sans")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        <b>Cara baca:</b> baris menunjukkan kondisi sebenarnya, kolom menunjukkan hasil prediksi model.
        Semakin besar angka di diagonal, semakin akurat modelnya.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode="lines",
                                  line=dict(color="#0E7C7B", width=3), name="Model"))
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(color="#D1D5DB", width=2, dash="dash"), name="Acak"))
        fig2.update_layout(title=f"Kurva ROC (AUC = {metrics['roc_auc']:.3f})",
                           xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           font_family="Plus Jakarta Sans")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div class="info-box">
        <b>Cara baca:</b> semakin kurva mendekati pojok kiri-atas, semakin baik model.
        AUC mendekati 1.0 berarti model sangat baik.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1rem;">Fitur Paling Berpengaruh</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({
        "Fitur": [FRIENDLY_NAMES.get(k, k) for k in feature_importance.keys()],
        "Tingkat Pengaruh": list(feature_importance.values()),
    })
    fig3 = px.bar(fi_df, x="Tingkat Pengaruh", y="Fitur", orientation="h",
                  color="Tingkat Pengaruh", color_continuous_scale=["#E6F4F3", "#0E7C7B"])
    fig3.update_layout(font_family="Plus Jakarta Sans", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)


# =============================================================================
# HALAMAN 4 — PREDIKSI
# =============================================================================
def render_predict():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#075E54 0%,#0E7C7B 60%,#1a9f8f 100%);
                border-radius:18px; padding:2rem 2.4rem; margin-bottom:1.6rem;
                box-shadow:0 6px 24px rgba(7,94,84,0.2);">
        <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#9FE0DA;margin-bottom:0.5rem;">
            🎯 Machine Learning · Random Forest
        </div>
        <div style="font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:800;color:white;margin-bottom:0.5rem;">
            Check Risiko Penyakit Diabetes
        </div>
        <div style="color:#D4F5F1;font-size:0.93rem;max-width:680px;line-height:1.65;">
            Lengkapi data kesehatan kamu di bawah ini untuk mendapatkan estimasi risiko diabetes secara instan.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Data Pribadi**")
            gender = st.selectbox("Jenis Kelamin", options=list(encoders["gender"].keys()))
            age = st.slider("Usia (tahun)", 1, 100, 35)
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=24.5, step=0.1,
                                  help="Normal: 18.5–24.9 | Overweight: 25–29.9 | Obesitas: ≥30")
        with c2:
            st.markdown("**🩺 Riwayat Kesehatan**")
            hypertension = st.radio("Memiliki Hipertensi?", options=["Tidak", "Ya"], horizontal=True)
            heart_disease = st.radio("Riwayat Penyakit Jantung?", options=["Tidak", "Ya"], horizontal=True)
            smoking_history = st.selectbox("Riwayat Merokok", options=list(encoders["smoking_history"].keys()))
        with c3:
            st.markdown("**🧪 Hasil Laboratorium**")
            hba1c = st.number_input("Kadar HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1,
                                    help="Normal: <5.7% | Prediabetes: 5.7–6.4% | Diabetes: ≥6.5%")
            glucose = st.number_input("Kadar Gula Darah (mg/dL)", min_value=50, max_value=350, value=110, step=1,
                                      help="Normal puasa: 70–99 | Prediabetes: 100–125 | Diabetes: ≥126")

        submitted = st.form_submit_button("🔍 Analisis Risiko Sekarang", use_container_width=True)

    if submitted:
        input_dict = {
            "gender": encoders["gender"][gender],
            "age": age,
            "hypertension": 1 if hypertension == "Ya" else 0,
            "heart_disease": 1 if heart_disease == "Ya" else 0,
            "smoking_history": encoders["smoking_history"][smoking_history],
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }
        X_input = pd.DataFrame([input_dict])[feature_columns]
        proba = model.predict_proba(X_input)[0][1]

        if proba < 0.3:
            risk_label, color, bg_color, icon_risk = "Risiko Rendah", "#2D9D78", "#EBF9F4", "✅"
            tips = [
                ("🥗", "Pertahankan pola makan sehat dan bergizi seimbang"),
                ("🏃", "Lanjutkan rutinitas olahraga minimal 30 menit/hari"),
                ("💧", "Cukupi asupan air putih minimal 8 gelas per hari"),
                ("🩺", "Tetap lakukan pemeriksaan kesehatan rutin setahun sekali"),
            ]
        elif proba < 0.6:
            risk_label, color, bg_color, icon_risk = "Risiko Sedang", "#F0A202", "#FEF6E4", "⚠️"
            tips = [
                ("🥦", "Kurangi konsumsi gula, makanan olahan, dan karbohidrat sederhana"),
                ("⚖️", "Jaga berat badan ideal, targetkan BMI di bawah 25"),
                ("🚭", "Hindari rokok dan batasi konsumsi alkohol"),
                ("🩸", "Cek kadar gula darah secara berkala setiap 3–6 bulan"),
            ]
        else:
            risk_label, color, bg_color, icon_risk = "Risiko Tinggi", "#D64550", "#FEF0F0", "🚨"
            tips = [
                ("👨‍⚕️", "Segera konsultasikan hasil ini kepada dokter atau tenaga medis"),
                ("🩸", "Lakukan pemeriksaan HbA1c dan gula darah puasa secara menyeluruh"),
                ("🏥", "Ikuti program pengelolaan diabetes jika disarankan dokter"),
                ("📋", "Catat pola makan dan aktivitas harian untuk dipantau bersama dokter"),
            ]

        r1, r2 = st.columns([1, 1.5])
        with r1:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{color}dd,{color});
                        border-radius:20px;padding:2rem 1.5rem;text-align:center;
                        box-shadow:0 8px 28px {color}55;color:white;">
                <div style="font-size:3rem;margin-bottom:0.5rem;">{icon_risk}</div>
                <div style="font-size:0.8rem;opacity:0.85;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Hasil Estimasi</div>
                <div style="font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:800;margin-bottom:0.6rem;">{risk_label}</div>
                <div style="background:rgba(255,255,255,0.2);border-radius:50px;padding:0.5rem 1.2rem;display:inline-block;font-size:1.1rem;font-weight:700;">
                    Probabilitas: {proba*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 40, "color": color}},
                delta={"reference": 30, "increasing": {"color": "#D64550"}, "decreasing": {"color": "#2D9D78"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#D6F5EC"},
                        {"range": [30, 60], "color": "#FEF3CD"},
                        {"range": [60, 100], "color": "#FDDEDE"},
                    ],
                    "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": proba * 100},
                },
                title={"text": "Skor Risiko Diabetes", "font": {"size": 14, "color": "#6B7280"}},
            ))
            gauge.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30),
                                font_family="Plus Jakarta Sans", paper_bgcolor="white")
            st.plotly_chart(gauge, use_container_width=True)

        tips_html = ""
        for tip_icon, tip_text in tips:
            tips_html += f"""
            <div style="background:white;border-radius:10px;padding:0.75rem 1rem;
                        display:flex;align-items:flex-start;gap:0.7rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <span style="font-size:1.3rem;">{tip_icon}</span>
                <span style="color:#374151;font-size:0.85rem;line-height:1.5;">{tip_text}</span>
            </div>"""

        st.markdown(f"""
        <div style="background:{bg_color};border:1.5px solid {color}33;border-radius:16px;
                    padding:1.4rem 1.6rem;margin-top:1.2rem;">
            <div style="font-family:'Sora',sans-serif;font-weight:700;color:{color};font-size:1rem;margin-bottom:1rem;">
                💡 Rekomendasi untuk Kamu
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.75rem;">
                {tips_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# ROUTING
# =============================================================================
if page == "🏠 Beranda":
    render_beranda()
elif page == "📊 Eksplorasi Data":
    render_eda()
elif page == "📈 Performa Model":
    render_model()
elif page == "🎯 Check Risiko Penyakit Diabetes":
    render_predict()
