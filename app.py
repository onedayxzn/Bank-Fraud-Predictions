"""
Aplikasi Streamlit untuk Deteksi Fraud Transaksi Bank
Menggunakan model machine learning yang telah dilatih
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-warning {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
    }
    .legit-success {
        background-color: #000000;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00aa00;
    }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk load model


@st.cache_resource
def load_model_resources():
    """Load model, scaler, dan informasi lainnya"""
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        model_info = joblib.load('models/model_info.pkl')

        # Coba load label encoders jika ada
        le_dict = None
        if os.path.exists('models/label_encoders.pkl'):
            le_dict = joblib.load('models/label_encoders.pkl')

        return model, scaler, feature_names, model_info, le_dict
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None


@st.cache_data
def load_dataset():
    """Load dataset untuk referensi"""
    try:
        df = pd.read_csv(
            "dataset/bank-transaction-dataset-for-fraud-detection/bank_transactions_data_2.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


# Load resources
model, scaler, feature_names, model_info, le_dict = load_model_resources()
df = load_dataset()

# Header
st.markdown("""
# BANK FRAUD DETECTION SYSTEM
**Sistem Deteksi Fraud Transaksi Bank**

Aplikasi machine learning untuk mendeteksi transaksi yang mencurigakan menggunakan model klasifikasi.
""")

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### Menu Navigasi")
    page = st.radio("Pilih halaman:",
                    ["Prediksi Fraud", "Informasi Model", "Dataset Overview"])

    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.info("""
    Aplikasi ini menggunakan Random Forest Classifier untuk mendeteksi fraud.

    **Fitur:**
    - Prediksi real-time
    - Confidence score
    - Batch prediction
    - Model insights
    """)

# ============================================================
# PAGE 1: PREDIKSI FRAUD
# ============================================================
if page == "Prediksi Fraud":
    st.markdown("## Prediksi Transaksi Bank")

    if model is None:
        st.error(
            "Model tidak berhasil dimuat. Pastikan file model ada di folder 'models'")
        st.stop()

    # Tabs untuk input method
    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

    with tab1:
        st.markdown("### Input Data Transaksi")

        # Ambil tipe data untuk setiap feature
        numeric_cols = model_info['numeric_cols']
        categorical_cols = model_info['categorical_cols']

        input_data = {}

        # Buat columns untuk layout yang lebih rapi
        col1, col2 = st.columns(2)

        # Input numeric features
        col_idx = 0
        cols = [col1, col2]

        for feature in numeric_cols:
            with cols[col_idx % 2]:
                min_val = float(df.drop(columns=['isFraud'] if 'isFraud' in df.columns else [
                                df.columns[-1]])[feature].min())
                max_val = float(df.drop(columns=['isFraud'] if 'isFraud' in df.columns else [
                                df.columns[-1]])[feature].max())
                mean_val = float(df.drop(columns=['isFraud'] if 'isFraud' in df.columns else [
                                 df.columns[-1]])[feature].mean())

                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
            col_idx += 1

        # Input categorical features
        for feature in categorical_cols:
            with cols[col_idx % 2]:
                if le_dict and feature in le_dict:
                    options = le_dict[feature].classes_
                    selected = st.selectbox(f"{feature}", options)
                    input_data[feature] = le_dict[feature].transform([selected])[
                        0]
                else:
                    input_data[feature] = st.number_input(
                        f"{feature}", value=0)
            col_idx += 1

        # Tombol prediksi
        if st.button("Prediksi Fraud", use_container_width=True, type="primary"):
            try:
                # Prepare input
                input_df = pd.DataFrame([input_data])

                # Reorder columns sesuai training features
                input_df = input_df[feature_names]

                # Scale data
                input_scaled = scaler.transform(input_df)

                # Prediksi
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]

                # Display hasil
                st.markdown("---")
                st.markdown("### Hasil Prediksi")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if prediction == 0:
                        st.markdown("""
                        <div class="legit-success">
                        <h3>TRANSAKSI AMAN (LEGITIMATE)</h3>
                        <p>Transaksi ini diprediksi aman dan sah</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="fraud-warning">
                        <h3>TRANSAKSI MENCURIGAKAN (FRAUD)</h3>
                        <p>Transaksi ini diprediksi sebagai kemungkinan fraud</p>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.metric(
                        "Confidence Legit",
                        f"{prediction_proba[0]*100:.2f}%",
                        delta=None
                    )

                with col3:
                    st.metric(
                        "Confidence Fraud",
                        f"{prediction_proba[1]*100:.2f}%",
                        delta=None
                    )

                # Risk gauge
                st.markdown("#### Fraud Risk Level")
                col1, col2 = st.columns([3, 1])

                fraud_prob = prediction_proba[1]
                with col1:
                    st.progress(fraud_prob)
                with col2:
                    if fraud_prob < 0.3:
                        st.success("LOW RISK")
                    elif fraud_prob < 0.7:
                        st.warning("MEDIUM RISK")
                    else:
                        st.error("HIGH RISK")

            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")

    with tab2:
        st.markdown("### Upload File CSV - Batch Prediction")

        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

        if uploaded_file is not None:
            try:
                # Load CSV
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview Data:")
                st.dataframe(df_upload.head())

                # Check columns
                missing_cols = [
                    col for col in feature_names if col not in df_upload.columns]
                if missing_cols:
                    st.error(
                        f"Error: Kolom yang hilang dalam file: {missing_cols}")
                    st.stop()

                # Prediksi batch
                if st.button("Prediksi Batch", use_container_width=True, type="primary"):
                    try:
                        # Prepare data - select only required features
                        df_predict = df_upload[feature_names].copy()

                        # Apply label encoding untuk categorical columns
                        categorical_cols = model_info['categorical_cols']
                        if le_dict:
                            for col in categorical_cols:
                                if col in df_predict.columns and col in le_dict:
                                    # Handle unseen categories
                                    df_predict[col] = df_predict[col].map(
                                        lambda x: le_dict[col].transform(
                                            [x])[0] if x in le_dict[col].classes_ else 0
                                    )

                        # Scale
                        df_predict_scaled = scaler.transform(df_predict)

                        # Prediksi
                        predictions = model.predict(df_predict_scaled)
                        probabilities = model.predict_proba(df_predict_scaled)

                        # Add results
                        results_df = df_upload.copy()
                        results_df['Prediction'] = ['Legit' if p ==
                                                    0 else 'Fraud' for p in predictions]
                        results_df['Fraud_Probability'] = probabilities[:, 1]
                        results_df['Confidence'] = np.where(
                            results_df['Fraud_Probability'] > 0.5,
                            results_df['Fraud_Probability'],
                            1 - results_df['Fraud_Probability']
                        ) * 100

                        # Display results
                        st.markdown("---")
                        st.markdown("### Hasil Prediksi Batch")

                        # Summary
                        fraud_count = (predictions == 1).sum()
                        legit_count = (predictions == 0).sum()
                        fraud_percentage = (
                            fraud_count / len(predictions)) * 100

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Transaksi", len(predictions))
                        col2.metric("Fraud Terdeteksi",
                                    f"{fraud_count} ({fraud_percentage:.1f}%)")
                        col3.metric(
                            "Legit", f"{legit_count} ({100-fraud_percentage:.1f}%)")

                        # Tampilkan tabel
                        st.dataframe(
                            results_df[['Prediction',
                                        'Fraud_Probability', 'Confidence']],
                            use_container_width=True
                        )

                        # Download hasil
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Hasil Prediksi (CSV)",
                            data=csv,
                            file_name=f"fraud_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Error dalam batch prediction: {str(e)}")

            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")

# ============================================================
# PAGE 2: INFORMASI MODEL
# ============================================================
elif page == "Informasi Model":
    st.markdown("## Informasi Model")

    if model_info is None:
        st.error("Model information tidak tersedia")
    else:
        # Model Performance
        st.markdown("### Performa Model")
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", f"{model_info['accuracy']:.4f}")
        col2.metric("Precision", f"{model_info['precision']:.4f}")
        col3.metric("Recall", f"{model_info['recall']:.4f}")
        col4.metric("F1-Score", f"{model_info['f1_score']:.4f}")
        col5.metric("ROC-AUC", f"{model_info['roc_auc']:.4f}")

        # Model Details
        st.markdown("### Detail Model")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Nama Model:** {model_info['model_name']}

            **Jumlah Features:** {model_info['n_features']}

            **Jumlah Numeric Features:** {len(model_info['numeric_cols'])}

            **Jumlah Categorical Features:** {len(model_info['categorical_cols'])}
            """)

        with col2:
            st.success("""
            ✅ **Model Status:** Siap untuk Produksi

            🔄 **Preprocessing:** StandardScaler

            🤖 **Algorithm:** Random Forest Classifier

            📦 **Deployment:** Streamlit App
            """)

        # Feature List
        st.markdown("### Daftar Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Numeric Features:**")
            for i, feat in enumerate(model_info['numeric_cols'], 1):
                st.caption(f"{i}. {feat}")

        with col2:
            st.markdown("**Categorical Features:**")
            for i, feat in enumerate(model_info['categorical_cols'], 1):
                st.caption(f"{i}. {feat}")

        # Metrik Penjelasan
        st.markdown("---")
        st.markdown("### Penjelasan Metrik")

        with st.expander("Klik untuk melihat penjelasan detail"):
            st.markdown("""
            **Accuracy:** Proporsi prediksi yang benar dari total prediksi.
            - Formula: (TP + TN) / (TP + TN + FP + FN)
            - Range: 0-1 (semakin tinggi semakin baik)

            **Precision:** Dari transaksi yang diprediksi fraud, berapa yang benar-benar fraud.
            - Formula: TP / (TP + FP)
            - Penting untuk mengurangi false positives (alarm palsu)

            **Recall:** Dari transaksi fraud yang sebenarnya, berapa yang terdeteksi.
            - Formula: TP / (TP + FN)
            - Penting untuk menangkap semua fraud yang ada

            **F1-Score:** Keseimbangan antara Precision dan Recall.
            - Formula: 2 * (Precision * Recall) / (Precision + Recall)
            - Range: 0-1 (semakin tinggi semakin baik)

            **ROC-AUC:** Area Under the Receiver Operating Characteristic Curve.
            - Range: 0.5-1.0 (0.5 = random, 1.0 = perfect)
            - Mengukur kemampuan model membedakan fraud dan legit di berbagai threshold
            """)

# ============================================================
# PAGE 3: DATASET OVERVIEW
# ============================================================
elif page == "Dataset Overview":
    st.markdown("## Dataset Overview")

    # Load data
    if df is None:
        st.error("Error: Dataset tidak berhasil dimuat")
        st.stop()

    st.markdown("### Statistik Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transaksi", len(df))
    col2.metric("Jumlah Kolom", len(df.columns))
    col3.metric("Waktu Analisis",
                datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Data shape dan info
    st.markdown("### Informasi Data")

    tab1, tab2, tab3 = st.tabs(["Head", "Info", "Statistik"])

    with tab1:
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        buffer = ""
        for col in df.columns:
            buffer += f"**{col}:** {df[col].dtype}\n\n"
        st.markdown(buffer)

    with tab3:
        st.dataframe(df.describe(), use_container_width=True)

    # Missing values
    st.markdown("### Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("Tidak ada missing values dalam dataset")
    else:
        st.warning(f"Ada {missing.sum()} missing values")
        st.dataframe(missing[missing > 0])

    # Target distribution
    if 'isFraud' in df.columns:
        st.markdown("### Distribusi Target (Fraud)")

        col1, col2 = st.columns(2)

        with col1:
            fraud_dist = df['isFraud'].value_counts()
            fig = st.bar_chart(
                pd.DataFrame({
                    'Kategori': ['Legit', 'Fraud'],
                    'Jumlah': [fraud_dist[0], fraud_dist[1]]
                }).set_index('Kategori')
            )

        with col2:
            st.metric("Total Fraud", fraud_dist.get(1, 0))
            st.metric("Total Legit", fraud_dist.get(0, 0))
            st.metric(
                "% Fraud", f"{(fraud_dist.get(1, 0) / len(df) * 100):.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><strong>Bank Fraud Detection System</strong> | Machine Learning for Transaction Security</p>
<p style='font-size: 0.8em; color: gray;'>© 2026 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
