#  Bank Fraud Detection System

**Sistem deteksi fraud transaksi bank dengan AI/ML - Akses langsung di web, tanpa setup lokal**

---

##  AKSES APLIKASI SEKARANG

### **[BUKA APLIKASI](https://bank-fraud-detection-app.streamlit.app/)**

Tidak perlu install apapun. Langsung bisa gunakan untuk:

- Prediksi fraud secara real-time
- Analisis batch transaksi
- Lihat model insights

---

##  JALANKAN LOKAL (OPSIONAL)

Jika ingin jalankan di komputer lokal:

### **CARA MUDAH (Windows):**
1. Double-click file `run_app.bat`
2. Tunggu browser membuka otomatis
3. Selesai!

### **CARA MANUAL:**
```cmd
# 1. Masuk folder project
cd "path\to\Bank Fraud Detection"

# 2. Aktivasi virtual environment
env\Scripts\activate.bat

# 3. Install streamlit (jika belum)
pip install streamlit

# 4. Jalankan aplikasi
streamlit run app.py
```

> 🆘 **Error?** Baca file **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

---

##  HASIL ANALISIS DATA

| Metric        | Hasil  | Status       |
| ------------- | ------ | ------------ |
| **Accuracy**  | 93.24% |  Excellent |
| **Precision** | 60.94% |  Very Good |
| **Recall**    | 81.25% |  Excellent |
| **ROC-AUC**   | 0.8940 |  Excellent |

**Kesimpulan:** Model sangat akurat untuk deteksi fraud! 

### Data Summary

- **Total Transaksi:** 2,512
- **Fraud:** 238 (9.47%)
- **Legitimate:** 2,274 (90.53%)
- **Features:** 16 variables

### Top 3 Fraud Indicators

1. **TransactionDuration** (22.35% influence)
2. **TransactionAmount** (20.93% influence)
3. **LoginAttempts** (15.01% influence)

---

##  FITUR APLIKASI

**Prediksi Manual** - Input 1 transaksi, dapatkan hasil fraud detection  
**Batch Prediction** - Upload CSV untuk ribuan transaksi sekaligus  
**Model Insights** - Lihat detail metrics, features, dan explanation  
**Dataset Overview** - Analisis data lengkap dengan statistik

---

## Informasi Teknis

**Model:** Random Forest Classifier  
**Training Data:** 2,009 transaksi (80%)  
**Testing Data:** 503 transaksi (20%)  
**Framework:** Scikit-learn + Streamlit  
**Deployment:** Streamlit Cloud

---

##  Dokumentasi

- **[README_ANALYSIS.md](README_ANALYSIS.md)** - Hasil analisis detail lengkap
- **[QUICK_ACCESS.md](QUICK_ACCESS.md)** - Akses cepat & ringkas
- **[PANDUAN_LENGKAP.md](PANDUAN_LENGKAP.md)** - Setup lokal (jika perlu)
- **[fraudDetection.ipynb](fraudDetection.ipynb)** - Notebook dengan penjelasan

---

##  Bagaimana Menggunakan (JIKA SETUP LOKAL)


>  **CATATAN:** Untuk akses cepat tanpa setup, gunakan link aplikasi di atas.
>
> Section ini hanya untuk mereka yang ingin setup lokal.

### 1. Aktivasi Environment

```bash
# Windows
.\env\Scripts\activate

# macOS/Linux
source env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan membuka di `http://localhost:8501`

---

### 4. Menghentikan Aplikasi
Tekan `CTRL + C` di terminal untuk menghentikan server Streamlit.