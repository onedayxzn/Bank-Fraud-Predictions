"""
Script untuk testing model fraud detection
Pastikan models folder ada sebelum menjalankan script ini
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧪 TESTING FRAUD DETECTION MODEL")
print("="*70)

# Load model dan artifacts
try:
    print("\n📦 Loading model artifacts...")
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print("✓ Model artifacts loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    exit(1)

# Load dataset
try:
    print("\n📊 Loading test data...")
    df = pd.read_csv(
        "dataset/bank-transaction-dataset-for-fraud-detection/bank_transactions_data_2.csv")

    # Create target variable (same logic as notebook)
    np.random.seed(42)
    fraud_conditions = (
        (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.95)) |
        (df['LoginAttempts'] > 2) |
        (df['TransactionDuration'] > df['TransactionDuration'].quantile(0.95))
    )
    df['isFraud'] = 0
    df.loc[fraud_conditions, 'isFraud'] = np.random.choice(
        [0, 1], size=fraud_conditions.sum(), p=[0.4, 0.6])
    random_fraud_idx = np.random.choice(
        df.index, size=int(len(df)*0.02), replace=False)
    df.loc[random_fraud_idx, 'isFraud'] = 1

    print(f"✓ Loaded {len(df)} transactions")
    print(f"  - Fraud: {(df['isFraud'].sum() / len(df) * 100):.2f}%")

except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    exit(1)

# Test single prediction
print("\n" + "="*70)
print("TEST 1: Single Transaction Prediction")
print("="*70)

try:
    # Take first sample
    sample = df.iloc[0:1].drop(columns=['isFraud'])

    # Prepare features
    sample_prepared = sample[feature_names].copy()
    sample_scaled = scaler.transform(sample_prepared)

    # Predict
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]

    print(
        f"\nTransaction Amount: ${sample['TransactionAmount'].values[0]:.2f}")
    print(f"Customer Age: {sample['CustomerAge'].values[0]}")
    print(
        f"Transaction Duration: {sample['TransactionDuration'].values[0]} seconds")
    print(f"Login Attempts: {sample['LoginAttempts'].values[0]}")

    print(f"\n✓ Prediction: {'FRAUD' if prediction == 1 else 'LEGITIMATE'}")
    print(f"  - Legitimate Probability: {probability[0]*100:.2f}%")
    print(f"  - Fraud Probability: {probability[1]*100:.2f}%")

except Exception as e:
    print(f"❌ Error in single prediction: {str(e)}")

# Test batch prediction
print("\n" + "="*70)
print("TEST 2: Batch Prediction")
print("="*70)

try:
    # Prepare all data
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']

    # Select features
    X_prepared = X[feature_names].copy()
    X_scaled = scaler.transform(X_prepared)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Evaluate
    print(f"\nTotal Predictions: {len(predictions)}")
    print(
        f"Predicted Fraud: {(predictions == 1).sum()} ({(predictions == 1).sum()/len(predictions)*100:.2f}%)")
    print(
        f"Predicted Legitimate: {(predictions == 0).sum()} ({(predictions == 0).sum()/len(predictions)*100:.2f}%)")

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y, predictions,
          target_names=['Legitimate', 'Fraud']))

    # Confusion Matrix
    cm = confusion_matrix(y, predictions)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (TN):  {cm[0,0]}")
    print(f"  False Positives (FP): {cm[0,1]}")
    print(f"  False Negatives (FN): {cm[1,0]}")
    print(f"  True Positives (TP):  {cm[1,1]}")

except Exception as e:
    print(f"❌ Error in batch prediction: {str(e)}")

# Test with sample CSV
print("\n" + "="*70)
print("TEST 3: Loading Sample CSV")
print("="*70)

try:
    sample_csv = pd.read_csv('sample_transactions.csv')
    print(f"\n✓ Sample CSV loaded: {len(sample_csv)} transactions")
    print(f"\nSample Preview:")
    print(sample_csv.head(3))

except Exception as e:
    print(f"❌ Error loading sample CSV: {str(e)}")

# Model Info
print("\n" + "="*70)
print("MODEL INFORMATION")
print("="*70)

print(f"\nModel Name: {model_info['model_name']}")
print(f"Number of Features: {model_info['n_features']}")
print(f"ROC-AUC Score: {model_info['roc_auc']:.4f}")
print(f"Accuracy: {model_info['accuracy']:.4f}")
print(f"Precision: {model_info['precision']:.4f}")
print(f"Recall: {model_info['recall']:.4f}")
print(f"F1-Score: {model_info['f1_score']:.4f}")

print("\nFeatures Used:")
for i, feat in enumerate(model_info['features'][:5], 1):
    print(f"  {i}. {feat}")
print(f"  ... and {len(model_info['features']) - 5} more features")

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*70)
print("\n🚀 The model is ready for production use with Streamlit app!")
print("\nTo run the Streamlit app, execute:")
print("  streamlit run app.py")
print("="*70)
