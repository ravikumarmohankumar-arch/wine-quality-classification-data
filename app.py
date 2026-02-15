"""
Wine Quality Classification - Streamlit App
Minimal Working Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Classification System")
st.markdown("### Machine Learning Model Comparison Platform")
st.markdown("---")

# Sidebar
st.sidebar.title("🎛️ Control Panel")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["📊 Model Performance", "🔮 Make Predictions"]
)

# Load models
@st.cache_resource
def load_models():
    """Load all models and resources"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, file in model_files.items():
        path = os.path.join('model', file)
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except:
                pass
    
    # Load scaler
    scaler = None
    scaler_path = os.path.join('model', 'feature_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except:
            pass
    
    if scaler is None:
        scaler = StandardScaler()
    
    # Load results
    results = None
    results_path = os.path.join('model', 'model_results.csv')
    if os.path.exists(results_path):
        try:
            results = pd.read_csv(results_path)
        except:
            pass
    
    # Feature names
    features = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ]
    
    return models, scaler, results, features

# Load everything
models, scaler, results_df, features = load_models()

st.sidebar.success(f"✅ {len(models)} models loaded")

# MODE 1: Performance
if app_mode == "📊 Model Performance":
    st.header("📊 Model Performance Dashboard")
    
    if results_df is not None:
        st.dataframe(results_df, use_container_width=True)
        
        best_idx = results_df['F1 Score'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_f1 = results_df.loc[best_idx, 'F1 Score']
        
        st.success(f"🏆 Best Model: {best_model} (F1: {best_f1:.4f})")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df['Model'], results_df['Accuracy'])
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Comparison')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        st.warning("Model results not found")

# MODE 2: Predictions
elif app_mode == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")
    
    selected_model = st.selectbox("Choose Model:", list(models.keys()))
    
    st.markdown("---")
    
    input_method = st.radio("Input Method:", ["📝 Manual Input", "📁 Upload CSV"])
    
    if input_method == "📝 Manual Input":
        st.subheader("Enter Wine Properties")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 10.0, 0.1)
            volatile_acidity = st.number_input("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
            citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3, 0.01)
            residual_sugar = st.number_input("Residual Sugar", 0.9, 16.0, 6.0, 0.1)
        
        with col2:
            chlorides = st.number_input("Chlorides", 0.01, 0.62, 0.08, 0.001)
            free_sulfur = st.number_input("Free SO₂", 1.0, 72.0, 30.0, 1.0)
            total_sulfur = st.number_input("Total SO₂", 6.0, 289.0, 100.0, 1.0)
            density = st.number_input("Density", 0.990, 1.010, 0.997, 0.0001)
        
        with col3:
            ph = st.number_input("pH", 2.7, 4.0, 3.3, 0.01)
            sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6, 0.01)
            alcohol = st.number_input("Alcohol %", 8.0, 15.0, 10.5, 0.1)
        
        if st.button("🔍 Predict", type="primary"):
            input_data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol
            ]])
            
            try:
                input_scaled = scaler.transform(input_data)
            except:
                # Fit scaler if needed
                scaler.fit(input_data)
                input_scaled = input_data
            
            model = models[selected_model]
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            
            st.markdown("---")
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", selected_model)
            with col2:
                quality = "Good Quality" if prediction == 1 else "Bad Quality"
                st.metric("Prediction", quality)
            with col3:
                conf = proba[1] if prediction == 1 else proba[0]
                st.metric("Confidence", f"{conf*100:.1f}%")
    
    else:  # CSV Upload
        st.subheader("Upload Dataset")
        st.info("Required: 11 wine features (with or without quality column)")
        
        uploaded_file = st.file_uploader("Choose CSV", type=['csv'])
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded: {data.shape[0]} samples")
                st.dataframe(data.head())
                
                has_quality = 'quality' in data.columns or 'quality_binary' in data.columns
                
                if st.button("🚀 Run Predictions", type="primary"):
                    with st.spinner("Processing..."):
                        # Prepare features
                        if 'quality_binary' in data.columns:
                            y_true = data['quality_binary']
                            X = data[features]
                        elif 'quality' in data.columns:
                            y_true = (data['quality'] >= 6).astype(int)
                            X = data[features]
                        else:
                            y_true = None
                            X = data[features]
                        
                        # Scale
                        try:
                            X_scaled = scaler.transform(X)
                        except:
                            scaler.fit(X)
                            X_scaled = X
                        
                        # Predict
                        model = models[selected_model]
                        y_pred = model.predict(X_scaled)
                        y_proba = model.predict_proba(X_scaled)
                        
                        # Results
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        results = data.copy()
                        results['Predicted'] = ['Good' if p == 1 else 'Bad' for p in y_pred]
                        results['Confidence'] = [max(p) for p in y_proba]
                        
                        if has_quality:
                            results['Actual'] = ['Good' if a == 1 else 'Bad' for a in y_true]
                            results['Correct'] = ['✓' if p == a else '✗' for p, a in zip(y_pred, y_true)]
                        
                        st.dataframe(results)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            good = (y_pred == 1).sum()
                            st.metric("Predicted Good", f"{good} ({good/len(y_pred)*100:.1f}%)")
                        with col2:
                            bad = (y_pred == 0).sum()
                            st.metric("Predicted Bad", f"{bad} ({bad/len(y_pred)*100:.1f}%)")
                        with col3:
                            avg_conf = results['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                        
                        # If has labels
                        if has_quality:
                            st.markdown("---")
                            acc = accuracy_score(y_true, y_pred)
                            st.metric("Accuracy", f"{acc:.4f}")
                            
                            # Confusion matrix
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                       xticklabels=['Bad', 'Good'],
                                       yticklabels=['Bad', 'Good'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                        
                        # Download
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Wine Quality Classification | M.Tech Data Science*")
