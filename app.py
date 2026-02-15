"""
EMERGENCY WINE QUALITY CLASSIFIER - WORKS WITHOUT PRE-TRAINED MODELS
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")

st.title("🍷 Wine Quality Classification System")
st.markdown("### Real-Time Machine Learning Predictions")
st.markdown("---")

st.sidebar.title("🎛️ Control Panel")
mode = st.sidebar.radio("Select Mode:", ["🔮 Make Predictions", "📊 Train & Compare Models"])

# Feature names
FEATURES = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
            'density', 'pH', 'sulphates', 'alcohol']

@st.cache_data
def train_models_on_data(df):
    """Train all models on uploaded data"""
    # Create binary target
    if 'quality_binary' in df.columns:
        y = df['quality_binary']
        X = df[FEATURES]
    elif 'quality' in df.columns:
        y = (df['quality'] >= 6).astype(int)
        X = df[FEATURES]
    else:
        return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC Score': roc_auc_score(y_test, y_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred)
        })
    
    return models, scaler, pd.DataFrame(results)

if mode == "🔮 Make Predictions":
    st.header("🔮 Upload Data & Get Predictions")
    
    st.info("📋 Upload your wine dataset (CSV) with quality labels to train models and make predictions")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {data.shape[0]} samples")
            
            with st.expander("👀 Preview Data"):
                st.dataframe(data.head(10))
            
            if st.button("🚀 Train Models & Predict", type="primary"):
                with st.spinner("Training models on your data..."):
                    
                    # Train models
                    models, scaler, results_df = train_models_on_data(data)
                    
                    if models is None:
                        st.error("❌ No quality column found. Please include 'quality' column.")
                    else:
                        st.success("✅ Models trained successfully!")
                        
                        # Select best model
                        best_idx = results_df['F1 Score'].idxmax()
                        best_model_name = results_df.loc[best_idx, 'Model']
                        best_model = models[best_model_name]
                        
                        st.markdown("---")
                        st.subheader("🏆 Best Model Selected")
                        st.info(f"**{best_model_name}** - F1 Score: {results_df.loc[best_idx, 'F1 Score']:.4f}")
                        
                        # Make predictions
                        if 'quality_binary' in data.columns:
                            y_true = data['quality_binary']
                            X = data[FEATURES]
                        else:
                            y_true = (data['quality'] >= 6).astype(int)
                            X = data[FEATURES]
                        
                        X_scaled = scaler.transform(X)
                        y_pred = best_model.predict(X_scaled)
                        y_proba = best_model.predict_proba(X_scaled)
                        
                        # Results
                        results = data.copy()
                        results['Predicted_Quality'] = ['Good' if p == 1 else 'Bad' for p in y_pred]
                        results['Confidence'] = [max(p) for p in y_proba]
                        results['Actual_Quality'] = ['Good' if a == 1 else 'Bad' for a in y_true]
                        results['Correct'] = ['✓' if p == a else '✗' for p, a in zip(y_pred, y_true)]
                        
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        st.dataframe(results)
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            acc = accuracy_score(y_true, y_pred)
                            st.metric("Accuracy", f"{acc:.2%}")
                        with col2:
                            good = (y_pred == 1).sum()
                            st.metric("Predicted Good", f"{good} ({good/len(y_pred)*100:.1f}%)")
                        with col3:
                            bad = (y_pred == 0).sum()
                            st.metric("Predicted Bad", f"{bad} ({bad/len(y_pred)*100:.1f}%)")
                        with col4:
                            avg_conf = results['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                        
                        # Confusion Matrix
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Confusion Matrix")
                            cm = confusion_matrix(y_true, y_pred)
                            fig, ax = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                       xticklabels=['Bad', 'Good'],
                                       yticklabels=['Bad', 'Good'])
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            st.pyplot(fig)
                        
                        with col2:
                            st.subheader("📈 Model Comparison")
                            st.dataframe(results_df.style.format({
                                'Accuracy': '{:.4f}',
                                'AUC Score': '{:.4f}',
                                'Precision': '{:.4f}',
                                'Recall': '{:.4f}',
                                'F1 Score': '{:.4f}',
                                'MCC': '{:.4f}'
                            }))
                        
                        # Download
                        st.markdown("---")
                        csv = results.to_csv(index=False)
                        st.download_button(
                            "📥 Download Predictions",
                            csv,
                            "wine_predictions.csv",
                            "text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your CSV has the 11 required features and a 'quality' column")

else:  # Train & Compare Models
    st.header("📊 Train Models & Compare Performance")
    
    st.info("Upload your wine dataset to train all 6 models and compare their performance")
    
    uploaded_file = st.file_uploader("Choose training dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {data.shape[0]} samples for training")
            
            if st.button("🎯 Train All Models", type="primary"):
                with st.spinner("Training 6 models... This may take a moment..."):
                    
                    models, scaler, results_df = train_models_on_data(data)
                    
                    if models is None:
                        st.error("❌ No quality column found")
                    else:
                        st.success("✅ All models trained!")
                        
                        st.markdown("---")
                        st.subheader("📊 Model Performance Comparison")
                        
                        # Styled dataframe
                        st.dataframe(
                            results_df.style.highlight_max(
                                axis=0,
                                subset=['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                                color='lightgreen'
                            ).format({
                                'Accuracy': '{:.4f}',
                                'AUC Score': '{:.4f}',
                                'Precision': '{:.4f}',
                                'Recall': '{:.4f}',
                                'F1 Score': '{:.4f}',
                                'MCC': '{:.4f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Best model
                        best_idx = results_df['F1 Score'].idxmax()
                        best_model = results_df.loc[best_idx, 'Model']
                        best_f1 = results_df.loc[best_idx, 'F1 Score']
                        
                        st.success(f"🏆 **Best Model:** {best_model} (F1 Score: {best_f1:.4f})")
                        
                        # Visualization
                        st.markdown("---")
                        st.subheader("📈 Visual Comparison")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        x = range(len(results_df))
                        width = 0.15
                        
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
                        
                        for i, metric in enumerate(metrics):
                            ax.bar([p + width*i for p in x], results_df[metric],
                                   width, label=metric, color=colors[i], alpha=0.8)
                        
                        ax.set_xlabel('Models', fontweight='bold')
                        ax.set_ylabel('Score', fontweight='bold')
                        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
                        ax.set_xticks([p + width*1.5 for p in x])
                        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                        ax.legend()
                        ax.grid(axis='y', alpha=0.3)
                        
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

st.markdown("---")
st.markdown("*Wine Quality Classification | M.Tech Data Science Assignment*")
