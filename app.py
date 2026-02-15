# ============================================================
# WINE QUALITY CLASSIFICATION - STREAMLIT WEB APPLICATION
# M.Tech Data Science Assignment
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8B0000;
    }
    .stButton>button {
        background-color: #8B0000;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE AND HEADER
# ============================================================
st.markdown('<p class="main-header">🍷 Wine Quality Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Model Comparison & Prediction Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Navigation
app_mode = st.sidebar.radio(
    "**Select Application Mode:**",
    ["📊 Model Performance", "🔮 Make Predictions", "ℹ️ About Dataset"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Guide:**
- **Model Performance**: Compare all 6 models
- **Make Predictions**: Test models with new data
- **About Dataset**: Learn about the data
""")

# ============================================================
# LOAD MODELS AND DATA
# ============================================================
@st.cache_resource
def load_all_resources():
    """Load all trained models, scaler, and results"""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'k-nearest_neighbor.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    # Load models
    for model_name, filename in model_files.items():
        filepath = os.path.join('model', filename)
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
    
    # Load scaler
    scaler_path = os.path.join('model', 'feature_scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    
    # Load results
    results_path = os.path.join('model', 'model_results.csv')
    results = pd.read_csv(results_path) if os.path.exists(results_path) else None
    
    # Load feature names
    features_path = os.path.join('model', 'feature_names.txt')
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = [line.strip() for line in f.readlines()]
    else:
        features = None
    
    return models, scaler, results, features

# Load resources
try:
    trained_models, data_scaler, performance_df, feature_names = load_all_resources()
    st.sidebar.success(f"✅ {len(trained_models)} models loaded!")
    if feature_names:
        st.sidebar.info(f"📋 {len(feature_names)} features ready")
except Exception as e:
    st.sidebar.error(f"❌ Error loading resources: {str(e)}")
    st.error("Could not load models. Please ensure the 'model' directory exists with all required files.")
    st.stop()

# ============================================================
# MODE 1: MODEL PERFORMANCE DASHBOARD
# ============================================================
if app_mode == "📊 Model Performance":
    st.header("📊 Model Performance Comparison Dashboard")
    
    if performance_df is not None:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Best Accuracy",
                f"{performance_df['Accuracy'].max():.4f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Best AUC Score",
                f"{performance_df['AUC Score'].max():.4f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "Best F1 Score",
                f"{performance_df['F1 Score'].max():.4f}",
                delta=None
            )
        
        with col4:
            best_idx = performance_df['F1 Score'].idxmax()
            best_model = performance_df.loc[best_idx, 'Model']
            st.metric(
                "Best Model",
                best_model,
                delta=None
            )
        
        st.markdown("---")
        
        # Detailed comparison table
        st.subheader("📋 Detailed Metrics Comparison")
        
        # Style the dataframe
        styled_df = performance_df.style.highlight_max(
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
        })
        
        st.dataframe(styled_df, use_container_width=True, height=280)
        
        # Winner announcement
        best_row = performance_df.loc[best_idx]
        st.success(f"""
        ### 🏆 Best Performing Model: **{best_model}**
        - **Accuracy:** {best_row['Accuracy']:.4f}
        - **F1 Score:** {best_row['F1 Score']:.4f}
        - **AUC Score:** {best_row['AUC Score']:.4f}
        - **MCC:** {best_row['MCC']:.4f}
        """)
        
        st.markdown("---")
        
        # Visualization section
        st.subheader("📈 Performance Visualizations")
        
        # Metric selector
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_metric = st.selectbox(
                "Select Metric to Visualize:",
                ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC'],
                index=4
            )
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#3498db' if i != best_idx else '#e74c3c' 
                  for i in range(len(performance_df))]
        
        bars = ax.bar(
            performance_df['Model'], 
            performance_df[selected_metric],
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5
        )
        
        ax.set_xlabel('Classification Models', fontweight='bold', fontsize=12)
        ax.set_ylabel(selected_metric, fontweight='bold', fontsize=12)
        ax.set_title(f'{selected_metric} Comparison Across All Models', 
                     fontweight='bold', fontsize=14, pad=20)
        ax.set_ylim([0, max(1, performance_df[selected_metric].max() * 1.1)])
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        
        # Radar chart for best model
        st.markdown("---")
        st.subheader(f"🎯 {best_model} - Multi-Metric Analysis")
        
        fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metrics = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
        values = best_row[metrics].values
        # Normalize MCC from [-1,1] to [0,1]
        values_normalized = values.copy()
        values_normalized[-1] = (values[-1] + 1) / 2
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_normalized = np.concatenate((values_normalized, [values_normalized[0]]))
        angles += angles[:1]
        
        ax2.plot(angles, values_normalized, 'o-', linewidth=2, color='#e74c3c')
        ax2.fill(angles, values_normalized, alpha=0.25, color='#e74c3c')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_title(f'{best_model} Performance Profile', 
                      fontweight='bold', fontsize=14, pad=20)
        ax2.grid(True)
        
        st.pyplot(fig2)
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")
        csv = performance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table (CSV)",
            data=csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )

# ============================================================
# MODE 2: MAKE PREDICTIONS
# ============================================================
elif app_mode == "🔮 Make Predictions":
    st.header("🔮 Wine Quality Prediction System")
    
    # Model selection
    st.subheader("1️⃣ Select Model")
    selected_model_name = st.selectbox(
        "Choose a classification model:",
        list(trained_models.keys()),
        index=1  # Default to Decision Tree (best model)
    )
    
    if selected_model_name:
        st.info(f"✓ Selected: **{selected_model_name}**")
    
    st.markdown("---")
    
    # Input method selection
    st.subheader("2️⃣ Choose Input Method")
    input_method = st.radio(
        "How would you like to provide wine data?",
        ["📝 Manual Input", "📁 Upload CSV File"],
        horizontal=True
    )
    
    # ============================================================
    # MANUAL INPUT
    # ============================================================
    if input_method == "📝 Manual Input":
        st.markdown("---")
        st.subheader("3️⃣ Enter Wine Properties")
        st.write("Provide the physicochemical properties of the wine:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fixed_acidity = st.number_input(
                "Fixed Acidity (g/dm³)", 
                min_value=4.0, max_value=16.0, value=10.0, step=0.1,
                help="Tartaric acid content"
            )
            volatile_acidity = st.number_input(
                "Volatile Acidity (g/dm³)", 
                min_value=0.1, max_value=1.6, value=0.5, step=0.01,
                help="Acetic acid content"
            )
            citric_acid = st.number_input(
                "Citric Acid (g/dm³)", 
                min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                help="Citric acid content"
            )
            residual_sugar = st.number_input(
                "Residual Sugar (g/dm³)", 
                min_value=0.9, max_value=16.0, value=6.0, step=0.1,
                help="Remaining sugar after fermentation"
            )
        
        with col2:
            chlorides = st.number_input(
                "Chlorides (g/dm³)", 
                min_value=0.01, max_value=0.62, value=0.08, step=0.001,
                help="Salt content"
            )
            free_sulfur = st.number_input(
                "Free Sulfur Dioxide (mg/dm³)", 
                min_value=1.0, max_value=72.0, value=30.0, step=1.0,
                help="Free SO₂ content"
            )
            total_sulfur = st.number_input(
                "Total Sulfur Dioxide (mg/dm³)", 
                min_value=6.0, max_value=289.0, value=100.0, step=1.0,
                help="Total SO₂ content"
            )
            density = st.number_input(
                "Density (g/cm³)", 
                min_value=0.990, max_value=1.010, value=0.997, step=0.0001,
                help="Wine density"
            )
        
        with col3:
            ph_value = st.number_input(
                "pH Level", 
                min_value=2.7, max_value=4.0, value=3.3, step=0.01,
                help="Acidity level (0-14 scale)"
            )
            sulphates = st.number_input(
                "Sulphates (g/dm³)", 
                min_value=0.3, max_value=2.0, value=0.6, step=0.01,
                help="Potassium sulphate content"
            )
            alcohol = st.number_input(
                "Alcohol (% vol)", 
                min_value=8.0, max_value=15.0, value=10.5, step=0.1,
                help="Alcohol percentage"
            )
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔍 Predict Wine Quality", type="primary", use_container_width=True):
            # Prepare input
            input_data = np.array([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur, total_sulfur, density, ph_value,
                sulphates, alcohol
            ]])
            
            # Scale input
            input_scaled = data_scaler.transform(input_data)
            
            # Make prediction
            selected_model = trained_models[selected_model_name]
            prediction = selected_model.predict(input_scaled)[0]
            prediction_proba = selected_model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Create columns for results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Used", selected_model_name)
            
            with col2:
                quality_label = "🟢 Good Quality" if prediction == 1 else "🔴 Bad Quality"
                st.metric("Prediction", quality_label)
            
            with col3:
                confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
                st.metric("Confidence", f"{confidence*100:.2f}%")
            
            with col4:
                recommendation = "✓ Recommended" if prediction == 1 else "⚠ Not Recommended"
                st.metric("Status", recommendation)
            
            # Probability visualization
            st.markdown("---")
            st.subheader("📈 Prediction Probabilities")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            classes = ['Bad Quality', 'Good Quality']
            colors = ['#e74c3c', '#2ecc71']
            
            bars = ax.barh(classes, prediction_proba, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Probability', fontweight='bold', fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_title('Classification Probabilities', fontweight='bold', fontsize=14)
            
            for i, (bar, prob) in enumerate(zip(bars, prediction_proba)):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                       f'{prob:.4f} ({prob*100:.2f}%)',
                       ha='left', va='center', fontweight='bold', fontsize=11)
            
            st.pyplot(fig)
            
            # Feature values summary
            with st.expander("📋 View Input Feature Summary"):
                input_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': input_data[0]
                })
                st.dataframe(input_df, use_container_width=True)
    
    # ============================================================
    # CSV UPLOAD
    # ============================================================
    else:
        st.markdown("---")
        st.subheader("3️⃣ Upload Wine Dataset")
        st.info("📋 **Required columns:** " + ", ".join(feature_names))
        
        uploaded_file = st.file_uploader(
            "Choose CSV file", 
            type=['csv'],
            help="Upload a CSV file with wine samples"
        )
        
        if uploaded_file is not None:
            try:
                # Read CSV
                test_data = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(test_data.head(10), use_container_width=True)
                
                # Check for quality column
                has_labels = 'quality' in test_data.columns or 'quality_binary' in test_data.columns
                
                if has_labels:
                    st.info("✓ Dataset contains labels - will evaluate model performance")
                else:
                    st.warning("⚠ No labels found - will only generate predictions")
                
                st.markdown("---")
                
                if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        
                        # Prepare data
                        if has_labels:
                            if 'quality_binary' in test_data.columns:
                                y_true = test_data['quality_binary']
                                X_test = test_data[feature_names]
                            else:
                                y_true = (test_data['quality'] >= 6).astype(int)
                                X_test = test_data[feature_names]
                        else:
                            X_test = test_data[feature_names]
                        
                        # Scale features
                        X_test_scaled = data_scaler.transform(X_test)
                        
                        # Make predictions
                        selected_model = trained_models[selected_model_name]
                        y_pred = selected_model.predict(X_test_scaled)
                        y_pred_proba = selected_model.predict_proba(X_test_scaled)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        # Add predictions to dataframe
                        results_df = test_data.copy()
                        results_df['Predicted_Quality'] = ['Good' if p == 1 else 'Bad' for p in y_pred]
                        results_df['Confidence'] = [max(prob) for prob in y_pred_proba]
                        
                        if has_labels:
                            results_df['Actual_Quality'] = ['Good' if a == 1 else 'Bad' for a in y_true]
                            results_df['Correct'] = ['✓' if p == a else '✗' 
                                                     for p, a in zip(y_pred, y_true)]
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            good_count = (y_pred == 1).sum()
                            st.metric("Predicted Good", f"{good_count} ({good_count/len(y_pred)*100:.1f}%)")
                        
                        with col2:
                            bad_count = (y_pred == 0).sum()
                            st.metric("Predicted Bad", f"{bad_count} ({bad_count/len(y_pred)*100:.1f}%)")
                        
                        with col3:
                            avg_conf = results_df['Confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_conf*100:.2f}%")
                        
                        # If labels available, show evaluation
                        if has_labels:
                            st.markdown("---")
                            st.subheader("📈 Model Evaluation")
                            
                            # Metrics
                            accuracy = accuracy_score(y_true, y_pred)
                            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Accuracy", f"{accuracy:.4f}")
                            with col2:
                                st.metric("AUC Score", f"{auc:.4f}")
                            with col3:
                                correct = (y_pred == y_true).sum()
                                st.metric("Correct Predictions", f"{correct}/{len(y_true)}")
                            
                            # Classification report
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("📋 Classification Report")
                                report = classification_report(
                                    y_true, y_pred, 
                                    target_names=['Bad Quality', 'Good Quality'],
                                    output_dict=True
                                )
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                            
                            with col2:
                                st.subheader("🎯 Confusion Matrix")
                                cm = confusion_matrix(y_true, y_pred)
                                
                                fig, ax = plt.subplots(figsize=(6, 5))
                                sns.heatmap(
                                    cm, annot=True, fmt='d', cmap='RdYlGn', 
                                    xticklabels=['Bad', 'Good'],
                                    yticklabels=['Bad', 'Good'],
                                    ax=ax, cbar_kws={'label': 'Count'}
                                )
                                ax.set_ylabel('Actual Quality', fontweight='bold')
                                ax.set_xlabel('Predicted Quality', fontweight='bold')
                                ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
                                st.pyplot(fig)
                        
                        # Download button
                        st.markdown("---")
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv,
                            file_name="wine_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct columns and format.")

# ============================================================
# MODE 3: ABOUT DATASET
# ============================================================
elif app_mode == "ℹ️ About Dataset":
    st.header("ℹ️ Wine Quality Dataset Information")
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", "600")
    with col2:
        st.metric("Features", "11")
    with col3:
        st.metric("Classes", "2 (Binary)")
    with col4:
        st.metric("Format", "CSV")
    
    st.markdown("---")
    
    # Features description
    st.subheader("🔬 Feature Descriptions")
    
    features_info = {
        'fixed_acidity': 'Tartaric acid content (g/dm³) - Non-volatile acids',
        'volatile_acidity': 'Acetic acid content (g/dm³) - Can lead to vinegar taste',
        'citric_acid': 'Citric acid content (g/dm³) - Adds freshness and flavor',
        'residual_sugar': 'Remaining sugar after fermentation (g/dm³) - Determines sweetness',
        'chlorides': 'Salt content (g/dm³) - Affects taste',
        'free_sulfur_dioxide': 'Free SO₂ content (mg/dm³) - Prevents microbial growth',
        'total_sulfur_dioxide': 'Total SO₂ content (mg/dm³) - Preservative',
        'density': 'Wine density (g/cm³) - Related to alcohol and sugar content',
        'pH': 'Acidity level (0-14 scale) - Lower pH means more acidic',
        'sulphates': 'Potassium sulphate content (g/dm³) - Wine additive',
        'alcohol': 'Alcohol percentage (% vol) - Alcohol content by volume'
    }
    
    feature_df = pd.DataFrame({
        'Feature': list(features_info.keys()),
        'Description': list(features_info.values())
    })
    
    st.dataframe(feature_df, use_container_width=True, height=450)
    
    st.markdown("---")
    
    # Classification task
    st.subheader("🎯 Classification Task")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Binary Classification:**
        - **Class 0 (Bad Quality):** Original quality rating < 6
        - **Class 1 (Good Quality):** Original quality rating ≥ 6
        
        **Original Quality Scale:** 0 (worst) to 10 (best)
        
        **Dataset Balance:**
        - Bad Quality: 308 samples (51.33%)
        - Good Quality: 292 samples (48.67%)
        - Status: ✓ Balanced dataset
        """)
    
    with col2:
        st.markdown("""
        **Data Quality:**
        - ✓ No missing values
        - ✓ All numeric features
        - ✓ Clean and preprocessed
        - ✓ Ready for modeling
        
        **Preprocessing Applied:**
        - Feature standardization (StandardScaler)
        - Binary target creation (quality ≥ 6)
        - Train-test split (75-25)
        - Stratified sampling
        """)
    
    st.markdown("---")
    
    # Model information
    st.subheader("🤖 Models Trained")
    
    model_info = {
        'Logistic Regression': 'Linear probabilistic classifier',
        'Decision Tree': 'Tree-based rule learning (Best Model)',
        'K-Nearest Neighbor': 'Instance-based learning (k=5)',
        'Naive Bayes': 'Probabilistic classifier using Bayes theorem',
        'Random Forest': 'Ensemble of 100 decision trees',
        'XGBoost': 'Gradient boosting ensemble'
    }
    
    model_df = pd.DataFrame({
        'Model': list(model_info.keys()),
        'Description': list(model_info.values())
    })
    
    st.dataframe(model_df, use_container_width=True)
    
    st.markdown("---")
    
    # References
    st.subheader("📚 References & Resources")
    st.markdown("""
    - **Source:** UCI Machine Learning Repository
    - **Domain:** Wine Chemistry & Quality Assessment
    - **Purpose:** Academic Machine Learning Assignment
    - **Tools:** Python, scikit-learn, pandas, Streamlit
    
    **Citation:**  
    M.Tech Data Science - Machine Learning Classification Models Comparison
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p><strong>Wine Quality Classification System</strong></p>
    <p>M.Tech Data Science | Machine Learning Assignment</p>
    <p>Developed with Python, scikit-learn, and Streamlit</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        🍷 © 2026 | For Academic Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)
