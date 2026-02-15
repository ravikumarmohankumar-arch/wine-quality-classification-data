# ============================================================
# WINE QUALITY CLASSIFICATION - COMPLETE TRAINING SCRIPT
# M.Tech Data Science Assignment
# ============================================================

# ============================================================
# STEP 1: IMPORT ALL NECESSARY LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try to import XGBoost, fall back to GradientBoosting if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available, using GradientBoostingClassifier instead")

import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("WINE QUALITY CLASSIFICATION - MODEL TRAINING")
print("="*70)
print("\n✓ All libraries imported successfully!\n")

# ============================================================
# STEP 2: LOAD DATASET
# ============================================================
print("Loading dataset...")
dataset_path = '/mnt/user-data/uploads/wine_quality_dataset.csv'
wine_data = pd.read_csv(dataset_path)

print(f"✓ Dataset loaded successfully!")
print(f"\nDataset Shape: {wine_data.shape}")
print(f"Total Features: {wine_data.shape[1] - 1}")
print(f"Total Instances: {wine_data.shape[0]}")
print(f"\nColumn Names: {wine_data.columns.tolist()}")

# Display basic information
print(f"\nFirst 5 rows:")
print(wine_data.head())

print(f"\nMissing Values Check:")
missing_values = wine_data.isnull().sum()
if missing_values.sum() == 0:
    print("✓ No missing values found - dataset is clean!")
else:
    print(missing_values[missing_values > 0])

print(f"\nQuality Distribution (Original):")
print(wine_data['quality'].value_counts().sort_index())

# ============================================================
# STEP 3: CREATE BINARY CLASSIFICATION TARGET
# ============================================================
print("\n" + "="*70)
print("CREATING BINARY CLASSIFICATION TARGET")
print("="*70)

# Convert quality to binary: quality >= 6 is 'Good' (1), else 'Bad' (0)
wine_data['quality_binary'] = (wine_data['quality'] >= 6).astype(int)

print("\nBinary Classification Rule:")
print("  - Quality >= 6 → Good Wine (Class 1)")
print("  - Quality < 6  → Bad Wine  (Class 0)")

print(f"\nBinary Target Distribution:")
print(wine_data['quality_binary'].value_counts())
print(f"\nClass Balance:")
balance = wine_data['quality_binary'].value_counts(normalize=True)
print(f"  Class 0 (Bad):  {balance[0]:.2%}")
print(f"  Class 1 (Good): {balance[1]:.2%}")

# ============================================================
# STEP 4: SEPARATE FEATURES AND TARGET
# ============================================================
print("\n" + "="*70)
print("PREPARING FEATURES AND TARGET")
print("="*70)

# Get feature columns (exclude original quality and binary target)
feature_columns = [col for col in wine_data.columns if col not in ['quality', 'quality_binary']]
X_features = wine_data[feature_columns]
y_target = wine_data['quality_binary']

print(f"\nFeature Columns ({len(feature_columns)}):")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i}. {col}")

print(f"\nFeature Matrix Shape: {X_features.shape}")
print(f"Target Vector Shape: {y_target.shape}")

# ============================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================
print("\n" + "="*70)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_features, 
    y_target, 
    test_size=0.25,  # 75% training, 25% testing
    random_state=42,  # For reproducibility
    stratify=y_target  # Maintain class balance in both sets
)

print(f"\nSplit Configuration:")
print(f"  Training Set: 75% ({X_train.shape[0]} instances)")
print(f"  Testing Set:  25% ({X_test.shape[0]} instances)")
print(f"  Random State: 42 (for reproducibility)")

print(f"\nTraining Set Class Distribution:")
print(y_train.value_counts())
print(f"\nTesting Set Class Distribution:")
print(y_test.value_counts())

# ============================================================
# STEP 6: FEATURE SCALING
# ============================================================
print("\n" + "="*70)
print("APPLYING FEATURE STANDARDIZATION")
print("="*70)

# Initialize StandardScaler
scaler_transformer = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler_transformer.fit_transform(X_train)
X_test_scaled = scaler_transformer.transform(X_test)

print("\n✓ Feature scaling completed using StandardScaler")
print("  Method: Zero mean and unit variance normalization")
print(f"  Mean of scaled training data: {X_train_scaled.mean():.6f}")
print(f"  Std of scaled training data: {X_train_scaled.std():.6f}")

# ============================================================
# STEP 7: DEFINE EVALUATION FUNCTION
# ============================================================
def calculate_performance_metrics(true_labels, predicted_labels, prediction_probabilities):
    """
    Compute all required evaluation metrics for classification.
    
    Parameters:
    -----------
    true_labels : array-like
        Actual target values
    predicted_labels : array-like
        Predicted class labels
    prediction_probabilities : array-like
        Predicted probabilities for positive class
    
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    metrics = {
        'Accuracy': accuracy_score(true_labels, predicted_labels),
        'AUC Score': roc_auc_score(true_labels, prediction_probabilities),
        'Precision': precision_score(true_labels, predicted_labels, zero_division=0),
        'Recall': recall_score(true_labels, predicted_labels, zero_division=0),
        'F1 Score': f1_score(true_labels, predicted_labels, zero_division=0),
        'MCC': matthews_corrcoef(true_labels, predicted_labels)
    }
    return metrics

print("\n✓ Evaluation function defined")
print("  Metrics: Accuracy, AUC, Precision, Recall, F1 Score, MCC")

# ============================================================
# STEP 8: INITIALIZE ALL MODELS
# ============================================================
print("\n" + "="*70)
print("INITIALIZING CLASSIFICATION MODELS")
print("="*70)

ml_models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, 
        max_iter=1000,
        solver='lbfgs'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, 
        max_depth=10,
        min_samples_split=5
    ),
    'K-Nearest Neighbor': KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean'
    ),
    'Naive Bayes': GaussianNB(
        var_smoothing=1e-9
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42, 
        n_estimators=100,
        max_depth=15
    ),
}

# Add XGBoost or GradientBoosting depending on availability
if XGBOOST_AVAILABLE:
    ml_models['XGBoost'] = XGBClassifier(
        random_state=42, 
        n_estimators=100,
        learning_rate=0.1,
        eval_metric='logloss'
    )
else:
    ml_models['XGBoost'] = GradientBoostingClassifier(
        random_state=42, 
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

print("\n✓ All 6 models initialized:")
for idx, model_name in enumerate(ml_models.keys(), 1):
    print(f"  {idx}. {model_name}")

# ============================================================
# STEP 9: TRAIN MODELS AND EVALUATE
# ============================================================
print("\n" + "="*70)
print("TRAINING MODELS AND COMPUTING METRICS")
print("="*70 + "\n")

performance_results = []

for model_name, model_obj in ml_models.items():
    print(f"Training: {model_name}...")
    
    # Train the model
    model_obj.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = model_obj.predict(X_test_scaled)
    y_pred_proba = model_obj.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate all metrics
    metrics = calculate_performance_metrics(y_test, y_pred, y_pred_proba)
    metrics['Model'] = model_name
    
    # Store results
    performance_results.append(metrics)
    
    # Display progress
    print(f"  ✓ {model_name} - Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1 Score']:.4f}")

print("\n✓ All models trained successfully!")

# ============================================================
# STEP 10: CREATE COMPARISON TABLE
# ============================================================
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("="*70 + "\n")

# Create DataFrame
results_dataframe = pd.DataFrame(performance_results)

# Reorder columns
column_sequence = ['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
results_dataframe = results_dataframe[column_sequence]

# Display table
print(results_dataframe.to_string(index=False))

# Find best model
best_model_index = results_dataframe['F1 Score'].idxmax()
best_model_info = results_dataframe.iloc[best_model_index]

print("\n" + "="*70)
print("BEST PERFORMING MODEL")
print("="*70)
print(f"\nModel Name: {best_model_info['Model']}")
print(f"Accuracy:   {best_model_info['Accuracy']:.4f}")
print(f"AUC Score:  {best_model_info['AUC Score']:.4f}")
print(f"Precision:  {best_model_info['Precision']:.4f}")
print(f"Recall:     {best_model_info['Recall']:.4f}")
print(f"F1 Score:   {best_model_info['F1 Score']:.4f}")
print(f"MCC:        {best_model_info['MCC']:.4f}")

# ============================================================
# STEP 11: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("CREATING PERFORMANCE VISUALIZATIONS")
print("="*70)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Comparison - Wine Quality Classification', 
             fontsize=16, fontweight='bold', y=1.00)

metrics_list = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for idx, metric in enumerate(metrics_list):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Create bar plot
    bars = ax.bar(results_dataframe['Model'], 
                   results_dataframe[metric], 
                   color=colors[idx], 
                   alpha=0.8,
                   edgecolor='black',
                   linewidth=1.2)
    
    ax.set_title(metric, fontweight='bold', fontsize=13, pad=10)
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: model_comparison.png")

# Create a second visualization - Grouped bar chart
fig2, ax2 = plt.subplots(figsize=(14, 8))

x_positions = np.arange(len(ml_models))
bar_width = 0.14
multiplier = 0

for idx, metric in enumerate(metrics_list):
    offset = bar_width * multiplier
    bars = ax2.bar(x_positions + offset, 
                   results_dataframe[metric], 
                   bar_width, 
                   label=metric,
                   alpha=0.8)
    multiplier += 1

ax2.set_xlabel('Models', fontweight='bold', fontsize=12)
ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
ax2.set_title('Comprehensive Model Performance Comparison', fontweight='bold', fontsize=14)
ax2.set_xticks(x_positions + bar_width * 2.5)
ax2.set_xticklabels(results_dataframe['Model'], rotation=45, ha='right')
ax2.legend(loc='upper left', fontsize=10)
ax2.set_ylim([0, 1.05])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('/home/claude/model_comparison_grouped.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: model_comparison_grouped.png")

# ============================================================
# STEP 12: SAVE MODELS
# ============================================================
print("\n" + "="*70)
print("SAVING TRAINED MODELS")
print("="*70)

# Create model directory
model_dir = '/home/claude/model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"\n✓ Created directory: {model_dir}")

# Save all trained models
print("\nSaving models...")
for model_name, model_obj in ml_models.items():
    # Create safe filename
    safe_filename = model_name.replace(' ', '_').lower() + '.pkl'
    model_path = os.path.join(model_dir, safe_filename)
    
    # Save model
    joblib.dump(model_obj, model_path)
    print(f"  ✓ {safe_filename}")

# Save the scaler
scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
joblib.dump(scaler_transformer, scaler_path)
print(f"  ✓ feature_scaler.pkl")

# Save results DataFrame
results_csv_path = os.path.join(model_dir, 'model_results.csv')
results_dataframe.to_csv(results_csv_path, index=False)
print(f"  ✓ model_results.csv")

# Save feature names
feature_names_path = os.path.join(model_dir, 'feature_names.txt')
with open(feature_names_path, 'w') as f:
    for feature in feature_columns:
        f.write(f"{feature}\n")
print(f"  ✓ feature_names.txt")

# Save dataset copy for reference
dataset_copy_path = os.path.join(model_dir, 'wine_quality_dataset.csv')
wine_data.to_csv(dataset_copy_path, index=False)
print(f"  ✓ wine_quality_dataset.csv (dataset copy)")

print("\n" + "="*70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n📁 All files saved in: {model_dir}/")
print(f"📊 Total models trained: {len(ml_models)}")
print(f"🏆 Best model: {best_model_info['Model']}")
print(f"✓ Ready for deployment!\n")
