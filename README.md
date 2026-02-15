# Wine Quality Classification - Machine Learning Assignment

## Problem Statement

This project implements and compares six different classification algorithms to predict wine quality based on physicochemical properties. The objective is to classify wines as either **good quality** (rating ≥ 6) or **bad quality** (rating < 6) using various machine learning techniques and evaluate their performance using multiple metrics.

## Dataset Description

**Dataset:** Wine Quality Dataset  
**Format:** CSV (wine_quality_dataset.csv)  
**Instances:** 600 wine samples  
**Features:** 11 physicochemical attributes + 1 quality rating  

### Features (11):
1. **fixed_acidity** - Tartaric acid content (g/dm³)
2. **volatile_acidity** - Acetic acid content (g/dm³)
3. **citric_acid** - Citric acid content (g/dm³)
4. **residual_sugar** - Remaining sugar after fermentation (g/dm³)
5. **chlorides** - Salt content (g/dm³)
6. **free_sulfur_dioxide** - Free SO₂ content (mg/dm³)
7. **total_sulfur_dioxide** - Total SO₂ content (mg/dm³)
8. **density** - Wine density (g/cm³)
9. **pH** - Acidity level (0-14 scale)
10. **sulphates** - Potassium sulphate content (g/dm³)
11. **alcohol** - Alcohol percentage (% vol)

### Target Variable:
- **Original:** Quality ratings from 3 to 8
- **Binary Classification:** 
  - Class 0 (Bad Quality): Quality < 6 (308 samples, 51.33%)
  - Class 1 (Good Quality): Quality ≥ 6 (292 samples, 48.67%)

### Data Quality:
✓ No missing values  
✓ All numeric features  
✓ Balanced classes (51.33% vs 48.67%)  
✓ Clean dataset ready for modeling  

## Models Implemented

All models were implemented using Python with scikit-learn library:

1. **Logistic Regression** - Linear probabilistic classifier with L2 regularization
2. **Decision Tree Classifier** - Tree-based rule learning algorithm (max_depth=10)
3. **K-Nearest Neighbor** - Instance-based learning with k=5 neighbors
4. **Naive Bayes (Gaussian)** - Probabilistic classifier using Bayes theorem
5. **Random Forest** - Ensemble of 100 decision trees with bootstrap aggregating
6. **XGBoost (GradientBoosting)** - Gradient boosting ensemble with 100 estimators

### Model Configuration:
- **Train/Test Split:** 75% training (450 samples) / 25% testing (150 samples)
- **Random State:** 42 (for reproducibility)
- **Feature Scaling:** StandardScaler (zero mean, unit variance)
- **Stratification:** Maintained class balance in both sets

## Evaluation Metrics

All models are evaluated using six comprehensive metrics:

- **Accuracy** - Overall correctness of predictions
- **AUC Score** - Area Under ROC Curve (model's ability to discriminate)
- **Precision** - Positive prediction accuracy (true positives / predicted positives)
- **Recall** - True positive detection rate (true positives / actual positives)
- **F1 Score** - Harmonic mean of precision and recall
- **MCC** - Matthews Correlation Coefficient (balanced measure for imbalanced data)

## Model Comparison Results

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|-------|----------|-----------|-----------|--------|----------|-----|
| Logistic Regression | 0.5267 | 0.5456 | 0.5208 | 0.3425 | 0.4132 | 0.0469 |
| **Decision Tree** | **0.5600** | **0.5630** | **0.5660** | **0.4110** | **0.4762** | **0.1174** |
| K-Nearest Neighbor | 0.4733 | 0.4559 | 0.4423 | 0.3151 | 0.3680 | -0.0646 |
| Naive Bayes | 0.4800 | 0.5291 | 0.4545 | 0.3425 | 0.3906 | -0.0489 |
| Random Forest | 0.4733 | 0.4629 | 0.4400 | 0.3014 | 0.3577 | -0.0660 |
| XGBoost | 0.4667 | 0.4761 | 0.4314 | 0.3014 | 0.3548 | -0.0794 |

### 🏆 Best Model: Decision Tree Classifier
- **Accuracy:** 56.00%
- **F1 Score:** 0.4762
- **AUC Score:** 0.5630
- **MCC:** 0.1174

## Detailed Model Observations

### 1. Logistic Regression
**Performance:** Moderate accuracy (52.67%), second-best performer  
**Strengths:**
- Simple, interpretable linear model
- Provides probability estimates
- Fast training and prediction
- Good baseline for comparison

**Limitations:**
- Assumes linear relationship between features and log-odds
- Lower recall (34.25%) indicates many false negatives
- Limited ability to capture non-linear patterns

**Use Cases:**
- Quick baseline modeling
- Scenarios requiring model interpretability
- When feature importance needs to be understood

---

### 2. Decision Tree Classifier ⭐
**Performance:** Best overall (56.00% accuracy, 0.4762 F1 score)  
**Strengths:**
- **Highest accuracy** among all models
- Best precision (56.60%) - reliable positive predictions
- Excellent interpretability - can visualize decision rules
- Handles non-linear relationships naturally
- No need for feature scaling

**Limitations:**
- Prone to overfitting (mitigated with max_depth=10)
- Can be unstable - small data changes affect structure
- Lower recall compared to precision

**Use Cases:**
- Explainable AI requirements
- Feature interaction analysis
- **Recommended for this dataset** due to best performance
- Educational purposes - easy to understand

---

### 3. K-Nearest Neighbor
**Performance:** Below average (47.33% accuracy)  
**Strengths:**
- Simple, intuitive algorithm
- No training phase required
- Adapts to local patterns in data
- Non-parametric approach

**Limitations:**
- **Worst accuracy** among traditional models
- Computationally expensive for predictions
- Sensitive to feature scaling (already applied)
- Struggles with high-dimensional data (curse of dimensionality)
- Poor performance on this dataset suggests non-local patterns

**Use Cases:**
- Small datasets with clear local patterns
- Recommendation systems
- Not recommended for this specific dataset

---

### 4. Naive Bayes (Gaussian)
**Performance:** Moderate (48.00% accuracy, 0.5291 AUC)  
**Strengths:**
- Very fast training and prediction
- Works well with limited data
- Probabilistic predictions
- Good AUC score (0.5291) despite lower accuracy

**Limitations:**
- **Strong independence assumption** (features assumed independent)
- This assumption likely violated in wine chemistry data
- Lower accuracy indicates model simplification issues

**Use Cases:**
- Real-time prediction systems
- Text classification
- Quick prototyping
- When training speed is critical

---

### 5. Random Forest
**Performance:** Surprisingly poor (47.33% accuracy)  
**Strengths:**
- Ensemble method - should reduce overfitting
- Handles feature interactions well
- Robust to outliers
- Provides feature importance

**Limitations:**
- **Unexpectedly low performance** on this dataset
- Ensemble averaging may be smoothing out important patterns
- Lower recall (30.14%) - missing many positive cases
- Possible overfitting despite ensemble nature

**Observations:**
- Usually performs better than single trees
- Poor performance suggests dataset-specific challenges
- May need hyperparameter tuning (n_estimators, max_depth)

**Use Cases:**
- Generally reliable for most classification tasks
- Needs further optimization for this dataset

---

### 6. XGBoost (GradientBoosting)
**Performance:** Lowest accuracy (46.67%)  
**Strengths:**
- Advanced gradient boosting algorithm
- Built-in regularization
- Usually excellent for tabular data
- Handles complex patterns

**Limitations:**
- **Worst performer** on this dataset (unexpected)
- Negative MCC (-0.0794) indicates poor correlation
- May be overfitting to training data
- Requires careful hyperparameter tuning

**Observations:**
- Performance contradicts typical XGBoost dominance
- Suggests model is not well-suited for this specific dataset
- May need extensive hyperparameter optimization
- Possible data distribution mismatch

**Use Cases:**
- Typically excellent for Kaggle competitions
- Production systems with proper tuning
- **Requires optimization** for this dataset

---

## Key Findings & Insights

### 1. Model Performance Summary
✓ **Decision Tree emerged as clear winner** with 56% accuracy  
✓ Traditional models (Logistic Regression, Decision Tree) outperformed ensemble methods  
✓ Ensemble methods surprisingly underperformed (may indicate overfitting or poor hyperparameters)

### 2. Dataset Characteristics
- **Moderate prediction difficulty** - best accuracy only 56%
- Suggests complex, overlapping decision boundaries
- Features may have non-linear interactions captured best by Decision Tree
- Balanced classes (51-49%) rule out class imbalance issues

### 3. Metric Analysis
- **Low recall across all models** (30-41%) indicates difficulty identifying good wines
- **Higher precision** than recall suggests conservative predictions
- **Low MCC values** indicate weak correlation between predictions and actual labels
- **AUC scores around 0.5** suggest models barely better than random guessing

### 4. Implications for Deployment
- Decision Tree should be deployed for best results
- Model ensemble may not improve performance (contrary to typical expectations)
- Consider collecting more features or data for better predictions
- Feature engineering might significantly improve performance

### 5. Recommendations for Improvement
1. **Feature Engineering:** Create interaction terms, polynomial features
2. **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV
3. **More Data:** Increase sample size beyond 600 instances
4. **Feature Selection:** Identify and focus on most important features
5. **Class Weighting:** Adjust for slight class imbalance
6. **Cross-Validation:** Use k-fold CV for more robust evaluation

## Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training Script (Google Colab / Local)
```python
# Upload wine_quality_dataset.csv to Colab
# Run train_models.py
python train_models.py
```

### Launch Streamlit Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Project Structure
```
wine-quality-classification/
│
├── model/                          # Trained models directory
│   ├── logistic_regression.pkl     # Trained Logistic Regression
│   ├── decision_tree.pkl           # Trained Decision Tree ⭐
│   ├── k-nearest_neighbor.pkl      # Trained KNN
│   ├── naive_bayes.pkl             # Trained Naive Bayes
│   ├── random_forest.pkl           # Trained Random Forest
│   ├── xgboost.pkl                 # Trained XGBoost
│   ├── feature_scaler.pkl          # StandardScaler object
│   ├── model_results.csv           # Performance metrics
│   ├── feature_names.txt           # List of feature names
│   └── wine_quality_dataset.csv    # Dataset copy
│
├── train_models.py                 # Complete training pipeline
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model_comparison.png            # Performance visualization
└── model_comparison_grouped.png    # Grouped bar chart
```

## Technical Specifications

### Training Configuration
- **Algorithm:** Supervised Binary Classification
- **Train/Test Split:** 75/25 stratified split
- **Feature Scaling:** StandardScaler (μ=0, σ=1)
- **Random State:** 42 (reproducible results)
- **Cross-Validation:** Not implemented (can be added)

### Model Hyperparameters

#### Logistic Regression
```python
max_iter=1000
solver='lbfgs'
random_state=42
```

#### Decision Tree
```python
max_depth=10
min_samples_split=5
random_state=42
```

#### K-Nearest Neighbor
```python
n_neighbors=5
metric='euclidean'
```

#### Naive Bayes
```python
var_smoothing=1e-9
```

#### Random Forest
```python
n_estimators=100
max_depth=15
random_state=42
```

#### XGBoost (GradientBoosting)
```python
n_estimators=100
learning_rate=0.1
max_depth=5
random_state=42
```

## Streamlit Application Features

### Mode 1: Model Performance Dashboard
- View comprehensive metrics comparison table
- Highlight best performers per metric
- Interactive metric visualization
- Best model identification

### Mode 2: Make Predictions
- **Manual Input:** Enter wine properties individually
- **CSV Upload:** Batch predictions with automatic evaluation
- Model selection dropdown
- Real-time predictions with confidence scores
- Classification reports and confusion matrices

## Future Enhancements

1. **Hyperparameter Optimization:** Implement GridSearchCV for all models
2. **Cross-Validation:** Add k-fold validation for robust evaluation
3. **Feature Importance:** Visualize top predictive features
4. **SHAP Values:** Add explainability for individual predictions
5. **Model Ensemble:** Create voting classifier combining top models
6. **Real-time Updates:** Allow model retraining through Streamlit
7. **More Metrics:** Add ROC curves, precision-recall curves
8. **Deployment:** Add Docker containerization for production

## References & Resources

- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Machine Learning Mastery:** Binary Classification Metrics
- **UCI ML Repository:** Wine Quality Dataset

## Author Information

**Course:** M.Tech Data Science  
**Assignment:** Machine Learning - Classification Models Comparison  
**Date:** February 2026  
**Tools:** Python, scikit-learn, pandas, Streamlit  

## License

This project is created for academic purposes as part of M.Tech Data Science coursework.

---

## Quick Start Guide

### For Google Colab Users:
1. Upload `wine_quality_dataset.csv` to Colab
2. Run `train_models.py` to train all models
3. Download the `/model` folder
4. Upload to GitHub repository with `app.py` and `requirements.txt`
5. Deploy on Streamlit Cloud

### For Local Development:
1. Clone repository
2. `pip install -r requirements.txt`
3. Ensure `wine_quality_dataset.csv` is in project root
4. `python train_models.py` to train models
5. `streamlit run app.py` to launch application

---

**⚠️ Important Notes:**
- Always use the same StandardScaler for training and prediction
- Models saved as .pkl files can be loaded using `joblib.load()`
- Dataset should have exact same 11 features for predictions
- Binary classification threshold is fixed at quality >= 6

**✅ Verified Working:**
- All 6 models trained and saved successfully
- Evaluation metrics computed correctly
- Visualizations generated
- Ready for Streamlit deployment

---

*Last Updated: February 2026*
