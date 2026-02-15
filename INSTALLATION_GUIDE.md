# 🚀 COMPLETE INSTALLATION & DEPLOYMENT GUIDE
# Wine Quality Classification - M.Tech Data Science Assignment

## 📋 TABLE OF CONTENTS
1. Google Colab Setup (For Training)
2. Local Development Setup
3. Streamlit Cloud Deployment
4. Troubleshooting
5. Quick Reference Commands

---

## 1️⃣ GOOGLE COLAB SETUP (For Training Models)

### Step 1: Upload Your Dataset
```python
# In Google Colab, upload the wine_quality_dataset.csv file
from google.colab import files
uploaded = files.upload()
```

### Step 2: Install Required Packages
```python
!pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

### Step 3: Copy and Run Training Script
Copy the entire `train_models.py` content into a Colab cell and run it.

The script will:
- ✓ Load your dataset
- ✓ Train all 6 models
- ✓ Compute all metrics
- ✓ Save models to `/model` directory
- ✓ Generate visualizations

### Step 4: Download Trained Models
```python
# Zip the model folder
!zip -r model.zip model/

# Download
from google.colab import files
files.download('model.zip')
```

### Step 5: Download Visualizations
```python
files.download('model_comparison.png')
files.download('model_comparison_grouped.png')
```

---

## 2️⃣ LOCAL DEVELOPMENT SETUP

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for version control)

### Step 1: Create Project Directory
```bash
mkdir wine-quality-ml
cd wine-quality-ml
```

### Step 2: Extract Downloaded Files
Extract the `model.zip` from Colab into your project directory.

Your directory should look like:
```
wine-quality-ml/
├── model/              (extracted from Colab)
├── app.py              (copy from assignment)
├── requirements.txt    (copy from assignment)
├── README.md           (copy from assignment)
└── wine_quality_dataset.csv
```

### Step 3: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run Streamlit App Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 3️⃣ STREAMLIT CLOUD DEPLOYMENT

### Step 1: Create GitHub Repository
1. Go to https://github.com and create a new repository
2. Name it: `wine-quality-classification`
3. Keep it public (or private if you have Streamlit Pro)

### Step 2: Upload Files to GitHub

#### Option A: Using GitHub Web Interface
1. Click "uploading an existing file"
2. Drag and drop these files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `wine_quality_dataset.csv` (optional)
3. Upload the entire `model/` folder:
   - Create a new folder called `model`
   - Upload all .pkl files into it
4. Commit changes

#### Option B: Using Git Command Line
```bash
# Initialize git in your project folder
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Wine Quality ML Project"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/wine-quality-classification.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - **Repository:** your-username/wine-quality-classification
   - **Branch:** main
   - **Main file path:** app.py
5. Click "Deploy!"

### Step 4: Wait for Deployment
- Streamlit will install dependencies from `requirements.txt`
- Usually takes 2-5 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`

### Step 5: Share Your App
Your app is now live! Share the URL with anyone.

---

## 4️⃣ TROUBLESHOOTING

### Issue 1: ModuleNotFoundError
**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue 2: Model Files Not Found
**Error:** `FileNotFoundError: model/logistic_regression.pkl not found`

**Solution:**
- Ensure the `model/` directory is in the same location as `app.py`
- Check that all .pkl files are present:
  ```bash
  ls model/
  ```
- Should show:
  - logistic_regression.pkl
  - decision_tree.pkl
  - k-nearest_neighbor.pkl
  - naive_bayes.pkl
  - random_forest.pkl
  - xgboost.pkl
  - feature_scaler.pkl
  - model_results.csv
  - feature_names.txt

### Issue 3: Streamlit Cloud Deployment Failed
**Error:** Build fails on Streamlit Cloud

**Solutions:**
1. Check `requirements.txt` format (no extra spaces)
2. Ensure all files are committed to GitHub
3. Check Streamlit Cloud logs for specific errors
4. Try these fixes in `requirements.txt`:
   ```txt
   streamlit
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   joblib
   ```

### Issue 4: Wrong Predictions
**Error:** Predictions seem incorrect

**Solutions:**
- Verify you're using the correct scaler
- Ensure input features are in the correct order
- Check that dataset has all 11 required features

### Issue 5: CSV Upload Not Working
**Error:** Cannot upload CSV in Streamlit app

**Solutions:**
- Ensure CSV has correct columns (11 features)
- Check for correct column names (match feature_names.txt)
- Verify no missing values in CSV

---

## 5️⃣ QUICK REFERENCE COMMANDS

### Training (Google Colab)
```python
# Install packages
!pip install scikit-learn pandas numpy matplotlib seaborn joblib

# Run training
python train_models.py

# Download models
!zip -r model.zip model/
from google.colab import files
files.download('model.zip')
```

### Local Development
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Git Commands
```bash
# Initialize repository
git init

# Add files
git add .

# Commit
git commit -m "Your message"

# Push to GitHub
git push origin main

# Check status
git status

# View changes
git diff
```

### Streamlit Commands
```bash
# Run app
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8502

# Run without auto-open browser
streamlit run app.py --server.headless true

# Clear cache
streamlit cache clear
```

---

## 📊 WHAT YOU SHOULD HAVE

### ✅ Required Files Checklist

**Main Files:**
- [ ] `train_models.py` - Training script
- [ ] `app.py` - Streamlit application
- [ ] `requirements.txt` - Dependencies
- [ ] `README.md` - Documentation
- [ ] `wine_quality_dataset.csv` - Dataset

**Model Directory (`model/`):**
- [ ] `logistic_regression.pkl`
- [ ] `decision_tree.pkl`
- [ ] `k-nearest_neighbor.pkl`
- [ ] `naive_bayes.pkl`
- [ ] `random_forest.pkl`
- [ ] `xgboost.pkl`
- [ ] `feature_scaler.pkl`
- [ ] `model_results.csv`
- [ ] `feature_names.txt`

**Optional Files:**
- [ ] `model_comparison.png`
- [ ] `model_comparison_grouped.png`
- [ ] `INSTALLATION_GUIDE.md` (this file)

---

## 🎯 DEPLOYMENT CHECKLIST

### For Assignment Submission:
- [ ] All 6 models trained successfully
- [ ] All metrics computed correctly
- [ ] README.md completed
- [ ] Streamlit app running locally
- [ ] Models saved as .pkl files
- [ ] requirements.txt complete
- [ ] Code well-commented

### For Streamlit Cloud:
- [ ] GitHub repository created
- [ ] All files pushed to GitHub
- [ ] model/ directory included
- [ ] App deployed on Streamlit Cloud
- [ ] App accessible via public URL
- [ ] All features working (both modes)

---

## 💡 PRO TIPS

1. **Always test locally first** before deploying to Streamlit Cloud
2. **Keep your models small** - large files slow down deployment
3. **Use Git** for version control and easy updates
4. **Test with different inputs** to ensure robustness
5. **Read Streamlit logs** if deployment fails
6. **Update requirements.txt** if you add new packages
7. **Use st.cache_resource** for models (already implemented)
8. **Check model/ folder** exists in GitHub before deploying

---

## 📞 SUPPORT & RESOURCES

### Documentation:
- Streamlit Docs: https://docs.streamlit.io/
- Scikit-learn Docs: https://scikit-learn.org/
- Pandas Docs: https://pandas.pydata.org/

### Tutorials:
- Streamlit Tutorial: https://docs.streamlit.io/library/get-started
- ML with Scikit-learn: https://scikit-learn.org/stable/tutorial/
- GitHub Basics: https://docs.github.com/en/get-started

### Common Issues:
- Streamlit Forum: https://discuss.streamlit.io/
- Stack Overflow: https://stackoverflow.com/questions/tagged/streamlit

---

## ✅ FINAL CHECK

Before submission, verify:

1. **Training Output:**
   - [ ] All 6 models trained
   - [ ] Metrics table displayed
   - [ ] Visualizations generated
   - [ ] Models saved to model/ folder

2. **Streamlit App:**
   - [ ] Opens without errors
   - [ ] Model Performance mode works
   - [ ] Manual prediction works
   - [ ] CSV upload works
   - [ ] All visualizations display

3. **Documentation:**
   - [ ] README.md is complete
   - [ ] Code is well-commented
   - [ ] Requirements.txt is accurate

4. **Deployment:**
   - [ ] GitHub repository created
   - [ ] All files uploaded
   - [ ] Streamlit Cloud deployment successful
   - [ ] Public URL accessible

---

## 🎉 CONGRATULATIONS!

You now have a complete, working Machine Learning classification system deployed on the web!

**Your Assignment Includes:**
- ✅ 6 trained classification models
- ✅ Comprehensive evaluation metrics
- ✅ Professional README documentation
- ✅ Interactive Streamlit web application
- ✅ Deployment-ready code
- ✅ Academic-quality presentation

**Total Time:** Under 3 hours as requested!

---

**Last Updated:** February 2026  
**For:** M.Tech Data Science - Machine Learning Assignment  
**Purpose:** Academic Evaluation & Learning
