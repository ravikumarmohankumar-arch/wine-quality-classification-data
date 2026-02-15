# Gracefully handles missing scaler
try:
    input_scaled = data_scaler.transform(input_data)
except:
    st.warning("⚠️ Scaler not properly loaded. Using temporary scaling.")
    temp_scaler = StandardScaler()
    temp_scaler.fit(sample_data)
    input_scaled = temp_scaler.transform(input_data)
```

---

## ⚡ **STEP-BY-STEP FIX:**

### **1. Update app.py on GitHub**
```
https://github.com/ravikumarmohankumar-arch/wine-quality-classification-data
