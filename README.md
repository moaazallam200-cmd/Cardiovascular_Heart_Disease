# Cardiovascular Disease Risk Prediction Using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moaazallam200-cmd/Cardiovascular_Heart_Disease/blob/main/cardiovascular_disease_ml.ipynb.)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 1. Project Overview & Business Problem
Cardiovascular diseases (CVDs) are the leading cause of mortality globally, making early risk stratification a critical priority for healthcare providers, clinical triage systems, and health insurance analytics. However, real-world medical examination records often contain data-entry corruptions and non-linear risk interactions.

**Objective:** Build a reproducible, leakage-free machine learning classification pipeline in Python to predict the presence of cardiovascular disease (`cardio`) from routine clinical and lifestyle examination data, prioritizing **ROC-AUC** and **Recall** to support preventive screening.

---

## 2. Dataset Description
The dataset (`cardio_train.csv`) consists of **70,000 patient examination records** across **12 features and 1 binary target variable** (`cardio`):

| Feature | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `id` | Index | Patient record identifier (dropped prior to modeling) | Integer |
| `age` | Objective | Patient age at examination | Days (converted to `age_years`) |
| `gender` | Objective | Patient gender | `1`: Female, `2`: Male |
| `height` | Objective | Patient height | `cm` |
| `weight` | Objective | Patient weight | `kg` |
| `ap_hi` | Examination | Systolic blood pressure | `mmHg` |
| `ap_lo` | Examination | Diastolic blood pressure | `mmHg` |
| `cholesterol` | Examination | Cholesterol level tier | `1`: Normal, `2`: Above Normal, `3`: Well Above Normal |
| `gluc` | Examination | Glucose level tier | `1`: Normal, `2`: Above Normal, `3`: Well Above Normal |
| `smoke` | Subjective | Current smoking status | `0`: No, `1`: Yes |
| `alco` | Subjective | Alcohol intake status | `0`: No, `1`: Yes |
| `active` | Subjective | Physical activity status | `0`: No, `1`: Yes |
| **`cardio`** | **Target** | **Presence of cardiovascular disease** | **`0`: Absence (50.03%), `1`: Presence (49.97%)** |

---

## 3. Methodology & Engineering Workflow

### A. Data Cleaning & Physiological Outlier Filtering
Initial inspection revealed zero missing values (`0` nulls), but descriptive statistics exposed severe data-entry anomalies in blood pressure (`ap_hi` ranging from `-150` to `16,020 mmHg`; `ap_lo` ranging from `-70` to `11,000 mmHg`) and anthropometrics (`height` down to `55 cm`, `weight` down to `10 kg`).
To ensure clinical validity, records were filtered using strict physiological bounds:
- Systolic BP: `80 <= ap_hi <= 250 mmHg`
- Diastolic BP: `50 <= ap_lo <= 150 mmHg`
- Pulse Pressure Validity: `ap_hi > ap_lo` (Systolic must strictly exceed Diastolic)
- Anthropometrics: `100 <= height <= 250 cm` and `weight >= 30 kg`

**Cleaning Result:** Removed **1,387 corrupted records (1.98%)**, retaining **68,613 valid patient records**.

### B. Feature Engineering
- **`age_years`**: Converted `age` from days to integer years (`age / 365.25`).
- **`bmi`**: Calculated Body Mass Index (`weight / (height / 100) ** 2`) to capture obesity-related cardiovascular risk.

### C. Preprocessing & Leakage Prevention
- **Train/Test Split:** Stratified 80/20 split (`54,890` training records / `13,723` testing records) preserving the balanced target distribution.
- **Scikit-Learn `ColumnTransformer` & `Pipeline`:** Applied `StandardScaler` to continuous numerical features and `OneHotEncoder(drop='first')` to ordinal clinical categories (`cholesterol`, `gluc`) strictly inside the cross-validation loop to prevent data leakage.

---

## 4. Model Comparison & Evaluation Results

Three classification algorithms were benchmarked against a `DummyClassifier(strategy='most_frequent')` baseline (**50.53% Test Accuracy**) using **5-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`):

| Model | CV Accuracy | CV Recall | CV ROC-AUC |
| :--- | :---: | :---: | :---: |
| **HistGradientBoostingClassifier** | **0.7345** | 0.6885 | **0.8003** |
| **Logistic Regression** | 0.7278 | 0.6643 | 0.7914 |
| **Random Forest Classifier** | 0.7009 | **0.6908** | 0.7586 |
| *Dummy Baseline (Most Frequent)* | *0.5053* | *0.0000* | *0.5000* |

### Final Hold-Out Test Set Performance (`HistGradientBoostingClassifier`, $N = 13,723$)
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No CVD)** | 0.71 | 0.78 | 0.74 | 6,934 |
| **1 (CVD Present)** | 0.75 | 0.68 | 0.71 | 6,789 |
| **Macro / Weighted Avg** | **0.73** | **0.73** | **0.73** | **13,723** |

---

## 5. Key Findings & Clinical Recommendations
1. **Blood Pressure Is the Primary Risk Driver:** Systolic (`ap_hi`, `r = 0.43`) and diastolic (`ap_lo`, `r = 0.34`) blood pressure exhibit the strongest linear correlation with cardiovascular disease presence.
2. **Age & Cholesterol Compound Risk:** Median patient age is significantly higher in the CVD-positive cohort (~56 years vs. ~52 years in the negative cohort), and patients in Cholesterol Tier 3 ("Well Above Normal") show a ~3x higher disease prevalence than Tier 1.
3. **Clinical Triage Recommendation:** While the default `0.50` probability threshold yields balanced accuracy (`73%`) and precision (`75%`), clinical screening deployments should lower the decision threshold (e.g., to `0.35–0.40`) to prioritize **Recall (>80%)**, ensuring high-risk patients are routed for secondary diagnostic testing.

---

## 6. How to Run This Project

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook notebooks/cardiovascular_disease_ml.ipynb
```
