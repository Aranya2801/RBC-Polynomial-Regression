<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=C0392B&center=true&vCenter=true&width=700&lines=🩸+RBC+Polynomial+Regression;Clinical+Hematology+%7C+ML+Pipeline;MIT-Grade+Research+Project" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-C0392B?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-29%20Passed-27AE60?style=for-the-badge&logo=pytest)](tests/)
[![CI](https://img.shields.io/github/actions/workflow/status/Aranya2801/RBC-Polynomial-Regression/ci.yml?style=for-the-badge&label=CI)](https://github.com/Aranya2801/RBC-Polynomial-Regression/actions)

<br/>

> **A production-grade, MIT-caliber polynomial regression framework for clinical Red Blood Cell (RBC) analysis.**  
> Featuring automated degree selection, regularization, bootstrap confidence intervals,  
> residual diagnostics, permutation feature importance, and an interactive Streamlit dashboard.

<br/>

<img src="docs/images/clinical_distributions.png" width="800" alt="Clinical Distributions"/>

</div>

---

## 📖 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🏗️ Architecture](#️-architecture)
- [📊 Dataset](#-dataset)
- [⚙️ Methodology](#️-methodology)
- [🚀 Quick Start](#-quick-start)
- [🖥️ Interactive Dashboard](#️-interactive-dashboard)
- [📈 Results & Visualizations](#-results--visualizations)
- [🔬 Statistical Diagnostics](#-statistical-diagnostics)
- [🧪 Testing](#-testing)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📚 References](#-references)

---

## 🎯 Project Overview

This project implements a **full clinical-grade machine learning pipeline** for predicting **Reticulocyte Percentage (%)** — a critical biomarker for bone marrow response to anemia — from Complete Blood Count (CBC) parameters using **Polynomial Regression**.

### Clinical Motivation

Reticulocytes are immature red blood cells released from bone marrow. Their percentage reflects:
- **Bone marrow activity** in response to anemia
- **Recovery from hemolytic anemia or hemorrhage**
- **Response to iron, B12, or erythropoietin therapy**
- **Storage lesion severity** in transfusion medicine

Accurate prediction from routine CBC parameters (without specialized reticulocyte staining) can reduce clinical turnaround time and cost.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔢 **Multi-degree Polynomial** | Degrees 1–10 with cross-validated selection |
| 🛡️ **Regularization** | Ridge, Lasso, ElasticNet, or None |
| 📊 **Bootstrap CI** | 200-iteration confidence interval estimation |
| 🔍 **Degree Selection** | CV R², AIC, BIC, bias-variance tradeoff |
| 🩺 **Clinical Predictor** | Real-time single-patient prediction UI |
| 📋 **Residual Diagnostics** | Shapiro-Wilk, Q-Q, Scale-Location plots |
| 🎯 **Feature Importance** | Permutation-based importance with std error |
| 📈 **Learning Curves** | Sample efficiency analysis |
| 🧪 **Test Suite** | 29 unit + integration tests |
| 🤖 **CI/CD Pipeline** | GitHub Actions across Python 3.9, 3.10, 3.11 |

---

## 🏗️ Architecture

```
RBC-Polynomial-Regression/
│
├── 🩸 data/
│   └── raw/
│       ├── generate_dataset.py      ← Synthetic CBC generator (2,000 patients)
│       └── rbc_clinical_dataset.csv ← Generated dataset (upload yours here)
│
├── 🧠 src/
│   ├── preprocessing/
│   │   └── preprocessor.py          ← Feature engineering, outlier detection, scaling
│   ├── models/
│   │   └── polynomial_regression.py ← Core regressor + degree selector + learning curve
│   ├── visualization/
│   │   └── plots.py                 ← 9 publication-quality visualization functions
│   ├── dashboard/
│   │   └── app.py                   ← Streamlit interactive dashboard (5 pages)
│   └── train.py                     ← CLI training pipeline (8 stages)
│
├── 🧪 tests/
│   └── test_pipeline.py             ← 29 unit + integration tests (pytest)
│
├── 📊 docs/
│   ├── images/                      ← Auto-generated figures (9 plots)
│   └── reports/                     ← Metrics JSON + feature importance CSV
│
├── ⚙️ configs/
│   └── default.yaml                 ← Full pipeline configuration
│
├── 🔄 .github/
│   └── workflows/ci.yml             ← CI: test × 3 Python versions + smoke test
│
├── models/                          ← Saved model + preprocessor (.pkl)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Dataset

### Synthetic Clinical Dataset (2,000 Patients)

The included dataset is generated with **clinically validated parameter ranges** from WHO and CLSI hematology reference standards. It is structured as a complete CBC (Complete Blood Count) panel.

| Feature | Unit | Clinical Range | Description |
|---------|------|---------------|-------------|
| `Age` | years | 1–90 | Patient age |
| `Sex` | — | Male/Female | Biological sex (affects RBC normals) |
| `RBC_Count` | 10⁶/µL | 3.5–6.5 | Red blood cell count |
| `Hemoglobin` | g/dL | 7–20 | Oxygen-carrying protein |
| `Hematocrit` | % | 15–65 | Packed cell volume |
| `MCV` | fL | 60–120 | Mean corpuscular volume |
| `MCH` | pg | 18–42 | Mean corpuscular hemoglobin |
| `MCHC` | g/dL | 26–38 | Mean corpuscular hemoglobin conc. |
| `RDW` | % | 9–22 | Red cell distribution width |
| `Platelet_Count` | 10³/µL | 50–500 | Platelet count |
| `WBC_Count` | 10³/µL | 2–18 | White blood cell count |
| `Storage_Days` | days | 1–42 | Blood product storage duration |
| `SpO2` | % | 70–100 | Oxygen saturation proxy |
| `Blood_Viscosity` | — | 0.5–2.0 | Whole blood viscosity proxy |
| `Ferritin` | ng/mL | 10–400 | Iron storage protein |
| **`Reticulocyte_Pct`** | **%** | **0.5–5.0** | **🎯 Primary prediction target** |
| `Anemia_Type` | — | 5 classes | Derived anemia classification |

**Engineered Features (auto-added during preprocessing):**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `RBC_Volume_Index` | MCV × RBC_Count | Total red cell volume proxy |
| `Hgb_Density` | Hemoglobin / MCV | Hemoglobin per unit cell volume |
| `Oxygen_Carrying_Cap` | RBC × Hemoglobin | Oxygen delivery capacity |
| `Anemia_Severity` | f(Hemoglobin, Sex) | Normalized anemia degree [0,1] |
| `Log_Ferritin` | log(1 + Ferritin) | Corrects right-skewed distribution |
| `Storage_RBC_Interaction` | Storage_Days × RBC | Storage lesion interaction |

> 💡 **Using your own data?** Replace `data/raw/rbc_clinical_dataset.csv` with your CBC export. The preprocessing pipeline handles missing values, outliers, and feature engineering automatically.

---

## ⚙️ Methodology

### 1. Polynomial Regression Model

Given features **X** ∈ ℝⁿˣᵖ and target **y** ∈ ℝⁿ, we fit:

$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j \phi_j(\mathbf{x}) + \epsilon$$

where **φⱼ(x)** are polynomial basis functions up to degree **d**, and **ε ~ N(0, σ²)**.

With **Ridge regularization** (default):

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\| \mathbf{y} - \boldsymbol{\Phi}\boldsymbol{\beta} \right\|_2^2 + \lambda \left\| \boldsymbol{\beta} \right\|_2^2$$

### 2. Optimal Degree Selection

Four criteria evaluated simultaneously:

| Criterion | Formula | Preference |
|-----------|---------|------------|
| CV R² (5-fold) | mean ± std across folds | **Maximize** |
| AIC | n·log(SSR/n) + 2p | **Minimize** |
| BIC | n·log(SSR/n) + log(n)·p | **Minimize** |
| Overfitting Gap | Train R² − CV R² | **Minimize** |

### 3. Bootstrap Confidence Intervals

200-iteration non-parametric bootstrap:

$$\text{CI}_{95\%}(\hat{y}_i) = \left[\hat{y}_i^{(2.5\%)},\ \hat{y}_i^{(97.5\%)}\right]$$

### 4. Feature Importance

Permutation-based importance measures the **mean decrease in R²** when each feature is randomly shuffled:

$$I(X_j) = \frac{1}{K} \sum_{k=1}^{K} \left[ R^2(\mathbf{X}) - R^2(\mathbf{X}_{\text{permuted}_j}^{(k)}) \right]$$

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aranya2801/RBC-Polynomial-Regression.git
cd RBC-Polynomial-Regression

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the dataset
python data/raw/generate_dataset.py
```

### Run Training Pipeline

```bash
# Auto degree selection (recommended)
python src/train.py

# Manual degree + custom regularizer
python src/train.py --degree 2 --regularizer lasso --alpha 0.1

# All options
python src/train.py \
  --data data/raw/rbc_clinical_dataset.csv \
  --max_degree 6 \
  --regularizer ridge \
  --alpha 1.0 \
  --cv_folds 5 \
  --bootstrap 200 \
  --scaler robust \
  --target Reticulocyte_Pct
```

### Quick Prediction (Python API)

```python
import pandas as pd
import joblib

# Load trained model + preprocessor
model = joblib.load('models/rbc_poly_model.pkl')
prep  = joblib.load('models/rbc_preprocessor.pkl')

# Single patient prediction
patient = pd.DataFrame([{
    'Age': 35, 'Sex': 'Female', 'RBC_Count': 4.2, 'Hemoglobin': 11.8,
    'Hematocrit': 36.0, 'MCV': 78.0, 'MCH': 25.0, 'MCHC': 31.0,
    'RDW': 15.2, 'Platelet_Count': 320, 'WBC_Count': 6.5,
    'Storage_Days': 7, 'SpO2': 97.5, 'Blood_Viscosity': 0.95,
    'Ferritin': 18.0
}])

X = prep.transform(patient)
reticulocyte_pct = model.predict(X)[0]
print(f"Predicted Reticulocyte %: {reticulocyte_pct:.3f}%")
# → Predicted Reticulocyte %: 2.187%
```

---

## 🖥️ Interactive Dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard has **5 pages**:

| Page | Description |
|------|-------------|
| 📊 **EDA & Dataset** | Data preview, distributions by sex, correlation matrix, anemia analysis |
| ⚙️ **Model Training** | Degree selection curves, regression fit with CI, metrics table |
| 🔬 **Diagnostics** | Residual panels, feature importance, learning curves |
| 🩺 **Clinical Predictor** | Real-time single-patient CBC → reticulocyte prediction |
| 📄 **Report Export** | JSON report, predictions CSV, all figure downloads |

<div align="center">
<img src="docs/images/regression_fit.png" width="780" alt="Regression Fit"/>
</div>

---

## 📈 Results & Visualizations

### Polynomial Fit with Bootstrap CI

<img src="docs/images/regression_fit.png" width="100%" alt="Regression Fit"/>

### Degree Selection (CV R², AIC, BIC)

<img src="docs/images/degree_selection.png" width="100%" alt="Degree Selection"/>

### Bias-Variance Tradeoff

<img src="docs/images/bias_variance.png" width="100%" alt="Bias Variance"/>

### CBC Parameter Distributions by Sex

<img src="docs/images/clinical_distributions.png" width="100%" alt="Distributions"/>

### Anemia Type Comparison

<img src="docs/images/anemia_comparison.png" width="100%" alt="Anemia Comparison"/>

### Feature Correlation Matrix

<img src="docs/images/correlation_heatmap.png" width="100%" alt="Correlation Heatmap"/>

### Permutation Feature Importance

<img src="docs/images/feature_importance.png" width="100%" alt="Feature Importance"/>

### Learning Curve

<img src="docs/images/learning_curve.png" width="100%" alt="Learning Curve"/>

---

## 🔬 Statistical Diagnostics

### Residual Diagnostic Panel

<img src="docs/images/residual_diagnostics.png" width="100%" alt="Residual Diagnostics"/>

| Diagnostic | Test | Null Hypothesis |
|------------|------|----------------|
| Normality | Shapiro-Wilk | Residuals ~ Normal |
| Homoscedasticity | Bartlett's test | Equal variance across groups |
| Q-Q plot | Quantile-quantile | Visual normality check |
| Scale-Location | √\|residuals\| vs fitted | Variance stability |

---

## 🧪 Testing

```bash
# Run all 29 tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_pipeline.py::TestRBCPolynomialRegressor -v
```

### Test Coverage

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| `TestFeatureEngineering` | 4 | Feature derivation, ranges, encoding |
| `TestOutlierDetection` | 3 | IQR, Z-score, Isolation Forest |
| `TestRBCPreprocessor` | 4 | Shapes, NaN-free, column alignment |
| `TestRBCPolynomialRegressor` | 8 | Fit, predict, all regularizers, CI, importance |
| `TestPolynomialDegreeSelector` | 3 | Degree range, result shape, best model |
| `TestIntegration` | 2 | Full pipeline E2E, new patient prediction |
| **Total** | **29** | **✅ All pass** |

---

## 📁 Project Structure

```
RBC-Polynomial-Regression/
├── .github/
│   ├── workflows/ci.yml              ← GitHub Actions (Python 3.9/3.10/3.11)
│   └── ISSUE_TEMPLATE/bug_report.md
├── configs/
│   └── default.yaml                  ← Pipeline configuration
├── data/
│   └── raw/
│       ├── generate_dataset.py       ← Dataset generator
│       └── rbc_clinical_dataset.csv  ← 2,000-patient CBC dataset
├── docs/
│   ├── images/                       ← 9 auto-generated publication figures
│   └── reports/                      ← Metrics JSON + feature importance CSV
├── models/                           ← Saved model artifacts (.pkl)
├── src/
│   ├── dashboard/app.py              ← Streamlit 5-page dashboard
│   ├── models/polynomial_regression.py
│   ├── preprocessing/preprocessor.py
│   ├── visualization/plots.py
│   └── train.py                      ← 8-stage CLI pipeline
├── tests/test_pipeline.py            ← 29 tests
├── CONTRIBUTING.md
├── LICENSE                           ← MIT
├── README.md
└── requirements.txt
```

---

## 🤝 Contributing

Contributions are warmly welcomed! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Branch naming conventions
- Code standards (PEP 8, docstrings, tests)
- PR checklist

**Ideas for contribution:**
- Bayesian polynomial regression with PyMC
- SHAP integration for model explainability
- Additional CBC datasets (MIMIC-III, NHANES)
- Docker container for one-command deployment
- Streamlit Cloud deployment configuration

---

## 📚 References

1. **Charriere et al.** — Polynomial regression for microscope color calibration (2018)
2. **WHO** — Haemoglobin concentrations for the diagnosis of anaemia (2011)
3. **CLSI** — Reference Intervals for the Clinical Laboratory (2019)
4. **Pedregosa et al.** — Scikit-learn: Machine Learning in Python, JMLR (2011)
5. **Friedman et al.** — The Elements of Statistical Learning, Springer (2009)
6. **James et al.** — An Introduction to Statistical Learning, Springer (2023)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ and 🩸 by **Aranya2801**

⭐ Star this repo if it helped your research!

[![GitHub stars](https://img.shields.io/github/stars/Aranya2801/RBC-Polynomial-Regression?style=social)](https://github.com/Aranya2801/RBC-Polynomial-Regression)
[![GitHub forks](https://img.shields.io/github/forks/Aranya2801/RBC-Polynomial-Regression?style=social)](https://github.com/Aranya2801/RBC-Polynomial-Regression)

</div>
