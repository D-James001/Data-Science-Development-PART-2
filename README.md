# Data Science Development — Part 2: Machine Learning Workflow & Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4c72b0.svg)](https://seaborn.pydata.org/)
[![Curriculum](https://img.shields.io/badge/AltSchool%20Africa-Data%20Science%20Semester%203-darkgreen.svg)](https://altschoolafrica.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Data Science and Machine Learning development portfolio documenting applied machine learning workflows, advanced exploratory data analysis (EDA), feature engineering, statistical modeling, and model evaluation diagnostics.

---

> [!IMPORTANT]
> ### ⚠️ Educational Use Disclaimer & Caveat
> The dataset utilized in this repository is statistically modeled **strictly for educational and instructional purposes** . **It is not an official census representation or empirical reflection of the actual United States Educational System, individual school districts, or real operating institutions.** All metrics, identifiers, and institutional figures are synthetic models designed to demonstrate data science and machine learning techniques.

---

## 📌 Executive Summary

Modern machine learning is far more than tuning hyperparameters—it is an iterative, rigorous engineering discipline where visualization and statistical exploration drive data quality, model selection, and diagnostic troubleshooting. 

This repository (**Part 2** of the Data Science Development series) captures hands-on development milestones across the full Machine Learning lifecycle:
- **Phase 1: Exploratory Data Analysis & Preprocessing** — Uncovering structural data distributions, skewness, socioeconomic gradients, and missingness patterns.
- **Phase 2: Feature Engineering & Transformation** — Scaling, categorical encodings, interaction variables, and handling hierarchical domain metrics.
- **Phase 3: Model Diagnostics & Evaluation** — Residual analyses, confusion matrices, bias vs. variance tradeoffs, and decision boundary interpretation.

---

## 📋 Activity Tracker & Curriculum Roadmap

The table below tracks the progressive activities, modules, and laboratory milestones maintained in this repository.

| Day / Milestone | Module / Activity | Core Focus & Techniques | Artifacts & Notebooks | Status |
|---|---|---|---|:---:|
| **Day 1** | **Machine Learning Visualization Pipeline** | • Computational role of visualization in ML<br>• Univariate distribution & skewness analysis (KDE, Histograms)<br>• Subgroup equity analysis via Title I Boxplots<br>• Multivariate interaction scatter plots (Socioeconomic factors vs. Math proficiency across governance controls) | [`day-1 ML Visualization.ipynb`](day-1%20ML%20Visualization.ipynb)<br>[`us_education_system_data_dictionary.md`](us_education_system_data_dictionary.md) | ✅ **Completed** |
| **Day 2** | **Data Cleaning & Missing Value Imputation** | • Handling structural vs. random missingness (SAT/ACT/AP across grade tiers)<br>• Median/mode grouping & KNN imputation strategies<br>• Outlier detection & robust capping methods | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 3** | **Feature Engineering & Transformation** | • Log-transformation for right-skewed funding variables<br>• One-Hot & Target encoding for high-cardinality locales & controls<br>• Interaction terms & multicollinearity diagnostics (VIF) | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 4** | **Supervised Modeling (Regression & Classification)** | • Predicting 4-year graduation rates & math proficiency<br>• Baseline models (Linear/Ridge/Lasso vs. Random Forest & XGBoost)<br>• Cross-validation & hyperparameter tuning | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 5** | **Model Evaluation, Diagnostics & Fairness** | • Residual diagnostic plots & learning curves<br>• Confusion matrices, ROC-AUC, Precision-Recall curves<br>• Bias vs. Variance diagnostic audits across socioeconomic subgroups | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 6+** | **Unsupervised Clustering & Pipeline Deployment** | • K-Means & Hierarchical Clustering for institutional segmentation<br>• Dimensionality reduction (PCA / t-SNE)<br>• Reproducible ML pipeline scripts | *Upcoming Milestone* | 🔄 **In Pipeline** |

---

## 🔬 Deep Dive: Day 1 — Machine Learning Visualization Pipeline

Effective data visualization is not about cosmetic presentations; it is a vital computational lens that informs every stage of model development:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       THE ML VISUALIZATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ├──► Phase 1: Exploratory Data Analysis & Preprocessing
          │      • Distribution shapes, outlier boundaries & log-normal skewness
          │      • Structural missingness & demographic correlations
          │
          ├──► Phase 2: Feature Engineering & Domain Interactions
          │      • Multivariate interaction mapping (Color/Hue, Size, Style)
          │      • Multicollinearity checks & feature selection signals
          │
          └──► Phase 3: Model Diagnostics & Evaluation
                 • Confusion Matrices & Error Clustering
                 • Bias vs. Variance validation curves
                 • High-dimensional decision boundary projections
```

### Key Analytical Highlights from Day 1 Notebook

1. **Univariate Analysis — Expenditure Distribution (`per_pupil_expenditure_usd`)**:
   - Evaluated operating budgets across 5,000 institutions.
   - Revealed marked right-skewness (log-normal profile) with typical spending clustering between \$12,000 and \$18,000, and a long tail extending toward \$65,000.
   - Highlights the necessity of log transformations prior to training linear or distance-based estimators.

2. **Bivariate Analysis — Academic Outcomes vs. Title I Status**:
   - Evaluated math proficiency distributions across institutional federal funding categories (`Title I Schoolwide`, `Title I Targeted`, `Not Title I Eligible`, and `Not Applicable`).
   - Highlighted persistent achievement gaps correlated with economic disadvantage, providing critical target signals for equity-focused predictive modeling.

3. **Multivariate Feature Interaction — Poverty, Governance & Resources**:
   - Rendered high-dimensional scatter plots mapping:
     - **X-axis**: Free/Reduced-Price Lunch Percentage (`percent_free_reduced_lunch` — key SES proxy)
     - **Y-axis**: Standardized Math Proficiency (`math_proficiency_pct`)
     - **Hue (Color)**: Institutional Governance (`control`: Public, Public Charter, Private Non-Profit, Private Religious, Private For-Profit)
     - **Size**: Per-Pupil Expenditure (`per_pupil_expenditure_usd`)
   - Unveiled strong negative correlation between economic disadvantage and standardized test outcomes, while illustrating how governance models and per-pupil spending mediate performance.

---

## 📊 Dataset Architecture & NCES Alignment

The core laboratory dataset models an educational ecosystem of **5,000 institutions** across all 50 US States and the District of Columbia, aligned with U.S. Department of Education (NCES Common Core of Data / IPEDS) reporting frameworks.

- **Dataset File:** [`us_education_system_dataset.csv`](us_education_system_dataset.csv) (5,000 rows × 35 variables)
- **Data Dictionary:** [`us_education_system_data_dictionary.md`](us_education_system_data_dictionary.md)

### Dimensional Summary

| Dimension Group | Key Features | Educational & Modeling Significance |
|---|---|---|
| **Institutional Context** | `institution_id`, `institution_name`, `state_code`, `county`, `nces_locale`, `level`, `control` | Establishes governance tier (Public, Charter, Private) and NCES urbanicity density. |
| **Resources & Finance** | `number_of_teachers`, `student_teacher_ratio`, `per_pupil_expenditure_usd`, `federal_funding_pct`, `state_funding_pct`, `local_funding_pct` | Captures financial inputs, instructional capacity, and local tax dependence. |
| **Demographics & SES** | `percent_free_reduced_lunch`, `title_i_status`, `racial_*_pct`, `special_education_pct`, `english_learner_pct`, `broadband_access_pct` | Captures student body diversity, federal aid eligibility, and digital divide indices. |
| **Academic Outcomes** | `math_proficiency_pct`, `reading_proficiency_pct`, `graduation_rate_pct`, `chronic_absenteeism_pct`, `avg_sat_score`, `avg_act_score`, `ap_ib_participation_pct`, `postsecondary_enrollment_pct` | Target variables for regression, classification, and longitudinal progress tracking. |

---

## 📂 Repository File Structure

```text
Data-Science-Development-PART-2/
├── .gitignore                                 # Git ignore patterns for Python, Jupyter, and OS artifacts
├── LICENSE                                    # Open-source MIT License
├── README.md                                  # Comprehensive portfolio documentation & activity tracker
├── requirements.txt                           # Pinned Python package dependencies for reproducibility
├── day-1 ML Visualization.ipynb               # Day 1: ML visualization pipeline & exploratory analysis
├── us_education_system_data_dictionary.md     # Detailed data dictionary, schema descriptions, & analytical cases
└── us_education_system_dataset.csv            # Modeled dataset of 5,000 educational institutions
```

---

## 🛠️ Getting Started & Reproducibility

### 1. Prerequisites
- **Python 3.10+**
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/D-James001/Data-Science-Development-PART-2.git
cd Data-Science-Development-PART-2
```

### 3. Create and Activate a Virtual Environment
```bash
# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Jupyter Notebook Environment
```bash
jupyter lab
# or
jupyter notebook
```
Open [`day-1 ML Visualization.ipynb`](day-1%20ML%20Visualization.ipynb) and run all cells sequentially. The notebook automatically detects local dataset availability or Kaggle execution paths.

---

## 💻 Tech Stack & Tooling

- **Programming Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Interactive Computing:** `jupyter`, `ipykernel`
- **Version Control & Collaboration:** Git, GitHub

---

## 👤 Author & Acknowledgments

**James Awofiranye**  
- **GitHub:** [@D-James001](https://github.com/D-James001)  
- **Program:** AltSchool Africa — Data Science Track (Semester 3)

*Developed as part of the Data Science Development series tracking applied machine learning workflows and production-ready data science implementations.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the LICENSE file for details.
