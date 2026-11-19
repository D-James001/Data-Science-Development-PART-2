# Data Science Development — Part 2: Machine Learning Workflow & Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-4c72b0.svg)](https://seaborn.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
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
- **Phase 3: Model Diagnostics & Evaluation** — Confusion matrices, bias vs. variance tradeoffs, dimensionality reduction, and decision boundary interpretation.
- **Phase 4: Business Value & Stakeholder Communication** — Translating ML outputs into actionable, stakeholder-ready business insights and strategic resource allocation frameworks.

---

## 📋 Activity Tracker & Curriculum Roadmap

The table below tracks the progressive activities, modules, and laboratory milestones maintained in this repository.

| Day / Milestone | Module / Activity | Core Focus & Techniques | Artifacts & Notebooks | Status |
|---|---|---|---|:---:|
| **Day 1** | **Machine Learning Visualization Pipeline** | • Computational role of visualization in ML<br>• Univariate distribution & skewness analysis (KDE, Histograms)<br>• Subgroup equity analysis via Title I Boxplots<br>• Multivariate interaction scatter plots (Socioeconomic factors vs. Math proficiency across governance controls) | [`day-1 ML Visualization.ipynb`](day-1%20ML%20Visualization.ipynb)<br>[`us_education_system_data_dictionary.md`](us_education_system_data_dictionary.md) | ✅ **Completed** |
| **Day 2** | **ML Visualization Continuation — Correlation, Classification & Dimensionality Reduction** | • Correlation matrix heatmaps for feature redundancy detection<br>• Supervised classification with Random Forest (above/below median math proficiency)<br>• Confusion matrix visualization & model diagnostics<br>• Dimensionality reduction via PCA (2D projections of feature space)<br>• Standardization with `StandardScaler` | [`day-2-ml-visualization-continuation.ipynb`](day-2-ml-visualization-continuation.ipynb)<br>[`Correlation matrix.png`](Correlation%20matrix.png)<br>[`Confusion Matrix.png`](Confusion%20Matrix.png)<br>[`2D PCA Projection.png`](2D%20PCA%20Projection.png) | ✅ **Completed** |
| **Day 3** | **Business Value of Data Visualization for ML** | • Bridging the gap between ML model accuracy and stakeholder trust<br>• Resource allocation audit: per-pupil expenditure distribution analysis<br>• Title I funding tier proficiency gap analysis (16–22 point structural deficit)<br>• Strategic \$50M intervention fund allocation framework across four evidence-based pillars<br>• ROI-driven data storytelling for executive decision-making | [`day-3-business-value-of-data-visualization-for-ml.ipynb`](day-3-business-value-of-data-visualization-for-ml.ipynb)<br>[`Resources Allocation.png`](Resources%20Allocation.png) | ✅ **Completed** |
| **Day 4** | **Data Cleaning & Missing Value Imputation** | • Handling structural vs. random missingness (SAT/ACT/AP across grade tiers)<br>• Median/mode grouping & KNN imputation strategies<br>• Outlier detection & robust capping methods | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 5** | **Feature Engineering & Transformation** | • Log-transformation for right-skewed funding variables<br>• One-Hot & Target encoding for high-cardinality locales & controls<br>• Interaction terms & multicollinearity diagnostics (VIF) | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 6** | **Supervised Modeling (Regression & Classification)** | • Predicting 4-year graduation rates & math proficiency<br>• Baseline models (Linear/Ridge/Lasso vs. Random Forest & XGBoost)<br>• Cross-validation & hyperparameter tuning | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 7** | **Model Evaluation, Diagnostics & Fairness** | • Residual diagnostic plots & learning curves<br>• ROC-AUC, Precision-Recall curves<br>• Bias vs. Variance diagnostic audits across socioeconomic subgroups | *Upcoming Milestone* | 🔄 **In Pipeline** |
| **Day 8+** | **Unsupervised Clustering & Pipeline Deployment** | • K-Means & Hierarchical Clustering for institutional segmentation<br>• Dimensionality reduction (PCA / t-SNE)<br>• Reproducible ML pipeline scripts | *Upcoming Milestone* | 🔄 **In Pipeline** |

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

## 🔬 Deep Dive: Day 2 — Correlation, Classification & Dimensionality Reduction

Day 2 builds upon the exploratory analysis from Day 1, advancing into three critical ML workflow stages: feature correlation analysis, supervised classification, and unsupervised dimensionality reduction.

### Key Analytical Highlights from Day 2 Notebook

1. **Correlation Matrix & Heatmap — Feature Redundancy Detection**:
   - Computed pairwise Pearson correlations across all continuous numeric performance indicators (expenditure, student-teacher ratio, proficiency scores, graduation rates, etc.).
   - Visualized the correlation structure using an annotated heatmap (`sns.heatmap`), enabling rapid identification of redundant or highly collinear features prior to model training.
   - Key finding: Math and reading proficiency scores exhibit strong positive correlation, suggesting potential for dimensionality reduction or feature selection.

2. **Supervised Classification — Random Forest (Model Diagnostics)**:
   - Framed math proficiency as a binary classification task (above/below median proficiency).
   - Trained a `RandomForestClassifier` (100 estimators) on a 70/30 train-test split.
   - Evaluated model performance using a **confusion matrix heatmap**, enabling visual inspection of true/false positive and negative rates.
   - Demonstrates the full supervised ML workflow: data cleaning → feature/target separation → train/test split → model training → prediction → evaluation.

3. **Dimensionality Reduction — PCA (2D Projection)**:
   - Applied `StandardScaler` to normalize features before PCA projection.
   - Reduced the multi-dimensional feature space to 2 principal components using `PCA(n_components=2)`.
   - Visualized the 2D PCA embeddings with class-based color coding, revealing cluster separability and overlap between above/below-median proficiency groups.
   - Provides foundational understanding for unsupervised clustering and advanced dimensionality reduction techniques (t-SNE, UMAP) in future milestones.

---

## 🔬 Deep Dive: Day 3 — Business Value of Data Visualization for ML

Day 3 shifts from technical ML workflows to the **strategic business communication layer** — demonstrating how visualization bridges the gap between model accuracy and executive stakeholder trust.

### The Core Problem

A 95% model accuracy is meaningless to business stakeholders if they cannot understand *why* the model makes its decisions. Data Visualization acts as the translation layer across the entire ML pipeline, converting technical outputs into actionable business intelligence.

### Business-Valued ML Visualization Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               BUSINESS-VALUED ML VISUALIZATION FRAMEWORK                     │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ├──► Phase 1: Risk & Data Quality
          │      • Business Problem: Corrupted, heavily skewed, incomplete data
          │      • Solution: Visualize errors, extreme outliers, missing information
          │
          ├──► Phase 2: ROI & Lever Discoveries
          │      • Business Problem: Investing in factors that won't impact goals
          │      • Solution: Visualize how spending directly impacts ROI
          │
          └──► Phase 3: Model Auditing & Trust
                 • Business Problem: Understanding the cost of model mistakes
                 • Solution: Visualize model errors in simple business terms
```

### Real-World Scenario

The notebook models a real-world advisory scenario: **advising a State Department of Education on allocating a \$50 Million intervention fund to improve student math proficiency**.

### Key Analytical Highlights from Day 3 Notebook

1. **Resource Allocation Audit — Financial Spending Distribution**:
   - Analyzed per-pupil expenditure distribution across 5,000 institutions.
   - Revealed a **clustered baseline of \$10,000–\$22,000** with a high-spending right skew reaching \$52,000+, exposing wealth-driven funding disparities.
   - **Audit Implication**: Uniform funding increases disproportionately reward already affluent districts without addressing baseline inadequacies.

2. **Title I Funding Tier Proficiency Gap Analysis**:
   - Uncovered a **16-to-22 percentage point structural achievement deficit** between Title I Schoolwide institutions (~51% median math proficiency) and non-eligible wealthier schools (~67–73%).
   - Identified **"Beat-the-Odds" outlier institutions** — Title I Schoolwide schools achieving 80%+ math proficiency despite socioeconomic barriers — offering high-value pedagogical blueprints.

3. **Strategic \$50M Intervention Fund Allocation Framework**:
   - **Weighted Student Funding (\$20M / 40%)**: Needs-weighted formula targeting Title I Schoolwide institutions.
   - **High-Dosage Math Tutoring (\$15M / 30%)**: Intensive 1:3 ratio in-school tutoring for 2x–3x academic year gains.
   - **Master Teacher Retention Stipends (\$10M / 20%)**: Retention bonuses to keep experienced educators in high-poverty classrooms.
   - **Foundational Numeracy Screening (\$5M / 10%)**: Early numeracy diagnostics to resolve math learning gaps before high school.

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
├── .gitignore                                     # Git ignore patterns for Python, Jupyter, and OS artifacts
├── LICENSE                                        # Open-source MIT License
├── README.md                                      # Comprehensive portfolio documentation & activity tracker
├── requirements.txt                               # Pinned Python package dependencies for reproducibility
├── day-1 ML Visualization.ipynb                   # Day 1: ML visualization pipeline & exploratory analysis
├── day-2-ml-visualization-continuation.ipynb      # Day 2: Correlation, classification & PCA workflows
├── day-3-business-value-of-data-visualization-for-ml.ipynb  # Day 3: Business value & stakeholder communication
├── us_education_system_data_dictionary.md         # Detailed data dictionary, schema descriptions, & analytical cases
├── us_education_system_dataset.csv                # Modeled dataset of 5,000 educational institutions
├── Correlation matrix.png                         # Day 2 Output: Heatmap of feature correlations
├── Confusion Matrix.png                           # Day 2 Output: Random Forest classification confusion matrix
├── 2D PCA Projection.png                          # Day 2 Output: PCA 2D embedding of feature space
└── Resources Allocation.png                       # Day 3 Output: Financial spending & Title I proficiency audit
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
Open the Day 1, Day 2, or Day 3 notebook and run all cells sequentially. The notebooks automatically detect local dataset availability or Kaggle execution paths.

---

## 💻 Tech Stack & Tooling

- **Programming Language:** Python 3.10+
- **Data Manipulation:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (Random Forest, PCA, StandardScaler, train/test split, confusion matrix)
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
