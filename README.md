# PathFinder-AI
Multi-Stage Machine Learning Predictive Placement and Salary Analytics Engine.

PathFinder AI is a full-stack predictive web application designed to forecast academic graduate outcomes, recommend tailored career positions, and estimate starting salary variables. The engine integrates a React/Vite/TypeScript frontend with an active Python Flask API backend running synchronized scikit-learn models.

**Live Deployment Portal:** [https://capstone-pathfinder.onrender.com/](https://capstone-pathfinder.onrender.com/)

---

## Project Overview & Operational Capabilities

PathFinder AI operates as a continuous evaluation pipeline, translating multidimensional student vectors into clear career insights.

### Core Deliverables & Functionality
* **Predictive Placement Classification:** Evaluates graduate profiling parameters to classify the binary probability of corporate placement.
* **Compensation Estimation (Regression Modeling):** Forecasts starting entry-level salary metrics utilizing multi-variable continuous regression models.
* **Feature Importance & Impact Analysis:** Quantifies and maps which specific academic or technical variables exert the highest statistical weight on employment success.
* **Algorithmic Career Alignment Tracking:** Employs geometric spatial positioning to automatically map a student's domain skills to the most compatible industry role.

---

## Core Architecture Blueprint

The application employs a 3-Stage Pipeline executing linear, spatial, and probabilistic calculations:

1. **Stage 1 (Classification):** A Random Forest Classifier determines historical deployment probability based on academic marks (GWA), internship parameters, and active backlog indices.
2. **Stage 2 (Regression):** A Multiple Linear Regression model dynamically scales entry-level salary compensation forecasts matching corporate tier environments.
3. **Stage 3 (Spatial Optimization):** A K-Nearest Neighbors (KNN) model assigns optimal career paths based on localized skill alignment across core domains (Python, DSA, Web Dev, ML).

---

## Data Logic & Modeling Framework

The underlying data engine processes student records across two synchronized mathematical logic layers:

### 1. Placement Propensity Parameters
A student's probability of placement is calculated by analyzing structural features across multiple domains:
* **Academic Performance:** Evaluated via continuous General Weighted Average (GWA) calculations and active academic backlogs.
* **Core Technological Competencies:** Binary switches tracking specialized skill paths including **Python Stacks, Data Structures & Algorithms (DSA), Machine Learning, and Web Architecture frameworks**.
* **Quantitative Assessments:** Evaluated using standardized Coding Competency Evaluations, Aptitude Indices, and Communication Ratings.
* **Practical Experience Nodes:** Tracks tokenized resume scores, portfolio project counts, and institutional internship completions.

### 2. Salary Engine Architecture
Continuous compensation values are simulated using structured algorithmic transformations rather than simple linear lookups:
* **Non-Linear Quantile Scale Transformations:** Maps inputs dynamically against non-linear distributions to mimic real-world income curves.
* **Corporate Tier Multipliers:** Adjusts core predictions by applying weighting coefficients based on target company environments (e.g., *Top Tech Titans* vs. *Bootstrap Startups*).
* **Controlled Stochastic Randomness:** Injects localized Gaussian noise variables to simulate realistic competitive variations in starting wage scales.

---

## Dataset Specifications

* **Observation Volume:** ~9,000 distinct historical graduate profile rows.
* **Feature Dimensions:** 20+ independent quantitative and categorical attribute columns.
* **File Format:** High-performance Comma-Separated Values (`.csv`) data structure.
* **Data Integrity:** 0% missing values; fully imputed and scaled before model ingestion.
* **Target Vectors:** `placed_status` (Categorical Classifier) and `predicted_salary` (Continuous Dependent Variable).

---

## Project Repository Structure

## 🛠️ Project Repository Structure

## 🛠️ Project Repository Structure

```text
PathFinderAI/
├── dataset/                        # Raw data files
│   └── philippine_gwa_dataset.csv
├── models/                         # Saved machine learning models
├── static/                         # Assets delivered directly to browser
│   ├── css/
│   │   └── style.css
│   ├── images/                     # Backgrounds, UI icons, illustrations
│   └── js/
│       └── script.js
├── templates/                      # Dynamic HTML layout files 
│   ├── index.html                  # Landing page
│   ├── dashboard.html              # Main workspace user portal
│   ├── predictor.html              # Core KNN input form 
│   ├── skills.html                 # Extra metrics / profile attributes
│   └── about.html                  # Project info page
├── app.py                          # Main application router/entrypoint
├── model.py                        # Model functions (prediction logic)
├── train_model.py                  # Training script to build & save model
├── requirements.txt                # Production server dependencies
└── .gitignore                      # Prevents local cache folder leaks
