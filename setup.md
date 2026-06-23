# 🛠️ Setup & Installation Guide

This document provides step-by-step instructions to set up, configure, and run the Event-Driven Congestion ML Pipeline and Streamlit Dashboard.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Python**: Version `3.9` to `3.11` is recommended. (Python 3.12+ might have issues with some older pre-compiled wheels for spatial libraries like `hdbscan`).
- **Git**: For cloning the repository (if applicable).
- **C++ Build Tools (Windows Only)**: Required for building package dependencies like `hdbscan` from source if binary wheels are unavailable.
  - Download and install the [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select **Desktop development with C++**.

---

## 🚀 Step 1: Clone the Repository

Navigate to your workspace directory and open your terminal:

```bash
git clone <repository-url>
cd "round 2"
```

---

## 🐍 Step 2: Environment Setup

We highly recommend using a virtual environment (`venv` or `conda`) to avoid package conflicts with your system python.

### Option A: Using Python standard `venv`

```bash
# 1. Create the virtual environment
python -m venv venv

# 2. Activate the environment
# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### Option B: Using Anaconda / Miniconda

```bash
# 1. Create a conda environment with Python 3.9 (recommended)
conda create -n congestion_env python=3.9 -y

# 2. Activate the conda environment
conda activate congestion_env
```

---

## 📦 Step 3: Install Dependencies

With your virtual environment active, run the following command to install all the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!WARNING]
> **Windows Installation Note:** If the installation of `hdbscan` fails, it is usually because the C++ compiler is missing. Make sure Visual Studio C++ Build Tools are installed, or install `hdbscan` via conda if you are using a conda environment:
> ```bash
> conda install -c conda-forge hdbscan
> ```

---

## 📊 Step 4: Placing the Dataset

The pipeline expects the raw Astram event CSV dataset to be in the project root directory.

1. Ensure your raw CSV file is named exactly:
   `Astram event data_anonymized - Astram event data_anonymizedb40ac87 (1).csv`
2. Place this file in the project root directory alongside `config.yaml` and `requirements.txt`.

> [!NOTE]
> If you have a different filename or want to change the location, you can update the `data.raw_csv` configuration path inside [config.yaml](file:///C:/Users/DELL/Desktop/flipkart%20chall/round%202/config.yaml).

---

## ⚙️ Step 5: Configuration Check

The pipeline relies on [config.yaml](file:///C:/Users/DELL/Desktop/flipkart%20chall/round%202/config.yaml) for hyperparameters, spatial configurations, and resource multipliers.

- **Data Paths**: Set source CSV and output directory targets.
- **Spatial Bounds**: Spatial constraints for Bengaluru coordinates.
- **Model Params**: Specific training parameters for LightGBM, XGBoost, CatBoost, and Random Forest.
- **Resource Recommender**: Adjust base manpower requirements (e.g., 10 for protests, 2 for vehicle breakdowns) and multipliers (e.g., rush hour multiplier: 1.3).

---

## 🏋️ Step 6: Train the ML Models & Run Pipeline

Run the end-to-end orchestrator pipeline. This script processes the raw CSV, performs feature engineering, trains the four models (Severity, Duration, Road Closure, and Spatial Clustering), evaluates performance, and outputs serialization artifacts.

```bash
python -m src.pipeline
```

### Optional: Tune the Duration Regressor
If you want to perform hyperparameter optimization for the LightGBM Quantile Duration model:
```bash
python -m src.tune_duration_model
```
*Note: This will output recommended parameters to configure in `config.yaml`.*

---

## 🖥️ Step 7: Launch the Dashboard

Once the models are trained (and joblib files are generated in `outputs/models/`), you can start the interactive Streamlit UI:

```bash
streamlit run dashboard/app.py
```

Your web browser should automatically open `http://localhost:8501`. If it doesn't, copy-paste the URL printed in the terminal.

---

## 📁 Output Artifacts Structure

After successfully running the pipeline, the following files will be generated in the `outputs/` directory:

```text
outputs/
├── processed_events.parquet      # Preprocessed event parquet file
├── models/
│   ├── severity_ensemble.joblib  # Trained stacked severity classification model
│   ├── duration_lgbm.joblib      # Trained quantile duration model
│   ├── road_closure_cat.joblib   # Trained road closure model
│   └── hdbscan_clustering.joblib # Spatial clustering model
├── plots/
│   ├── correlation_matrix.png    # Correlation plot of features
│   ├── feature_importance_*.png  # Importance rankings for each model
│   └── actual_vs_predicted_*.png # Performance plots
└── reports/
    ├── pipeline_metrics.json     # ROC-AUC, F1, MAE, and Interval coverage logs
    └── feedback_corrections.json # Post-event loop adjustments/bias factors
```

---

## 🛠️ Troubleshooting & Common Issues

### 1. `ModuleNotFoundError: No module named 'src'`
**Cause:** Attempting to run python files directly (e.g., `python src/pipeline.py`) instead of as packages.
**Solution:** Always execute python commands from the project root directory using the `-m` flag:
```bash
python -m src.pipeline
```

### 2. `error: Microsoft Visual C++ 14.0 or greater is required`
**Cause:** Missing build tools for building `hdbscan` or other binary compilation packages on Windows.
**Solution:** Install Build Tools for Visual Studio from the official Microsoft site. Select C++ desktop development.

### 3. File not found: `outputs/models/severity_ensemble.joblib`
**Cause:** You are running the Streamlit dashboard before running the ML pipeline.
**Solution:** Run the pipeline first to generate the model files: `python -m src.pipeline`.

### 4. Memory or Out Of Memory (OOM) Errors
**Cause:** Feature engineering generates a large number of spatial/temporal features and historical event lags, which may consume several gigabytes of RAM.
**Solution:** Ensure you have at least 8GB of free RAM, or increase virtual page file size if run on Windows.
