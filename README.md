# AQI Forecasting Models

This directory contains the machine learning pipelines and exploratory data analysis (EDA) used for predicting the Air Quality Index (AQI) across various Indian cities.

The forecasting models included in this repository are:
- **ARIMA** (AutoRegressive Integrated Moving Average)
- **SARIMA** (Seasonal ARIMA)
- **LSTM** (Long Short-Term Memory Neural Networks)
- **XGBoost** (Extreme Gradient Boosting)

---

## Getting Started on a Local Machine

If you have downloaded this repository and wish to run the `.ipynb` Jupyter Notebooks locally, follow the steps below to ensure your environment is configured correctly.

### 1. Prerequisites

You must have Python 3.8+ installed on your system. It is highly recommended to use a virtual environment (`venv`, `conda`, etc.) to isolate dependencies.

### 2. Install Dependencies

A `requirements.txt` file is provided in the root directory to help you install all necessary packages. Open your terminal or command prompt, navigate to the project directory, and run:

```bash
pip install -r requirements.txt
```

This will install data science essentials like `pandas`, `numpy`, `scikit-learn`, `matplotlib`, along with model-specific libraries like `pmdarima`, `xgboost`, and `tensorflow`.

### 3. Fixing Hardcoded Paths (Crucial Step)

The Jupyter Notebooks currently use **hardcoded absolute paths** (e.g., `D:/AQI_Project/...`) which will cause `FileNotFoundError` exceptions when run on your computer.

Before running the models, you must update the paths to match your local setup. You can do this in one of two ways:

#### Option A: Find and Replace (Recommended)
1. Open the `.ipynb` notebook you wish to run in Jupyter.
2. Press `Ctrl + F` (or `Cmd + F` on Mac) to open the Find and Replace dialog.
3. Search for the string `D:/AQI_Project/`
4. Replace it with the absolute path to where you saved the repository (e.g., `C:/Users/YourName/Documents/AQI_Project/` on Windows, or `/Users/YourName/AQI_Project/` on Mac/Linux).
5. Ensure you maintain forward slashes `/` even on Windows, as they are safer for cross-platform compatibility in Python.

#### Option B: Use Relative Paths
You can manually edit the cells reading CSV files or saving models to use relative paths. For example, change:
```python
df = pd.read_csv('D:/AQI_Project/data/AQI_Data.csv')
```
To:
```python
df = pd.read_csv('../data/AQI_Data.csv')
```
*(Note: Relative paths depend on the location of the specific notebook relative to the data file.)*

### 4. Running the Notebooks

1. Launch Jupyter Notebook from your terminal:
   ```bash
   jupyter notebook
   ```
2. Navigate to the desired model's folder (e.g., `XGBOOST/`).
3. Open a city-specific notebook (e.g., `03_XGBoost_AQI_Patia.ipynb`).
4. Run the cells sequentially from top to bottom.

---

## Output Expectations

- **Images**: All generated plots (training histories, forecasts, residual graphs, and confusion matrices) are automatically saved into an `images/` directory generated inside the same folder as the notebook.
- **Saved Models**: Trained models (e.g., `.pkl` or `.h5` files) are saved inside their respective directories to be consumed by the Flask backend application.
- **Classification Metrics**: The final cell of each model notebook will output an AQI classification report with a visual confusion matrix, giving insight into how well the continuous regression model predicts standard Indian AQI bands.
