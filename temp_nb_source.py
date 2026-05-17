try:
    import xgboost
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings, os
warnings.filterwarnings('ignore')

# ── Config — only change these paths ──────────────────────────
INPUT_FILE   = r'D:\AQI_Project_new\data\clean\AnandVihar_AQI_cleaned.csv'
SAVE_DIR     = r'D:\AQI_Project_new\XGBOOST\saved_models'
IMAGE_DIR    = 'images'
STATION_NAME = 'AnandVihar'
RANDOM_SEED  = 42

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR,  exist_ok=True)

print('Libraries loaded successfully.')
print(f'XGBoost version: {xgb.__version__}')
print(f'Station: {STATION_NAME}')


# ── 1. Load hourly feature-engineered CSV (output of EDA §11) ─
df = pd.read_csv(INPUT_FILE, parse_dates=['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f'Shape      : {df.shape}')
print(f'Date range : {df["Timestamp"].min()}  →  {df["Timestamp"].max()}')
print(f'Columns    : {df.columns.tolist()}')
print(f'Nulls      : {df.isnull().sum().sum()}')
df.head(3)


# ── 2. Feature engineering on hourly data ─────────────────────
# The EDA export already contains:
#   • AQI_lag_1h … AQI_lag_48h
#   • AQI_roll_mean_3/6/24h, AQI_roll_std_3/6/24h
#   • Hour_sin/cos, Month_sin/cos, IsWeekend, Season_Enc, AQI_Category_Enc
# We add a few extra lags and then define the final feature list.

df_feat = df.copy()

# Additional hourly lags not in the EDA export
for lag in [72, 168]:          # 3-day, 1-week
    col = f'AQI_lag_{lag}h'
    if col not in df_feat.columns:
        df_feat[col] = df_feat['AQI'].shift(lag)

# Day-of-year cyclic (annual seasonality signal for XGBoost)
if 'DayOfYear' not in df_feat.columns:
    df_feat['DayOfYear'] = df_feat['Timestamp'].dt.dayofyear
df_feat['doy_sin'] = np.sin(2 * np.pi * df_feat['DayOfYear'] / 365)
df_feat['doy_cos'] = np.cos(2 * np.pi * df_feat['DayOfYear'] / 365)

# Drop columns that are not model inputs
DROP_COLS = ['Timestamp', 'AQI_Category', 'Season', 'AQI_Category_Enc']
lag_roll_cols = [c for c in df_feat.columns if 'lag' in c or 'roll' in c]

# Feature list = everything except target and drop cols
feature_cols = [c for c in df_feat.columns
                if c not in DROP_COLS + ['AQI']
                and df_feat[c].dtype in [np.float64, np.int64, float, int]]

# Drop rows where any lag/roll feature is NaN (first ~168 rows)
df_model = df_feat.dropna(subset=lag_roll_cols).copy()

X = df_model[feature_cols]
y = df_model['AQI']

print(f'Feature matrix : {X.shape}')
print(f'Null count in X: {X.isnull().sum().sum()}')
print(f'\nFeatures ({len(feature_cols)}):')
for f in feature_cols:
    print(f'  • {f}')


# ── 3. Chronological 70/15/15 split ───────────────────────────
# EDA §10.7: always use time-based splits to avoid data leakage
n = len(df_model)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df_model.iloc[:train_end]
val_df   = df_model.iloc[train_end:val_end]
test_df  = df_model.iloc[val_end:]

X_train, y_train = train_df[feature_cols], train_df['AQI']
X_val,   y_val   = val_df[feature_cols],   val_df['AQI']
X_test,  y_test  = test_df[feature_cols],  test_df['AQI']

print(f'Train : {len(train_df):,} rows  '
      f'({train_df["Timestamp"].iloc[0].date()} → {train_df["Timestamp"].iloc[-1].date()})')
print(f'Val   : {len(val_df):,} rows  '
      f'({val_df["Timestamp"].iloc[0].date()} → {val_df["Timestamp"].iloc[-1].date()})')
print(f'Test  : {len(test_df):,} rows  '
      f'({test_df["Timestamp"].iloc[0].date()} → {test_df["Timestamp"].iloc[-1].date()})')


# ── 4. Train XGBoost ──────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators      = 500,
    max_depth         = 5,
    learning_rate     = 0.03,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    early_stopping_rounds = 30,
    eval_metric       = 'rmse'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
print(f'Best iteration: {model.best_iteration}')


# ── 5a. Direct forecast (uses actual lag values from test set) ─
pred_direct = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred_direct))
mae  = mean_absolute_error(y_test, pred_direct)
mask = y_test.values != 0
mape = np.mean(np.abs((y_test.values[mask] - pred_direct[mask]) / y_test.values[mask])) * 100

print('XGBoost Direct Forecast (hourly):')
print(f'  RMSE : {rmse:.2f}')
print(f'  MAE  : {mae:.2f}')
print(f'  MAPE : {mape:.2f}%')


# ── 5b. Recursive (multi-step) forecast ───────────────────────
def recursive_forecast(model, train_df, test_df, feature_cols, lag_cols):
    all_data    = pd.concat([train_df, test_df]).copy()
    predictions = []

    for i in range(len(train_df), len(all_data)):
        feat = all_data.iloc[i][feature_cols].values.reshape(1, -1)
        pred = model.predict(feat)[0]
        predictions.append(pred)
        if i + 1 < len(all_data):
            for lag_col in lag_cols:
                lag_n = int(lag_col.replace('AQI_lag_','').replace('h',''))
                if len(predictions) >= lag_n:
                    all_data.at[all_data.index[i + 1], lag_col] = predictions[-lag_n]

    return np.array(predictions)

lag_cols     = [c for c in feature_cols if c.startswith('AQI_lag_')]
pred_recursive = recursive_forecast(model, train_df, test_df, feature_cols, lag_cols)

rmse_r = np.sqrt(mean_squared_error(y_test, pred_recursive))
mae_r  = mean_absolute_error(y_test, pred_recursive)
print(f'XGBoost Recursive Forecast: RMSE={rmse_r:.2f}  MAE={mae_r:.2f}')


# ── 5c. Plot forecast ─────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 13), gridspec_kw={'height_ratios': [3, 3, 2]})

# ── Panel 1: Full timeline ──────────────────────────────────
ax = axes[0]
last_train = train_df.set_index('Timestamp')['AQI'].iloc[-168:]
last_train.plot(ax=ax, label='Train (last 7d)', color='steelblue', alpha=0.6, lw=1)

val_ts = val_df.set_index('Timestamp')
val_ts['AQI'].plot(ax=ax, label='Validation (Actual)', color='orange', lw=1.0, alpha=0.7)

test_ts = test_df.set_index('Timestamp')
test_ts['AQI'].plot(ax=ax, label='Test (Actual)', color='black', lw=1.0)
ax.plot(test_ts.index, pred_direct,    label=f'Direct   (RMSE={rmse:.1f})',    color='green', linestyle='--', lw=1.0)
ax.plot(test_ts.index, pred_recursive, label=f'Recursive (RMSE={rmse_r:.1f})', color='red',   linestyle=':',  lw=1.0)

ax.axvline(x=val_df['Timestamp'].iloc[0],  color='gray', linestyle='--', lw=0.8)
ax.axvline(x=test_df['Timestamp'].iloc[0], color='gray', linestyle='-',  lw=0.8)
ax.set_title(f'{STATION_NAME} — XGBoost Hourly AQI Forecast (Full)', fontsize=12, fontweight='bold')
ax.set_ylabel('AQI')
ax.legend(fontsize=8, loc='upper left')

# ── Panel 2: Zoomed — last 14 days of test ─────────────────
ax2 = axes[1]
zoom_n = 14 * 24   # 14 days × 24 hours
test_zoom_idx   = test_ts.index[-zoom_n:]
actual_zoom     = test_ts['AQI'].values[-zoom_n:]
direct_zoom     = pred_direct[-zoom_n:]
recursive_zoom  = pred_recursive[-zoom_n:]

ax2.plot(test_zoom_idx, actual_zoom,    label='Actual',   color='black', lw=1.5)
ax2.plot(test_zoom_idx, direct_zoom,    label=f'Direct   (RMSE={rmse:.1f})',    color='green', linestyle='--', lw=1.5)
ax2.plot(test_zoom_idx, recursive_zoom, label=f'Recursive (RMSE={rmse_r:.1f})', color='red',   linestyle=':',  lw=1.5)
ax2.set_title('Zoomed — Last 14 Days of Test Period', fontsize=11)
ax2.set_ylabel('AQI')
ax2.legend(fontsize=8)

# ── Panel 3: Residual errors ────────────────────────────────
ax3 = axes[2]
errors_direct    = test_ts['AQI'].values - pred_direct
errors_recursive = test_ts['AQI'].values - pred_recursive
ax3.plot(test_ts.index, errors_direct,    label='Error Direct',    color='green', lw=0.8, alpha=0.8)
ax3.plot(test_ts.index, errors_recursive, label='Error Recursive',  color='red',   lw=0.8, alpha=0.8, linestyle='--')
ax3.axhline(0, color='black', lw=0.8, linestyle='--')
ax3.set_title('Residual Errors (Actual − Predicted)', fontsize=11)
ax3.set_ylabel('Error (AQI)')
ax3.set_xlabel('Timestamp')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/xgboost_forecast_{STATION_NAME}.png', dpi=150, bbox_inches='tight')
plt.show()


# ── 6. Feature Importance ─────────────────────────────────────
plt.close('all')
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
importance.head(20).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title(f'{STATION_NAME} — XGBoost Top 20 Feature Importances', fontsize=13)
ax.set_xlabel('Importance (gain)')
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/xgboost_feature_importance_{STATION_NAME}.png', dpi=130, bbox_inches='tight')
plt.show()

print('\nTop 10:')
print(importance.head(10).to_string())


# ── 7. Learning Curve ─────────────────────────────────────────
evals_result = model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(evals_result['validation_0']['rmse'], label='Validation RMSE', color='crimson')
ax.axvline(model.best_iteration, linestyle='--', color='gray',
           label=f'Best iteration = {model.best_iteration}')
ax.set_title(f'{STATION_NAME} — XGBoost Validation RMSE by Iteration')
ax.set_xlabel('Iteration'); ax.set_ylabel('RMSE')
ax.legend()
plt.tight_layout()
plt.savefig(f'{IMAGE_DIR}/xgboost_learning_curve_{STATION_NAME}.png', dpi=150, bbox_inches='tight')
plt.show()


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Define AQI categories based on Indian standards
def categorize_aqi(aqi_values):
    categories = []
    for val in aqi_values:
        if val <= 50:
            categories.append('Good')
        elif val <= 100:
            categories.append('Satisfactory')
        elif val <= 200:
            categories.append('Moderate')
        elif val <= 300:
            categories.append('Poor')
        elif val <= 400:
            categories.append('Very Poor')
        else:
            categories.append('Severe')
    return np.array(categories)

# Ensure values are flattened arrays
y_true_cont = np.array(y_test.values).flatten()
y_pred_cont = np.array(pred_recursive).flatten()

# Convert continuous predictions to discrete AQI classes
y_true_class = categorize_aqi(y_true_cont)
y_pred_class = categorize_aqi(y_pred_cont)

# Labels for standard Indian AQI
labels = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
# Filter labels to only those present in the actual or predicted to avoid empty rows/cols if desired
# But keeping all labels shows a full matrix
present_labels = [l for l in labels if l in y_true_class or l in y_pred_class]

# Calculate Metrics
acc = accuracy_score(y_true_class, y_pred_class)
# weighted average for precision and recall since it's multiclass
prec = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
rec = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)

print(f"Classification Metrics for AQI Categories:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f} (Weighted)")
print(f"Recall:    {rec:.4f} (Weighted)\n")

print("Classification Report:")
print(classification_report(y_true_class, y_pred_class, labels=present_labels, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true_class, y_pred_class, labels=present_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=present_labels, yticklabels=present_labels)
plt.title('AQI Classification Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

os.makedirs('images', exist_ok=True)
plt.savefig(f'{IMAGE_DIR}/xgboost_confusion_matrix_{STATION_NAME}.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 9. Save Model ─────────────────────────────────────────────
import joblib

model_path = os.path.join(SAVE_DIR, f'{STATION_NAME}_xgboost_aqi.pkl')
joblib.dump(model, model_path)
print(f'✅ Model saved : {model_path}')

# Also save feature list so it can be reloaded consistently
import json as _json
feat_path = os.path.join(SAVE_DIR, f'{STATION_NAME}_xgboost_features.json')
with open(feat_path, 'w') as fp:
    _json.dump(feature_cols, fp)
print(f'✅ Feature list: {feat_path}')
