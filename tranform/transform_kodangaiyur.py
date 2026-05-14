import pandas as pd

# Load both datasets
anandvihar = pd.read_csv('D:/AQI_Project/Data/cleaned/AnandVihar_AQI_data.csv')
kodungaiyur = pd.read_csv('D:/AQI_Project/Data/raw/Kodungaiyur_AQI.csv')

print("AnandVihar columns:", list(anandvihar.columns))
print("kodungaiyur columns (before):", list(kodungaiyur.columns))

# ── Transformation ──────────────────────────────────────────────────────────
# kodungaiyur already has all the required sub-columns and AQI fields.
# We just need to keep only the 10 columns that match AnandVihar's structure
# and drop the extra raw measurement + weather columns.

target_columns = [
    'Timestamp',
    'PM25_sub',
    'PM10_sub',
    'NO2_sub',
    'SO2_sub',
    'NH3_sub',
    'CO_sub',
    'O3_sub',
    'AQI',
    'AQI_Category'
]

kodungaiyur_transformed = kodungaiyur[target_columns].copy()

# Ensure dtypes match AnandVihar exactly
float_cols = ['PM25_sub', 'PM10_sub', 'NO2_sub', 'SO2_sub',
              'NH3_sub', 'CO_sub', 'O3_sub', 'AQI']
kodungaiyur_transformed[float_cols] = kodungaiyur_transformed[float_cols].astype(float)
kodungaiyur_transformed['Timestamp'] = kodungaiyur_transformed['Timestamp'].astype(str)
kodungaiyur_transformed['AQI_Category'] = kodungaiyur_transformed['AQI_Category'].astype(str)

# ── Validation ───────────────────────────────────────────────────────────────
print("\nkodungaiyur columns (after):", list(kodungaiyur_transformed.columns))
print("Columns match AnandVihar:", list(kodungaiyur_transformed.columns) == list(anandvihar.columns))
print("\nDtypes match:")
print(kodungaiyur_transformed.dtypes)
print("\nShape:", kodungaiyur_transformed.shape)
print("\nSample rows:")
print(kodungaiyur_transformed.head())

# ── Save ─────────────────────────────────────────────────────────────────────
kodungaiyur_transformed.to_csv(r'D:\AQI_Project\Data\cleaned\Kodungaiyur_AQI_data.csv', index=False)
print("\nSaved to kodungaiyur_AQI_data.csv")
