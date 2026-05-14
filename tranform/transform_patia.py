import pandas as pd

# Load both datasets
anandvihar = pd.read_csv('D:/AQI_Project/Data/cleaned/AnandVihar_AQI_data.csv')
patia = pd.read_csv('D:/AQI_Project/Data/raw/Patia_AQI.csv')

print("AnandVihar columns:", list(anandvihar.columns))
print("patia columns (before):", list(patia.columns))

# ── Transformation ──────────────────────────────────────────────────────────
# patia already has all the required sub-columns and AQI fields.
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

patia_transformed = patia[target_columns].copy()

# Ensure dtypes match AnandVihar exactly
float_cols = ['PM25_sub', 'PM10_sub', 'NO2_sub', 'SO2_sub',
              'NH3_sub', 'CO_sub', 'O3_sub', 'AQI']
patia_transformed[float_cols] = patia_transformed[float_cols].astype(float)
patia_transformed['Timestamp'] = patia_transformed['Timestamp'].astype(str)
patia_transformed['AQI_Category'] = patia_transformed['AQI_Category'].astype(str)

# ── Validation ───────────────────────────────────────────────────────────────
print("\npatia columns (after):", list(patia_transformed.columns))
print("Columns match AnandVihar:", list(patia_transformed.columns) == list(anandvihar.columns))
print("\nDtypes match:")
print(patia_transformed.dtypes)
print("\nShape:", patia_transformed.shape)
print("\nSample rows:")
print(patia_transformed.head())

# ── Save ─────────────────────────────────────────────────────────────────────
patia_transformed.to_csv(r'D:\AQI_Project\Data\cleaned\Patia_AQI_data.csv', index=False)
print("\nSaved to Patia_AQI_data.csv")
