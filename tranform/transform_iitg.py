import pandas as pd

# Load both datasets
anandvihar = pd.read_csv('D:/AQI_Project/Data/cleaned/AnandVihar_AQI_data.csv')
iitg = pd.read_csv('D:/AQI_Project/Data/raw/IITG_AQI.csv')

print("AnandVihar columns:", list(anandvihar.columns))
print("iitg columns (before):", list(iitg.columns))

# ── Transformation ──────────────────────────────────────────────────────────
# iitg already has all the required sub-columns and AQI fields.
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

iitg_transformed = iitg[target_columns].copy()

# Ensure dtypes match AnandVihar exactly
float_cols = ['PM25_sub', 'PM10_sub', 'NO2_sub', 'SO2_sub',
              'NH3_sub', 'CO_sub', 'O3_sub', 'AQI']
iitg_transformed[float_cols] = iitg_transformed[float_cols].astype(float)
iitg_transformed['Timestamp'] = iitg_transformed['Timestamp'].astype(str)
iitg_transformed['AQI_Category'] = iitg_transformed['AQI_Category'].astype(str)

# ── Validation ───────────────────────────────────────────────────────────────
print("\niitg columns (after):", list(iitg_transformed.columns))
print("Columns match AnandVihar:", list(iitg_transformed.columns) == list(anandvihar.columns))
print("\nDtypes match:")
print(iitg_transformed.dtypes)
print("\nShape:", iitg_transformed.shape)
print("\nSample rows:")
print(iitg_transformed.head())

# ── Save ─────────────────────────────────────────────────────────────────────
iitg_transformed.to_csv('IITG_AQI_data.csv', index=False)
print("\nSaved to IITG_AQI_data.csv")
