import pandas as pd

# Load both datasets
anandvihar = pd.read_csv('D:/AQI_Project/Data/cleaned/AnandVihar_AQI_data.csv')
colaba = pd.read_csv('D:/AQI_Project/Data/raw/Colaba_AQI.csv')

print("AnandVihar columns:", list(anandvihar.columns))
print("Colaba columns (before):", list(colaba.columns))

# ── Transformation ──────────────────────────────────────────────────────────
# Colaba already has all the required sub-columns and AQI fields.
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

colaba_transformed = colaba[target_columns].copy()

# Ensure dtypes match AnandVihar exactly
float_cols = ['PM25_sub', 'PM10_sub', 'NO2_sub', 'SO2_sub',
              'NH3_sub', 'CO_sub', 'O3_sub', 'AQI']
colaba_transformed[float_cols] = colaba_transformed[float_cols].astype(float)
colaba_transformed['Timestamp'] = colaba_transformed['Timestamp'].astype(str)
colaba_transformed['AQI_Category'] = colaba_transformed['AQI_Category'].astype(str)

# ── Validation ───────────────────────────────────────────────────────────────
print("\nColaba columns (after):", list(colaba_transformed.columns))
print("Columns match AnandVihar:", list(colaba_transformed.columns) == list(anandvihar.columns))
print("\nDtypes match:")
print(colaba_transformed.dtypes)
print("\nShape:", colaba_transformed.shape)
print("\nSample rows:")
print(colaba_transformed.head())

# ── Save ─────────────────────────────────────────────────────────────────────
colaba_transformed.to_csv(r'D:\AQI_Project\Data\cleaned\Colaba_AQI_data.csv', index=False)
print("\nSaved to Colaba_AQI_data.csv")
