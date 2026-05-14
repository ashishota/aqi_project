import pandas as pd

# Load both datasets
anandvihar = pd.read_csv('D:/AQI_Project/Data/cleaned/AnandVihar_AQI_data.csv')
jayanagar = pd.read_csv('D:/AQI_Project/Data/raw/Jayanagar_AQI.csv')

print("AnandVihar columns:", list(anandvihar.columns))
print("jayanagar columns (before):", list(jayanagar.columns))

# ── Transformation ──────────────────────────────────────────────────────────
# jayanagar already has all the required sub-columns and AQI fields.
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

jayanagar_transformed = jayanagar[target_columns].copy()

# Ensure dtypes match AnandVihar exactly
float_cols = ['PM25_sub', 'PM10_sub', 'NO2_sub', 'SO2_sub',
              'NH3_sub', 'CO_sub', 'O3_sub', 'AQI']
jayanagar_transformed[float_cols] = jayanagar_transformed[float_cols].astype(float)
jayanagar_transformed['Timestamp'] = jayanagar_transformed['Timestamp'].astype(str)
jayanagar_transformed['AQI_Category'] = jayanagar_transformed['AQI_Category'].astype(str)

# ── Validation ───────────────────────────────────────────────────────────────
print("\njayanagar columns (after):", list(jayanagar_transformed.columns))
print("Columns match AnandVihar:", list(jayanagar_transformed.columns) == list(anandvihar.columns))
print("\nDtypes match:")
print(jayanagar_transformed.dtypes)
print("\nShape:", jayanagar_transformed.shape)
print("\nSample rows:")
print(jayanagar_transformed.head())

# ── Save ─────────────────────────────────────────────────────────────────────
jayanagar_transformed.to_csv(r'D:\AQI_Project\Data\cleaned\Jayanagar_AQI_data.csv', index=False)
print("\nSaved to Jayanagar_AQI_data.csv")
