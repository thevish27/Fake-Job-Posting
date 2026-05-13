import pandas as pd
from data_preprocessing import preprocess_data

# Load raw dataset
data_path = "../data/raw/DataSet.csv"
df = pd.read_csv(data_path)

print("Original shape:", df.shape)

# Apply preprocessing
processed_df = preprocess_data(df)

print("Processed shape:", processed_df.shape)

# Show sample rows
print(processed_df.head())

# Save processed data
output_path = "../data/processed/processed_jobs.csv"
processed_df.to_csv(output_path, index=False)

print("Processed data saved at:", output_path)
