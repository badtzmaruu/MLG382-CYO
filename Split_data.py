import os
import pandas as pd
from sklearn.model_selection import train_test_split

#Load dataset
data = pd.read_csv('../Data/weather_cleaned.csv', delimiter=",")

#Split into train/test sets (80/20) with stratified sampling
train_data, test_data = train_test_split(data, test_size=0.2,random_state=42, stratify=data["Rain"]
)

#Save to CSV files inside the existing Data folder
output_path = "../Data"
os.makedirs(output_path, exist_ok=True)

train_data.to_csv(os.path.join(output_path, "train_data.csv"), index=False)
test_data.to_csv(os.path.join(output_path, "test_data.csv"), index=False)

#Print class distribution info
print("\nRain Distribution in Training Set:")
counts = train_data["Rain"].value_counts().sort_index()
percentages = (counts / len(train_data) * 100).round(2)
distribution_df = pd.DataFrame({
    "Count": counts,
    "Percentage (%)": percentages
})

print(distribution_df)