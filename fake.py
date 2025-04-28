from sdv.tabular import GaussianCopula
import pandas as pd

# Load dữ liệu gốc
data = pd.read_csv("Data/Cardiovascular_Disease_Dataset.csv")

# Xóa 'patientid' nếu có
if 'patientid' in data.columns:
    data = data.drop(columns=['patientid'])

# Train mô hình tạo dữ liệu giả
model = GaussianCopula()
model.fit(data)

# Tạo ra 1000 dòng giả
synthetic_data = model.sample(1000)

# Save
synthetic_data.to_csv("Data/Synthetic_Cardiovascular_Disease(2).csv", index=False)

print("✅ Đã tạo dữ liệu giả bằng GaussianCopula thành công!")