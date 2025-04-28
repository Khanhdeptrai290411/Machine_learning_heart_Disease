import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ file heart.csv
data = pd.read_csv('heart.csv')

# Tách đặc trưng (X) và biến mục tiêu (y)
X = data.drop('target', axis=1)  # 13 đặc trưng
y = data['target']  # 0 hoặc 1

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Giảm chiều từ 13 chiều xuống 2 chiều bằng PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Huấn luyện SVM với kernel tuyến tính trên dữ liệu đã giảm chiều
svm_pca = SVC(kernel='linear', C=1.0)
svm_pca.fit(X_train_pca, y_train)

# Dự đoán và tính độ chính xác
y_pred_pca = svm_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"Độ chính xác (Linear Kernel, sau PCA): {accuracy_pca:.2f}")

# Vẽ siêu phẳng trong không gian 2D (dữ liệu đã giảm chiều)
h = .02  # Bước lưới
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Dự đoán trên lưới
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Vẽ siêu phẳng và lề
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Thành phần chính 1 (PCA)')
plt.ylabel('Thành phần chính 2 (PCA)')
plt.title('Siêu phẳng SVM (Linear Kernel) sau khi giảm chiều từ 13 chiều xuống 2 chiều')

# Vẽ vector hỗ trợ
support_vectors_pca = svm_pca.support_vectors_
plt.scatter(support_vectors_pca[:, 0], support_vectors_pca[:, 1], s=100, facecolors='none', edgecolors='k', label='Vector hỗ trợ')
plt.legend()
plt.show()

# In số vector hỗ trợ
print(f"Số vector hỗ trợ: {len(svm_pca.support_vectors_)}")

# In tỷ lệ phương sai được giải thích bởi 2 thành phần chính
print(f"Tỷ lệ phương sai được giải thích bởi 2 thành phần chính: {pca.explained_variance_ratio_}")