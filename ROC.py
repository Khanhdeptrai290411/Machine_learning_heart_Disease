import matplotlib.pyplot as plt
import numpy as np

# Giả lập dữ liệu cho ROC
# AUC = 1: TPR = [0, 1, 1], FPR = [0, 0, 1]
# AUC = 0.5: TPR = FPR = [0, 0.5, 1]
# AUC = 0: TPR = [0, 0, 1], FPR = [0, 1, 1]

fpr_perfect = [0, 0, 1]
tpr_perfect = [0, 1, 1]
fpr_random = [0, 0.5, 1]
tpr_random = [0, 0.5, 1]
fpr_reverse = [0, 1, 1]
tpr_reverse = [0, 0, 1]

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.plot(fpr_perfect, tpr_perfect, label='AUC = 1 (Perfect)', color='blue', linewidth=2)
plt.plot(fpr_random, tpr_random, label='AUC = 0.5 (Random)', color='green', linestyle='--')
plt.plot(fpr_reverse, tpr_reverse, label='AUC = 0 (Reverse)', color='red', linewidth=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Baseline')

# Cấu hình biểu đồ
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for AUC = 1, 0.5, 0')
plt.legend()
plt.grid(True)
plt.show()