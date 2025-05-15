import matplotlib.pyplot as plt
from xgboost import Booster, plot_importance

# Tạo đối tượng Booster và load model từ file JSON
model = Booster()
model.load_model("CDD_Model/xgboost_model_CDD(final).json")

# Vẽ biểu đồ importance
plot_importance(model)
plt.show()
