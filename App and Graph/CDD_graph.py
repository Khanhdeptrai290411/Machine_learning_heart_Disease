import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df = pd.read_csv('History_CDD\model_comparison_results_CDD(3).csv')

# Debug: Print column names to verify
print(df.columns)

# Clean column names by removing leading/trailing spaces
df.columns = df.columns.str.strip()

# Clean model names for display
df['Model'] = df['Model'].replace({
    'logistic_regression_model_ML(CDD2).pkl': 'Logistic Regression',
    'svm_model_CDD(final).pkl': 'SVM',
    'xgboost_model_CDD(final).json': 'XGBoost'
})

# Metrics to plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss']

# Set up the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.15
index = range(len(df['Model']))

# Plot bars for each metric
for i, metric in enumerate(metrics):
    plt.bar([x + bar_width * i for x in index], df[metric], bar_width, label=metric)

# Customize the chart
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison CDD')
plt.xticks([i + bar_width * 2 for i in index], df['Model'])
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('model_comparison_bar_chart22.png')
plt.show()