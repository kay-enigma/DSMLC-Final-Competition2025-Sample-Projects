import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# File path to the dataset
file_path = "employees_cleaned.xlsx"  # Replace with the correct file path

# Step 1: Load the dataset
employees = pd.read_excel(file_path)

# Step 2: Convert hire_date to datetime
employees['hire_date'] = pd.to_datetime(employees['hire_date'])

# Step 3: Create a 'status' column based on hire_date and target_ratio
current_year = pd.Timestamp.now().year
employees['status'] = employees.apply(
    lambda row: 'Terminated' if row['hire_date'].year < (current_year - 5) or row['target_ratio'] == 0 else 'Active',
    axis=1
)

# Step 4: Prepare features (X) and target (y)
X = employees[['target_ratio']]  # Only one feature: target_ratio
y = employees['status'].apply(lambda x: 1 if x == 'Terminated' else 0)  # Binary target: 1 for Terminated, 0 for Active

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train a simple Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


########### Part 2 of the Code ###########
########### We create a visual to understand the above ###############

import matplotlib.pyplot as plt
import numpy as np

# Metrics from the classification report
metrics = {
    "Active (0)": {"precision": 0.53, "recall": 0.31, "f1-score": 0.39},
    "Terminated (1)": {"precision": 0.80, "recall": 0.91, "f1-score": 0.85}
}

# Data preparation for plotting
categories = list(metrics.keys())
precision = [metrics[cat]['precision'] for cat in categories]
recall = [metrics[cat]['recall'] for cat in categories]
f1_score = [metrics[cat]['f1-score'] for cat in categories]

x = np.arange(len(categories))  # the label locations
width = 0.25  # the width of the bars

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - width, precision, width, label='Precision')
bar2 = ax.bar(x, recall, width, label='Recall')
bar3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add labels, title, and legend
ax.set_xlabel('Class')
ax.set_ylabel('Scores')
ax.set_title('Model Performance by Class')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()