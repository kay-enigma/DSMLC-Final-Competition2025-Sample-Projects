
# Employee Status Classification Using Logistic Regression - Example 1 

# Overview

This project serves as a practical **Machine Learning (ML) example** aligned with the **Advanced Category**
of the **Final Competition** hosted by the **University of Calgary’s Data Science and Machine Learning Club**. 
The dataset used in this project is similar to those provided in the competition, where participants are expected 
to analyze real-world data, extract insights, and apply machine learning techniques.

In this example, we predict **employee status (Active or Terminated)** using **Logistic Regression**, demonstrating
how machine learning can be used for **classification tasks**. The dataset contains employee records, including 
**hire date** and **target ratio**, which are key factors in determining employment status.

## This example showcases:

- **Data preprocessing:** Handling datetime conversions and filtering based on conditions.
- **Feature engineering:** Creating a `status` column based on hire date and target ratio.
- **Machine learning pipeline:** Training a classification model to predict employee status.
- **Performance evaluation:** Using precision, recall, and F1-score to assess model performance.
- **Data visualization:** Presenting the model's performance metrics using a bar chart.

This project aligns with the **Machine Learning Approach category** of the competition, where participants are 
expected to **clean and preprocess data, select appropriate ML algorithms, evaluate model performance, 
and derive insights**. It serves as an example of how competitors can structure their projects, apply 
statistical methods, and visualize results to communicate findings effectively.

## Dataset

The dataset is stored in an Excel file (`employees_cleaned.xlsx`) and contains employee records, including:

- **hire_date**: The date the employee was hired.
- **target_ratio**: A numerical value used to determine the employee’s activity.
- **status**: A generated column that labels employees as **"Active"** or **"Terminated"** based on conditions.

## Workflow

### Step 1: Load and Process the Data

- The dataset is loaded using **Pandas**.
- The `hire_date` column is converted to **datetime** format.
- A **new column** (`status`) is created based on:
  - If the employee was hired **more than 5 years ago**, they are labeled as `"Terminated"`.
  - If their **target ratio is 0**, they are also labeled as `"Terminated"`.
  - Otherwise, they are labeled as `"Active"`.

```python
import pandas as pd

# Load dataset
file_path = "employees_cleaned.xlsx"  # Replace with the actual file path
employees = pd.read_excel(file_path)

# Convert hire_date to datetime format
employees['hire_date'] = pd.to_datetime(employees['hire_date'])

# Define current year
current_year = pd.Timestamp.now().year

# Create 'status' column
employees['status'] = employees.apply(
    lambda row: 'Terminated' if row['hire_date'].year < (current_year - 5) or row['target_ratio'] == 0 else 'Active',
    axis=1
)
```

### Step 2: Define Features and Target Variable

- The **only feature used** is `target_ratio`, but this can be modified for further analysis.
- The **target variable (y)** is binary:
  - "Terminated" → 1
  - "Active" → 0

```python
# Define feature (X) and target (y)
X = employees[['target_ratio']]
y = employees['status'].apply(lambda x: 1 if x == 'Terminated' else 0)
```

### Step 3: Train and Test the Model

- The data is split into **training (70%)** and **testing (30%)** sets.
- A **Logistic Regression model** is trained on the training data.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Step 4: Predict and Evaluate Model Performance

- The model makes predictions on the test data.
- A **classification report** is generated, displaying:
  - **Precision**
  - **Recall**
  - **F1-Score**

```python
from sklearn.metrics import classification_report

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### Step 5: Visualize Model Performance

A **bar chart** is created using Matplotlib to compare the precision, recall, and F1-score for each class (Active vs. Terminated).

```python
import matplotlib.pyplot as plt
import numpy as np

# Define classification metrics manually (example values)
metrics = {
    "Active (0)": {"precision": 0.53, "recall": 0.31, "f1-score": 0.39},
    "Terminated (1)": {"precision": 0.80, "recall": 0.91, "f1-score": 0.85}
}

# Data preparation for plotting
categories = list(metrics.keys())
precision = [metrics[cat]['precision'] for cat in categories]
recall = [metrics[cat]['recall'] for cat in categories]
f1_score = [metrics[cat]['f1-score'] for cat in categories]

x = np.arange(len(categories))  # Label locations
width = 0.25  # Bar width

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width, precision, width, label='Precision')
ax.bar(x, recall, width, label='Recall')
ax.bar(x + width, f1_score, width, label='F1-Score')

# Add labels and title
ax.set_xlabel('Class')
ax.set_ylabel('Scores')
ax.set_title('Model Performance by Class')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
```

## Conclusion

This project demonstrates a simple yet effective approach to predicting employee status using **logistic regression**. The model provides insights into employee attrition trends based on the **target ratio** and can be further improved by incorporating additional features.
