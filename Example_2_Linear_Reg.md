# Employee Revenue Prediction Using Linear Regression

## Overview

This project applies **Linear Regression** to predict employee revenue based on their **hourly rate** and **hours worked**. The dataset includes billing and employee records, and the model is trained to estimate total revenue.

This example showcases:

- **Data preprocessing:** Merging employee rate data with billing records and handling missing values.
- **Feature engineering:** Calculating revenue as a product of hours worked and employee rate.
- **Machine learning pipeline:** Training a regression model to predict revenue.
- **Performance evaluation:** Using Mean Squared Error (MSE) and R-squared (R²) to assess model accuracy.
- **Data visualization:** Plotting actual vs. predicted revenue for comparison.

## Dataset

The dataset consists of two Excel files:

- **billing_cleaned.xlsx**: Contains billing records, including hours worked.
- **employees_cleaned.xlsx**: Contains employee information, including hourly rates.

## Workflow

### Step 1: Load the Datasets

```python
import pandas as pd

# File paths to the datasets
billing_path = "Files/Data/billing_cleaned.xlsx"  # Replace with the correct file path
employees_path = "Files/Data/employees_cleaned.xlsx"  # Replace with the correct file path

# Load datasets
billing = pd.read_excel(billing_path)
employees = pd.read_excel(employees_path)
```

### Step 2: Merge Employee Rate into Billing Data

```python
# Merge 'rate' column from employees into billing
billing = pd.merge(billing, employees[['employee_id', 'rate']], on='employee_id', how='left')
```

### Step 3: Handle Missing Values

```python
# Replace missing rates with the mean
billing['rate'] = billing['rate'].fillna(billing['rate'].mean())

# Calculate revenue
billing['revenue'] = billing['hours'] * billing['rate']
```

### Step 4: Prepare Features and Target Variable

```python
# Define features and target
X = billing[['rate', 'hours']]  # Features: employee rate and hours worked
y = billing['revenue']  # Target: total revenue generated
```

### Step 5: Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 6: Train the Linear Regression Model

```python
from sklearn.linear_model import LinearRegression

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
```

### Step 7: Make Predictions

```python
# Predict revenue
y_pred = model.predict(X_test)
```

### Step 8: Evaluate the Model

```python
from sklearn.metrics import mean_squared_error, r2_score

# Print evaluation metrics
print("\nRegression Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))
```

### Step 9: Visualize Actual vs. Predicted Revenue

```python
import matplotlib.pyplot as plt

# Scatter plot of actual vs. predicted revenue
plt.figure(figsize=(6, 4))
plt.scatter(X_test['hours'], y_test, color='blue', label='Actual Revenue')
plt.scatter(X_test['hours'], y_pred, color='red', label='Predicted Revenue')
plt.xlabel('Hours Worked')
plt.ylabel('Revenue')
plt.title('Actual vs Predicted Revenue')
plt.legend()
plt.show()
```

## Conclusion

This project demonstrates how **Linear Regression** can be used to predict employee revenue based on hourly rates and hours worked. The model provides insights into billing trends and can be further enhanced with additional features.

