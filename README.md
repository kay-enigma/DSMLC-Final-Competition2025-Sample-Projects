# Data Science and Machine Learning Club (DSMLC) Projects

## Overview

This repository contains **example projects** aligned with the **Advanced Category** of the **Final Competition** hosted by the **University of Calgary’s Data Science and Machine Learning Club**. These examples showcase **Machine Learning (ML) approaches** and **data-driven insights** using real-world datasets.

Each project follows a **structured pipeline**:
1. **Data Preprocessing** – Cleaning and handling missing values.
2. **Feature Engineering** – Creating meaningful features for ML models.
3. **Model Training** – Applying Logistic and Linear Regression.
4. **Model Evaluation** – Assessing performance using relevant metrics.
5. **Data Visualization** – Communicating insights effectively.

These projects **serve as examples**, and competitors are encouraged to **explore other ML models, feature engineering techniques, and visualization tools**.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Example 1: Employee Status Classification (Logistic Regression)](#example-1-employee-status-classification-logistic-regression)
- [Example 2: Employee Revenue Prediction (Linear Regression)](#example-2-employee-revenue-prediction-linear-regression)
- [How to Run the Code](#how-to-run-the-code)
- [Dataset Details](#dataset-details)
- [Future Improvements](#future-improvements)

---

## Project Structure

```
DSMLC_Projects-main/
│── Example_1_Regression.md      # Documentation for Example 1
│── Example_2_Linear_Reg.md      # Documentation for Example 2
│── README.md                    # Main documentation (this file)
│── Files/
│   ├── Code/
│   │   ├── Example1.py          # Python script for Example 1 (Logistic Regression)
│   │   ├── Example2.py          # Python script for Example 2 (Linear Regression)
│   ├── Data/
│   │   ├── employees_cleaned.xlsx  # Employee data
│   │   ├── billing_cleaned.xlsx    # Billing data
│   ├── Screenshots/               # Images for visualizations
```

---

## Example 1: Employee Status Classification (Logistic Regression)

This project **predicts employee status (Active or Terminated)** using **Logistic Regression**.

### Workflow
1. **Preprocessing:**
   - Convert `hire_date` to datetime.
   - Create `status` column based on `hire_date` and `target_ratio`.
2. **Feature Engineering:** 
   - Use `target_ratio` as the predictor variable.
3. **Model Training:**
   - Train a **Logistic Regression model** on the dataset.
4. **Evaluation:**
   - Assess model performance using **precision, recall, and F1-score**.
5. **Visualization:**
   - Create bar charts to compare model performance metrics.

📄 **See full details in [Example_1_Regression.md](Example_1_Regression.md)**.

---

## Example 2: Employee Revenue Prediction (Linear Regression)

This project **predicts total revenue generated by employees** using **Linear Regression**.

### Workflow
1. **Preprocessing:**
   - Merge employee hourly rates with billing records.
   - Handle missing values.
2. **Feature Engineering:** 
   - Create `revenue = hours_worked × hourly_rate`.
3. **Model Training:**
   - Train a **Linear Regression model**.
4. **Evaluation:**
   - Use **Mean Squared Error (MSE)** and **R² score** to assess performance.
5. **Visualization:**
   - Scatter plot of actual vs. predicted revenue.

📄 **See full details in [Example_2_Linear_Reg.md](Example_2_Linear_Reg.md)**.

---

## How to Run the Code

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib numpy openpyxl
```

### Steps to Run
```bash
cd Files/Code

# Run Example 1
python Example1.py

# Run Example 2
python Example2.py
```

---

## Dataset Details

The dataset consists of two key files:

### 1. `employees_cleaned.xlsx`
- **employee_id**: Unique identifier for each employee.
- **hire_date**: Date when the employee was hired.
- **target_ratio**: The percentage of time an employee is expected to spend on billable work.
- **rate**: Hourly rate of the employee.
- **status**: Employee status (Active or Terminated).

### 2. `billing_cleaned.xlsx`
- **employee_id**: Foreign key linking to employees.
- **hours**: Number of hours worked.
- **revenue**: Total revenue generated (`hours worked × hourly rate`).

---

## Future Improvements
- **Expand Feature Engineering**: Add new variables such as department, years_of_experience, etc.
- **Test Different Models**: Experiment with Decision Trees, Random Forests, or Neural Networks.
- **Hyperparameter Tuning**: Optimize model performance using GridSearchCV.
- **Interactive Dashboards**: Use Tableau or Power BI to enhance data storytelling.
- **Graph Database Implementation**: Try Neo4j for relationship-based insights.

---

## Conclusion

This repository serves as a resource for participants in the **Final Competition’s Machine Learning Approach** category. 
It provides structured examples of **Logistic Regression (classification)** and **Linear Regression (prediction)** applied to real-world HR and financial datasets.

📢 **Contributions & Enhancements:**
Participants are encouraged to modify, extend, and improve these models to develop unique insights and competitive solutions for the Final Competition.

🚀 **Start exploring and get ready for the competition!** 🚀
