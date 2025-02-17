# SQL Guide: Working with a Dataset Using SQL and Queries

## Overview
This guide demonstrates how to create tables, run queries, and retrieve meaningful insights from a dataset using SQL. The dataset includes billing transactions, employees, projects, and branches. You will learn how to structure your database, execute queries to join tables, and apply filtering techniques. Additionally, common issues and solutions are covered to help troubleshoot any potential problems.

## 🛠 Step 1: Creating Tables

The following SQL statements create tables for the required data structure:

```sql
-- Create table for projects_cleaned
CREATE TABLE projects_cleaned (
    project_key INTEGER PRIMARY KEY,
    branch_id TEXT,
    project_leader TEXT,
    project_coordinator TEXT,
    project_type TEXT
);

-- Create table for employees_cleaned
CREATE TABLE employees_cleaned (
    employee_id INTEGER PRIMARY KEY,
    employee_name TEXT,
    hire_date TEXT,
    branch_id TEXT,
    coach TEXT,
    target_ratio INTEGER,
    rate INTEGER
);

-- Create table for branches_cleaned
CREATE TABLE branches_cleaned (
    branch_id TEXT PRIMARY KEY,
    branch_name TEXT
);

-- Create table for billing_cleaned
CREATE TABLE billing_cleaned (
    billing_id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    project_key INTEGER,
    regular_hours REAL,
    transfer_date TEXT,
    category TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees_cleaned(employee_id),
    FOREIGN KEY (project_key) REFERENCES projects_cleaned(project_key)
);
```

---

## 🛠 Step 2: Running Queries

### Query: Retrieve Billing Details with Employee and Branch Info

This query joins multiple tables to retrieve billing details along with employee and branch information:

```sql
SELECT 
    b.billing_id,
    b.employee_id,
    b.project_key AS project_id,
    b.regular_hours AS amount,
    b.transfer_date AS billing_date,
    e.employee_name,
    e.rate AS employee_rate,
    br.branch_name AS branch_name,
    p.project_leader,
    p.project_coordinator,
    p.project_type
FROM 
    billing_cleaned b
JOIN 
    employees_cleaned e 
ON 
    b.employee_id = e.employee_id
JOIN 
    branches_cleaned br 
ON 
    e.branch_id = br.branch_id
JOIN 
    projects_cleaned p 
ON 
    b.project_key = p.project_key;
```

### ✅ **Explanation of the Query**
- `billing_cleaned` contains transaction records.
- The `JOIN` operations fetch related data:
  - `employees_cleaned` to get the employee’s name and rate.
  - `branches_cleaned` to retrieve the branch name.
  - `projects_cleaned` to include project details.
- The final result provides a complete view of each billing entry.

---

## 🛠 Step 3: Filtering Data

### Example: Retrieve Only Billable Entries
```sql
SELECT * FROM billing_cleaned WHERE category = 'Billable';
```
### Example: Retrieve Data for a Specific Employee
```sql
SELECT * FROM employees_cleaned WHERE employee_id = 10286;
```
### Example: Retrieve All Projects in a Specific Branch
```sql
SELECT * FROM projects_cleaned WHERE branch_id = '1-10';
```

---

## 🛠 Step 4: Common Issues and Solutions

### 🔍 Issue: No Data Appears in Query Results
**Possible Cause:** One or more tables are empty. 
**Solution:** Check if tables contain data.
```sql
SELECT COUNT(*) FROM billing_cleaned;
SELECT COUNT(*) FROM employees_cleaned;
SELECT COUNT(*) FROM branches_cleaned;
SELECT COUNT(*) FROM projects_cleaned;
```

### 🔍 Issue: Some Employees or Branches Are Missing in Results
**Possible Cause:** Missing foreign key references in related tables.
**Solution:** Find missing records.
```sql
SELECT employee_id FROM billing_cleaned WHERE employee_id NOT IN (SELECT employee_id FROM employees_cleaned);
SELECT branch_id FROM employees_cleaned WHERE branch_id NOT IN (SELECT branch_id FROM branches_cleaned);
```

### 🔍 Issue: Data Type Mismatches in Joins
**Possible Cause:** Data type inconsistencies between joined columns.
**Solution:** Check column data types.
```sql
PRAGMA table_info(billing_cleaned);
PRAGMA table_info(employees_cleaned);
PRAGMA table_info(branches_cleaned);
PRAGMA table_info(projects_cleaned);
```

---

This guide provides an overview of table creation, essential queries, and data retrieval techniques. Let me know if you need further refinements! 🚀
