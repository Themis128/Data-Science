
# Import necessary libraries
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the paths
venv_path = os.getenv('VIRTUAL_ENV')  # Ensure that the virtual environment is activated
if venv_path is None:
    raise ValueError("Virtual environment path not found. Please ensure the virtual environment is activated.")
data_path = os.path.join(venv_path, 'data.csv')  # Path to the CSV file in the venv
output_path = r'C:\Users\baltz\OneDrive - Ελληνικό Ανοικτό Πανεπιστήμιο\Επιφάνεια εργασίας\Model_Graphs'  # Output path

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load dataset
data = pd.read_csv(data_path, delimiter=';')

# Define columns based on your dataset
categorical_columns = ['job_title', 'job_category', 'salary_currency', 'employee_residence', 'experience_level', 'employment_type', 'work_setting', 'company_location', 'company_size']  # Replace with your actual categorical columns
numerical_columns = ['work_year', 'salary', 'salary_in_usd']     # Replace with your actual numerical columns

# Check if the target column exists in the dataset
if 'salary_in_usd' not in data.columns:
    raise ValueError("Column 'salary_in_usd' not found in the dataset.")

# Preprocessing: One-Hot Encoding for categorical data and Standard Scaling for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ])

# Apply the transformations to the data
data_processed = preprocessor.fit_transform(data)

# Split the data into features and target variable
y = data['salary_in_usd']
X = data_processed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Calculate and print metrics
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Further analysis and visualization code goes here

# Scatter plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Salaries')
regression_plot_dir = output_path
if not os.path.exists(regression_plot_dir):
    os.makedirs(regression_plot_dir)
regression_plot_path = os.path.join(regression_plot_dir, 'regression_plot.png')
plt.savefig(regression_plot_path)
print(f"Regression scatter plot saved to: {regression_plot_path}")

# K-Means Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans_labels = kmeans.labels_

# DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
dbscan_labels = dbscan.labels_

# Visualization and Analysis
# Ensure 'company_location' column exists for geographical analysis
if 'company_location' not in data.columns:
    raise ValueError("Column 'company_location' not found in the dataset for geographical analysis.")

# Group by 'company_location' and calculate relevant statistics
geo_data = data.groupby('company_location').agg({
    'salary_in_usd': ['mean', 'median', 'count']
}).reset_index()
geo_data.columns = ['Company Location', 'Average Salary', 'Median Salary', 'Job Count']

# Visualize the geographical distribution
plt.figure(figsize=(12, 6))
sns.barplot(x='Company Location', y='Job Count', data=geo_data.sort_values('Job Count', ascending=False))
plt.xticks(rotation=45)
plt.title('Number of Data Science Jobs by Company Location')
plt.xlabel('Company Location')
plt.ylabel('Number of Jobs')
plt.tight_layout()
plot_path_jobs = os.path.join(output_path, 'jobs_by_location.png')
plt.savefig(plot_path_jobs)
print(f"Job count by location plot saved to: {plot_path_jobs}")

plt.figure(figsize=(12, 6))
sns.barplot(x='Company Location', y='Average Salary', data=geo_data.sort_values('Average Salary', ascending=False))
plt.xticks(rotation=45)
plt.title('Average Data Science Salary by Company Location')
plt.xlabel('Company Location')
plt.ylabel('Average Salary')
plt.tight_layout()
plot_path_salary = os.path.join(output_path, 'salary_by_location.png')
plt.savefig(plot_path_salary)
print(f"Average salary by location plot saved to: {plot_path_salary}")

# Save the Geographical Analysis Data
geo_data_path = os.path.join(output_path, 'geographical_analysis.xlsx')
geo_data.to_excel(geo_data_path, index=False)
print(f"Geographical analysis data saved to: {geo_data_path}")
