# Import necessary libraries
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

# Define the paths
venv_path = os.getenv('VIRTUAL_ENV')  # Ensure the virtual environment is activated
if venv_path is None:
    raise ValueError("Virtual environment path not found. Please ensure the virtual environment is activated.")
data_path = os.path.join(venv_path, 'data.csv')  # Path to the CSV file in the venv
output_path = r'C:\Users\baltz\OneDrive - Ελληνικό Ανοικτό Πανεπιστήμιο\Επιφάνεια εργασίας\Model_Graphs'  # Output path for graphs and Excel files

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load dataset
try:
    data = pd.read_csv(data_path, delimiter=';')
except FileNotFoundError:
    print(f"File not found: {data_path}")
    exit()

# Define columns for pre-processing
categorical_columns = ['job_title', 'job_category', 'salary_currency', 'employee_residence', 'experience_level', 'employment_type', 'work_setting', 'company_location', 'company_size']
numerical_columns = ['work_year', 'salary', 'salary_in_usd']

# Check for target column
if 'salary_in_usd' not in data.columns:
    raise ValueError("Target column 'salary_in_usd' not found in the dataset.")

# Preprocessing: One-Hot Encoding for categorical data and Standard Scaling for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ])

# Apply transformations to the data
data_processed = preprocessor.fit_transform(data)

# Convert processed data to DataFrame
if isinstance(data_processed, np.ndarray):
    processed_data_df = pd.DataFrame(data_processed, columns=preprocessor.get_feature_names_out())
else:
    processed_data_df = pd.DataFrame(data_processed.toarray(), columns=preprocessor.get_feature_names_out())

# Save the pre-processed data to an Excel file
processed_data_path = os.path.join(output_path, 'processed_data.xlsx')
processed_data_df.to_excel(processed_data_path, index=False)
print(f"Pre-processed data saved to: {processed_data_path}")

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

# Scatter plot for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Salaries')
regression_plot_path = os.path.join(output_path, 'regression_plot.png')
plt.savefig(regression_plot_path)
print(f"Regression scatter plot saved to: {regression_plot_path}")

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans_labels = kmeans.labels_

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X)
dbscan_labels = dbscan.labels_

# Visualization and Analysis
# Group by 'company_location' and calculate relevant statistics including 'employment_type'
geo_data = data.groupby('company_location').agg({
    'salary_in_usd': ['mean', 'median', 'max', 'count'],
    'employment_type': lambda x: x.mode()[0]  # Most common employment type
}).reset_index()
geo_data.columns = ['Company Location', 'Average Salary', 'Median Salary', 'Max Salary', 'Job Count', 'Common Employment Type']

# Save the Geographical Analysis Data with Employment Type
geo_data_path = os.path.join(output_path, 'geographical_analysis.xlsx')
geo_data.to_excel(geo_data_path, index=False)
print(f"Geographical analysis data with employment type saved to: {geo_data_path}")

# Visualize the geographical distribution
plt.figure(figsize=(12, 6))
sns.barplot(x='Company Location', y='Job Count', data=geo_data.sort_values('Job Count', ascending=False))
plt.xticks(rotation=45)
plt.title('Number of Data Science Jobs by Company Location')
plt.xlabel('Company Location')
plt.ylabel('Number of Jobs')
plot_path_jobs = os.path.join(output_path, 'jobs_by_location.png')
plt.savefig(plot_path_jobs)
print(f"Job count by location plot saved to: {plot_path_jobs}")

plt.figure(figsize=(12, 6))
sns.barplot(x='Company Location', y='Average Salary', data=geo_data.sort_values('Average Salary', ascending=False))
plt.xticks(rotation=45)
plt.title('Average Data Science Salary by Company Location')
plt.xlabel('Company Location')
plt.ylabel('Average Salary')
plot_path_salary = os.path.join(output_path, 'salary_by_location.png')
plt.savefig(plot_path_salary)
print(f"Average salary by location plot saved to: {plot_path_salary}")
