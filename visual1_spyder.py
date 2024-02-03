# Import necessary libraries
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the paths
data_path = r'C:\Users\baltz\OneDrive - Ελληνικό Ανοικτό Πανεπιστήμιο\Επιφάνεια εργασίας\model\data.csv'
output_path = r'C:\Users\baltz\OneDrive - Ελληνικό Ανοικτό Πανεπιστήμιο\Επιφάνεια εργασίας\Model_Graphs'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load dataset with correct delimiter
data = pd.read_csv(data_path, delimiter=';')

# Define columns for pre-processing
categorical_columns = ['job_title', 'job_category', 'salary_currency', 'employee_residence', 'experience_level', 'employment_type', 'work_setting', 'company_size']
numerical_columns = ['work_year', 'salary', 'salary_in_usd']

# Encode the categorical target variable
le = LabelEncoder()
y_encoded = le.fit_transform(data['company_location'])



# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', StandardScaler(), numerical_columns)
    ])

# Apply transformations to the data
X = preprocessor.fit_transform(data)
y = data['company_location']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)



# Initialize and train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# This is where you calculate the number of unique classes
num_classes = np.unique(np.concatenate((y_test, y_pred))).shape[0]

# Then you can generate the target names
target_names = ['class ' + str(i) for i in range(num_classes)]

# Add debug print statements here
print('Number of classes:', num_classes)
print('Length of target_names:', len(target_names))

# And finally call classification_report
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=1))

# Visualization and Analysis Suggestions:
# 1. Feature Importance Visualization: Show which features are most important for predicting company location.
feature_importances = clf.feature_importances_
feature_names = preprocessor.get_feature_names_out()
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importance for Predicting Company Location')
plt.show()

# 2. Confusion Matrix: Visualize the model's performance across different locations.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
cm = confusion_matrix(y_test, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.show()

# 3. Salary Distribution by Location: Compare the distribution of salaries across different locations.
sns.boxplot(x='company_location', y='salary_in_usd', data=data)
plt.xticks(rotation=45)
plt.title('Salary Distribution by Company Location')
plt.show()

# Note: Update the 'data_path' and 'output_path' variables to match your local environment.
