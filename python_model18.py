import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Refresh the output directory
output_dir = "Model Report & Graphs"

# Check if the directory exists
if os.path.exists( output_dir ):
    # If it exists, delete it and recreate
    shutil.rmtree( output_dir )
os.makedirs( output_dir, exist_ok=True )

# Job category mapping
category_mapping = {
    0: 'Data Scientist',
    1: 'Data Analyst',
    2: 'Machine Learning Engineer',
    3: 'Business Intelligence Analyst',
    4: 'Data Engineer',
    5: 'Database Administrator',
    6: 'Statistician',
    7: 'Data Architect',
    8: 'Big Data Engineer',
    9: 'BI Developer'
}


# Preprocess data
def preprocess_data( data, categorical_columns ):
    impute = SimpleImputer( strategy='most_frequent' )
    data [categorical_columns] = impute.fit_transform( data [categorical_columns] )

    # Include 'work_year' column if it's not present
    if 'work_year' not in data.columns:
        data ['work_year'] = pd.to_datetime( data ['start_date'] ).dt.year

    le = LabelEncoder()
    for column in categorical_columns:
        data [column] = le.fit_transform( data [column] )
        if column == 'job_category':
            data [column] = data [column].map( category_mapping )
    return data


# Filter top 10 categories
def filter_top_10_categories( data ):
    top_10_categories = data ['job_category'].value_counts().nlargest( 10 ).index
    return data [data ['job_category'].isin( top_10_categories )]


# Visualization functions (excluding plot_global_job_distribution_interactive)

def plot_job_category_frequency(data, output_dir="Model Report & Graphs"):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create the bar plot
    plt.figure(figsize=(10, 8))
    chart = sns.countplot(y='job_category', data=data, order=data['job_category'].value_counts().index)

    plt.title('Frequency of Job Categories')
    plt.xlabel('Frequency')
    plt.ylabel('Job Category')

    # Set the y-axis ticks and labels
    num_ticks = len(data['job_category'].value_counts())
    tick_positions = range(num_ticks)
    tick_labels = data['job_category'].value_counts().index
    plt.yticks(tick_positions, tick_labels)

    # Improve readability of the y-axis labels
    chart.set_yticklabels(chart.get_yticklabels(), rotation=0, horizontalalignment='right')

    plt.tight_layout()

    # Save the plot to the specified directory
    plot_path = os.path.join(output_dir, 'job_category_frequency.png')
    plt.savefig(plot_path)

    # Optionally, you can still display the plot if desired
    # plt.show()

    # Clear the current figure to free memory and avoid interference with subsequent plots
    plt.clf()


def plot_salary_distribution_by_category(data, output_dir="Model Report & Graphs"):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Calculate the average salary for each job category
    avg_salary_by_category = data.groupby('job_category')['salary_in_usd'].mean().sort_values()

    # Create a bar plot
    plt.figure(figsize=(10, 8))
    avg_salary_by_category.plot(kind='barh', color='skyblue')
    plt.xlabel('Average Salary in USD')
    plt.ylabel('Job Category')
    plt.title('Average Salary by Job Category')
    plt.tight_layout()

    # Save the plot to the specified directory
    plot_path = os.path.join(output_dir, 'salary_distribution_by_category.png')
    plt.savefig(plot_path)

    # Optionally, you can still display the plot if desired
    # plt.show()

    # Clear the current figure to free memory and avoid interference with subsequent plots
    plt.clf()



def plot_experience_level_distribution( data, output_dir="Model Report & Graphs" ):
    experience_levels = data ['experience_level'].value_counts()

    plt.figure( figsize=(8, 8) )
    experience_levels.plot( kind='pie', autopct='%1.1f%%', startangle=140 )
    plt.ylabel( '' )
    plt.title( 'Experience Level Distribution for Data Professionals' )
    plt.tight_layout()
    plt.savefig( os.path.join( output_dir, 'experience_level_distribution.png' ) )
    plt.close()


def plot_skills_demand(data, output_dir="Model Report & Graphs"):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Access the 'job_title' column instead of 'required_skills'
    all_job_titles = data['job_title'].str.split(',').explode().str.strip()
    job_title_frequency = all_job_titles.value_counts().head(20)  # Focus on top 20 job titles for clarity

    plt.figure(figsize=(10, 8))
    sns.barplot(x=job_title_frequency, y=job_title_frequency.index)
    plt.xlabel('Frequency')
    plt.ylabel('Job Titles')
    plt.title('Top 20 In-Demand Job Titles')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'job_title_demand_distribution.png'))
    plt.close()


# Dynamic Visualization Runner
def run_visualizations( data, output_dir="Model Report & Graphs" ):
    visualization_functions = [
        lambda: plot_job_category_frequency( data, output_dir ),
        lambda: plot_salary_distribution_by_category( data, output_dir ),
        lambda: plot_experience_level_distribution( data, output_dir ),
        lambda: plot_skills_demand( data, output_dir )
    ]
    for func in visualization_functions:
        func()


# Main workflow
def main():
    dataset_path = 'jobs_in_data.csv'  # Adjust the path as necessary
    data = pd.read_csv( dataset_path )
    categorical_columns = ['job_category', 'salary_currency', 'company_size', 'employee_residence', 'experience_level',
                           'employment_type', 'work_setting', 'company_location']

    data_processed = preprocess_data( data, categorical_columns )
    data_filtered = filter_top_10_categories( data_processed )

    # Visualization calls with dynamic execution
    run_visualizations( data_filtered, output_dir )


if __name__ == "__main__":
    main()
