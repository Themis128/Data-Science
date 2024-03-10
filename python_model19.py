import os
import textwrap
import inspect
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure the output directory exists
output_dir = "Model Report & Graphs"
os.makedirs( output_dir, exist_ok=True )


# Helper Functions
def preprocess_data( data, categorical_columns ):
    """
    Preprocesses the given data by handling missing data and encoding categorical columns.

    Args:
    - data: Pandas DataFrame containing the data.
    - categorical_columns: List of columns in `data` that are categorical.

    Returns:
    - The preprocessed data.
    """
    # Handling missing data
    impute = SimpleImputer( strategy='most_frequent' )
    data [categorical_columns] = impute.fit_transform( data [categorical_columns] )

    # Encoding categorical columns
    le = LabelEncoder()
    for column in categorical_columns:
        data [column] = le.fit_transform( data [column] )
    return data


def plot_accuracy_graph( model_configurations, accuracies ):
    """
    Plots the relationship between model configurations (specifically the number of estimators)
    and their corresponding accuracies.

    Args:
    - model_configurations: A list of the number of estimators used in the RandomForestClassifier.
    - accuracies: A list of accuracy scores corresponding to each configuration.
    """
    plt.figure( figsize=(10, 6) )
    plt.plot( model_configurations, accuracies, marker='o', linestyle='-', color='blue' )
    plt.title( 'Accuracy vs. Number of Estimators' )
    plt.xlabel( 'Number of Estimators' )
    plt.ylabel( 'Accuracy' )
    plt.xticks( model_configurations )
    plt.grid( True )

    # Save the plot
    plt.savefig( os.path.join( output_dir, 'accuracy_vs_estimators.png' ) )
    plt.close()
    print( f"Accuracy graph saved to {output_dir}/accuracy_vs_estimators.png" )


# Define other helper functions (e.g., `train_random_forest_classifier`, `save_plot`, etc.) here...

def plot_evaluation_metrics( metrics, metric_names ):
    """
    Plots evaluation metrics for a model.

    Args:
    - metrics: A list of metric scores (e.g., precision, recall, f1-score).
    - metric_names: A list of names corresponding to the metrics in `metrics`.
    """
    plt.figure( figsize=(10, 6) )
    bar_positions = range( len( metrics ) )
    plt.bar( bar_positions, metrics, color='skyblue' )
    plt.xticks( bar_positions, metric_names )
    plt.title( 'Model Evaluation Metrics' )
    description = f"Highest metric is {metric_names [metrics.index( max( metrics ) )]} with a score of {max( metrics ):.2f}."
    plt.figtext( 0.5, -0.1, description, wrap=True, horizontalalignment='center', fontsize=10 )
    plt.ylabel( 'Scores' )
    plt.savefig( os.path.join( output_dir, 'model_evaluation_metrics.png' ) )
    plt.close()
    print( f"Model evaluation metrics graph saved to {output_dir}/model_evaluation_metrics.png" )


# Define other visualization functions (e.g., `plot_global_job_distribution`, `plot_salary_distribution`, etc.) here...

def run_visualizations( data ):
    """
    Runs a series of visualization functions on the data.

    Args:
    - data: Pandas DataFrame containing the data.
    """
    plot_salary_distribution( data )
    plot_job_category_frequency( data )
    plot_global_job_distribution( data, geo_data_path='ne_10m_admin_0_countries.shp' )


# Main Workflow
def main():
    dataset_path = 'jobs_in_data.csv'
    data = pd.read_csv( dataset_path )
    categorical_columns = ['job_category', 'salary_currency', 'company_size', 'employee_residence', 'experience_level',
                           'employment_type', 'work_setting', 'company_location']
    data_processed = preprocess_data( data, categorical_columns )

    # Assume 'job_title' is the target for the purpose of this example
    X = data_processed.drop( 'job_title', axis=1 )
    y = data_processed ['job_title']
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

    # Example usage of RandomForestClassifier, assume we're testing different configurations
    model_configurations = [10, 50, 100, 200]
    accuracies = []
    for config in model_configurations:
        model = RandomForestClassifier( n_estimators=config, random_state=42 )
        model.fit( X_train, y_train )
        y_pred = model.predict( X_test )
        accuracies.append( accuracy_score( y_test, y_pred ) )

    plot_accuracy_graph( model_configurations, accuracies )

    # Calculate and plot evaluation metrics
    precision = precision_score( y_test, y_pred, average='macro', zero_division=1 )
    recall = recall_score( y_test, y_pred, average='macro', zero_division=1 )
    f1 = f1_score( y_test, y_pred, average='macro' )
    plot_evaluation_metrics( [precision, recall, f1], ['Precision', 'Recall', 'F1 Score'] )

    # Visualization calls
    run_visualizations( data )


if __name__ == "__main__":
    main()
