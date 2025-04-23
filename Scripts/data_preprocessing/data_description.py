import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime


def describe_data(df):
    print("***Describing the data:***")
    num_rows, num_columns = df.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")

    print("\nColumn details:")
    for column in df.columns:
        col_data = df[column]
        col_dtype = col_data.dtype
        print(f"\nColumn: {column}, Type: {col_dtype}")

        if pd.api.types.is_numeric_dtype(col_data):
            min_val = col_data.min()
            max_val = col_data.max()
            mean_val = col_data.mean()
            median_val = col_data.median()
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Mean: {mean_val:.2f}")
            print(f"  Median: {median_val}")
        elif pd.api.types.is_categorical_dtype(col_data) or col_data.dtype == 'object':
            num_categories = col_data.nunique()
            print(f"  Number of categories: {num_categories}")
            if num_categories <= 10:
                print("  Counts per category:")
                category_counts = col_data.value_counts()
                for index, value in category_counts.items():
                    print(f"    {index}: {value}")
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            min_date = col_data.min()
            max_date = col_data.max()
            print(f"  Date Range: {min_date} to {max_date}")
            print(f"  Number of unique dates: {col_data.nunique()}")
        else:
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:
                print("  Unique values:")
                for val in unique_vals:
                    print(f"    {val}")

    return num_rows, num_columns


def count_nulls(df):
    print("Describing Nulls in the data:")

    null_counts_columns = df.isnull().sum()
    print("Null counts per variable:")
    print(null_counts_columns)

    null_counts_rows = df.isnull().sum(axis=1)
    max_nulls = null_counts_rows.max()
    rows_with_most_nulls = null_counts_rows[null_counts_rows == max_nulls].index.tolist()

    total_rows = len(df)
    rows_with_any_nulls = (null_counts_rows > 0).sum()
    percentage_with_nulls = (rows_with_any_nulls / total_rows) * 100

    print(f"\nRows with the highest number of nulls ({max_nulls} nulls):")
    print(rows_with_most_nulls)
    print(f"Percentage of rows with any nulls: {percentage_with_nulls:.2f}%")

    directory = "Images"
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), gridspec_kw={'width_ratios': [3, 1]})

    sns.histplot(null_counts_rows, bins=max_nulls, kde=False, color='blue', ax=ax1)
    ax1.set_title('Histogram of Nulls Per Row')
    ax1.set_xlabel('Number of Nulls')
    ax1.set_ylabel('Frequency of Rows')
    ax1.grid(True)

    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', color='black', xytext=(0, 5), textcoords='offset points')

    sns.boxplot(y=null_counts_rows, color='green', ax=ax2)
    ax2.set_title('Box Plot of Nulls Per Row')
    ax2.set_ylabel('Number of Nulls')

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{directory}/Null_distributions_{current_time}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved histogram and boxplot as: {filename}")


def describe_numeric(df):
    print("***Reporting on Numeric variables:***")
    numeric_vars = df.select_dtypes(include=['int64', 'float64'])
    descriptions = numeric_vars.describe()
    print(descriptions)

    for column in numeric_vars:
        data = numeric_vars[column].dropna()
        if data.empty:
            print(f"No data available for histogram of {column} after removing NaNs.")
            continue

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), gridspec_kw={'width_ratios': [3, 1]})

        sns.histplot(data, ax=ax1, color='blue', alpha=0.7, kde=False, binwidth=None, element='bars', stat='count')
        bin_width = ax1.patches[0].get_width() if ax1.patches else 0  # Calculate bin width from the first patch
        ax1.set_title(f'Histogram of {column}')
        ax1.set_xlabel(f"{column} (Bin width: {bin_width:.2f})")
        ax1.set_ylabel('Frequency')
        ax1.grid(True)

        sns.boxplot(y=data, ax=ax2, color='green')
        ax2.set_title(f'Box Plot of {column}')
        ax2.set_ylabel('Values')
        ax2.set_xlabel('Box plot')

        plt.tight_layout()

        filename = f"Images/Numeric/{column}.png"
        plt.savefig(filename, format='png', dpi=300)
        plt.close(fig)
        print(f"{filename} has been saved")


def plot_correlations(df, target_var):
    numeric_vars = df.select_dtypes(include=['int64', 'float64', 'float32', 'int32'])

    if target_var not in numeric_vars:
        print(f"The target variable '{target_var}' is not in the DataFrame or is not numeric.")
        return

    num_vars = numeric_vars.columns.size - 1
    n_cols = 3
    n_rows = (num_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 5))
    fig.suptitle('Scatter Plots of ' + target_var + ' with Other Numerical Variables', fontsize=16, y=1.02)

    ax = axes.ravel()

    for i, var in enumerate([col for col in numeric_vars.columns if col != target_var]):
        sns.scatterplot(x=numeric_vars[var], y=numeric_vars[target_var], ax=ax[i], alpha=0.6)
        ax[i].set_xlabel(var)
        ax[i].set_ylabel(target_var)
        ax[i].grid(True)

    for j in range(i + 1, n_cols * n_rows):
        ax[j].axis('off')

    plt.tight_layout()
    filename = "Images/Numeric/correlations_{target_var}.png"
    plt.savefig(filename)
    print(f"{filename} has been saved")