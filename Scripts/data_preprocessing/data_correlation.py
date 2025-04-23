import pandas as pd
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def correcting_datatypes(df, date_cols=None, categorical_cols=None, float_cols=None):
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
                    print(f"Converted '{col}' to datetime")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to datetime. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype('category')
                    print(f"Converted '{col}' to category")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to category. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    if float_cols:
        for col in float_cols:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                    print(f"Converted '{col}' to float")
                except ValueError as e:
                    print(f"Warning: Could not convert column '{col}' to float. Error: {e}")
            else:
                print(f"Warning: Column '{col}' does not exist in the DataFrame.")

    return df


def remove_duplicates(df):
    duplicate_indices = df[df.duplicated()].index.tolist()
    if duplicate_indices:
        print("Removed duplicate rows at indices:", duplicate_indices)
    else:
        print("No duplicate rows found.")
    return df.drop_duplicates()


def detect_outliers_democratic(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    outlier_summary = {}

    print("*** Outlier Detection Summary ***\n")

    for col in num_cols:
        values = df[col].dropna()

        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = values[(values < iqr_lower) | (values > iqr_upper)].index

        z_scores = zscore(values)
        z_outliers = values[np.abs(z_scores) > 3].index

        median = np.median(values)
        mad = np.median(np.abs(values - median))
        mad_threshold = 3 * mad
        mad_outliers = values[np.abs(values - median) > mad_threshold].index

        all_outliers = list(iqr_outliers) + list(z_outliers) + list(mad_outliers)
        outlier_counts = pd.Series(all_outliers).value_counts()
        final_outliers = outlier_counts[outlier_counts >= 2].index.tolist()

        if final_outliers:
            outlier_summary[col] = final_outliers
            print(f"Variable: {col}")
            print(f"Outliers detected at indices: {final_outliers}\n")

    if not outlier_summary:
        print("No significant outliers detected.")
    else:
        print("*** Final Outlier Report ***")
        for col, indices in outlier_summary.items():
            print(f"{col}: {len(indices)} outliers detected at {indices}")

    return outlier_summary


def median_group_impute(df, target_variable, group_by_vars, outliers=None):
    if outliers and target_variable in outliers:
        df_cleaned = df.drop(index=outliers[target_variable]).copy()
    else:
        df_cleaned = df.copy()

    missing_before = df_cleaned[target_variable].isnull().sum()
    print(f"Missing values in '{target_variable}' before imputation: {missing_before}")

    median_values = df_cleaned.groupby(group_by_vars, observed=False)[target_variable].median()

    for index, row in df_cleaned.iterrows():
        if pd.isnull(row[target_variable]):
            group_key = tuple(row[group_by_vars])
            if group_key in median_values.index:
                df_cleaned.at[index, target_variable] = median_values[group_key]

    missing_after = df_cleaned[target_variable].isnull().sum()
    print(f"Missing values in '{target_variable}' after imputation: {missing_after}")

    if outliers and target_variable in outliers:
        df_outliers = df.loc[outliers[target_variable]]
        df_final = pd.concat([df_cleaned, df_outliers])
    else:
        df_final = df_cleaned

    return df_final


def check_missing_patterns(df, target_variable):
    df['missing_indicator'] = df[target_variable].isnull().astype(int)

    categorical_vars = df.select_dtypes(include=['category', 'object']).columns.tolist()
    if categorical_vars:
        for cat_var in categorical_vars:
            print(f"Missing values distribution across {cat_var}:")
            print(df.groupby(cat_var)['missing_indicator'].sum(), "\n")
            plt.figure(figsize=(8, 4))
            sns.countplot(x=df[cat_var], hue=df['missing_indicator'])
            plt.title(f"Missingness Pattern by {cat_var}")
            plt.xticks(rotation=45)
            plt.savefig(f"Images/Missing/pattern_{target_variable}_{cat_var}.png")
            plt.show()

    numerical_vars = df.select_dtypes(include=['number']).columns.tolist()
    if numerical_vars:
        for num_var in numerical_vars:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df['missing_indicator'], y=df[num_var])
            plt.title(f"Boxplot of {num_var} by Missingness in {target_variable}")
            plt.savefig(f"Images/Missing/pattern_{target_variable}_{num_var}.png")
            plt.show()

    df.drop(columns=['missing_indicator'], inplace=True)