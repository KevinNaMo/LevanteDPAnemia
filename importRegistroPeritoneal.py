import pandas as pd


# File extraction from xls/xlsx files
def load_excel_files(file_paths):
    # Initialize a dictionary to hold all dataframes
    all_dfs = {}

    # Loop over all file paths
    for file_path in file_paths:
        # Load all sheets from the current file into dataframes
        xls = pd.ExcelFile(file_path)
        dfs = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}

        # Loop over all dataframes and merge them with the corresponding ones in all_dfs
        for sheet_name, df in dfs.items():
            # Drop columns where all elements are NaN
            df.dropna(axis=1, how='all', inplace=True)

            # If this sheet name already exists in all_dfs, append the new dataframe to it
            if sheet_name in all_dfs:
                all_dfs[sheet_name] = pd.concat([all_dfs[sheet_name], df], ignore_index=True)
            # Otherwise, just add the new dataframe to all_dfs
            else:
                all_dfs[sheet_name] = df

    return all_dfs


def merge_dataframes(main_df, second_df, column_name):
    # Merge the dataframes on the specified column
    merged_df = pd.merge(main_df, second_df, on=column_name, how='left')

    return merged_df


def calculate_baseline(input_df, cat_cols, lab_cols, aggregation='average'):
    # Initialize the result dictionary
    result = {}

    # Calculate the counts and percentages for the categorical columns
    for col in cat_cols:
        # Replace missing values with a specific label
        input_df[col] = input_df[col].fillna('Missing_value')
        # Count the number of occurrences of each value
        counts = input_df.groupby('REGISTRO')[col].first().value_counts(dropna=False)
        # Calculate the percentages
        percentages = counts / len(input_df['REGISTRO'].unique()) * 100
        # Add the counts and percentages to the result dictionary
        result[col] = {'counts': counts, 'percentages': percentages}

    # Select one row per patient based on the aggregation policy
    if aggregation == 'first':
        input_df = input_df.sort_values('FECHA').groupby('REGISTRO').first()
    elif aggregation == 'last':
        input_df = input_df.sort_values('FECHA').groupby('REGISTRO').last()
    elif aggregation == 'average':
        input_df = input_df.groupby('REGISTRO')[lab_cols].mean()

    # Calculate the average, min, max and standard deviation for the numerical columns
    for col in lab_cols:
        # Calculate the statistics
        avg = input_df[col].mean()
        min_val = input_df[col].min()
        max_val = input_df[col].max()
        std_dev = input_df[col].std()
        # Add the statistics to the result dictionary
        result[col] = {'average': avg, 'min': min_val, 'max': max_val, 'std_dev': std_dev}

    return result

def print_baseline(baseline):
    # Loop over all items in the baseline dictionary
    for col, stats in baseline.items():
        print(f'{col}:')
        # If the stats is another dictionary (for numerical columns), print each stat on a separate line
        if isinstance(stats, dict):
            for stat, value in stats.items():
                print(f'    {stat}: {value}')
        # If the stats is not a dictionary (for categorical columns), print it on a single line
        else:
            print(f'    counts: \n{stats["counts"]}')
            print(f'    percentages: \n{stats["percentages"]}')
        print()


def analyze_nans(df, print_col_results=False, print_row_results=False):
    # Initialize the result dictionaries
    col_results = {}
    row_results = {}

    # Calculate the count and percentage of NaNs in each column
    for col in df.columns:
        nan_count = df[col].isna().sum()
        nan_percentage = nan_count / len(df) * 100
        col_results[col] = {'count': nan_count, 'percentage': nan_percentage}

    # Calculate the count and percentage of NaNs in each row
    for i, row in df.iterrows():
        nan_count = row.isna().sum()
        nan_percentage = nan_count / len(row) * 100
        row_results[i] = {'count': nan_count, 'percentage': nan_percentage}

    # Print the column results if requested
    if print_col_results:
        print('Column results:')
        for col, results in sorted(col_results.items(), key=lambda x: x[1]['percentage'], reverse=True):
            print(f'{col}: count = {results["count"]}, percentage = {results["percentage"]}%')

    # Print the row results if requested
    if print_row_results:
        print('Row results:')
        for i, results in sorted(row_results.items(), key=lambda x: x[1]['percentage'], reverse=True):
            print(f'Row {i}: count = {results["count"]}, percentage = {results["percentage"]}%')

    return col_results, row_results

def print_nan_col_results(col_results):
    # Loop over all items in the col_results dictionary
    for col, results in col_results.items():
        print(f'{col}:')
        # Print each result on a separate line
        for result, value in results.items():
            print(f'    {result}: {value}')
        print()