import pandas as pd
import numpy as np

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


def filter_by_year(df_list, date_cols, from_year, last_year):
    for i in range(len(df_list)):
        df = df_list[i].copy()  # Make a copy to avoid SettingWithCopyWarning
        date_col = date_cols[i]

        # Convert the date_col to datetime using .loc to avoid warnings
        df.loc[:, date_col] = pd.to_datetime(df[date_col])

        # Create a mask for the dates within the range
        mask = (df[date_col] >= pd.Timestamp(from_year, 1, 1)) & (df[date_col] <= pd.Timestamp(last_year, 12, 31))

        # Apply the mask to the dataframe
        df_list[i] = df.loc[mask]

    return tuple(df_list)


def bool_col_convert(df, columns):
    for col in columns:
        df[col] = df[col].fillna('NO').map({'SI': True, 'NO': False})
    return df


def add_age_column(input_df):
    # Convert 'INICIO_DP' and 'NACIMIENTO' to datetime format
    input_df['INICIO_DP'] = pd.to_datetime(input_df['INICIO_DP'])
    input_df['NACIMIENTO'] = pd.to_datetime(input_df['NACIMIENTO'])

    # Calculate the difference in years and assign it to the new 'EDAD' column
    input_df['EDAD'] = (input_df['INICIO_DP'] - input_df['NACIMIENTO']).dt.days // 365

    # Convert 'EDAD' to int type
    input_df['EDAD'] = input_df['EDAD'].astype(int)


def add_ckd_column(df):
    # Define a function to apply to each row
    def calculate_ckd(row):
        # Check if 'SEXO', 'CREATININA' or 'EDAD' are NaN
        if pd.isna(row['SEXO']) or pd.isna(row['CREATININA']) or pd.isna(row['EDAD']):
            return np.nan

        # Calculate the CKD-EPI creatinine equation based on the sex
        if row['SEXO'] == '1. Hombre':
            if row['CREATININA'] <= 0.9:
                return 141 * (row['CREATININA'] / 0.9) ** -0.411 * 0.993 ** row['EDAD']
            else:
                return 141 * (row['CREATININA'] / 0.9) ** -1.209 * 0.993 ** row['EDAD']
        elif row['SEXO'] == '2. Mujer':
            if row['CREATININA'] <= 0.7:
                return 144 * (row['CREATININA'] / 0.7) ** -0.329 * 0.993 ** row['EDAD']
            else:
                return 144 * (row['CREATININA'] / 0.7) ** -1.209 * 0.993 ** row['EDAD']

    # Apply the function to each row and assign the results to the new 'CKD_CALC' column
    df['CKD_CALC'] = df.apply(calculate_ckd, axis=1)

    # Define a function to apply to each row
    def calculate_ckd_stage(row):
        # Check if 'CKD_CALC' is NaN
        if pd.isna(row['CKD_CALC']):
            return np.nan

        # Determine the CKD stage based on the 'CKD_CALC' value
        if row['CKD_CALC'] > 90:
            return 'Stage 1'
        elif 60 <= row['CKD_CALC'] <= 89:
            return 'Stage 2'
        elif 45 <= row['CKD_CALC'] <= 59:
            return 'Stage 3A'
        elif 30 <= row['CKD_CALC'] <= 44:
            return 'Stage 3B'
        elif 15 <= row['CKD_CALC'] <= 29:
            return 'Stage 4'
        else:
            return 'Stage 5'

    # Apply the function to each row and assign the results to the new 'CKD_STAGE' column
    df['CKD_STAGE'] = df.apply(calculate_ckd_stage, axis=1)

    return df
    

def exclude_patients(df, exclude_col, verbose=False):
    number_rows_before_deletion = df.shape[0]

    for col in exclude_col:
        df = df[df[col] != True]

    number_rows_after_deletion = df.shape[0]
    percent_deleted_rows = ((number_rows_before_deletion - number_rows_after_deletion) / number_rows_before_deletion) * 100

    if verbose:
        print(f'Rows before excluding patients: {number_rows_before_deletion}')
        print(f'Rows after excluding patients: {number_rows_after_deletion}')
        print(f'Percentage of deleted rows: {percent_deleted_rows}%')

    return df


def calculate_baseline(input_df, cat_cols, lab_cols, aggregation='first'):
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


# Clean rows with NaN values for the columns on the col_list

def clean_df(df, col_list, verbose=False):
    df_total_rows = df.shape[0]
    unique_patients_before = df['REGISTRO'].nunique()  # Count unique patients before clean up

    df = df.dropna(subset=col_list)
    new_df_rows = df.shape[0]
    unique_patients_after = df['REGISTRO'].nunique()  # Count unique patients after clean up

    if verbose:
        print(
            f"The dataframe had {df_total_rows} rows (Unique patients: {unique_patients_before}), after the clean up of missing values, it has {new_df_rows} rows (Unique patients: {unique_patients_after})")

    return df

def df_binner(df, bin_size, policy='first'):
    # Calculate the minimum date for each patient
    min_dates = df.groupby('REGISTRO')['FECHA'].min()

    # Calculate the number of days since the first date for each patient
    df['days_since_first'] = df.apply(lambda row: (row['FECHA'] - min_dates[row['REGISTRO']]).days, axis=1)

    # Calculate the bin number
    df['bin_num'] = np.floor(df['days_since_first'] / bin_size).astype(int)

    # Calculate the start and end date of each bin
    #df.reset_index(inplace=True)
    #df['start_bin'] = min_dates[df['REGISTRO']] + pd.to_timedelta(df['bin_num'] * bin_size, unit='D')
    #df['end_bin'] = min_dates[df['REGISTRO']] + pd.to_timedelta((df['bin_num'] + 1) * bin_size, unit='D')

    # Group by 'REGISTRO' and 'bin_num', and select the row based on the policy
    if policy == 'first':
        grouped = df.sort_values('FECHA').groupby(['REGISTRO', 'bin_num']).first()
    elif policy == 'last':
        grouped = df.sort_values('FECHA').groupby(['REGISTRO', 'bin_num']).last()
    else:
        raise ValueError(f"Invalid policy: {policy}. Valid options are 'first' and 'last'.")

    # Reset the index
    grouped.reset_index(inplace=True)

    return grouped