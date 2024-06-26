import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from lifelines import CoxPHFitter
from lifelines import CoxTimeVaryingFitter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sksurv.util import Surv
from scipy.stats import ttest_ind


# --------------------
# Data import
# --------------------


def load_and_merge_files(directory, file_prefix, file_extension, verbose=False):
    # Initialize a dictionary to hold all dataframes
    all_dfs = {}

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the file_prefix and file_extension
        if filename.startswith(file_prefix) and filename.endswith(file_extension):
            # Load all sheets from the current file into dataframes
            file_path = os.path.join(directory, filename)
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

                # If verbose is True, print the progress
                if verbose:
                    print(f"Added {len(df)} rows from {filename} to {sheet_name}")

    # Check for duplicates and save the merged dataframes as pickle files
    for sheet_name, df in all_dfs.items():
        # Check for duplicates
        duplicates = df.duplicated()
        if duplicates.any():
            print(f"Found {duplicates.sum()} duplicates in {sheet_name}")

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Save the dataframe as a pickle file
        df.to_pickle(f"{sheet_name}.pkl")

    return all_dfs

# --------------------
# Data preprocessing
# --------------------


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


def filter_df(df, col_name, min_val, max_val, verbose=False):
    # Count the number of rows and unique patients before the filter
    rows_before_filter = len(df)
    unique_patients_before_filter = df['REGISTRO'].nunique()

    # Filter the dataframe
    filtered_df = df[(df[col_name] >= min_val) & (df[col_name] <= max_val)]

    # Count the number of rows and unique patients after the filter
    rows_after_filter = len(filtered_df)
    unique_patients_after_filter = filtered_df['REGISTRO'].nunique()

    # If verbose is True, print the number of rows and unique patients before and after the filter
    if verbose:
        print(f"Rows before filter: {rows_before_filter} (Unique patients: {unique_patients_before_filter})")
        print(f"Rows after filter: {rows_after_filter} (Unique patients: {unique_patients_after_filter})")

    return filtered_df


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


def include_patients(df, max_diff_days, exclusion_point='before_after'):
    # Convert 'FECHA' and 'INICIO_DP' to datetime
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['INICIO_DP'] = pd.to_datetime(df['INICIO_DP'])

    # Group by 'REGISTRO' and get the earliest 'FECHA' for each patient
    earliest_dates = df.groupby('REGISTRO')['FECHA'].min()

    # List to store the 'REGISTRO' of patients to include
    include_patients = []

    for registro, fecha in earliest_dates.items():
        # Get the 'INICIO_DP' date for this patient
        inicio_dp = df.loc[df['REGISTRO'] == registro, 'INICIO_DP'].iloc[0]

        # Calculate the difference in days
        diff_days = (fecha - inicio_dp).days

        # Check if this patient should be included based on the exclusion_point
        if exclusion_point == 'before_only' and -max_diff_days <= diff_days < 0:
            include_patients.append(registro)
        elif exclusion_point == 'after_only' and 0 <= diff_days <= max_diff_days:
            include_patients.append(registro)
        elif exclusion_point == 'before_after' and -max_diff_days <= diff_days <= max_diff_days:
            include_patients.append(registro)

    # Return a dataframe with only the rows of the included patients
    return df[df['REGISTRO'].isin(include_patients)]


def time_follow_limit(lab_df, hosp_df, limit_days):
    # Initialize empty dataframes for the results
    lab_df_new = pd.DataFrame()
    hosp_df_new = pd.DataFrame()

    # Loop over each patient
    for registro in lab_df['REGISTRO'].unique():
        # Get the data for this patient
        lab_patient = lab_df[lab_df['REGISTRO'] == registro]
        hosp_patient = hosp_df[hosp_df['REGISTRO'] == registro]

        # Calculate the first and last date for this patient
        first_date = lab_patient['FECHA'].min()
        last_date = first_date + pd.Timedelta(days=limit_days)

        # Select the rows that fall between the first and last date
        lab_patient_new = lab_patient[(lab_patient['FECHA'] >= first_date) & (lab_patient['FECHA'] <= last_date)]
        hosp_patient_new = hosp_patient[(hosp_patient['FINGRESO'] >= first_date) & (hosp_patient['FINGRESO'] <= last_date)]
        print(hosp_patient_new)

        # Append the selected rows to the result dataframes
        lab_df_new = pd.concat([lab_df_new, lab_patient_new])
        hosp_df_new = pd.concat([hosp_df_new, hosp_patient_new])

    return lab_df_new, hosp_df_new


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


def clean_zero_values(df, col_list, verbose=False):
    # Initial number of rows
    initial_rows = len(df)
    
    # Check and remove rows with 0 or 0.0 in the specified columns
    for col in col_list:
        df = df[~(df[col].isin([0, 0.0]))]
    
    # Final number of rows
    final_rows = len(df)
    rows_deleted = initial_rows - final_rows
    
    # Verbose output
    if verbose:
        print(f"Number of rows before deleting: {initial_rows}")
        print(f"Number of rows deleted: {rows_deleted}")
        print(f"Total number of rows after deleting: {final_rows}")
    
    # Return the new dataframe without the deleted rows
    return df

    
# --------------------
# New columns 
# --------------------


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


def add_days_since_start(info_df, target_df, date_col):
    # Create a mapping from 'REGISTRO' to 'INICIO_DP'
    inicio_dp_mapping = info_df.drop_duplicates('REGISTRO').set_index('REGISTRO')['INICIO_DP'].to_dict()

    # Filter target_df to only include rows with 'REGISTRO' present in info_df
    target_df = target_df[target_df['REGISTRO'].isin(inicio_dp_mapping.keys())].copy()

    # Convert the date columns to datetime format
    target_df.loc[:, date_col] = pd.to_datetime(target_df[date_col])
    for registro, inicio_dp in inicio_dp_mapping.items():
        inicio_dp_mapping[registro] = pd.to_datetime(inicio_dp)

    # Add the 'days_since_start' column
    target_df.loc[:, 'days_since_start'] = target_df.apply(lambda row: (row[date_col] - inicio_dp_mapping[row['REGISTRO']]).days, axis=1)

    return target_df


def add_anemia_column(input_df, Hb_masc=13, Hb_fem=12):
    # Define a function to apply to each row
    def check_anemia(row):
        # Check if 'SEXO' or 'HEMOGLOBINA' are NaN
        if pd.isna(row['SEXO']) or pd.isna(row['HEMOGLOBINA']):
            #print(f"NaN value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']}")
            return np.nan
        # Check if the patient is a man and has anemia
        elif row['SEXO'] == '1. Hombre' and row['HEMOGLOBINA'] < Hb_masc:
            #print(f"True value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']} and {row['SEXO']}")
            return True
        # Check if the patient is a woman and has anemia
        elif row['SEXO'] == '2. Mujer' and row['HEMOGLOBINA'] < Hb_fem:
            #print(f"True value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']} and {row['SEXO']}")
            return True
        # If none of the above conditions are met, the patient does not have anemia
        else:
            return False

    # Apply the function to each row and assign the results to the new 'ANEMIA' column
    input_df['ANEMIA'] = input_df.apply(check_anemia, axis=1)

    
# --------------------
# Baseline 
# --------------------


def calculate_baseline(input_df, cat_cols, lab_cols, aggregation='first', print_results=False):
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

    if print_results:
        # Loop over all items in the result dictionary
        for col, stats in result.items():
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

    return result

    
# --------------------
# Stats
# --------------------

def hosp_stats(lab_df, hosp_df):
    # Convert 'FECHA' and 'FINGRESO' to datetime
    lab_df['FECHA'] = pd.to_datetime(lab_df['FECHA'])
    hosp_df['FINGRESO'] = pd.to_datetime(hosp_df['FINGRESO'])

    # Group by 'REGISTRO' and get the earliest 'FECHA' for each patient in lab_df
    earliest_dates = lab_df.groupby('REGISTRO')['FECHA'].min()

    # List to store the 'REGISTRO' of patients who had at least one row in hosp_df
    hosp_patients = []

    # List to store the year of hospitalisation for each patient
    hosp_years = []

    for registro in lab_df['REGISTRO'].unique():
        # Get the earliest 'FECHA' for this patient
        earliest_FECHA = earliest_dates[registro]

        # Get the rows in hosp_df for this patient
        hosp_rows = hosp_df[hosp_df['REGISTRO'] == registro]

        if not hosp_rows.empty:
            # This patient had at least one row in hosp_df
            hosp_patients.append(registro)

            for fingreso in hosp_rows['FINGRESO']:
                # Calculate the year of hospitalisation
                hosp_year = ((fingreso - earliest_FECHA).days // 365) + 1
                hosp_years.append(hosp_year)

    # Calculate and print the absolute count and percentage of patients in lab_df who had at least one row in hosp_df
    count = len(hosp_patients)
    percentage = (count / len(lab_df['REGISTRO'].unique())) * 100
    print(f"Absolute count of patients with hospitalisations: {count}")
    print(f"Percentage of patients with hospitalisations: {percentage}%")

    # Calculate and print the percentage of patients with hospitalisations during the first, second, third...year
    for i in range(1, max(hosp_years) + 1):
        count = hosp_years.count(i)
        percentage = (count / len(hosp_patients)) * 100
        print(f"Percentage of patients with hospitalisations during the {i} year: {percentage}%")


def lab_freq_stats(df, print_avg=True, print_patient=False):
    # Convert 'FECHA' to datetime
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Sort the dataframe by 'REGISTRO' and 'FECHA'
    df = df.sort_values(['REGISTRO', 'FECHA'])

    # Calculate the difference between each 'FECHA' and the next 'FECHA' for each 'REGISTRO'
    df['FECHA_diff'] = df.groupby('REGISTRO')['FECHA'].diff().dt.days

    # Calculate the average 'FECHA_diff' for each 'REGISTRO'
    avg_diff_per_patient = df.groupby('REGISTRO')['FECHA_diff'].mean()

    if print_patient:
        for registro, avg_diff in avg_diff_per_patient.items():
            print(f"Patient {registro}: average time between tests = {avg_diff} days")

    if print_avg:
        global_avg_diff = avg_diff_per_patient.mean()
        print(f"Global average time between tests = {global_avg_diff} days")

    return avg_diff_per_patient


# --------------------
# Checks and internal stats
# --------------------


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


def check_common_values(base_df, second_df, base_col, second_col, registro_col):
    # Initialize the counters
    empty_count = 0
    coincidence_count = 0
    no_coincidence_count = 0

    # Loop over each row in the second dataframe
    for i, row in second_df.iterrows():
        # Skip if the value in 'second_col' is empty
        if pd.isna(row[second_col]):
            #print(f"The patient {row[registro_col]} has an empty row, it's skipped")
            empty_count += 1
            continue

        # Select the rows in base_df with the same 'registro_col' value
        base_rows = base_df[base_df[registro_col] == row[registro_col]]

        # Check if there's any row in base_rows with the same 'base_col' value as 'second_col' in the current row
        if any(base_rows[base_col] == row[second_col]):
            #print(f"For the patient {row[registro_col]} there is a lab result at {row[second_col]} same as {base_col}")
            coincidence_count += 1
        else:
            #print(f"For the patient {row[registro_col]} with date {row[second_col]} there is no coincidence in base_df")
            no_coincidence_count += 1

    # Print the counts
    print(f"Number of empty rows: {empty_count}")
    print(f"Number of coincidences: {coincidence_count}")
    print(f"Number of no coincidences: {no_coincidence_count}")


def print_diff_days(df, max_diff_days):
    # Convert 'FECHA' and 'INICIO_DP' to datetime
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['INICIO_DP'] = pd.to_datetime(df['INICIO_DP'])

    # Group by 'REGISTRO' and get the earliest 'FECHA' for each patient
    earliest_dates = df.groupby('REGISTRO')['FECHA'].min()

    diff_days_list = []  # List to store all diff_days
    count = 0  # Counter for patients with diff_days within max_diff_days

    for registro, fecha in earliest_dates.items():
        # Get the 'INICIO_DP' date for this patient
        inicio_dp = df.loc[df['REGISTRO'] == registro, 'INICIO_DP'].iloc[0]

        # Calculate the difference in days
        diff_days = (fecha - inicio_dp).days
        diff_days_list.append(diff_days)  # Add diff_days to the list

        # Check if diff_days is within max_diff_days
        if -max_diff_days <= diff_days <= max_diff_days:
            count += 1

        # Print the result
        print(
            f"The patient {registro} started PD in {inicio_dp} and had the first lab at {fecha} that makes it {diff_days} days.")

    # Calculate and return the average of diff_days
    avg_diff_days = sum(diff_days_list) / len(diff_days_list)
    print(f"Average diff_days: {avg_diff_days}")
    print(f"Number of patients with diff_days within {-max_diff_days} to {max_diff_days}: {count}")

    # Calculate and print the percentage of patients with diff_days within max_diff_days
    percentage = (count / len(diff_days_list)) * 100
    print(f"Percentage of patients with diff_days within {-max_diff_days} to {max_diff_days}: {percentage}%")


def follow_up_periods(df, print_patients=False):
    # Convert 'FECHA' to datetime
    df['FECHA'] = pd.to_datetime(df['FECHA'])

    # Group by 'REGISTRO' and get the earliest and latest 'FECHA' for each patient
    earliest_dates = df.groupby('REGISTRO')['FECHA'].min()
    latest_dates = df.groupby('REGISTRO')['FECHA'].max()

    follow_up_list = []  # List to store all follow up periods

    for registro in df['REGISTRO'].unique():
        # Get the earliest and latest 'FECHA' for this patient
        earliest_FECHA = earliest_dates[registro]
        latest_FECHA = latest_dates[registro]

        # Calculate the follow up period
        follow_up = (latest_FECHA - earliest_FECHA).days
        follow_up_list.append(follow_up)  # Add follow up to the list

        # Print the result for this patient if print_patients is True
        if print_patients:
            print(
                f"The patient {registro} has the first lab at {earliest_FECHA} and the last lab at {latest_FECHA} with a follow up period of {follow_up} days.")

    # Calculate and print the average follow up period
    avg_follow_up = sum(follow_up_list) / len(follow_up_list)
    print(f"Average follow up period: {avg_follow_up} days")

    # Calculate and print the percentage of patients with a follow up period of at least one year, two years, three years, etc.
    for i in range(1, max(follow_up_list) // 365 + 1):
        count = sum(follow_up >= i * 365 for follow_up in follow_up_list)
        percentage = (count / len(follow_up_list)) * 100
        print(f"Percentage of patients with a follow up period of at least {i} year(s): {percentage}%")

    
# --------------------
# Anemia
# --------------------


def anemia_prevalence_stack(df, hb_limit_male, hb_limit_female, year_range, print_maxmin=False):
    # Convert 'FECHA' to datetime format and extract the year
    df['YEAR'] = pd.to_datetime(df['FECHA']).dt.year

    # Calculate the yearly average 'HEMOGLOBINA' value for each 'REGISTRO' and 'SEXO'
    df = df.groupby(['REGISTRO', 'SEXO', 'YEAR'])['HEMOGLOBINA'].mean().reset_index()

    # Rename the 'HEMOGLOBINA' column to 'HB_AVG'
    df.rename(columns={'HEMOGLOBINA': 'HB_AVG'}, inplace=True)

    # Initialize the result dictionaries
    male_results = {year: [0, 0, 0] for year in range(year_range[0], year_range[1] + 1)}
    female_results = {year: [0, 0, 0] for year in range(year_range[0], year_range[1] + 1)}
    total_results = {year: [0, 0, 0] for year in range(year_range[0], year_range[1] + 1)}
    male_counts = {year: 0 for year in range(year_range[0], year_range[1] + 1)}
    female_counts = {year: 0 for year in range(year_range[0], year_range[1] + 1)}
    total_counts = {year: 0 for year in range(year_range[0], year_range[1] + 1)}

    # Loop over each row in the dataframe
    for i, row in df.iterrows():
        # Skip if the year is not within the range
        if not year_range[0] <= row['YEAR'] <= year_range[1]:
            continue

        # Calculate the anemia level based on the sex and 'HB_AVG' value
        if row['SEXO'] == '1. Hombre':
            male_counts[row['YEAR']] += 1
            total_counts[row['YEAR']] += 1
            if row['HB_AVG'] < hb_limit_male[0]:
                male_results[row['YEAR']][0] += 1
                total_results[row['YEAR']][0] += 1
            elif row['HB_AVG'] < hb_limit_male[1]:
                male_results[row['YEAR']][1] += 1
                total_results[row['YEAR']][1] += 1
            elif row['HB_AVG'] < hb_limit_male[2]:
                male_results[row['YEAR']][2] += 1
                total_results[row['YEAR']][2] += 1
        elif row['SEXO'] == '2. Mujer':
            female_counts[row['YEAR']] += 1
            total_counts[row['YEAR']] += 1
            if row['HB_AVG'] < hb_limit_female[0]:
                female_results[row['YEAR']][0] += 1
                total_results[row['YEAR']][0] += 1
            elif row['HB_AVG'] < hb_limit_female[1]:
                female_results[row['YEAR']][1] += 1
                total_results[row['YEAR']][1] += 1
            elif row['HB_AVG'] < hb_limit_female[2]:
                female_results[row['YEAR']][2] += 1
                total_results[row['YEAR']][2] += 1

    # Calculate the prevalence for each year and sex
    for year in range(year_range[0], year_range[1] + 1):
        for i in range(3):
            male_results[year][i] = male_results[year][i] / male_counts[year] * 100 if male_counts[year] != 0 else 0
            female_results[year][i] = female_results[year][i] / female_counts[year] * 100 if female_counts[year] != 0 else 0
            total_results[year][i] = total_results[year][i] / total_counts[year] * 100 if total_counts[year] != 0 else 0
    def plot_histogram(results, title):
        plt.figure(figsize=(10, 5))
        plt.bar(results.keys(), [results[year][0] for year in results.keys()], color='#bf211e', label='Hb < 10 g/dl')
        plt.bar(results.keys(), [results[year][1] for year in results.keys()], bottom=[results[year][0] for year in results.keys()], color='#f9dc5c', label='Hb 11-10 g/dl')
        plt.bar(results.keys(), [results[year][2] for year in results.keys()], bottom=[results[year][0] + results[year][1] for year in results.keys()], color='#688E26', label='Hb 13-11 g/dl')
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Prevalence (%)')
        plt.legend(loc='lower right')
        plt.show()
    # Plot the histograms
    plot_histogram(male_results, 'Prevalencia de anemia en hombres')
    plot_histogram(female_results, 'Prevalencia de anemia en mujeres')
    plot_histogram(total_results, 'Prevalencia de anemia en total')

    # Convert the results to a more human-readable format
    male_results_readable = {year: {'Hb < 10 g/dl': results[0], 'Hb 11-10 g/dl': results[1], 'Hb 13-11 g/dl': results[2]} for year, results in male_results.items()}
    female_results_readable = {year: {'Hb < 10 g/dl': results[0], 'Hb 11-10 g/dl': results[1], 'Hb 13-11 g/dl': results[2]} for year, results in female_results.items()}
    total_results_readable = {year: {'Hb < 10 g/dl': results[0], 'Hb 11-10 g/dl': results[1], 'Hb 13-11 g/dl': results[2]} for year, results in total_results.items()}

    if print_maxmin:
         # Initialize variables to store the highest and lowest sums and their corresponding years
        highest_sum_all = 0
        highest_year_all = None
        lowest_sum_all = float('inf')
        lowest_year_all = None
        highest_sum_10 = 0
        highest_year_10 = None
        lowest_sum_10 = float('inf')
        lowest_year_10 = None
        highest_sum_11 = 0
        highest_year_11 = None
        lowest_sum_11 = float('inf')
        lowest_year_11 = None

        # Iterate over the dictionary
        for year, categories in total_results_readable.items():
            # Calculate the sum of the percentages for the current year
            current_sum = sum(categories.values())
            
            # If the current sum is higher than the highest sum found so far, update the highest sum and year
            if current_sum > highest_sum_all:
                highest_sum_all = current_sum
                highest_year_all = year
    
            # If the current sum is lower than the lowest sum found so far, update the lowest sum and year
            if current_sum < lowest_sum_all:
                lowest_sum_all = current_sum
                lowest_year_all = year
                
         # Iterate over the dictionary
        for year, categories in total_results_readable.items():
            # Calculate the sum of the percentages for the current year
            current_sum = categories['Hb < 10 g/dl']
            
            # If the current sum is higher than the highest sum found so far, update the highest sum and year
            if current_sum > highest_sum_10:
                highest_sum_10 = current_sum
                highest_year_10 = year
    
            # If the current sum is lower than the lowest sum found so far, update the lowest sum and year
            if current_sum < lowest_sum_10:
                lowest_sum_10 = current_sum
                lowest_year_10 = year

        for year, categories in total_results_readable.items():
            # Calculate the sum of the percentages for the current year
            current_sum = first_value = categories['Hb 11-10 g/dl']
            
            # If the current sum is higher than the highest sum found so far, update the highest sum and year
            if current_sum > highest_sum_11:
                highest_sum_11 = current_sum
                highest_year_11 = year
    
            # If the current sum is lower than the lowest sum found so far, update the lowest sum and year
            if current_sum < lowest_sum_11:
                lowest_sum_11 = current_sum
                lowest_year_11 = year
                
        # Print the results
        print(f'Lowest value year_all {lowest_year_all}: {lowest_sum_all}\nHighest value year_all {highest_year_all}: {highest_sum_all}')
        print(f'Lowest value year < 10g/dl {lowest_year_10}: {lowest_sum_10}\nHighest value year <10 g/dl {highest_year_10}: {highest_sum_10}')
        print(f'Lowest value year < 11 g/dl{lowest_year_11}: {lowest_sum_11}\nHighest value year < 11 g/dl{highest_year_11}: {highest_sum_11}')

    return male_results_readable, female_results_readable, total_results_readable, df

# --------------------
# Time Trend Analysis
# --------------------


def time_trend_analysis(df_input, time_frame, col_names, follow_up_limit, plot_results=False, t_test=False, normality_test=False):
    # Create a copy of df_input to avoid SettingWithCopyWarning
    df = df_input.copy()

    # Add 'days_since_start' column
    df_input['days_since_start'] = (pd.to_datetime(df_input['FECHA']) - pd.to_datetime(df_input['INICIO_DP'])).dt.days

    # Initialize the output dataframe
    time_trend_df = pd.DataFrame()

    # Calculate the number of time frames
    num_time_frames = df_input['days_since_start'].max() // time_frame + 1

    # Loop over each time frame
    for i in range(num_time_frames):
        # Calculate the start and end day of the current time frame
        start_day = i * time_frame
        end_day = (i + 1) * time_frame

        # Select the rows that fall within the current time frame
        df_time_frame = df_input[(df_input['days_since_start'] >= start_day) & (df_input['days_since_start'] < end_day)]

        # Calculate the average and standard deviation for each column in col_names
        for col in col_names:
            time_trend_df.loc[i, f'{col}_avg'] = df_time_frame[col].mean()
            time_trend_df.loc[i, f'{col}_std'] = df_time_frame[col].std()

    # Delete the rows that exceed the follow_up_limit
    time_trend_df = time_trend_df[time_trend_df.index * time_frame <= follow_up_limit]

    # Perform t-test if t_test is True
    if t_test:
        results = {}
        for col in col_names:
            # Select the rows that fall within the first and last time frames
            df_first_time_frame = df_input[(df_input['days_since_start'] >= 0) & (df_input['days_since_start'] < time_frame)]
            df_last_time_frame = df_input[(df_input['days_since_start'] >= (num_time_frames - 1) * time_frame) & (df_input['days_since_start'] < num_time_frames * time_frame)]

            # Perform the normality test if normality_test is True
            if normality_test:
                combined_data = pd.concat([df_first_time_frame[col], df_last_time_frame[col]]).dropna()
                if len(combined_data) >= 3:
                    stat, p = shapiro(combined_data)
                    print(f"Normality test for {col}: Statistics={stat}, p={p}")
                else:
                    print(f"Not enough data to perform normality test for {col}")

            # Perform the t-test on the data from the first and last time frames
            t_stat, p_val = ttest_ind(df_first_time_frame[col], df_last_time_frame[col], nan_policy='omit')

            results[col] = p_val
            print(f"{col}: p-value = {p_val}")

    # Plot the results if plot_results is True
    if plot_results:
        for col in col_names:
            plt.figure(figsize=(10, 5))
            plt.plot(time_trend_df.index * time_frame, time_trend_df[f'{col}_avg'], label='Media')
            plt.fill_between(time_trend_df.index * time_frame, time_trend_df[f'{col}_avg'] - time_trend_df[f'{col}_std'], time_trend_df[f'{col}_avg'] + time_trend_df[f'{col}_std'], color='b', alpha=0.1, label='Desviación estándar')
            if t_test:
                plt.title(f'Evolución temporal de {col} (p={results[col]:.3f})')
            else:
                plt.title(f'Evolución temporal de {col}')
            plt.xlabel('Días desde el inicio de DP')
            plt.ylabel(col)
            plt.legend()
            plt.show()

    return time_trend_df


def stationary_test(df, col_str='_avg'):
    for column in df.columns:
        if column.endswith(col_str):
            print(f'Testing for stationarity in column: {column}')
            result = adfuller(df[column])
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value:.3f}')
            if result[1] < 0.05:
                print(f'The data in {column} is stationary.\n')
            else:
                print(f'The data in {column} is not stationary and may need differencing.\n')

    
# --------------------
# Cox Model Related Functions
# --------------------


def calculate_residuals(cph, df, martingale=False, schonenfeld=False):
    if martingale:
        martingale_residuals = cph.compute_residuals(df, 'martingale')
        print("Martingale Residuals:\n", martingale_residuals)
    if schonenfeld:
        schonenfeld_residuals = cph.check_assumptions(df, p_value_threshold=0.05, show_plots=True)
        print("Schonenfeld Residuals:\n", schonenfeld_residuals)

def cox_visualization(cph_results, df, survival_function=False, baseline_survival=False, baseline_cumulative_hazard=False, assumption=False):
    if survival_function:
        # Create a new DataFrame with the covariate values of interest
        covariates = pd.DataFrame({'HEMOGLOBINA': [1], 'IST': [0], 'HIERRO': [0]})
        # Calculate the survival function for these covariates
        survival_function = cph_results.predict_survival_function(covariates)
        # Plot the survival function
        survival_function.plot()
        plt.title('Survival function for specific covariate patterns')
        plt.show()

    if baseline_survival:
        # Plot the baseline survival function
        cph_results.baseline_survival_.plot()
        plt.title('Baseline survival function')
        plt.show()

    if baseline_cumulative_hazard:
        # Plot the baseline cumulative hazard function
        cph_results.baseline_cumulative_hazard_.plot()
        plt.title('Baseline cumulative hazard function')
        plt.show()

    if assumption:
        # Check the proportional hazards assumption
        cph_results.check_assumptions(df, p_value_threshold=0.05, show_plots=True)
        


def prepare_cox_df(lab_df, hosp_df, covariate_list):
    # Calculate the mean values for each 'REGISTRO' in the lab_df
    lab_df_mean = lab_df.groupby('REGISTRO')[covariate_list].mean().reset_index()

    # Initialize the cox_df
    cox_df = pd.DataFrame()

    # Add the 'REGISTRO' column
    cox_df['REGISTRO'] = lab_df_mean['REGISTRO']

    # Add the 'days_since_start' column
    #cox_df['days_since_start'] = 0

    # Add the covariate columns
    for covariate in covariate_list:
        cox_df[covariate] = lab_df_mean[covariate]

    # Add the 'finish_days' and 'event_col' columns
    cox_df['finish_days'] = cox_df['REGISTRO'].apply(lambda x: hosp_df[hosp_df['REGISTRO'] == x]['days_since_start'].min() if x in hosp_df['REGISTRO'].values else 365)
    cox_df['event_col'] = cox_df['REGISTRO'].apply(lambda x: True if x in hosp_df['REGISTRO'].values else False)

    # Drop the 'REGISTRO' column
    cox_df = cox_df.drop(columns=['REGISTRO'])
    
    return cox_df


def cox_time_varying_prep(lab_df, hosp_df, covariate_list, study_time=365):
    # Initialize the output dataframe
    cox_df = pd.DataFrame()

    # Get a list of unique 'REGISTRO' codes
    registro_list = lab_df['REGISTRO'].unique()

    # Go through the list of 'REGISTRO' codes
    for registro in registro_list:
        # Get all rows for the current 'REGISTRO' in lab_df and hosp_df
        lab_rows = lab_df[lab_df['REGISTRO'] == registro].sort_values('days_since_start')
        hosp_rows = hosp_df[hosp_df['REGISTRO'] == registro].sort_values('days_since_start')

        # Go through the lab_rows
        for i in range(len(lab_rows)):
            # Get the start_col
            start_col = lab_rows.iloc[i]['days_since_start']

            # Get the covariate_col values
            covariate_cols = lab_rows.iloc[i][covariate_list]

            # Calculate the finish_col and event_col
            if i < len(lab_rows) - 1:
                finish_col = lab_rows.iloc[i + 1]['days_since_start']
                event_col = False
            elif len(hosp_rows) > 0 and hosp_rows.iloc[0]['days_since_start'] > start_col:
                finish_col = hosp_rows.iloc[0]['days_since_start']
                event_col = True
            else:
                finish_col = study_time
                event_col = False

            # Add the row to the output dataframe
            row_df = pd.DataFrame({'REGISTRO': [registro], 'start_col': [start_col], 'finish_col': [finish_col], 'event_col': [event_col], **covariate_cols})
            cox_df = pd.concat([cox_df, row_df], ignore_index=True)

    return cox_df


def evaluate_cox_model(cph, df, duration_col, event_col, print_result=False):
    # Calculate the risk scores
    risk_scores = cph.predict_partial_hazard(df)
    
    # Calculate the true event values
    true_event_values = (df[duration_col] <= 365) & df[event_col]
    
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(true_event_values, risk_scores)
    
    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    
    # Print the AUC and ROC curve if print_result is True
    if print_result:
        print(f'AUC: {roc_auc}')
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    
    return roc_auc


def stratified_cox(df, duration_col, event_col, strat_col, strat_limits):
    # Initialize a dictionary to store the Cox models and AUCs
    cph_models = {}
    aucs = {}

    # Loop over the strat_limits
    for i in range(len(strat_limits) - 1):
        # Select the rows in df that fall within the current stratum
        stratum_df = df[(df[strat_col] >= strat_limits[i]) & (df[strat_col] < strat_limits[i + 1])]

        # Initialize the CoxPHFitter
        cph = CoxPHFitter()

        # Fit the data to the model
        cph.fit(stratum_df, duration_col=duration_col, event_col=event_col)

        # Print the summary of the model
        print(f'CoxPHFitter results for {strat_col} between {strat_limits[i]} and {strat_limits[i + 1]}:')
        cph.print_summary()

        # Plot the coefficients of the model
        cph.plot()
        plt.title(f'CoxPHFitter coefficients for {strat_col} between {strat_limits[i]} and {strat_limits[i + 1]}')
        plt.show()

        # Calculate the risk scores
        risk_scores = cph.predict_partial_hazard(stratum_df)

        # Calculate the true event values
        true_event_values = (stratum_df[duration_col] <= 365) & stratum_df[event_col]

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(true_event_values, risk_scores)

        # Calculate the AUC
        roc_auc = auc(fpr, tpr)
        aucs[f'{strat_col}_{strat_limits[i]}_{strat_limits[i + 1]}'] = roc_auc

        # Print the AUC and ROC curve
        print(f'AUC for {strat_col} between {strat_limits[i]} and {strat_limits[i + 1]}: {roc_auc}')
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

        # Store the model in the dictionary
        cph_models[f'{strat_col}_{strat_limits[i]}_{strat_limits[i + 1]}'] = cph

    return cph_models, aucs


def prepare_extended_cox_df(lab_df, hosp_df, covariate_list, study_time=365):
    # Initialize the output dataframe
    cox_df = pd.DataFrame()

    # Get a list of unique 'REGISTRO' codes
    registro_list = lab_df['REGISTRO'].unique()

    # Go through the list of 'REGISTRO' codes
    for registro in registro_list:
        # Get all rows for the current 'REGISTRO' in lab_df and hosp_df
        lab_rows = lab_df[lab_df['REGISTRO'] == registro].sort_values('days_since_start')
        hosp_rows = hosp_df[hosp_df['REGISTRO'] == registro].sort_values('days_since_start')

        # Go through the lab_rows
        for i in range(len(lab_rows)):
            # Get the start_col
            start_col = lab_rows.iloc[i]['days_since_start']

            # Get the covariate_col values
            covariate_cols = lab_rows.iloc[i][covariate_list]

            # Calculate the finish_col and event_col
            if i < len(lab_rows) - 1:
                finish_col = lab_rows.iloc[i + 1]['days_since_start']
                event_col = False
            elif len(hosp_rows) > 0 and hosp_rows.iloc[0]['days_since_start'] > start_col:
                finish_col = hosp_rows.iloc[0]['days_since_start']
                event_col = True
            else:
                finish_col = study_time
                event_col = False

            # Add the row to the output dataframe
            row_df = pd.DataFrame({'REGISTRO': [registro], 'start_col': [start_col], 'finish_col': [finish_col], 'event_col': [event_col], **covariate_cols})
            cox_df = pd.concat([cox_df, row_df], ignore_index=True)

    return cox_df


# --------------------
# Random survival forest
# --------------------

def prepare_rsf_df(lab_df, hosp_df, covariate_list):
    # Calculate the mean values for each 'REGISTRO' in the lab_df
    lab_df_mean = lab_df.groupby('REGISTRO')[covariate_list].mean().reset_index()

    # Initialize the rsf_df
    rsf_df = pd.DataFrame()

    # Add the 'REGISTRO' column
    rsf_df['REGISTRO'] = lab_df_mean['REGISTRO']

    # Add the covariate columns
    for covariate in covariate_list:
        rsf_df[covariate] = lab_df_mean[covariate]

    # Calculate the 'time' and 'event' columns
    rsf_df['time'] = rsf_df['REGISTRO'].apply(lambda x: hosp_df[hosp_df['REGISTRO'] == x]['days_since_start'].min() if x in hosp_df['REGISTRO'].values else 365)
    rsf_df['event'] = rsf_df['REGISTRO'].apply(lambda x: True if x in hosp_df['REGISTRO'].values else False)

    # Drop the 'REGISTRO' column
    rsf_df = rsf_df.drop(columns=['REGISTRO'])

    # Convert the 'time' and 'event' columns to a structured array
    y = Surv.from_dataframe('event', 'time', rsf_df)

    # Drop the 'time' and 'event' columns from rsf_df
    rsf_df = rsf_df.drop(columns=['time', 'event'])

    return rsf_df, y


def plot_survival_function(rsf, rsf_df, patient_index):
    # Get the survival function for the specified patient
    survival_function = rsf.predict_survival_function(rsf_df.iloc[patient_index:patient_index+1, :])

    # Get the time points and survival probabilities from the StepFunction object
    time_points = survival_function.x
    survival_probabilities = survival_function.y

    # Plot the survival function
    plt.step(time_points, survival_probabilities, where="post")
    plt.title('Survival Function')
    plt.ylabel('Survival Probability')
    plt.xlabel('Time')
    plt.show()


# --------------------
# Andersen-gill 
# --------------------

def prepare_andersen_gill_df(lab_df, event_df, covariate_list, study_time):
    # Initialize the output dataframe
    ag_df = pd.DataFrame()

    # Get a list of unique 'REGISTRO' codes
    registro_list = lab_df['REGISTRO'].unique()

    # Go through the list of 'REGISTRO' codes
    for registro in registro_list:
        # Get all rows for the current 'REGISTRO' in lab_df and event_df
        lab_rows = lab_df[lab_df['REGISTRO'] == registro].sort_values('days_since_start')
        event_rows = event_df[event_df['REGISTRO'] == registro].sort_values('days_since_start')

        # Go through the lab_rows
        for i in range(len(lab_rows)):
            # Get the start_col
            start_col = lab_rows.iloc[i]['days_since_start']

            # Get the covariate_col values
            covariate_cols = lab_rows.iloc[i][covariate_list]

            # Calculate the finish_col and event_col
            if i < len(lab_rows) - 1:
                finish_col = lab_rows.iloc[i + 1]['days_since_start']
                event_col = False
            elif len(event_rows) > 0 and event_rows.iloc[0]['days_since_start'] > start_col:
                finish_col = event_rows.iloc[0]['days_since_start']
                event_col = True
            else:
                finish_col = study_time
                event_col = False

            # Add the row to the output dataframe
            row_df = pd.DataFrame({'REGISTRO': [registro], 'start_col': [start_col], 'finish_col': [finish_col], 'event_col': [event_col], **covariate_cols})
            ag_df = pd.concat([ag_df, row_df], ignore_index=True)

    return ag_df


# --------------------
# Joint-model
# --------------------


def prepare_joint_model_df(lab_df, event_df, covariate_list, study_time):
    # Initialize the output dataframe
    joint_df = pd.DataFrame()

    # Get a list of unique 'REGISTRO' codes
    registro_list = lab_df['REGISTRO'].unique()

    # Go through the list of 'REGISTRO' codes
    for registro in registro_list:
        # Get all rows for the current 'REGISTRO' in lab_df and event_df
        lab_rows = lab_df[lab_df['REGISTRO'] == registro].sort_values('days_since_start')
        event_rows = event_df[event_df['REGISTRO'] == registro].sort_values('days_since_start')

        # Go through the lab_rows
        for i in range(len(lab_rows)):
            # Get the start_col
            start_col = lab_rows.iloc[i]['days_since_start']

            # Get the covariate_col values
            covariate_cols = lab_rows.iloc[i][covariate_list]

            # Calculate the finish_col and event_col
            if i < len(lab_rows) - 1:
                finish_col = lab_rows.iloc[i + 1]['days_since_start']
                event_col = False
            elif len(event_rows) > 0 and event_rows.iloc[0]['days_since_start'] > start_col:
                finish_col = event_rows.iloc[0]['days_since_start']
                event_col = True
            else:
                finish_col = study_time
                event_col = False

            # Add the row to the output dataframe
            row_df = pd.DataFrame({'REGISTRO': [registro], 'start_col': [start_col], 'finish_col': [finish_col], 'event_col': [event_col], **covariate_cols})
            joint_df = pd.concat([joint_df, row_df], ignore_index=True)

    return joint_df

def execute_joint_model(joint_df, time_col, event_col, covariate_list):
    # Convert the dataframe to long format
    long_df = to_long_format(joint_df, duration_col=time_col)

    # Add the covariates to the timeline
    long_df = add_covariate_to_timeline(long_df, joint_df, id_col='REGISTRO', event_col=event_col, duration_col=time_col, start_col='start_col', stop_col='finish_col', covariates=covariate_list)

    # Initialize the CoxPHFitter
    cph = CoxPHFitter()

    # Fit the data to the model
    cph.fit(long_df, duration_col='duration', event_col=event_col, start_stop_times=['start', 'stop'])

    # Print the summary of the model
    cph.print_summary()

    return cph