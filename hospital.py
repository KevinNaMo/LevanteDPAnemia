import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# Check if there are (or not) labs for a given hospitalization date
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
