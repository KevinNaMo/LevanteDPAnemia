import pandas as pd

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