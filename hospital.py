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
