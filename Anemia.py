import pandas as pd
import numpy as np

def add_anemia_column(input_df, Hb_masc=13, Hb_fem=12):
    # Define a function to apply to each row
    def check_anemia(row):
        # Check if 'SEXO' or 'HEMOGLOBINA' are NaN
        if pd.isna(row['SEXO']) or pd.isna(row['HEMOGLOBINA']):
            print(f"NaN value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']}")
            return np.nan
        # Check if the patient is a man and has anemia
        elif row['SEXO'] == '1. Hombre' and row['HEMOGLOBINA'] < Hb_masc:
            print(f"True value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']} and {row['SEXO']}")
            return True
        # Check if the patient is a woman and has anemia
        elif row['SEXO'] == '2. Mujer' and row['HEMOGLOBINA'] < Hb_fem:
            print(f"True value for {row['REGISTRO']} since hb = {row['HEMOGLOBINA']} and {row['SEXO']}")
            return True
        # If none of the above conditions are met, the patient does not have anemia
        else:
            return False

    # Apply the function to each row and assign the results to the new 'ANEMIA' column
    input_df['ANEMIA'] = input_df.apply(check_anemia, axis=1)


def anemia_prevalence(df):
    # Create a temporary dataframe with the necessary columns
    anemia_df = df[['REGISTRO', 'FECHA', 'ANEMIA']].copy()

    # Convert 'FECHA' to datetime format and extract the year
    anemia_df['YEAR'] = pd.to_datetime(anemia_df['FECHA']).dt.year

    # Sort the dataframe by 'ANEMIA' in descending order (True before False)
    anemia_df.sort_values('ANEMIA', ascending=False, inplace=True)

    # Drop duplicate rows for the same patient and year, keeping the first occurrence (which is 'ANEMIA' == True if it exists)
    anemia_df.drop_duplicates(subset=['YEAR', 'REGISTRO'], keep='first', inplace=True)

    # Group by 'YEAR' and calculate the sum of 'ANEMIA' (True is 1, False is 0)
    yearly_anemia_sum = anemia_df.groupby('YEAR')['ANEMIA'].sum()
    print(f'yearly anemia: {yearly_anemia_sum}')

    # Group by 'YEAR' and get the number of unique patients
    yearly_patients = anemia_df.groupby('YEAR')['REGISTRO'].nunique()
    print(f'total_patients: {yearly_patients}')

    # Calculate the prevalence of 'ANEMIA' for each year
    prevalence = (yearly_anemia_sum / yearly_patients) * 100
    prevalence = np.round(prevalence, 2)

    return prevalence.to_dict()