import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from scipy.stats import ttest_ind

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

def anemia_prevalence(df, print_results=False, print_graph=False, tendency=False):
    # Create a temporary dataframe with the necessary columns
    anemia_df = df[['REGISTRO', 'FECHA', 'ANEMIA']].copy()

    # Convert 'FECHA' to datetime format and extract the year
    anemia_df['YEAR'] = pd.to_datetime(anemia_df['FECHA']).dt.year

    # Sort the dataframe by 'ANEMIA' in descending order (True before False)
    anemia_df.sort_values('ANEMIA', ascending=False, inplace=True)

    # Drop duplicate rows for the same patient and year, keeping the first occurrence (which is 'ANEMIA' == True if it exists)
    anemia_df.drop_duplicates(subset=['YEAR', 'REGISTRO'], keep='first', inplace=True)

    # Group by 'YEAR' and calculate the sum of 'ANEMIA' (True is 1, False is 0)
    yearly_anemia_sum = pd.to_numeric(anemia_df.groupby('YEAR')['ANEMIA'].sum())

    # Group by 'YEAR' and get the number of unique patients
    yearly_patients = anemia_df.groupby('YEAR')['REGISTRO'].nunique()

    # Calculate the prevalence of 'ANEMIA' for each year
    prevalence = (yearly_anemia_sum / yearly_patients) * 100
    prevalence = prevalence.round(2)

    # Print results if print_results is True
    if print_results:
        print("Prevalencia anual de anemia:")
        for year, value in prevalence.items():
            print(f"{year}: {value}%")

    # Render a bar graph if print_graph is True
    if print_graph:
        plt.figure(figsize=(10, 5))
        prevalence.plot(kind='bar')
        plt.title('Prevalencia anual de anemia')
        plt.xlabel('Año')
        plt.ylabel('Prevalencia (%)')

        plt.show()

    return prevalence.to_dict()

def time_trend_analysis(df_input, time_frame, col_names, follow_up_limit, plot_results=False, t_test=False):
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
        