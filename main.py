import pandas as pd
import pprint

import importPeritoneal as i
import Anemia as a
import hospital as h

file_path = ['/home/dedalo/PycharmProjects/LevanteDPAnemia/Data/LevanteDPC.xls']


dfs_base = i.load_excel_files(file_path)
df_analiticas = i.merge_dataframes(dfs_base['Analíticas'], dfs_base['Pacientes'], 'REGISTRO')


# Clean NaNs for interesting columns

df_analiticas = df_analiticas.dropna(subset=['HEMOGLOBINA'])
dfs_base['Ingresos'] = dfs_base['Ingresos'].dropna(subset=['FINGRESO'])

# Select years

mask = (df_analiticas['FECHA'] >= pd.Timestamp(2010, 1, 1)) & (df_analiticas['FECHA'] <= pd.Timestamp(2023, 12, 31))
df_analiticas = df_analiticas.loc[mask]


# Add anemia column based on 'HEMOGLOBINA' and 'SEXO' specified in KDIGO guidelines

a.add_anemia_column(df_analiticas)


# Include patients that have -30 to 30 days max between the INICIO_DP and the first lab test

included_df = h.include_patients(df_analiticas, 30)


# Baseline

cat_cols = ['SEXO', 'COD_ENFERM', 'TRASPLANTE', 'EXCLUSION', 'INCLUSION', 'HD', 'CARDIORENAL', 'PROVINCIA', 'SALIDA', 'CAUSA', 'EXITUS', 'PASO_A_HD', 'ARRITMIA', 'ITU', 'TBC', 'DM', 'VC', 'DIVERT', 'NEO', 'CH', 'SIST', 'EPOC', 'CARDIO', 'VP', 'DISLIPEMIA', 'HTA', 'FRAGNOS', 'DIURETICO', 'CALCIOANTA', 'IECA', 'ARAII', 'BBLOQUEANTE', 'ABLOQUEANTE', 'ABBLOQUEANTE', 'AGONISTASC', 'VASODILATADOR', 'OTROSFR', 'CHARLSON', 'ICEDAD', 'CATETER', 'CIRUGIA', 'SISTEMA', 'CAMBIOSIST', 'ICODEXTRINA', 'AMINOACIDOS', 'TAMPON', 'PERMA']
lab_cols = ['GLUCOSA', 'UREA', 'CREATININA', 'CKDEPI', 'URICO', 'SODIO', 'POTASIO', 'CALCIO', 'FOSFORO', 'HIERRO', 'TRANSFERRINA', 'IST', 'FERRITINA', 'COLESTEROL', 'TRIGLICERIDOS', 'HDL', 'LDL', 'APOLIPOA', 'APOLIPOB', 'MAGNESIO', 'PROTEINAS', 'ALBUMINA', 'GPT', 'ASTGOT', 'GGT', 'FALCALINA', 'ZINC', 'DIURESIS', 'UREAOR', 'UREA24H', 'CREATOR', 'CREAT24H', 'ACLARACREAT', 'PROTEINASOR', 'PROTEINAS24H', 'PROTCREATOR', 'SODIOR', 'SODIO24H', 'POTASIOR', 'POTASIO24H', 'CLOROR', 'CLORO24H', 'GLU24LP', 'CREAT24LP', 'UREA24LP', 'PROTEIN24LP', 'SODIO24LP', 'LEUCOCITOS', 'NEUTROFILOSP', 'LINFOCITOSP', 'MONOCITOSP', 'EOSINOFILOSP', 'BASOFILOSP', 'GRANULOCITOSP', 'NEUTROFILOS', 'LINFOCITOS', 'MONOCITOS', 'EOSINOFILOS', 'BASOFILOS', 'GRANULOCITOS', 'HEMATIES', 'HEMOGLOBINA', 'HEMATOCRITO', 'VCM', 'HCM', 'CHCM', 'PLAQUETAS', 'PLAQUETOCRITO', 'VPM', 'ANCHOPLQ', 'GRANINMADURO', 'RETICULOCITOSP', 'RETICULOCITOS', 'HBRETICULOCIT', 'TPROTROMBINA', 'RTP', 'QUICK', 'INR', 'APTT', 'TTROMBOPLASTINA', 'TTROMBINA', 'TTR', 'FIBRINOGENO', 'CA125', 'PSA', 'IGG', 'IGA', 'IGM', 'LIPOPROTEINA', 'PCR', 'NTPROBNP', 'FOLATO', 'VITB12', 'TSH', 'T4L', 'INSULINA', 'PTH', 'VITD', 'ALUMINIO', 'ANCHOERITRO', 'PREALBUMINA']

baseline_dict = i.calculate_baseline(df_analiticas, cat_cols, lab_cols, aggregation='average')


# NaN stats

col_nans, row_nans = i.analyze_nans(df_analiticas, print_col_results=False, print_row_results=False)
nan_results = i.analyze_nans(df_analiticas)




# Calculate prevalence for every year

anemia_dict = a.anemia_prevalence(included_df, print_results=True, print_graph=True)
