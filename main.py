import pandas as pd
import numpy as np

import importRegistroPeritoneal as imp
import Anemia as a


file_path = ['/home/dedalo/PycharmProjects/LevanteDPAnemia/Data/LevanteDPC.xls']


dfs_base = imp.load_excel_files(file_path)
df_analiticas = imp.merge_dataframes(dfs_base['Analíticas'], dfs_base['Pacientes'], 'REGISTRO')


# Baseline

cat_cols = ['SEXO', 'COD_ENFERM', 'TRASPLANTE', 'EXCLUSION', 'INCLUSION', 'HD', 'CARDIORENAL', 'PROVINCIA', 'SALIDA', 'CAUSA', 'EXITUS', 'PASO_A_HD', 'ARRITMIA', 'ITU', 'TBC', 'DM', 'VC', 'DIVERT', 'NEO', 'CH', 'SIST', 'EPOC', 'CARDIO', 'VP', 'DISLIPEMIA', 'HTA', 'FRAGNOS', 'DIURETICO', 'CALCIOANTA', 'IECA', 'ARAII', 'BBLOQUEANTE', 'ABLOQUEANTE', 'ABBLOQUEANTE', 'AGONISTASC', 'VASODILATADOR', 'OTROSFR', 'CHARLSON', 'ICEDAD', 'CATETER', 'CIRUGIA', 'SISTEMA', 'CAMBIOSIST', 'ICODEXTRINA', 'AMINOACIDOS', 'TAMPON', 'PERMA']
lab_cols = ['GLUCOSA', 'UREA', 'CREATININA', 'CKDEPI', 'URICO', 'SODIO', 'POTASIO', 'CALCIO', 'FOSFORO', 'HIERRO', 'TRANSFERRINA', 'IST', 'FERRITINA', 'COLESTEROL', 'TRIGLICERIDOS', 'HDL', 'LDL', 'APOLIPOA', 'APOLIPOB', 'MAGNESIO', 'PROTEINAS', 'ALBUMINA', 'GPT', 'ASTGOT', 'GGT', 'FALCALINA', 'ZINC', 'DIURESIS', 'UREAOR', 'UREA24H', 'CREATOR', 'CREAT24H', 'ACLARACREAT', 'PROTEINASOR', 'PROTEINAS24H', 'PROTCREATOR', 'SODIOR', 'SODIO24H', 'POTASIOR', 'POTASIO24H', 'CLOROR', 'CLORO24H', 'GLU24LP', 'CREAT24LP', 'UREA24LP', 'PROTEIN24LP', 'SODIO24LP', 'LEUCOCITOS', 'NEUTROFILOSP', 'LINFOCITOSP', 'MONOCITOSP', 'EOSINOFILOSP', 'BASOFILOSP', 'GRANULOCITOSP', 'NEUTROFILOS', 'LINFOCITOS', 'MONOCITOS', 'EOSINOFILOS', 'BASOFILOS', 'GRANULOCITOS', 'HEMATIES', 'HEMOGLOBINA', 'HEMATOCRITO', 'VCM', 'HCM', 'CHCM', 'PLAQUETAS', 'PLAQUETOCRITO', 'VPM', 'ANCHOPLQ', 'GRANINMADURO', 'RETICULOCITOSP', 'RETICULOCITOS', 'HBRETICULOCIT', 'TPROTROMBINA', 'RTP', 'QUICK', 'INR', 'APTT', 'TTROMBOPLASTINA', 'TTROMBINA', 'TTR', 'FIBRINOGENO', 'CA125', 'PSA', 'IGG', 'IGA', 'IGM', 'LIPOPROTEINA', 'PCR', 'NTPROBNP', 'FOLATO', 'VITB12', 'TSH', 'T4L', 'INSULINA', 'PTH', 'VITD', 'ALUMINIO', 'ANCHOERITRO', 'PREALBUMINA']

baseline_dict = imp.calculate_baseline(df_analiticas, cat_cols, lab_cols, aggregation='average')


# NaN stats

col_nans, row_nans = imp.analyze_nans(df_analiticas, print_col_results=False, print_row_results=False)
nan_results = imp.analyze_nans(df_analiticas)


# Add anemia column based on 'HEMOGLOBINA' and 'SEXO' specified in KDIGO guidelines

a.add_anemia_column(df_analiticas)


# Calculate prevalence for every year

anemia_dict = a.anemia_prevalence(df_analiticas)