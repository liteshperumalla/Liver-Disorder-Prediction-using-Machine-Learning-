import pandas as pd
file_1 = '/Users/liteshperumalla/Desktop/Files/projects/archive/dataset1.csv'
file_2 = '/Users/liteshperumalla/Desktop/Files/projects/archive/dataset2.csv'
df1 = pd.read_csv(file_1 , encoding='unicode_escape', sep=';')
df2 = pd.read_csv(file_2, encoding='unicode_escape', sep=';')
try:
    df1 = pd.read_csv(file_1, encoding='utf-8')
except UnicodeDecodeError:
    df1 = pd.read_csv(file_1, encoding='latin1')

try:
    df2 = pd.read_csv(file_2, encoding='utf-8')
except UnicodeDecodeError:
    df2 = pd.read_csv(file_2, encoding='latin1')
merge_df = pd.merge(df1, df2, on = ['age', 'Gender', 'Total_Billrubin', 'Direct_Billrubin', 'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase', 'Sgot_Aspartate_Aminotransferase', 'Total_Proteins', 'ALB_Albumin', 'A/G_Ratio_Albumin_and_Gobulin Ratio'], how = 'outer')
output_file = '/Users/liteshperumalla/Desktop/Files/projects/archive/traindataset.csv'
merge_df.to_csv(output_file, index=False)
print(f'Merged data saved to {output_file}')