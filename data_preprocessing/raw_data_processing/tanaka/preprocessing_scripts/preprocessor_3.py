import pandas as pd 
import numpy as np

df = pd.read_csv('final_trimmed_db.csv')

allergy = pd.read_csv('fix099_Supp.csv', na_filter=False)
allergy["ID"] = allergy["ID"].astype(str)

df['ID'] = df['subject'].astype(str)



df_new=df.merge(allergy[["ID","Group"]], on='ID', how='left') 
df.replace('NaN', np.nan)
   
df_new.dropna()

print(df_new.head(300))

df_new.to_csv('test_result.csv')


