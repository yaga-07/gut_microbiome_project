import pandas as pd
import numpy as np 




df = pd.read_csv('trimmed_db.csv', dtype=str)
df = df.astype(str)
print(df.dtypes)
text_to_be_trimmed = "DRP004173_infants stool microbiome_16s rrna amplicon sequencing of stool sample of japanese infant"
print(df['name'].head())
trimmed_db = df.copy()
trimmed_db['name'] = df['name'].apply(lambda x: x.replace(text_to_be_trimmed, ''))
print(trimmed_db.head())

trimmed_db.to_csv('trimmed_db_2.csv', index=False)









