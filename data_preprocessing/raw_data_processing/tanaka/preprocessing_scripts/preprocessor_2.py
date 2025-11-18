import pandas as pd 
import numpy as np 
import re
df = pd.read_csv('trimmed_db_2.csv',dtype=str)
df = df.dropna()

df = df.astype(str)

print(df.dtypes)

def filterSubjectandAge(x):
    x = str(x)
    # pattern 1 - "12a36_"
    # pattern 2 - " no.2 1 year_"
    a = re.findall(r'\d+', x)
    b,c = a[0],a[1] 
    return b,c

df[['subject', 'age (months)']] = pd.DataFrame(df['name'].apply(filterSubjectandAge).tolist(), index=df.index)   
print(df.tail())
del df['name']
df.to_csv('final_trimmed_db.csv')
    