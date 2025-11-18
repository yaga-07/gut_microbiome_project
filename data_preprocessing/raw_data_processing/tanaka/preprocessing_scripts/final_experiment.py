import pandas as pd 
import numpy as np 

db = pd.read_csv('/home/psvenom/Desktop/huggingscience/tanaka results/test_result.csv', index_col=False, dtype=str, na_filter=False)

#remvoe redundant values

db_clear = db.copy()

print(db['Group'][0])
print(db['Group'].unique())
db_clear =  db_clear[db['Group']!='']

print(db_clear.head())

#create a loop that fetches sids of a particular month, then saves it in a seperate csv

months = db_clear['age (months)'].unique()

print(months)

for age in months:
    slice_db = db_clear[db['age (months)'] == age][['#sid','Group']]
    slice_db.to_csv(f'tanaka results/month_{age}.csv', index=False)