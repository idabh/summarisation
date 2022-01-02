#THE RESULTS FROM IDA'S RUN OF THIS SCRIPT ARE ON THE GPU!

import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split

#load data
df = pd.read_csv(r'danewsroom.csv', nrows = 500000) #500k
df = df.rename(columns={'Unnamed: 0': 'idx'})
df_small = df[['text', 'summary', 'idx', 'density']]

#get 100k samples in each density category
abs = df_small[df_small.density <= 1.5] #126805
abs = abs[:100000] #only want to keep 100k
mix = df_small[(df_small.density > 1.5) & (df_small.density < 8.1875)] #101541
mix = mix[:100000] #only want to keep 100k
ex = df_small[df_small.density >= 8.1875] #271654
ex = ex[:100000] #only want to keep 100k

#drop density column
abs = abs.drop('density', 1)
mix = mix.drop('density', 1)
ex = ex.drop('density', 1)

#make dataset format
abs_data = Dataset.from_pandas(abs)
abs_data = abs_data.remove_columns('__index_level_0__')
mix_data = Dataset.from_pandas(mix)
mix_data = mix_data.remove_columns('__index_level_0__')
ex_data = Dataset.from_pandas(ex)
ex_data = ex_data.remove_columns('__index_level_0__')

#do test-train-val splits
abs_train, abs_test = abs_data.train_test_split(test_size=0.2).values()
abs_train, abs_val = abs_train.train_test_split(test_size=0.25).values()

mix_train, mix_test = mix_data.train_test_split(test_size=0.2).values()
mix_train, mix_val = mix_train.train_test_split(test_size=0.25).values()

ex_train, ex_test = ex_data.train_test_split(test_size=0.2).values()
ex_train, ex_val = ex_train.train_test_split(test_size=0.25).values()

#Save :)
abs_train.to_csv("abs_train.csv")
abs_test.to_csv("abs_test.csv")
abs_val.to_csv("abs_val.csv")

mix_train.to_csv("mix_train.csv")
mix_test.to_csv("mix_test.csv")
mix_val.to_csv("mix_val.csv")

ex_train.to_csv("ex_train.csv")
ex_test.to_csv("ex_test.csv")
ex_val.to_csv("ex_val.csv")

#test if they can be loaded:
test_train_abs = Dataset.from_pandas(pd.read_csv("abs_train.csv", usecols=['text','summary','idx']))
test_val_abs = Dataset.from_pandas(pd.read_csv("abs_val.csv", usecols=['text','summary','idx']))

