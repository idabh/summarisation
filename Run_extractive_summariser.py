from tf_idf_summariser_scr import summarise_danewsroom, word_tokenize, summarise_danewsroom
import pandas as pd
import numpy as np


train = pd.read_csv(r'train_d.csv')
val = pd.read_csv(r'val_d.csv')
test = pd.read_csv(r'test_d.csv')
frames = [train, val, test]
dd = pd.concat(frames)

import nltk
from nltk.tokenize import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')
dd = pd.DataFrame.reset_index(dd)

df = pd.read_csv(r'../danewsroom.csv', nrows = 10)
# Run on the first n samples
output = summarise_danewsroom(dd, 3)
output[0]

d = dd[48:50]
d = pd.DataFrame.reset_index(d)
output = summarise_danewsroom(d, 3)



# Run on the more extractive samples
df = pd.read_csv(r'../danewsroom.csv', nrows = 1500)

extractive = df[df["density"] > 8.1875] 
len(df[df["density"] > 8.1875])
extractive_500 = extractive[0:500]
extractive = pd.DataFrame.reset_index(extractive_500)
output_ex = summarise_danewsroom(extractive, 2)
output_ex[0]

#-- look at scores --
output[0] # make a table with the scores?

#---look at results of the texts summarisation ---
checking = pd.DataFrame(list(zip(output_ex[1], output_ex[2])), columns =['Human', 'Generated'])
#pd.set_option('display.max_colwidth', 50)
checking["Generated"][3]
output[1][2] #human
output[2][2] #extracted


# What is the mean length of the reference summaries?
length = []
text_length = []
for summary, text in zip(df["summary"], df["text"]):
    words = word_tokenize(summary)
    length.append(len(words))
    text_tokens = word_tokenize(text)
    text_length.append(len(text_tokens))

np.mean(length)
np.mean(text_length)

np.mean(length)/np.mean(text_length)

# From the 500 samples the summaries are only ~6% of the text


