from tf_idf_summariser_scr import summarise_danewsroom, word_tokenize
import pandas as pd
import numpy as np


import nltk
from nltk.tokenize import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')

df = pd.read_csv(r'../danewsroom.csv', nrows = 500)
# Run on the first n samples
output = summarise_danewsroom(df, 15)
output[0]

# Run on the more extractive samples
df = pd.read_csv(r'../danewsroom.csv', nrows = 1500)

extractive = df[df["density"] > 8.1875] 
len(df[df["density"] > 8.1875])
extractive_500 = extractive[0:500]
extractive = pd.DataFrame.reset_index(extractive_500)
output_ex = summarise_danewsroom(extractive, 15)
output_ex[0]

#-- look at scores --
output[0] # make a table with the scores?

#---look at results of the texts summarisation ---
checking = pd.DataFrame(list(zip(output[1], output[2])), columns =['Human', 'Generated'])
#pd.set_option('display.max_colwidth', 50)
checking
output[1][0] #human
output[2][0] #extracted


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


