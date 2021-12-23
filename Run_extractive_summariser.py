from tf_idf_summariser_scr import summarise_danewsroom, word_tokenize, summarise_danewsroom
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')
import json
import codecs
import csv

test = pd.read_csv(r'test_d.csv')
dd = pd.DataFrame.reset_index(test)
# Run on the altogether test samples
output = summarise_danewsroom(dd, 3)

with open('result_test_d_rouge.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output[0]))


with open('result_test_d_references.txt', 'w', encoding='utf8') as convert_file:
    for line in output[1]:
        convert_file.write(line)
        convert_file.write('\n')

with open('result_test_d_summaries.txt', 'w', encoding='utf8') as convert_file:
    for line in output[2]:
        convert_file.write(line)
        convert_file.write('\n')
############

### extractive
test = pd.read_csv(r'ex_test.csv')
dd = pd.DataFrame.reset_index(test)
# Run on the altogether test samples
output = summarise_danewsroom(dd, 3)

with open('result_test_ex_rouge.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output[0]))

with open('result_test_ex_references.txt', 'w', encoding='utf8') as convert_file:
    for line in output[1]:
        convert_file.write(line)
        convert_file.write('\n')

with open('result_test_ex_summaries.txt', 'w', encoding='utf8') as convert_file:
    for line in output[2]:
        convert_file.write(line)
        convert_file.write('\n')

# mixed
test = pd.read_csv(r'mix_test.csv')
dd = pd.DataFrame.reset_index(test)
# Run on the altogether test samples
output = summarise_danewsroom(dd, 3)

with open('result_test_mix_rouge.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output[0]))


with open('result_test_mix_references.txt', 'w', encoding='utf8') as convert_file:
    for line in output[1]:
        convert_file.write(line)
        convert_file.write('\n')

with open('result_test_mix_summaries.txt', 'w', encoding='utf8') as convert_file:
    for line in output[2]:
        convert_file.write(line)
        convert_file.write('\n')

# mixed
test = pd.read_csv(r'abs_test.csv')
dd = pd.DataFrame.reset_index(test)
# Run on the altogether test samples
output = summarise_danewsroom(dd, 3)

with open('result_test_abs_rouge.txt', 'w') as convert_file:
     convert_file.write(json.dumps(output[0]))

with open('result_test_abs_references.txt', 'w', encoding='utf8') as convert_file:
    for line in output[1]:
        convert_file.write(line)
        convert_file.write('\n')

with open('result_test_abs_summaries.txt', 'w', encoding='utf8') as convert_file:
    for line in output[2]:
        convert_file.write(line)
        convert_file.write('\n')
'''

# What is the mean length of the reference summaries compared to the generated summaries?
with open('result_test_ex_references.txt',  encoding='utf8') as f:
    refs = f.readlines()
with open('result_test_ex_summaries.txt',  encoding='utf8') as f:
    summaries = f.readlines()

length = []
refs_length = []
for summary, ref in zip(summaries, refs):
    words = word_tokenize(summary)
    length.append(len(words))
    refs_tokens = word_tokenize(ref)
    refs_length.append(len(refs_tokens))

np.mean(length)
np.mean(refs_length)
'''

''' Length for generated summary vs. reference summary
>>> np.mean(length)
41.612738680991285
>>> np.mean(refs_length)
25.4009840653636
'''
'''
with open('result_test_mix_references.txt',  encoding='utf8') as f:
    refs = f.readlines()
with open('result_test_mix_summaries.txt',  encoding='utf8') as f:
    summaries = f.readlines()

length = []
refs_length = []
for summary, ref in zip(summaries, refs):
    words = word_tokenize(summary)
    length.append(len(words))
    refs_tokens = word_tokenize(ref)
    refs_length.append(len(refs_tokens))

np.mean(length)
np.mean(refs_length)
'''
#np.mean(length)/np.mean(text_length)

# From the 500 samples the summaries are only ~6% of the text


# Read in stuff
#with open('result_test_ex_references.txt',  encoding='utf8') as f:
    #lines = f.readlines()

