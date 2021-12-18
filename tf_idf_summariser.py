import nltk
import os
import re
import math
import operator
#from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')
nltk.download('averaged_perceptron_tagger')
Stopwords = set(stopwords.words('danish'))
nltk.download('wordnet')
import lemmy
lemmatizer = lemmy.load("da")
lemmatizer.lemmatize("", "ordene")[0]

import spacy
nlp = spacy.load("da_core_news_sm")

import pandas as pd


#wordlemmatizer = WordNetLemmatizer()
def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(lemmatizer.lemmatize("", word)[0])
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    #regex = r'[^a-zA-Z0-9\s]' 
    #regex = r'[^a-åA-Å0-9\s]'
    text = re.sub('\W+',' ', text)
    #text = re.sub(regex,'',text)
    return text

def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq
text = "hej med dig, du er sød"

def pos_tagging_en(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

pos_tagging_en(text)

def pos_tagging(text):
    doc = nlp(text)
    pos_tagged_noun_verb = []
    for token in doc:
        tag = token.text
        if tag == "NOUN" or tag == "VERB":
            pos_tagged_noun_verb.append(token)
    return pos_tagged_noun_verb # Maybe problem with having no ' ' around the words
#pos_tagging(text)

def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf
def idf_score(no_of_sentences,word,sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [lemmatizer.lemmatize("", word)[0] for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf
def tf_idf_score(tf,idf):
    return tf*idf
def word_tfidf(dict_freq,word,sentences,sentence):
    word_tfidf = []
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf
def sentence_importance(sentence,dict_freq,sentences):
     sentence_score = 0
     sentence = remove_special_characters(str(sentence)) 
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = [] 
     no_of_sentences = len(sentences)
     pos_tagged_sentence = sentence # used to use pos_tagging function
     pos_list = pos_tagged_sentence.split() # therefore need to split up tokens
     for word in pos_list:
        if word.lower() not in Stopwords and word not in Stopwords and len(word)>1: 
                word = word.lower()
                word = lemmatizer.lemmatize("", word)[0]
                sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)
                
     return sentence_score

text = "John var en flink mand. Han boede to gader væk, men cyklede altid til skole. Vi elskede John så højt. Han havde tre katte, og en af dem var drægtig med killinger. En dag vågnede John ikke. Hele byen var så trist. De holdt ham en begravelse. Alle kom. Det var rart. Vi vil altid huske John."

def summarise(text, retain_input):
    tokenized_sentence = tokenizer.tokenize(text)
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text, language = 'danish')
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words) # maybe problem with lemmatizer given two options at places. here only giving first but sometimes it is wrong
    word_freq = freq(tokenized_words)
    #input_user = int(input('Percentage of information to retain(in percent):'))
    input_user = retain_input
    no_of_sentences = int((input_user * len(tokenized_sentence))/100)
    #print(no_of_sentences)
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence: # Problem with 0 importance given at the moment
        sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c+1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
            break
    sentence_no.sort()
    cnt = 1
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            
            summary.append(sentence)
            cnt = cnt+1
        else: 
            cnt = cnt+1
    summary = " ".join(summary)

    return summary

# Making it for the danewsroom dataset - problem with the iterate
import pandas as pd

# Rouge scores
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


df = pd.read_csv(r'../danewsroom.csv', nrows = 10)
df["summary"][0] # hmmm correct
d_text = df["text"][0]
summarise(d_text, 30)

# Lists for output
rouge_scores = []
summaries = []
filesummaries = []


def summarise_danewsroom(df, len_summary):
    for iter_num in range(len(df)):

        # do things with chunk
        filedata = df['text'][iter_num]
        filesummary = df['summary'][iter_num]
        
        summary = summarise(filedata, len_summary) 
        #summary = " ".join(map(str, summary)) # from list of sentences to a string object
            
        # Rouge scores
        scores = scorer.score(filesummary, summary)
            
        rouge_scores.append(scores)
        filesummaries.append(filesummary)
        summaries.append(summary)
    #else:
        #print("moving on")
        #continue
        
        # break
        if iter_num == 100:
            break 
    return list([rouge_scores, filesummaries, summaries])

output = summarise_danewsroom(df, 30)
output[0]
#outF = open('summary.txt',"w")
#outF.write(summary)

# To do 
# check that the code splits here \n\n
# Look through code again
# Why is the first hand-written summary only one sentence?