# from https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70


# coding: utf-8
import nltk
import math
nltk.download('stopwords')
from nltk.corpus import stopwords # Also contains stopwords in Danish
# A stopword is e.g. "and", "me", "can"
# check with print(stopwords.words('danish'))
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx # tool for graphs

#filedata = chunk['text'][3]
import re
# splits a txt file into sentences that are also tokenized
def prepare_article(file):
    filedata = file.replace("\n\n", ". ")
    article = filedata.split(". ")
    #article = filedata.split("\n\n") # I added this since it is in many of the articles instead of .
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
   
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w not in stopwords: # if the word in the sentence is not in the stopwords add 1 
            vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w not in stopwords:
            vector2[all_words.index(w)] += 1                                      

    dist = cosine_distance(vector1, vector2) # Meaning the cosine similarity is actually only between the stopwords?
    if math.isnan(dist): # if both vectors are 0 they give nan. We need this to be 1 to make the function work
        dist = 1
        
    return 1 - dist
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
            
    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('danish')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = prepare_article(file_name)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph, max_iter=1000)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))


    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: \n", ". ".join(summarize_text))

    return summarize_text


# Making it for the danewsroom dataset - problem with the iterate
import pandas as pd

# Rouge scores
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


import csv
#df = pd.read_csv('../../NP Exam/danewsroom.csv', chunksize=10000, iterator=True)
#df = pd.read_csv('NP Exam/danewsroom.csv', nrows=100)
df = pd.read_csv('../Summarization/danewsroom.csv', nrows=100)


def summarise_danewsroom(df):
    rouge_scores = []
    preds = []
    summaries = []
    fulltexts = []
    for iter_num in range(len(df)):

        text = df['text'][iter_num]
        summary = df['summary'][iter_num]
        
        pred = generate_summary(text, 3) 
        pred = " ".join(map(str, pred)) # from list of sentences to a string object
            
        scores = scorer.score(summary, pred)
            
        rouge_scores.append(scores)
        summaries.append(summary)
        preds.append(pred)

    #return list([fulltexts, summaries, preds, rouge_scores])
    return preds

# I am a bit confused. Seems to me that the similarity is only between the stopwords.. which is weird. It is supposed to be between everything else but stopwords
output = summarise_danewsroom(df)
print(output[0][23])

#inspect generated extractive summaries VS the human-written summary:




