# Inspiration from https://medium.com/voice-tech-podcast/automatic-extractive-text-summarization-using-tfidf-3fc9a7b26f5 
import nltk
import re
import math
import operator
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("danish")
import sys
import pandas as pd
from rouge_score import rouge_scorer

# Making print function for printing the iterations
def my_print(text):
    sys.stdout.write(str(text))
    sys.stdout.flush()

def stem_words(words):
    '''
    Stems words in danish using the Snowballstemmer from nltk
    Args: The words to be stemmed 
    Return: List of stemmed words detokenized
    '''   
    tokens = word_tokenize(words, language = 'danish')
    stemmed_words = []
    for word in tokens:
       stemmed_words.append(stemmer.stem(word))
    stemmed_words = TreebankWordDetokenizer().detokenize(stemmed_words)
    return stemmed_words

def remove_special_characters(text):
    '''
    Removes special characters but not æ, ø and å 
    Args: A sentence tokenized text
    Return: The text without special characters 
    '''    
    text = re.sub('\W+',' ', text)
    text = re.sub(r'_', '', text) # remove underscore from places
    return text


def pos_tagging(text):
    '''
    Finds pos tags for each word in a text
    Args: a sentence of words to be pos tagged
    Return: The words which are tagged as either a noun or a verb
    '''
    doc = nlp(text)
    pos_tagged_noun_verb = []
    for token in doc:
        tag = token.pos_
        if tag == "NOUN" or tag == "VERB":
            pos_tagged_noun_verb.append(token)
    return pos_tagged_noun_verb 

def tf_score(word,sentence):
    '''
    Calculates term frequency scores for a word in a given sentence
    Args: The word and the sentence where it is used.
    Return: The term frequency for that word in the sentence
    '''
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    '''
    Calculates inverse document frequency scores a word in a text meaning how many sentences use a specific word in a list of sentences. 
    Args: Number of sentences in the text, the word to get the score for and the text as tokenized sentences
    Return: The inverse document frequency score
    '''
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
    '''
    Calculates the tf_idf score
    Args: The tf and idf
    Return: The tf_idf score
    '''
    return tf*idf

def word_tfidf(word,sentences,sentence):
    '''
    Calculates the tf_idf score for a given word using the tf, idf and tf_idf functions.
    Args: The word, the text tokenized sentences and the sentence with the word. 
    Return: The tf_idf score for that word
    '''
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence,sentences):
    '''
    Calculates the sentence importance score using the tf-idf score function on the lemmatized nouns and verbs. 
    Args: The sentence to calculate score, all tokenized sentences from the text and the dictionary of frequencies. 
    Return: The sentence importance score
    '''
    
    sentence = remove_special_characters(str(sentence)) 
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = [] 
    pos_tagged_sentence = pos_tagging(sentence) 

    sentence_score = 0
    for word in pos_tagged_sentence:
        word = word.text
        if word.lower() not in Stopwords and word not in Stopwords and len(word)>1: 
            word = word.lower()
            word = lemmatizer.lemmatize("", word)[0]
            sentence_score = sentence_score + word_tfidf(word,sentences,sentence)          
    return sentence_score

def summarise(text, summary_length):
    '''
    Summarises a text by extracting the most important sentences using tf-idf scores
    Args: The text as a string to be summarised and the number of sentences to keep for the summary. 
    Return: A summary as a string
    '''
    # Tokenize sentences meaning seperating each sentence in a text 
    tokenized_sentence = tokenizer.tokenize(text) # also removes the \n\n
    
    # Calculate sentence importance from tokenized sentences
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c+1

    # sort the sentences according to performance
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    max_length = summary_length

    # take out the most important sentences indexes
    for word_prob in sentence_with_importance:
        if cnt < max_length:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
            break
    sentence_no.sort() # sort the sentences
    cnt = 1

    # join the sentences chose as most important
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
            summary.append(sentence)
            cnt = cnt+1
        else: 
            cnt = cnt+1
    summary = " ".join(summary)

    return summary


def rouge_output(pred, ref, rouge_type, stemmer):
    '''
    Calculates rouge scores as a mean of scores from all texts
    Args: The predicted summary, the reference summary and the rouge_type e.g. 'rouge1'. and stemmer = True/False if stemming should be used. 
    Return: A dictionary containing the mean value for precision, recall and fmeasure
    '''
    scorer = rouge_scorer.RougeScorer([rouge_type]) # Rouge uses porter stemmer which is not for Danish - try snowball
    # a dictionary that will contain the results
    results = {'precision': [], 'recall': [], 'fmeasure': []}

    # for each of the hypothesis and reference documents pair
    my_print("before stemmer")
    my_print("\n")
    for (h, r) in zip(pred, ref):
        if stemmer == True:
            h = stem_words(h)
            r = stem_words(r)
        # computing the ROUGE
        score = scorer.score(r, h)
        # separating the measurements
        precision, recall, fmeasure = score[rouge_type]
        # add them to the proper list in the dictionary
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['fmeasure'].append(fmeasure)

    precision = round(np.mean(results['precision']), 4)
    recall = round(np.mean(results['recall']), 4)
    fmeasure = round(np.mean(results['fmeasure']), 4)

    # a dictionary that will contain the results
    result = {'precision': precision, 'recall': recall, 'fmeasure': fmeasure}

    return result


def summarise_danewsroom(df, len_summary, stemmer):
    '''
    Loops over texts in the danewsroom dataset and summarises using the summarise function
    Args: The pandas df containing the danewsroom data and the length of the summary as percentage e.g. 20
    Return: A  list[dict, list, list] where the dict is the rouge mean scores overall. 
    The first list is the reference summaries and the second list is the predicted summaries.
    '''
    # Lists for output
    summaries = []
    filesummaries = []
    for iter_num in range(len(df)):
        my_print(iter_num)
        my_print("\n")
        # do things with chunk
        filedata = df['text'][iter_num]
        filesummary = df['summary'][iter_num]
        
        summary = summarise(filedata, len_summary) 

        # save the summary and reference
        filesummaries.append(filesummary)
        summaries.append(summary)

        # Output a mean rouge score
    mean_scores_r1 = rouge_output(summaries, filesummaries, 'rouge1', stemmer = stemmer)  
    mean_scores_r2 = rouge_output(summaries, filesummaries, 'rouge2',  stemmer = stemmer) 
    mean_scores_rL = rouge_output(summaries, filesummaries, 'rougeL',  stemmer = stemmer)   

    results = {'rouge1': mean_scores_r1, 'rouge2': mean_scores_r2, 'rougeL': mean_scores_rL}

    return [results, filesummaries, summaries]

