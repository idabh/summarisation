import nltk
import os
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

def lemmatize_words(words):
    '''
    Lemmatises words using the danish version of "lemmy" 
    Maybe problem with lemmatizer given two options at places. Index the first element to only get one. 
    Args: words as a tokenized list only lower case
    Return: Lemmatized words
    '''
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(lemmatizer.lemmatize("", word)[0])
    return lemmatized_words

def remove_special_characters(text):
    '''
    Removes special characters but not æ, ø and å 
    Args: A sentence tokenized text
    Return: The text without special characters 
    '''    
    text = re.sub('\W+',' ', text)
    return text

def freq(words):
    '''
    Calculates the frequency that each words is used in a list of words
    Args: Words as a tokenized list only lower case
    Return: a dictionary with each word as key and the frequency as the value
    '''
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq

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
    return pos_tagged_noun_verb # Maybe problem with having no ' ' around the words

def tf_score(word,sentence):
    '''
    Calculates term frequency scores for a word in a given sentence
    Args: The word and the sentence where it is used.
    Return: The term frequency for that word in the sentence
    '''
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences):
    '''
    Calculates inverse document frequency scores a word in a text
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

def word_tfidf(dict_freq,word,sentences,sentence):
    '''
    Calculates the tf_idf score for a given word using the tf, idf and tf_idf functions.
    Args: The dict of frequencies, the word, the text tokenized sentences and the sentence with the word. 
    Return: The tf_idf score for that word
    '''
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence,dict_freq,sentences):
    '''
    Calculates the sentence importance score. 
    Args: The sentence to calculate score, all tokenized sentences from the text and the dictionary of frequencies. 
    Return: The sentence importance score
    '''
     sentence_score = 0
     sentence = remove_special_characters(str(sentence)) 
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = [] 
     no_of_sentences = len(sentences)
     pos_tagged_sentence = pos_tagging(sentence) 
     for word in pos_tagged_sentence:
         word = word.text
         if word.lower() not in Stopwords and word not in Stopwords and len(word)>1: 
             word = word.lower()
             word = lemmatizer.lemmatize("", word)[0]
             sentence_score = sentence_score + word_tfidf(dict_freq,word,sentences,sentence)          
     return sentence_score

def summarise(text, retain_input):
    '''
    Summarises
    Args: The dict of frequencies, the word, the text tokenized sentences and the sentence with the word. 
    Return: The tf_idf score for that word
    '''
    tokenized_sentence = tokenizer.tokenize(text) # also removes the \n\n
    text = remove_special_characters(str(text)) # also removes the \n\n
    text = re.sub(r'\d+', '', text) # removes numbers
    tokenized_words_with_stopwords = word_tokenize(text, language = 'danish')
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords] # should take out "i" and so on
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words) 
    word_freq = freq(tokenized_words) # Get frequencies of the lemmatized words

    input_user = retain_input
    no_of_sentences = int((input_user * len(tokenized_sentence))/100) # number of sentecences for the summary output
    
    # Calculate sentence importance from tokenized sentences
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,word_freq,tokenized_sentence)
        sentence_with_importance[c] = sentenceimp
        c = c+1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []

    # take out the most important sentences indexes
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
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

# Making it for the danewsroom dataset - problem with the iterate
import pandas as pd

# Rouge scores
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True) # is stemmer good for Danish?

def rouge_output(pred, ref, rouge_type):
    # make a RougeScorer object with rouge_types=['rouge1']
    scorer = rouge_scorer.RougeScorer([rouge_type])

    # a dictionary that will contain the results
    results = {'precision': [], 'recall': [], 'fmeasure': []}

    # for each of the hypothesis and reference documents pair
    for (h, r) in zip(pred, ref):
        # computing the ROUGE
        score = scorer.score(h, r)
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

rouge_output(pred, hyp)
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
            break # remove

        # Output a mean rouge score
        rouge_scores.append(scores)
        
    mean_scores_r1 = rouge_output(summaries, filesummaries, 'rouge1')  
    mean_scores_r2 = rouge_output(summaries, filesummaries, 'rouge2') 
    mean_scores_rL = rouge_output(summaries, filesummaries, 'rougeL')   

    results = {'rouge1': mean_scores_r1, 'rouge2': mean_scores_r2, 'rougeL': mean_scores_rL}

    return list([results, filesummaries, summaries])

df = pd.read_csv(r'../danewsroom.csv', nrows = 2)
# Run on the first n samples
output = summarise_danewsroom(df, 70)
output[0]
output[1][6]
output[2][6]
df["text"][]

# Run on the more extractive samples
extractive = df[df["density"] > 8.1875] 
len(df[df["density"] > 8.1875])
extractive = pd.DataFrame.reset_index(extractive)
output_ex = summarise_danewsroom(extractive, 35)
output_ex[0]

# To do 
# Look through code again
# add the other two rouge measures