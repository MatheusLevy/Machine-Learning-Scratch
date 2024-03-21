import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words
import ast

vocabulary = {}
data = pd.read_csv('data/emails.csv')
nltk.download('words')
set_words = set(words.words())

def build_vocabulary(curr_email):
    idx = len(vocabulary)
    for word in curr_email:
        if word.lower() not in vocabulary and word.lower() in set_words:
            vocabulary[word.lower()] = idx
            idx+=1

if __name__ == '__main__':
    for i in range(data.shape[0]):
        curr_email = data.iloc[i, 0].split()
        print(f'Current email is {i}/{data.shape[0]} and the length of the vocabulary is {len(vocabulary)}')
        build_vocabulary(curr_email)

    file = open('data/vocabulary.txt', 'w')
    file.write(str(vocabulary))
    file.close()
        
    # Map a email to a list of word frequence of that email
    data = pd.read_csv('data/emails.csv')
    file = open('data/vocabulary.txt', 'r')
    contents = file.read()
    vocabulary = ast.literal_eval(contents)
    X = np.zeros((data.shape[0], len(vocabulary)))
    y = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        email = data.iloc[i, 0].split()
        for email_word in email:
            if email_word.lower() in vocabulary:
                X[i, vocabulary[email_word.lower()]] += 1
                y[i] = data.iloc[i, 1]

    np.save('data/X.npy', X)
    np.save('data/y.npy', y)