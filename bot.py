import numpy as np
import nltk.tokenize
from nltk.stem.porter import PorterStemmer
Stemmer=PorterStemmer()
#seperate words
def tokenize(sentence):
   return nltk.word_tokenize(sentence)



#give the base of words+lower
def stem(word):
    return Stemmer.stem(word.lower())
# coincidance entre la phrase et mes mots 
def bag_of_words(tokonized_sentence,words): #sentence tableau de string 
    sentence=[stem(w) for w in tokonized_sentence ]
    print("sentence"+str(sentence))
    bag=np.zeros(len(words),dtype=np.float32)
 
    for wor in sentence:   
       if wor in words:
          index=words.index(wor)
          bag[index]=1.0

    return bag


