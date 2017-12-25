import os
import enchant
import nltk
from nltk.stem import WordNetLemmatizer
import re
import math
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity



def get_bow1(file):
    f = open(file,'rb')
    content = f.read().splitlines()
    d = enchant.Dict("en_US")
    bowtemp = []
    for sentence in content:
        words = nltk.word_tokenize(str(sentence))
        for word,pos in nltk.pos_tag(words):
            if(len(word)>2):
                if(pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS' or pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
                    word = word.replace('0x96',"")
                    word = word.replace('.',"")
                    word = word.replace('0x94',"")
                    word = WordNetLemmatizer().lemmatize(word.lower(),pos = 'v')
                    #if(d.check(word)):
                        if bool(re.search(r'^[A-Z', word))==False:
                            bowtemp.append(word)
    return bowtemp

f = open("/Users/apple/temp/nltk/task2/verbsbow_set.txt","w")

def read_files():
    path = "/Users/apple/temp/nltk/task2/Task_2"
    content = os.listdir(path)
    bow = []
    i = 0
    for direc in content:
        if direc!=".DS_Store":
            files = os.listdir(path + "/" + direc)
            for file in files:
                print(i)
                i=i+1
                bowt = get_bow1(path + "/" + direc + "/" + file)
                bowt = list(set(bowt))
                for word in bowt:
                    f.write("%s " %word)
                f.write("\n")
                #print(len(bow))
    f.close()
read_files()