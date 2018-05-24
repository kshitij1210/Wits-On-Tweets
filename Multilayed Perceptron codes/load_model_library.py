from __main__ import *
import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()
from string import punctuation

def donut(label,pred):
    k= pred[0][np.argmax(pred)] * 100
    labels = ""
    colors = []  
    print(label)
    flag=0
    if k >= 70 and label == 0:
        flag=1
        print("The sentiment of tweet is NEGATIVE!!!\n")
    if k >= 70 and label == 1:
        flag=2
        print("The sentiment of tweet is POSITIVE!!!\n")
    if flag == 0: 
        print("The sentiment of tweet is somewhat NEUTRAL or NOT SURE !!!\n");
    
    if label == 0:
        labels="Negative","Positive"
        colors.append('red')
        colors.append('green')
    else:
        labels="Positive","Negative"
        colors.append('green')
        colors.append('red')
    
    sizes = [k,100-k]
    
    explode = (0, 0)  # explode a slice if required

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
        
    #draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    
    
    plt.show()

    
    


tokenizer = Tokenizer(num_words=3000)
# for human-friendly printing
labels = ['Negative', 'Positive']

# read in our saved dictionary
with open('dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print(("'%s' not in training corpus; ignoring." %(word)))
    return wordIndices


# read in your saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('model.h5')

# okay here's the interactive part
def sentiment_detect(line):
    stop_words = set(stopwords.words('english'))
    line=line+" ."
    line=line.lower()
    words = word_tokenize(line)
    lem_words=[lemmatizer.lemmatize(x) for x in words]
    new_str1=""
    new_str=""
    count=0
    flag=0
    new_words=[]
    for i in lem_words:
        flag=flag+1
        count=count+1
        if i == "not" or i == "no" or i == "n't":
            i=i+"_"+lem_words[count]
            flag=-1
        if flag !=0:
            new_words.append(i)
    for i in new_words:
        new_str=new_str+i+" "
    new_str =''.join([char for char in new_str if char not in punctuation])
    for word in word_tokenize(new_str.lower()):
        if word not in stop_words:
            new_str1=new_str1+word+" "
    print("Tweet after negation handling",new_str1);
    # format your input for the neural net
    testArr = convert_text_to_index_array(new_str1)
    input1 = tokenizer.sequences_to_matrix([testArr], mode='binary')
    # predict which bucket your input belongs in
    pred = model.predict(input1)
    
    # and print it for the humons
    return np.argmax(pred)
    #xx=1
    #if np.argmax(pred)==1:
    #    xx=1
    #else:
    #    xx=0
    #donut(xx,pred)
def sentiment_donut(line):
    stop_words = set(stopwords.words('english'))
    line=line+" ."
    line=line.lower()
    words = word_tokenize(line)
    lem_words=[lemmatizer.lemmatize(x) for x in words]
    new_str=""
    new_str1=""
    count=0
    flag=0
    new_words=[]
    for i in lem_words:
        flag=flag+1
        count=count+1
        if i == "not" or i == "no" or i == "n't":
            i=i+"_"+lem_words[count]
            flag=-1
        if flag !=0:
            new_words.append(i)
    for i in new_words:
        new_str=new_str+i+" "
    new_str =''.join([char for char in new_str if char not in punctuation])
    for word in word_tokenize(new_str.lower()):
        if word not in stop_words:
            new_str1=new_str1+word+" "
   
    print("Tweet after negation handling",new_str1);
    testArr = convert_text_to_index_array(new_str1)
    input1 = tokenizer.sequences_to_matrix([testArr], mode='binary')

    pred = model.predict(input1)
    
    xx=1
    if np.argmax(pred)==1:
        xx=1
    else:
        xx=0
    donut(xx,pred)
    return xx