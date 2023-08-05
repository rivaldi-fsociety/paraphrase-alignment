from flask import Flask, render_template,url_for,request

import numpy as np
import gensim
import string
import json
import pandas as pd
import re

from numpy import zeros
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.callbacks import LambdaCallback
from keras.layers import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential

app = Flask (__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data_latih')
def data_latih():
	df = pd.read_json('view_hasil_rivaldi_baru1.json')
	dff = df['data'][2]
	dt=pd.DataFrame(dff,columns=['id_case','sentence1','sentence2'])
	sentence1 = []
	sentence2 = []
	id_case1 = []
	id_case1 = dt['id_case'].tolist()
	sentence1 = dt['sentence1'].tolist()
	sentence2 = dt['sentence2'].tolist()
	return render_template('data_latih.html', len = len(id_case1), id_case_ = id_case1, sentence1_ = sentence1, sentence2_ = sentence2)

@app.route('/paraphrase')
def paraphrase():
    return render_template('paraphrase.html')

@app.route('/alignment', methods=['POST'])
def alignment():
    word_model = gensim.models.Word2Vec.load("word2vec.model")
  
    def to_lowercase(words):
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    #Menghapus tanda baca pada setiap kalimat data latih
    def remove_tandabaca(words):
        new_words = []
        for word in words:
            new_word = word.translate(str.maketrans("","",string.punctuation))
            new_words.append(new_word)
        return new_words

    #Menghapus jarak pada setiap kalimat data latih
    def remove_space(words):
        new_words = []
        for word in words:
            new_word = word.strip()
            #re.sub(' +', ' ',new_word)
            re.sub(r"^\s+|\s+$", "", new_word)
            new_words.append(new_word)
        return new_words

    #Menghapus jarak dan tanda pada setiap kalimat data latih
    def text_cleaning(words):
        words = remove_tandabaca(words)
        words = remove_space(words)
        return words

    def preprocessing(words):
        words = text_cleaning(words)
        words = to_lowercase(words)
        return words

    if request.method == 'POST':

        kalimat1 = []
        kalimat2 = []
        kalimat1 = request.form['kalimat1']
        kalimat2 = request.form['kalimat2']
        # kalimat1 = input("Masukkan kalimat 1 :")
        # kalimat2 = input("Masukkan kalimat 2 :")

        hasil1 = kalimat1.split(' ')
        hasil2 = kalimat2.split(' ')
        text1 = []
        text2 = []
        for i in hasil1:
            text1.append(i)
        print(text1)
        for i in hasil2:
            text2.append(i)
        print(text2)

        text1 = preprocessing(text1)
        text2 = preprocessing(text2)
        m = len(text1)
        n = len(text2)
        similar = zeros([n,m], float)
        #menghitung cosine similarity antar kata pada kalimat 1 dan kalimat 2
        for i in range(0, len(text2)):
            for j in range(0, len(text1)):
                similar[i][j] = word_model.similarity(text2[i],text1[j])
                #print(text1[j]," i=",i," j=",j, text2[i])
            #print("\n")

        #mencetak nilai cosine similarity
        for i in range(0, len(text2)):
            for j in range(0, len(text1)):
                print(similar[i][j]," ", end = '')
            print("\n")

        #cek Nilai terbesar pada setiap rows
        a = np.array(similar)
        maximun = np.amax(a, 1)

        for i in range(0, len(text2)):
            for j in range(0, len(text1)):
                if maximun[i] == similar[i][j]:
                    print("X"," ",end = '')
                else:
                    print("O"," ",end = '')
            print("\n")

        kata_1 = []
        kata_2 = []
        jumlah = 1
        total = 1
        for i in range(0, len(text2)):
            for j in range(0, len(text1)):
                if maximun[i] == similar[i][j]:
                    kata_1.append(text1[j])
                    kata_2.append(text2[i])
                    #print(text1[j]," i=",i," j=",j, text2[i])
                    print(jumlah,".",text1[j],"(",j,")","=",text2[i],"(",i,")"," ",end = '')
                    jumlah = jumlah + 1
            print("\n")
        #for i in range(0, len(text1)):
            #print(text1[i],"(",i,")", end = '')
        #print("\n")
        #for i in range(0, len(text1)):
            #print(text1[i],"(",i,")", end = '')
    return render_template('paraphrase_hasil.html',total=total,sim=similar,max=maximun,len_kata1 = len(text1),kata1 = text1,len_kata2 = len(text2),kata2 = text2,jum=jumlah , sentence1 = kalimat1, sentence2 = kalimat2 )

@app.route('/bantuan')
def bantuan():
    return render_template('bantuan.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)