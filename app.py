from vega_datasets import data
import streamlit as st
import altair as alt
import os
import pandas as pd
import re
import string
from tqdm import tqdm_notebook as tqdm
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from IPython import get_ipython
import numpy as np
from nltk.tokenize import word_tokenize 
import altair as alt
# get_ipython().magic(u'matplotlib inline')

def main():
    
    halaman = st.sidebar.selectbox("MENU", ["Tentang", "Data Preprocessing","Simulasi Word2vec","Prediksi Kalimat","Dokumentasi Hasil Training"])

    def load_css(css_file):
        with open(css_file) as f:
            st.markdown('<style>{}</style>'.format(f.read()),unsafe_allow_html=True)
    load_css("gaya.css")
    def load_images(image_name):
        img = Image.open(image_name)
        return st.image(img,width=300)

    if halaman == "Tentang":
        def loadpage():
            st.markdown('''
            <h1 style="margin-bottom:0px">Analisis Sentimen Media Sosial Twitter Terhadap Layanan Provider Telekomunikasi Menggunakan Metode Long Short Term Memory</h1>
            <h2 style="margin-top:0px">(Studi Kasus PT Telkomsel)</h2>
            <hr class="new5"
            <div>
                <h1 class="title">Abstrak</h1>
                <p class="abstrak">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.</p>
            </div>               
            ''',unsafe_allow_html=True)
            if st.checkbox("Tentang Penulis dan Pembimbing"):
                st.markdown('''
                <div id='container'>
                    <div id="conten">
                        <h1 class="nama">Fauzi Faruq Nabbani</h1>
                        <p class="biodata">Nama Panggilan : Fauzi<br>Perguruan Tinggi : Universitas Padjadjaran<br>Program Studi : Teknik Informatika<br>NPM : 140810160007<br>JK : Laki-laki<br>TTL : Tasikmalaya, 20 September 1997<br>Agama : Islam<br></p>
                    </div>
                </div>
                <div>
                    <div id='parent'>
                        <div id='wide'>
                            <h2 class="title">Pembimbing I</h2>
                            <p class="biodata1">Nama Lengkap : Dr. Intan Nurma Yulita , M.T.<br>NPM : 19850704 201504 2 003</p>
                        </div>
                        <div id='narrow'>
                            <h2 class="title">Pembimbing II</h2>
                            <p class="biodata1">Nama Lengkap : Drs. Ino Suryana, M.Kom.<br>NIP : 19600115 198701 1 002</p>
                        </div>
                    </div>
                </div>              
            ''',unsafe_allow_html=True)
        loadpage() 
        
    elif halaman == "Data Preprocessing":
        st.markdown(
        """
        <h1 class="title">Data Preprocessing</h1>
        """, unsafe_allow_html=True)
        st.text("Lorem Ipsum is simply dummy text of the printing and typesetting industry. \nLorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.\nIt has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.\nIt was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
        st.markdown(
        """
        <h2 class="title">Informasi Tentang Data</h2>
        <ul>
            <li>Sumber Data berasal dari Sosial Media Twitter Komentar Masyrakat terhadap akun @Telkomsel</li>
            <li>Data yang dikumpul yaitu dari Bulan Januari - Februari, dan melakuakan penambahan data pada minggu ke 1 April 2020</li>
            <li>Sebelum diproses kedalam tahap preprocessing menggunakan coding , dilakuakan pelabelan dan perubahan kata kata dalam sumber dalam sesuai dengan kbbi</li>
            <li>Jumlah data adalah 7346, dengan komposisi 1000 kalimat berlabel positif dan 1000 kalimat berlabel negatif</li>
        </ul>
        """, unsafe_allow_html=True)
        # def pilih_file(folder_path='./datamentah'):
        #     namafile = os.listdir(folder_path)
        #     fileterpilih = st.selectbox('Pilih Data set',namafile)
        #     return os.path.join(folder_path, fileterpilih)
        # data = pilih_file()
        # st.write('You selected `%s`' % data)

        dataTweet = pd.read_csv('./datamentah/DataTelkomsel.csv',usecols=["text","sentiment"], encoding = "latin-1")
        st.markdown(
        """
        <h2 class="title">Data Awal</h2>
        """, unsafe_allow_html=True)
        
        if st.checkbox('Menampilkan Seluruh data, bentuk data, jumlah setiap label',key=0):
            st.markdown(
                """
                <h3>Menampilkan Seluruh data</h3>
                """, unsafe_allow_html=True)
            st.dataframe(dataTweet)
            st.markdown(
                """
                <h3>Menampilkan Bentuk Data</h3>
                """, unsafe_allow_html=True) 
            st.write(dataTweet.shape)
            st.write("Jumlah baris ",dataTweet.shape[0] ," Jumlah kolom ", dataTweet.shape[1])
            st.markdown(
                """
                <h3>Menampilkan Jumlah Data Berlabel Negatif dan Positif</h3>
                """, unsafe_allow_html=True)
            st.write(dataTweet['sentiment'].value_counts())
            st.markdown(
                """
                <h3>Ringkasan Dataset</h3>
                """, unsafe_allow_html=True)
            st.write(dataTweet.describe())
    

        ################################## CASE FOLDING #######################################    

        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Case Folding</h2>
        """, unsafe_allow_html=True)
        def casefolding(text):
            text = text.lower().strip()
            return text
        hasilcasefolding = []
        for text in dataTweet.text:
            pro = casefolding(text)
            hasilcasefolding.append(pro)
            
        dataTweet.insert(1,"casefolding",hasilcasefolding)
        dataTweet.to_csv('Hasilcasefolding.csv')
        dataTweet.head(10)

        if st.checkbox('Hasil Preprocessing | Case Folding',key=1):
            st.dataframe(dataTweet)
            st.text("Simulasi Data Preprocessing Case Folding")
            cobacasefolding = st.text_input('Masukan kalimat',key=1)
            st.write('Hasil CaseFolding : ',casefolding(cobacasefolding))

        ################################## NOISE REMOVAL #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Noise Removal</h2>
        """, unsafe_allow_html=True)
        def clean(text):
            text = re.sub(r'\n', '', text)
            #Menghapus username 
            text = re.sub("(@[A-Za-z0-9]+)","",text)
            text = re.sub("(@_[A-Za-z0-9]+)","",text)
            text =  re.sub(r'\\x..', '', text)
            #Menghapus #
            text = re.sub("(#[A-Za-z0-9]+)","",text)
            text = re.sub("(\w+:\/\/\S+)","",text)
            #Menghapus link
            text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', '', text)
            #Menghapus sebuah kalimat bersama angka dalam satu kata
            text = re.sub("(\w*\d\w*)","",text)
            # menghapus simbol
            text = re.sub(r'[^A-Za-z\s\/]' ,' ', text)
            text = re.sub(r'_', '', text) #hapus simbol _
            # menghapus angka
            text = re.sub(r'\d+', '', text)  
            # menghapus spasi
            text = re.sub(r'\s{2,}', ' ', text)
            #menghapus b diawal kalimat tweet
            text = text.replace('b ','')
            return text
  
        hasilclean = []
        for desc in dataTweet.casefolding:
            pro = clean(desc)
            hasilclean.append(pro)
  
        dataTweet.insert(2,"noiseremoval",hasilclean)
        dataTweet.to_csv('Hasilcleaning.csv')
        dataTweet.rename(columns={"text": "Hasil Cleaning"}).head(10)

        if st.checkbox('Hasil Preprocessing | Noise Removal',key=2):
            st.write(dataTweet)
            st.text("Simulasi Data Preprocessing Noise Removal")
            cobaremovenoise = st.text_input('Masukan kalimat',key=2)
            st.write('Hasil Noise Removal : ',clean(cobaremovenoise))
        
        ################################## Tokenizer #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Tokenizer </h2>
        """, unsafe_allow_html=True)
        def tokenizer(text):
            return text.split()

        hasiltokenizer = []
        for desc in tqdm(dataTweet['noiseremoval']):
            pro = tokenizer(desc)
            hasiltokenizer.append(pro)
            
        dataTweet.insert(3,"tokenizer",hasiltokenizer)
        dataTweet.to_csv('hasiltokonizer.csv')
        dataTweet.rename(columns={"text": "Hasil Tokenizer"}).head(10)

        if st.checkbox('Hasil Preprocessing | Tokenizer ',key=3):
            st.write(dataTweet)
            st.text("Simulasi Data Preprocessing Tokenizer")
            cobatokenizer = st.text_input('Masukan kalimat',key=3)
            st.write('Hasil Tokenizer : ',tokenizer(cobatokenizer))
        
        ################################## StopWord Removal #######################################
        st.markdown(
        """
        <h2 class="title">Data Preprocessing | Stopword Removal </h2>
        """, unsafe_allow_html=True)

        def hapus_stopword(desc):
            StopWords = "stopwordbaru.txt"
            sw=open(StopWords,encoding='utf-8', mode='r');stop=sw.readlines();sw.close()
            stop=[kata.strip() for kata in stop];stop=set(stop)
            kata = [item for item in desc if item not in stop]
            return ' '.join(kata)
        hasilstopword = []

        for desc in tqdm(dataTweet['tokenizer']):
            pro = hapus_stopword(desc)
            hasilstopword.append(pro)
            
        dataTweet.insert(4,"stopword",hasilstopword)
        dataTweet.to_csv('Hasilstopwordfile.csv')
        if st.checkbox('Hasil Preprocessing | Stopword Removal ',key=4):
            st.write(dataTweet)
            st.text("Simulasi Data Preprocessing Stopword Removal")
            cobastopword = st.text_input('Masukan kalimat',key=4)
            word_tokens = word_tokenize(cobastopword)
            stop_words = open('stopwordbaru.txt','r').read().split()
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            v = ' '.join(filtered_sentence) 
            st.write('Hasil Stopword Removal : ',v)

    elif halaman == "Simulasi Word2vec":
        st.markdown(
        """
        <h1 class="title">Feature Extraction - Word2vec</h1>
        """, unsafe_allow_html=True)
        st.text("Lorem Ipsum is simply dummy text of the printing and typesetting industry. \nLorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.\nIt has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.\nIt was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
        st.markdown(
        """
        <h2 class="title">Hyperparameter yang digunakan</h2>
        <ul>
            <li>Size : 100 (Dimensi Kata Vektor)</li>
            <li>Window : 3 (jarak antara kata-kata konteks dengan posisi kata yang menjadi inputan)</li>
            <li>Min_count : 1 (semua kata dengan frekuensi total lebih rendah dari nilai min_count)</li>
            <li>Epoch : 100 (Melatih vektor kata dengan 100 epoch)</li>
        </ul>
        """, unsafe_allow_html=True)

        id_w2v = Word2Vec.load("hasilword2vec.w2v")

        dataTweet = pd.read_csv("word2vec.csv")
        st.markdown(
        """
        <h2 class="title">Menampilkan vektor setiap kata</h1>
        """, unsafe_allow_html=True)
        st.write(dataTweet)

        st.markdown(
        """
        <h2 class="title">Simulasi Kemiripan antar kata dengan Word2vec</h1>
        """, unsafe_allow_html=True)

        word2vec = st.text_input('Masukan satu kata untuk mencari similarity',key=5)
        if st.button("Submit",key=0):
            st.text("Vector dari kata " + word2vec + " adalah :")
            st.write(id_w2v[word2vec])
            # st.text("lima Kata yang paling mirip dengan : ",word2vec)
            st.write(id_w2v.wv.most_similar(word2vec, topn = 5))
        st.markdown(
        """
        <h2 class="title">Tingkat Kemiripan antar dua kata</h1>
        """, unsafe_allow_html=True)   
        kata1 = st.text_input('kata1 : ',key=6)
        kata2 = st.text_input('kata2 : ',key=7)
        if st.button("Submit",key=1):
            
            st.write("Kata : " + kata1 + " dengan kata " + kata2 + " memiliki tingat kemiripan : " , id_w2v.wv.similarity(kata1,kata2))
    
    elif halaman == "Prediksi Kalimat":
        st.markdown(
        """
        <h1 class="title">Prediksi Kalimat</h1>
        """, unsafe_allow_html=True)
        model = load_model('Malam29AprilSatu7.h5',compile=False)
        with open('malam29april.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        def load_css(css_file):
            with open(css_file) as f:
                st.markdown('<style>{}</style>'.format(f.read()),unsafe_allow_html=True)
        def loadh1():
            st.markdown('<h1 class="kelas">Hello Word</h1>',unsafe_allow_html=True)        

        url = st.text_input('Masukan kalimat',key=8)
        twt = [url]
        twt = tokenizer.texts_to_sequences(twt)
        twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
        sentiment = model.predict(twt,batch_size=1,verbose = 2)
        if st.button("Submit",key=2):
            if(sentiment[0] < 0.5):
                st.success("Negative with "+ "{0:.2f}".format((1 - sentiment[0][0]) * 100)+ "% Confidence.")
                load_images("negative.png")
            else:
                st.success("Positive with "+"{0:.2f}".format(sentiment[0][0] * 100)+"% Confidence.")
                load_images("positive.png")

        def analyseSentiment(data):
            twt = [data]
            twt = tokenizer.texts_to_sequences(twt)
            twt = pad_sequences(twt, maxlen=32, dtype='int32', value=0)
            sentiment = model.predict(twt,batch_size=1,verbose = 2)
            if(sentiment[0] < 0.5):
                return "Negative with "+ "{0:.2f}".format((1 - sentiment[0][0]) * 100)+ "% Confidence."
            else:
                return "Positive with "+"{0:.2f}".format(sentiment[0][0] * 100)+"% Confidence." 
        datavalidasi = pd.read_csv('Hasilstopwordfile.csv', usecols=["text","casefolding","noiseremoval","tokenizer","stopword","sentiment"], encoding = "latin-1")
        datavalidasi = datavalidasi[datavalidasi['stopword'].notnull()]
        datavalidasi['sentiment'] = datavalidasi['sentiment'].map({'positive' :1, 'negative' : 0})
        hasilpredik = []
        for desc in datavalidasi.stopword:
            pro = analyseSentiment(desc)
            hasilpredik.append(pro)   
        datavalidasi.insert(6,"prediksi",hasilpredik)
        datavalidasi['prediksi'] = hasilpredik
        if st.checkbox("Data Sample untuk Validation"):
            st.write(datavalidasi[["text","sentiment","prediksi"]].sample(40))         
        load_css("gaya.css")
        loadh1()

    elif halaman == "Dokumentasi Hasil Training":
        st.markdown(
        """
        <h1 class="title">Dokumentasi Hasil Training</h1>
        """, unsafe_allow_html=True)
        st.markdown(
        """
        <h2 class="title">Skema Pelatihan Data</h2>
        <ul>
            <li>Metode yang Digunakan adalah Long Short Term Memory</li>
            <li>Data yang digunakan untuk proses training adalah 80% dan 20% dijadikan sebagain data testing</li>
            <li>Menggunakan 10 Fold Cross Validation</li>
            <li>Telah dilakukan beberapa kali pelatihan ,dan pada website ini aka ditampilkan hasil training yang digunakan</li>
        </ul>
        """, unsafe_allow_html=True)
        if st.checkbox("Hasil Training"):
            st.markdown(
            """
            <h3 class="title">Hyperparameter yang digunakan</h3>
            <ul>
                <li>LSTM Hidden State : 100</li>
                <li>Dropout : 0.5</li>
                <li>Fungsi Aktivasi : Sigmoid</li>
                <li>Optimizer : Adama</li>
                <li>Batch Size :128 </li>
                <li>Epoch : 50</li>
                <li>Hyperparameter Word2vec = Size : 100 , Window : 3, min_count : 1, workers : 4, epoch : 100</li>
                <li>Lama Waktu Training : 45 menit</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown(
            """
            <h3 class="title">Skema 10 Fold Cross Validation</h3>
            """, unsafe_allow_html=True)
            if st.checkbox("Fold ke 1"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 2"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 3"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 4"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 5"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 6"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 7"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 8"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))
            if st.checkbox("Fold ke 10"):
                st.markdown(
                """
                <ul>
                    <li>Akurasi Training : 98 %</li>
                </ul>
                """, unsafe_allow_html=True)
                def load_image(image_name):
                    img = Image.open(image_name)
                    return st.image(img,width=600)
                load_image('NewImage1.png')
                st.markdown(
                """
                <ul>
                    <li>Akurasi Testing : 98 % , F-1 Score : 97 %, Precission : 97% , Recall :97%</li>
                </ul>
                """, unsafe_allow_html=True)
                data = pd.DataFrame({'cm': ['True Positif', 'True  Negatif', 'False Positif','False Negatif'],'jumlah': [666, 656, 18, 22],})
                st.subheader('Evaluasi Menggunakan Confusion Matrix')
                st.write(data)
                st.write(alt.Chart(data).mark_bar().encode(x=alt.X('cm', sort=None),y='jumlah',))

            
if __name__ == "__main__":
    main()