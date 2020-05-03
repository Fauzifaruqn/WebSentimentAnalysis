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


def main():
    
    halaman = st.sidebar.selectbox("MENU", ["Tentang", "Data Preprocessing","Simulasi Word2vec","Prediksi Kalimat","Hasil Training"])

    if halaman == "Tentang":
        def load_css(css_file):
            with open(css_file) as f:
                st.markdown('<style>{}</style>'.format(f.read()),unsafe_allow_html=True)
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
                        <p class="biodata">Nama Panggilan : Fauzi<br>NPM : 140810160007<br>JK : Laki-laki<br>TTL : Tasikmalaya, 20 September 1997<br>Agama : Islam<br></p>
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
        load_css("gaya.css")
        loadpage() 
        
    elif halaman == "Data Preprocessing":
        st.markdown(
        """
        <h1>Data Preprocessing</h1>
        """, unsafe_allow_html=True)
        st.text("Lorem Ipsum is simply dummy text of the printing and typesetting industry. \nLorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.\nIt has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.\nIt was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")
        
        def pilih_file(folder_path='./datamentah'):
            namafile = os.listdir(folder_path)
            fileterpilih = st.selectbox('Pilih Data set',namafile)
            return os.path.join(folder_path, fileterpilih)
        data = pilih_file()
        st.write('You selected `%s`' % data)

        dataTweet = pd.read_csv(data,usecols=["text","sentiment"], encoding = "latin-1")
        st.markdown(
        """
        <h2>Data Awal</h2>
        """, unsafe_allow_html=True)
        
        if st.checkbox('Data Awal | Menampilkan Semua Data',key=0):
            st.write(dataTweet) 
    

        ################################## CASE FOLDING #######################################    

        st.markdown(
        """
        <h2>Data Preprocessing | Case Folding</h2>
        """, unsafe_allow_html=True)
        st.text("Case Folding adalah merupakan bla bla bla")
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
        <h2>Data Preprocessing |Noise Removal</h2>
        """, unsafe_allow_html=True)
        st.text("Case Folding adalah merupakan bla bla bla")
        def clean(text):
            text = re.sub(r'\n', '', text)
            #Menghapus username 
            text = re.sub("(@[A-Za-z0-9]+)","",text)
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
        <h2>Data Preprocessing | Tokenizer </h2>
        """, unsafe_allow_html=True)
        st.text("Tokenizer adalah merupakan bla bla bla")
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
        <h2>Data Preprocessing | Stopword Removal </h2>
        """, unsafe_allow_html=True)
        st.text("Stopword Removal adalah merupakan bla bla bla")

        def hapus_stopword(desc):
            StopWords = "stpword.txt"
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
            st.write('Hasil Stopword Removal : ',hapus_stopword(cobastopword))

    elif halaman == "Simulasi Word2vec":
        st.markdown(
        """
        <h2>Feature Extraction - Word2vec</h1>
        """, unsafe_allow_html=True)
        st.text("Pada bagian ini akan dilakukan data preprocessing dari dataset yang anda berikan. Data preprocessing \nadalah suatu langkah yang dilakukan untuk membuat data mentah menjadi data yang siap digunakan \nuntuk proses selanjutnya. Preprocessing yang dilakukan pada aplikasi ini adalah mengambil \ndata yang memiliki nilai lebih dari nol.")

        id_w2v = Word2Vec.load("hasilword2vec.w2v")

        dataTweet = pd.read_csv("word2vec.csv")
        st.write(dataTweet)

        st.markdown(
        """
        <h2>Simulasi Kemiripan antar kata dengan Word2vec</h1>
        """, unsafe_allow_html=True)

        word2vec = st.text_input('Masukan satu kata untuk mencari similarity',key=5)
        if st.button("Submit",key=0):
            st.text("Vector dari kata " + word2vec + " adalah :")
            st.write(id_w2v[word2vec])
            # st.text("lima Kata yang paling mirip dengan : ",word2vec)
            st.write(id_w2v.wv.most_similar(word2vec, topn = 5))

        st.text("Kemiripan antar dua kata")    
        kata1 = st.text_input('kata1 : ',key=6)
        kata2 = st.text_input('kata2 : ',key=7)
        if st.button("Submit",key=1):
            
            st.write("Kata : " + kata1 + " dengan kata " + kata2 + " memiliki tingat kemiripan : " , id_w2v.wv.similarity(kata1,kata2))
    
    elif halaman == "Prediksi Kalimat":
        st.header("This is Prediksi Kalimat.")
        st.write("Please select a page on the right.")

        model = load_model('Malam29AprilSatu7.h5',compile=False)
        with open('malam29april.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        def load_images(image_name):
            img = Image.open(image_name)
            return st.image(img,width=300)
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

    elif halaman == "Hasil Training":
        st.header("This is Hasil Training")
        st.write("Please select a page on the right.")

if __name__ == "__main__":
    main()