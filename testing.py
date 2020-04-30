from vega_datasets import data
import streamlit as st
import altair as alt
import os
import pandas as pd
import re
import string
def main():
    
    halaman = st.sidebar.selectbox("MENU", ["Tentang", "Data Preprocessing","Simulasi Word2vec","Prediksi Kalimat","Hasil Training"])

    if halaman == "Tentang":
        st.markdown(
        """
        <link rel="stylesheet" href="https://mleibman.github.io/SlickGrid/slick.grid.css" type="text/css"/>
        <link rel="stylesheet" href="https://mleibman.github.io/SlickGrid/css/smoothness/jquery-ui-1.8.16.custom.css" type="text/css"/>
        <link rel="stylesheet" href="https://mleibman.github.io/SlickGrid/examples/examples.css" type="text/css"/>

        <script src="https://mleibman.github.io/SlickGrid/lib/jquery-1.7.min.js"></script>
        <script src="https://mleibman.github.io/SlickGrid/lib/jquery.event.drag-2.2.js"></script>
        <script src="https://mleibman.github.io/SlickGrid/slick.core.js"></script>
        <script src="https://mleibman.github.io/SlickGrid/slick.grid.js"></script>
        <script>
        var grid;
        var columns = [
            {id: "title", name: "Title", field: "title"},
            {id: "duration", name: "Duration", field: "duration"},
            {id: "%", name: "% Complete", field: "percentComplete"},
            {id: "start", name: "Start", field: "start"},
            {id: "finish", name: "Finish", field: "finish"},
            {id: "effort-driven", name: "Effort Driven", field: "effortDriven"}
        ];
        var options = {
            enableCellNavigation: true,
            enableColumnReorder: false
        };
        $(function () {
            var data = [];
            for (var i = 0; i < 500; i++) {
            data[i] = {
                title: "Task " + i,
                duration: "5 days",
                percentComplete: Math.round(Math.random() * 100),
                start: "01/01/2009",
                finish: "01/05/2009",
                effortDriven: (i % 5 == 0)
            };
            }
            grid = new Slick.Grid("#myGrid", data, columns, options);
        })
        </script>
        """, unsafe_allow_html=True)
        
    elif halaman == "Data Preprocessing":
        st.markdown(
        """
        <h1>Data Preprocessing</h1>
        """, unsafe_allow_html=True)
        st.text("Pada bagian ini akan dilakukan data preprocessing dari dataset yang anda berikan. Data preprocessing \nadalah suatu langkah yang dilakukan untuk membuat data mentah menjadi data yang siap digunakan \nuntuk proses selanjutnya. Preprocessing yang dilakukan pada aplikasi ini adalah mengambil \ndata yang memiliki nilai lebih dari nol.")
        
        def pilih_file(folder_path='./datamentah'):
            namafile = os.listdir(folder_path)
            fileterpilih = st.selectbox('Pilih Data set',namafile)
            return os.path.join(folder_path, fileterpilih)
        data = pilih_file()
        st.write('You selected `%s`' % data)

        dataTweet = pd.read_csv(data,usecols=["text","sentiment"], encoding = "latin-1")
        st.markdown(
        """
        <h1>Data Awal</h1>
        """, unsafe_allow_html=True)
        
        if st.checkbox('Data Awal | Menampilkan Semua Data',key=0):
            st.write(dataTweet) 
    

        ################################## CASE FOLDING #######################################    

        st.markdown(
        """
        <h1>Data Preprocessing | Case Folding</h1>
        """, unsafe_allow_html=True)
        st.text("Case Folding adalah merupakan bla bla bla")
        def casefolding(text):
            text = text.lower().strip()
            return text
        hasilcasefolding = []
        for text in dataTweet.text:
            pro = casefolding(text)
            hasilcasefolding.append(pro)
            
        dataTweet['text'] = hasilcasefolding
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
        <h1>Data Preprocessing |Noise Removal</h1>
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
        for desc in dataTweet.text:
            pro = clean(desc)
            hasilclean.append(pro)
  
        dataTweet['text'] = hasilclean
        dataTweet.to_csv('Hasilcleaning.csv')
        dataTweet.rename(columns={"text": "Hasil Cleaning"}).head(10)

        if st.checkbox('Hasil Preprocessing | Remove Noise ',key=2):
            st.write(dataTweet)
            st.text("Simulasi Data Preprocessing Case Folding")
            cobaremovenoise = st.text_input('Masukan kalimat',key=2)
            st.write('Hasil Noise Removal : ',clean(cobaremovenoise))
    






    elif halaman == "Simulasi Word2vec":
        st.header("This is your Simulasi Word2vec.")
        st.write("Please select a page on the right.")
    
    elif halaman == "Prediksi Kalimat":
        st.header("This is Prediksi Kalimat.")
        st.write("Please select a page on the right.")

    elif halaman == "Hasil Training":
        st.header("This is Hasil Training")
        st.write("Please select a page on the right.")




if __name__ == "__main__":
    main()