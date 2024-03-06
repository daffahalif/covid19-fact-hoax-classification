### IMPORT PAKET ###

import pandas as pd # For I/O, Data Transformation
from sklearn.model_selection import train_test_split

import time # digunakan untuk menghitung waktu compile
tm = time.ctime()
print("Waktu program dimulai :",tm)
start = time.time()

## Import data ke dataframe
data1 = pd.read_excel("beritacovid_jun.xlsx") # Dataframe untuk Fake News
data2 = pd.read_excel("beritacovidhoax_jun.xlsx")
data1 = data1.iloc[:,1:2]
data2 = data2.iloc[:,1:2]

data = pd.concat([data1,data2])
data = data.reset_index()
data = data.iloc[:,1:2]

# Data Cleaning, menghapus data yang tidak relevan dan duplikasi
def hapuss(data):
    if      data.find("(Update per") == -1 \
        and data.find("(Update Per") == -1 \
        and data.find("Update Vaksinasi") == -1 \
        and data.find("Update COVID-19 di") == -1 \
        and data.find("Update Penambahan Kasus COVID-19") == -1 \
        and data.find("Penambahan Kasus COVID-19") == -1 \
        and data.find("Penambahan Kasus Positif") == -1 \
        and data.find("Awas H") == -1 \
        and data.find("AWAS H") == -1 \
        and data.find("Awas Disinformasi") == -1 \
        and data.find("Grafik Kasus Aktif,") == -1 \
        and data.find("Infografis COVID-19") == -1 \
        and data.find("Kasus Positif COVID-19 Bertambah") == -1 \
        and data.find("Kasus Positif COVID-19 Baru") == -1 \
        and data.find("Kasus Positif COVID-19 Capai") == -1 \
        and data.find("Kasus Positif COVID-19 Jadi") == -1 \
        and data.find("Kasus Positif COVID-19 Naik") == -1 \
        and data.find("Kasus Positif COVID-19 Jadi") == -1 \
        and data.find("Kasus Positif COVID-19 Sentuh") == -1 \
        and data.find("Kasus Positif COVID-19 Me") == -1 \
        and data.find("Kasus Sembuh COVID-19") == -1 \
        and data.find("Kasus Terkonfirmasi Positif COVID-19") == -1 \
        and data.find("Keputusan Menteri") == -1 \
        and data.find("Keputusan Direktur") == -1 \
        and data.find("Kesembuhan COVID-19") == -1 \
        and data.find("Kesembuhan dari") == -1 \
        and data.find("Kesembuhan Dari") == -1 \
        and data.find("Kesembuhan Harian") == -1 \
        and data.find("Kesembuhan Kumulatif") == -1 \
        and data.find("Kesembuhan M") == -1 \
        and data.find("Kesembuhan Pesat") == -1 \
        and data.find("Kesembuhan Pasien COVID-19") == -1 \
        and data.find("Kesembuhan Total") == -1 \
        and data.find("Pasien Sembuh ") == -1 \
        and data.find("[TOP 5]") == -1 \
        and data.find("[Top 5]") == -1 \
        and data.find("Tahap Ke") == -1 \
        and data.find("Tahap ke") == -1 \
        and data.find("Total P") == -1 \
        and data.find("Total Kesembuhan ") == -1 \
        and data.find("Tips") == -1 \
        and data.find("Bagaimana") == -1 \
        and data.find("Apakah") == -1 \
        and data.find("Kenapa") == -1 \
        and data.find("[DOKUMENTASI]") == -1 \
        and data.find("[FALSE]") == -1 : # data berbahasa inggris tidak dapat dijadikan input
        data = data
    else:
        data = None
    return data
data = data['Judul'].apply(lambda x : hapuss(x))
print('\nTotal Data NA :',data.isna().sum())
data = data.dropna()
print('\nTotal Data Duplikasi :',data.duplicated().sum())
data = data.drop_duplicates()
data = data.reset_index() # karena proses selanjutnya memanfaatkan urutan index, maka harus di reset
data = data.iloc[:,1:3] # kolom index lama dihapus, sehingga tersisa kolom judul saja
print("\nCleaning Done :",round(time.time()-start,2),"seconds")

# MENGATUR RANGE DATA (HANYA UNTUK TES KODE, KARENA DATA BANYAK MEMAKAN WAKTU LAMA)
data = data.iloc[0:50,:]
data = data.reset_index()
data = data.iloc[:,1:3]

def labell(data):
    Label = 0
    if  data.find("[SALAH]") == -1 and \
        data.find("Misinformasi:") == -1 and \
        data.find("MITOS:") == -1 and \
        data.find("Hoaks:") == -1 and \
        data.find("HOAKS:") == -1 and \
        data.find("HOAKS ➡️") == -1:
        Label = 0 # ASLI
    else:
        Label = 1 # HOAKS
    return Label

def hl(judul): # Hapus label pada judul
    if  judul.find("[SALAH]") == -1 and \
        judul.find("Misinformasi:") == -1 and \
        judul.find("MITOS:") == -1 and \
        judul.find("Hoaks:") == -1 and \
        judul.find("HOAKS:") == -1 and \
        judul.find("HOAKS ➡️") == -1:
        judul = judul.replace("[BERITA]","")
        judul = judul.replace("[Update]","")
        judul = judul.replace("UPDATE!","")
    else:
        mitos = ["MITOS:","Faktanya","faktanya",
                 "[SALAH]","Hoaks:","HOAKS ➡️ ",
                 "HOAKS","Misinformasi"]
        for char in mitos:
            judul = judul.replace(char, "")
    return judul

data['Label'] = data['Judul'].apply(lambda x : labell(x))
data['Judul'] = data['Judul'].apply(lambda x : hl(x))
print("Labelling Done :",round(time.time()-start,2),"seconds")

data = data.sort_values(by='Label')
data = data.reset_index()
data = data.iloc[:,1:]
print(data.head(10),"\n")
print(data.groupby('Label').count(),"\n")

### PREPROCESSING ###

### CASE FOLDING ###
def case_folding(data):
    import re
    # Lowertext (tidak kapital)
    data = data.lower()
    # Menghilangkan Angka 0-9, Tanda Baca, Karakter Spesial, dan Emoji
    data = re.sub(r'[^A-Za-z]', ' ', data)
    # Menghilangkan Whitespace
    data = data.strip()
    return data
data['case_folding'] = data['Judul'].apply(lambda x : case_folding(x))
print("Case Folding Done :",round(time.time()-start,2),"seconds")

### TOKENIZING ###
def tokenize(teks):
    import nltk
    teks = nltk.tokenize.word_tokenize(teks)
    return teks
data['tokenizing'] = data['case_folding'].apply(lambda x : tokenize(x))
print("Tokenizing Done :",round(time.time()-start,2),"seconds")

### REMOVE STOPWORD ###
def rmstopwords(teks):
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    factory = StopWordRemoverFactory()
    swsastra = factory.get_stop_words()
    nostopwords = []
    for word in teks:
        if (word not in swsastra):
            nostopwords.append(word)
    return nostopwords
data['stopwords_removed'] = data['tokenizing'].apply(lambda x : rmstopwords(x))
print("Stopword Removed :",round(time.time()-start,2),"seconds")

### STEMMING ###
def stemming(teks):
    cleantext = []
    for word in teks:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factorystem = StemmerFactory()
        stemmer = factorystem.create_stemmer()
        stemmed_word = stemmer.stem(word)
        cleantext.append(stemmed_word)
    return cleantext
data['stemming'] = data['stopwords_removed'].apply(lambda x : stemming(x))
print("Stemming Done :",round((time.time()-start)/60,2),"mins")

### MENGGABUNGKAN TOKEN ###
def join(teks):
    teks = " ".join([char for char in teks])
    return teks
data['Text_Preprocessed'] = data['stemming'].apply(lambda x : join(x))

### MEMISAH INPUT DAN OUTPUT ###
X = data.iloc[:,6] # TEKS HASIL PREPROCESSING / INPUT
y = data.iloc[:,1] # KELAS / OUTPUT

### SELEKSI FITUR : TF-IDF ###
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer() # deklarasi
tfidf_model_fit = tfidf_model.fit(X) # menghitung tfidf berdasarkan X
X_vect = pd.DataFrame(tfidf_model_fit.transform(X).toarray())

tfidf_model_fit_col = pd.DataFrame(X_vect.columns) # mengambil indeks kata
X_vect.columns = tfidf_model_fit.get_feature_names()
tfidf_model_fit_col['Word'] = X_vect.columns # tabel indeks kata + kata

print("TF-IDF Done :",round((time.time()-start)/60,2),"mins\n")

### TRAINING MODEL (CLASSIFIER) ###

## Import
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
import numpy as np

### CONFUSION MATRIX ###
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt # For Plotting
import seaborn as sns # For Plotting

def ttsplit(X_vect,y):
    if tt == 0:
        X_train, X_test, y_train, y_test = train_test_split(X_vect,y,
                                    test_size=0.4, stratify=y) # 60:40
        print('\n\n###### TRAINING 60 : 40 TESTING ######') 
        ttsp = '60.40'
    elif tt == 1:
        X_train, X_test, y_train, y_test = train_test_split(X_vect,y,
                                    test_size=0.3, stratify=y) # 70:30
        print('\n\n###### TRAINING 70 : 30 TESTING ######') 
        ttsp = '70.30'
    elif tt == 2:
        X_train, X_test, y_train, y_test = train_test_split(X_vect,y,
                                    test_size=0.2, stratify=y) # 80:20
        print('\n\n###### TRAINING 80 : 20 TESTING ######')
        ttsp = '80.20'
    return X_train, X_test, y_train, y_test, ttsp

### RANDOM FOREST ###
rfhasil = []
rf=RandomForestClassifier(n_estimators=100) # Inisiasi Random Forest
for tt in range(3):
    mat = np.zeros((4,10))
    for i in range(10):
        ### MEMBAGI TRAINING TESTING
        X_train, X_test, y_train, y_test, ttsp = ttsplit(X_vect,y)
        print('\n#### RANDOM FOREST : RUN KE',i+1,'####')
        
        rf.fit(X_train,y_train) # Fit Random Forest
        
        ## Testing Model
        y_pred1=rf.predict(X_test) # Predict Random Forest
        
        print('Precision : %.2f' % (precision_score(y_test, y_pred1)*100))
        print('Recall    : %.2f' % (recall_score(y_test, y_pred1)*100))
        print('F1-Score  : %.2f' % (f1_score(y_test, y_pred1)*100))
        print('Accuracy  : %.2f' % (accuracy_score(y_test, y_pred1)*100))
        
        mat[0,i] = precision_score(y_test, y_pred1)*100
        mat[1,i] = recall_score(y_test, y_pred1)*100
        mat[2,i] = f1_score(y_test, y_pred1)*100
        mat[3,i] = accuracy_score(y_test, y_pred1)*100
        
        # confusion matrix rf
        cm1 = confusion_matrix(y_test, y_pred1)
        fig, ax= plt.subplots(figsize=(10,5))
        sns.heatmap(cm1, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels') # labels, title and ticks
        ax.set_ylabel('True labels')
        ax.set_title('CM - RF | Rasio={} | Iterasi ke={}'.format(ttsp, i+1))
        ax.xaxis.set_ticklabels(['ASLI','HOAKS'])
        ax.yaxis.set_ticklabels(['ASLI','HOAKS'])
        # fig.figure.savefig("hasil/confusion-matrix/2-rf-{}-{}.png".format(ttsp, i+1))
    
    rff = pd.DataFrame(mat, index=['Presisi','Recall','F1-Score','Akurasi'],
                       columns=[1,2,3,4,5,6,7,8,9,10])
    if tt == 0:
        rf1 = rff
    elif tt == 1:
        rf2 = rff
    elif tt == 2:
        rf3 = rff
rfhasil = rf1.append([rf2,rf3])
rfhasil = rfhasil.reset_index()
rfhasil = rfhasil.reset_index()

### NAIVE BAYES ###
nbhasil = []
gnb=GaussianNB() # Inisiasi Gaussian Naive Bayes
for tt in range(3):
    mat = np.zeros((4,10))
    for i in range(10):
        ### MEMBAGI TRAINING TESTING
        X_train, X_test, y_train, y_test, ttsp = ttsplit(X_vect,y)
        print('\n#### NAIVE BAYES : RUN KE',i+1,'####')
                
        gnb.fit(X_train,y_train) # Fit GaussianNB
        
        ## Testing Model
        y_pred2=gnb.predict(X_test) # Predict Gaussian Naive Bayes
        
        print('Precision : %.2f' % (precision_score(y_test, y_pred2)*100))
        print('Recall    : %.2f' % (recall_score(y_test, y_pred2)*100))
        print('F1-Score  : %.2f' % (f1_score(y_test, y_pred2)*100))
        print('Accuracy  : %.2f' % (accuracy_score(y_test, y_pred2)*100))
        
        mat[0,i] = precision_score(y_test, y_pred2)*100
        mat[1,i] = recall_score(y_test, y_pred2)*100
        mat[2,i] = f1_score(y_test, y_pred2)*100
        mat[3,i] = accuracy_score(y_test, y_pred2)*100
        
        # confusion matrix nb
        cm2 = confusion_matrix(y_test, y_pred2)
        fig, ax= plt.subplots(figsize=(10,5))
        sns.heatmap(cm2, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels') # labels, title and ticks
        ax.set_ylabel('True labels')
        ax.set_title('CM - NB | Rasio={} | Iterasi ke={}'.format(ttsp, i+1))
        ax.xaxis.set_ticklabels(['ASLI','HOAKS'])
        ax.yaxis.set_ticklabels(['ASLI','HOAKS'])
        # fig.figure.savefig("hasil/confusion-matrix/2-nb-{}-{}.png".format(ttsp, i+1))
    
    nbf = pd.DataFrame(mat, index=['Presisi','Recall','F1-Score','Akurasi'], 
                        columns=[1,2,3,4,5,6,7,8,9,10])
    if tt == 0:
        nb1 = nbf
    elif tt == 1:
        nb2 = nbf
    elif tt == 2:
        nb3 = nbf
nbhasil = nb1.append([nb2,nb3])
nbhasil = nbhasil.reset_index()
nbhasil = nbhasil.reset_index()

### SUPPORT VECTOR MACHINE ###
svmhasil = []
svml=svm.SVC(kernel='linear') # Inisiasi Support Vector Machine
for tt in range(3):
    mat = np.zeros((4,10))
    for i in range(10):
        ### MEMBAGI TRAINING TESTING
        X_train, X_test, y_train, y_test, ttsp = ttsplit(X_vect,y)
        print('\n#### SUPPORT VECTOR MACHINE : RUN KE',i+1,'####')

        svml.fit(X_train, y_train) # Fit SVM Kernel=Linear
        
        ## Testing Model
        y_pred3=svml.predict(X_test) # Predict Support Vector Machine
        
        print('Precision : %.2f' % (precision_score(y_test, y_pred3)*100))
        print('Recall    : %.2f' % (recall_score(y_test, y_pred3)*100))
        print('F1-Score  : %.2f' % (f1_score(y_test, y_pred3)*100))
        print('Accuracy  : %.2f' % (accuracy_score(y_test, y_pred3)*100))
        
        mat[0,i] = precision_score(y_test, y_pred3)*100
        mat[1,i] = recall_score(y_test, y_pred3)*100
        mat[2,i] = f1_score(y_test, y_pred3)*100
        mat[3,i] = accuracy_score(y_test, y_pred3)*100
        
        # confusion matrix svm
        cm3 = confusion_matrix(y_test, y_pred3)
        fig, ax= plt.subplots(figsize=(10,5))
        sns.heatmap(cm3, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels') # labels, title and ticks
        ax.set_ylabel('True labels')
        ax.set_title('CM - SVM | Rasio={} | Iterasi ke={}'.format(ttsp, i+1))
        ax.xaxis.set_ticklabels(['ASLI','HOAKS'])
        ax.yaxis.set_ticklabels(['ASLI','HOAKS'])
        # fig.figure.savefig("hasil/confusion-matrix/2-svm-{}-{}.png".format(ttsp, i+1))
    
    svmf = pd.DataFrame(mat, index=['Presisi','Recall','F1-Score','Akurasi'], 
                        columns=[1,2,3,4,5,6,7,8,9,10])
    if tt == 0:
        svm1 = svmf
    elif tt == 1:
        svm2 = svmf
    elif tt == 2:
        svm3 = svmf
svmhasil = svm1.append([svm2,svm3])
svmhasil = svmhasil.reset_index()
svmhasil = svmhasil.reset_index()

finalhasil = rfhasil.append([nbhasil, svmhasil])
finalhasil.rename(columns = {'level_0':'rasio','index':'evaluasi'}, inplace = True)
finalhasil = finalhasil.reset_index()
finalhasil.rename(columns = {'index':'algoritma'}, inplace = True)

finalhasil['rasio'] = finalhasil['rasio'].replace([0,1,2,3],'60:40')
finalhasil['rasio'] = finalhasil['rasio'].replace([4,5,6,7],'70:30')
finalhasil['rasio'] = finalhasil['rasio'].replace([8,9,10,11],'80:20')

finalhasil['algoritma'][0:12] = 'Random Forest'
finalhasil['algoritma'][12:24] = 'Naive Bayes'
finalhasil['algoritma'][24:36] = 'SVM'

# finalhasil.to_excel('finalhasil.xlsx')

# finalhasil2 = finalhasil.T

print("\nClassification Done :",round((time.time()-start)/60,2),"mins\n")

### WAKTU SELESAI PROGRAM
end = time.time()
print("Durasi waktu eksekusi program :",round(end-start,2), "seconds")
print("Durasi waktu eksekusi program :",round((end-start)/60,2), "minutes")
tm1 = time.ctime()
print("Waktu program selesai :",tm1,"\n")