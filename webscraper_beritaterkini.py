### WEB SCRAPING COVID.GO.ID ###

from urllib.request import urlopen, Request
import pandas as pd
from bs4 import BeautifulSoup as bs4
import time

# record start time
start = time.time()

judul = []
# isi = []
for pc in range(540):
    # CONNECT
    url = 'https://covid19.go.id/id/p/berita?page={}&search='.format(pc)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:107.0) Gecko/20100101 Firefox/107.0"}
    html = urlopen(Request(url,headers=headers))
    sop = bs4(html,'html.parser')
    
    # FIND DATA
    data = sop.find_all('h5')
    
    # GATHER LINKS
    links = []
    for i in data:
        links.append(i.a['href'])
    
    # ACCESS EACH LINK AND EXTRACT DATA
    for l in links:
        ua = Request(l,headers=headers)
        page = urlopen(ua)
        sopp = bs4(page,'html.parser')
        judul.append(sopp.find('div', {'class' : 'post-title'}).text)
        # isie = sopp.find_all('div', {'id' : 'konten-artikel'})
        # for i in isie:
        #     isipart = i.text.strip()
        # isi.append(isipart)
    
    print("PAGE :",pc+1," [DONE]")
    
berita = pd.DataFrame()
berita['Judul'] = judul
# berita['Isi'] = isi

print(len(data))
print(len(links))
print(len(judul))
# print(len(isi))

berita.to_excel("beritacovid_jun.xlsx")

### WAKTU SELESAI PROGRAM
end = time.time()
print("Durasi waktu eksekusi program :",round(end-start,2), "seconds")
print("Durasi waktu eksekusi program :",round((end-start)/60,2), "minutes")
tm1 = time.ctime()
print("Waktu program selesai :",tm1,"\n")