#--     Komut satırında Python kodu çalıştırma     --#

# Görev 1: PyCharm'da "hi.py" isminde python dosyası oluşturunuz.
# Görev 2: Bu dosyanın içerisine print kodunu kendinize göre düzenleyeniz ve kaydediniz.
# Görev 3: Konsoldan üzerinden "hi.py" dosyasının olduğu dizine gidiniz. PyCharm'da sol tarafta yer alan menüde hi.py dosyası hangi klasördeyse o klasöre sağ tuş ile tıklayıp şu seçimi yapınız: "open in > terminal".
#           PyCharm'ın alt tarafında terminal ekranı açılacak. Şu anda hi.py dosyası ile aynı dizindesini.
# Görev 4: Konsola hi.py yazarak, python kodunu çalıştırınız.







#--     Virtual Environment Oluşturma     --#

# Görev 1: Kendi isminizde bir Virtual Environment oluşturunuz, oluşturma esnasında Python3 kurulumu yapınız.
# Görev 2: Oluşturduğunuz environment'i aktif ediniz.
# Görev 3: Yüklü Paketleri listeleyeniz.
# Görev 4: Environment içerisinde NumPy'ın güncel versiyonunu ve Pandas'ın 1.2.1 versiyonunu aynı anda indiriniz.
# Görev 5: İndirilen NumPy versiyonu nedir?
# Görev 6: Pandas'ı upgrade ediniz. Yeni versiyon nedir?
# Görev 7: NumPy'ı environment'tan siliniz.
# Görev 8: Seaborn ve Matplotlib kütüphanesinin güncel versiyonlarını aynı anda indiriniz.
# Görev 9: Virtual Environment içindeki kütüphaneleri versiyon bilgisi ile beraber export ediniz ve yaml dosyasını inceleyiniz.
# Görev 10: Oluşturduğunuz environmenti siliniz.


# Çözüm 1: conda create -n name
# Çözüm 2: conda activate name
# Çözüm 3: conda list
# Çözüm 4: conda install numpy pandas=1.21
# Çözüm 5: conda list
# Çözüm 6: conda upgrade pandas
# Çözüm 7: conda remove numpy
# Çözüm 8: conda install seaborn matplotlib
# Çözüm 9: conda env export > environment.yaml
# Çözüm 10: conda env remove -n name











#--     Python Alıştırmaları     --#



# Görev 1: Veri tiplerini sorgulayınız.

# X= 8
# Y= 3.2
# Z= 8J+18
# S= "HELLO WORLD"
# B= True
# C= 23<22
# L= [1,2,3,4]
# D= {Name: "Jack", Age: 22}
# T= ("Machine learning" "Data science")
# S= {"Python","Veri"}

# Cevap 1: 
# X= İNTEGER
# Y= FLOAT
# Z= COMPLEX
# S= STRİNG
# B= BOOLEN
# C= FALSE
# L= LİST
# D= DİCTİONARY
# T= TUPLE
# S= SET



# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. virgül ve nokta yerine boşluk koyunuz.
text= "The goal is to turn data information, and information into insight."
# Cevap 2:
text.upper()
text.replace("," , " ")
text.replace("." , " ")
text.split()



# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.
list= ["D","A","T","A","S","C","İ","E","N","C","E",]
# Adım 1: Verilen listenin eleman sayısına bakınız.
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım 4: Sekizinci indeksteki elemanı siliniz.
# Adım 5: Yeni bir eleman ekleyiniz.
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

# Cevap3:
# Cevap_Adım1: 
len(list)
# Cevap_Adım2: 
list[0] , list[10]
# Cevap_Adım3: 
list2=list[:4]
# Cevap_Adım4: 
list.pop(8)
# Cevap_Adım5: 
list.append(25)
# Cevap_Adım6: 
list.insert(8,"N")



# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
dict= {"Christian" : ["America",18],
        "Daisy": ["England",12],
        "Antonio": ["Spain",22],
        "Dante": ["İtaly",25]}
# Adım1: Key değerlerine erişiniz.
# Adım2: Value'lara erişiniz.
# Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# Adım5: Antonio'yu dictionary'den siliniz.

# Cevap 4: 
# Cevap_Adım1: 
dict.keys()
# Cevap_Adım2: 
dict.values()
# Cevap_Adım3: 
dict["Daisy"] = ["England" , 13]
# Cevap_Adım4: 
dict["Ahmet"] = ["Turkey" , 24]
# Cevap_Adım5: 
dict.pop("Antonio")



# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.
# Cevap5 5:
l = [2, 13, 18, 93, 22]
even_list = []
odd_list = []

def func( a=[]):
    for i in l:
        if i % 2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return odd_list, even_list

odd_list, even_list = func(l)
print(odd_list, even_list)



# Görev 6: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
# Cevap 6:
import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns

num_col = ["NUM" + col.upper() for col in df.columns if df[col].dtype != "O"]



# Görev 7: List Comprehension yapısı kullanarak car_crashes verisinde isminde"no" barındırmayan değişkenlerin isimlerinin sonuna "FLAG" yazınız.
# Cevap 7:
import seaborn as sns

df=sns.load_dataset("car_crashes")
df.columns

new_col = [col.upper() + "FLAG" if "no" not in col else col.upper() for col in df.columns ]
df.columns = new_col



# Görev 8: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir data frame oluşturunuz.
# Cevap 8:
import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns
og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
