#   -- Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama --  #

#   İş Problemi: Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak level based (seviye tabanlı) persona (yeni müşteri tanımları) oluşturmak ve bu yeni müşteri tanımlarına göre
#                segmentler oluşturup bu segmentlere göre yen gelebilecek müşterilerin şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

#   Örneğin: Türkiye'den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.

#   Veri Seti Hikayesi: Persona.csv veri seti uluslararası bir oyun şirketinni sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı demografik bilgilerini barındırmaktadır.
#                       Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo tekilleştirilmemiştir. 
#                       Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.


            # __Değişkenler__ #
# Price   - Müşterinin harcama tutarı
# Source  - Müşterinin bağlandığı cihaz türü
# Sex     - Müşterinin cinsiyet
# Country - Müşterinin ülkesi
# Age     - Müşterinin yaşı


            #__Uygulama Öncesi Veri Seti__#
        
#   Price    Source      Sex      Country     Age
#    39      android     male     bra         17
#    39      andorid     male     bra         17
#    49      android     male     bra         17
#    29      android     male     tur         17
#    49      android     male     tur         17

                #__Hedeflenen Çıktı__#
                
#   CUSTOMERS_LEVEL_BASED        PRICE        SEGMENT      
#   BRA_ANDROID_FEMALE_0_18      35.6453        B
#   BRA_ANDROID_FEMALE_19_23     34.0773        C
#   BRA_ANDROID_FEMALE_24_30     33.8639        C
#   BRA_ANDROID_FEMALE_31_40     34.8983        B
#   BRA_ANDROID_FEMALE_41_66     36.7371        A




# Görev 1: Aşağıdaki soruları yanıtlayınız.
import pandas as pd

# Soru 1: Persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("C:/Users/emir/OneDrive/Masaüstü/Miuul ML Bootcamp/2- Data Analysis With Python/project/persona.csv")
df.head()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir? 
df["SOURCE"].unique()

# Soru 3: Kaç unique "PRICE" vardır?
df["PRICE"].unique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()

# Soru 7: SOURCE türlerine göre satışsayıları nedir?
df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean()

# Soru 9: SOURCE'laragöre PRICE ortalamaları nedir?
df.groupby("SOURCE")["PRICE"].mean()

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"])["PRICE"].mean()




# Görev 2: COUNTRY, SOURCE, SEX, AGE kırılımındaortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"])["PRICE"].mean()




# Görev 3: Çıktıyı PRICE'a göre sıralayınız.
agg_df = df.groupby (["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df = agg_df.sort_values("PRICE", ascending=False)
print(f"\nPRICE’a göre sıralama:\n{agg_df.head()}")




# Görev 4: Indekste yer alan isimleri değişken ismine çeviriniz.
agg_df = agg_df.reset_index()
print(f"\nIndekste yer alan isimleri değişken ismine çevirme:\n\n{agg_df.head()}")




# Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
def age_to_cat(x):
    if 0 <= x <= 18:
        return "0_18"
    elif 19 <= x <= 23:
        return "19_23"
    elif 24 <= x <= 30:
        return "24_30"
    elif 31 <= x <= 40:
        return "31_40"
    elif 41 <= x <= 70:
        return "41_70"
    else:
        return "70+"
agg_df["AGE_CAT"] = agg_df["AGE"].apply(lambda x: age_to_cat(x))
print(f"\nAge değişkenini kategorik değişkene çevirme:\n\n{agg_df.head()}")




# Görev 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız.
agg_df["customers_level_based"] = [country.upper() + "_" + source.upper() + "_" +
        sex.upper() + "_" + age_cat.upper() for country, source, sex, age_cat
        in zip(agg_df["COUNTRY"], agg_df["SOURCE"], agg_df["SEX"], agg_df["AGE_CAT"])]

agg_df = agg_df.groupby(["customers_level_based"]).agg({"PRICE": "mean"})
agg_df = agg_df.reset_index()
agg_df





# Görev 7: Yeni müşterileri (personaları) segmentlere ayırınız
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], q=4, labels=["D", "C", "B", "A"])

def analysisSegments(df):
    segment_list = list(df["SEGMENT"].unique())
    for segment in segment_list:
        print("Segment: ", segment,
              "\n", df[df["SEGMENT"] == segment].agg({"PRICE": ["mean", "max", "sum"]}), end="\n\n")
        
analysisSegments(agg_df)




# Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.
TUR_ANDROID_FEMALE_31_40 = agg_df[agg_df['customers_level_based'] == 'TUR_ANDROID_FEMALE_31_40']
FRA_IOS_FEMALE_31_40 = agg_df[agg_df['customers_level_based'] == 'FRA_IOS_FEMALE_31_40']
print("31-40 yaş arası Türk kadını Android: \n", "Ortalama Kazanç: ", TUR_ANDROID_FEMALE_31_40["PRICE"].mean().__round__(2), "Segment: ", TUR_ANDROID_FEMALE_31_40["SEGMENT"].unique())
print("31-40 yaş arası Fransız kadını iOS: \n", "Ortalama Kazanç: ", FRA_IOS_FEMALE_31_40["PRICE"].mean().__round__(2), "Segment: ", FRA_IOS_FEMALE_31_40["SEGMENT"].unique())



