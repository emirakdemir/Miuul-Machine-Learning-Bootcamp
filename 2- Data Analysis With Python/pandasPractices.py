                            #   -- Pandas Alıştırmaları --  #


import seaborn as sns
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
titanic_df = sns.load_dataset("titanic")


# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
male_count = titanic_df['sex'].value_counts()['male']
male_count
female_count = titanic_df['sex'].value_counts()['female']
female_count


# Görev 3: Her bir sütuna ait unique değerlerin sayısını bulunuz.
titanic_df.nunique()


# Görev 4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
titanic_df['pclass'].nunique()


# Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
titanic_df[['pclass', 'parch']].nunique()


# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
titanic_df['embarked'].dtype


# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.
titanic_df[titanic_df['embarked'] == 'C']


# Görev 8: embarked değeri 5 olmayanların tüm bilgilerini gösteriniz.
titanic_df[titanic_df['embarked'] != 'S']


# Görev 9: Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
titanic_df.query('(age < 30) & (sex == "female")')


# Görev 10: fare'i 500'den büyük veya yaşı 70'den büyük yolcuların bilgilerini gösteriniz.
titanic_df.query('(fare > 500) & (age > 70)')


# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
titanic_df.isnull().sum()


# Görev 12: who değişkenini dataFrame'den çıkarınız.
titanic_df.drop(columns='who', inplace=True)


# Görev 13: deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri(mod) ile doldurunuz.
deck_mode = titanic_df['deck'].mode()[0]
titanic_df['deck'].fillna(deck_mode, inplace=True)


# Görev 14: age değişkenindeki boş değerleri age değişkeninin medyanı ile doldurunuz.
age_median = titanic_df['age'].median()
titanic_df['age'].fillna(age_median, inplace=True)


# Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımında sum, count, mean değerlerini bulunuz.
sub_df = titanic_df[['survived', 'pclass', 'sex']]
sub_df.groupby(['pclass', 'sex']).survived.sum()
sub_df.groupby(['pclass', 'sex']).survived.count()
sub_df.groupby(['pclass', 'sex']).survived.mean()


# Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazın. Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız.)
age_flag_series = titanic_df['age'].apply(lambda x: 1 if (x < 30) else 0)
titanic_df['age_flag'] = age_flag_series


# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
tips_df = sns.load_dataset("tips")


# Görev 18: Time değişkeninin kategorilerine göre total_bill değerinin sum, min, max, mean değerlerini bulunuz.
tips_df.groupby('time')['total_bill'].sum()
tips_df.groupby('time')['total_bill'].min()
tips_df.groupby('time')['total_bill'].max()
tips_df.groupby('time')['total_bill'].mean()


# Görev 19: Day ve time'a göre total_bill değerlerinin sum, min, max ve mean değerlerini bulunuz.
tips_df.groupby(['day', 'time']).total_bill.sum()
tips_df.groupby(['day', 'time']).total_bill.min()
tips_df.groupby(['day', 'time']).total_bill.max()
tips_df.groupby(['day', 'time']).total_bill.mean()


# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre sum, min, max, mean değerlerini bulunuz.
lunch_female_df = tips_df.query('time == "Lunch" & sex == "Female"')
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').sum()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').min()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').max()
lunch_female_df[['day', 'total_bill', 'tip']].groupby('day').mean()


# Görev 21: size'ı 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
tips_df.query('size < 3 & total_bill > 10').loc[:, 'total_bill'].mean()


# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip'in toplamını versin.
tips_df["total_bill_tip_sum"] = tips_df['total_bill'] + tips_df['tip']


# Görev 23: Total_billdeğişkenininkadınveerkekiçinayrıayrıortalamasınıbulunuz. Bulduğunuzortalamalarınaltındaolanlara0, üstündeveeşitolanlara1 verildiğiyeni birtotal_bill_flagdeğişkenioluşturunuz.KadınlariçinFemale olanlarınınortalamaları, erkekleriçiniseMale olanlarınortalamalarıdikkatealınacktır. Parametreolarakcinsiyetvetotal_billalanbirfonksiyonyazarakbaşlayınız. (If-else koşullarıiçerecek)
mean_by_sex = tips_df.groupby('sex')['total_bill'].mean()
male_mean = mean_by_sex[0]
female_mean = mean_by_sex[1]
tips_df['total_bill_flag'] = np.where((((tips_df['sex'] == 'Male') & (tips_df['total_bill'] > male_mean)) | ((tips_df['sex'] == 'Female') & (tips_df['total_bill'] > female_mean))), 1, 0)


# Görev 24: total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyiniz. 
tips_df.groupby(['sex', 'total_bill_flag'])['total_bill'].count()


# Görev 25: Veriyi total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataFrame'e atayınız.
first_30_df = tips_df.sort_values('total_bill_tip_sum', ascending=False).head(30)
