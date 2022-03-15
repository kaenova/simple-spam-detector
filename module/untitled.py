# imported from iFest 2021 Data Cleaning Module by Yaudahlah Teams

import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class DataCleaning:
  # Initialization
  factory     = StemmerFactory()
  stemmer     = factory.create_stemmer()
  kamus_alay1 = pd.read_csv('https://raw.githubusercontent.com/fendiirfan/Kamus-Alay/main/Kamu-Alay.csv')
  kamus_alay1 = kamus_alay1.set_index('kataAlay')
  kamus_alay2 = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')
  kamus_alay2 = kamus_alay2.filter(['slang', 'formal'], axis=1)
  kamus_alay2 = kamus_alay2.drop_duplicates(subset=['slang'], keep='first')
  kamus_alay2 = kamus_alay2.set_index('slang')
  stopword1   = list(pd.read_csv('https://raw.githubusercontent.com/datascienceid/stopwords-bahasa-indonesia/master/stopwords_id_satya.txt', header = None)[0])
  custom_word = []

    
  def CleanDataFrame(cls, df, col_name, label_name, jum_minimum=None, minimum_kata=0, label_mapping=None, dropna=False):

    final_list_clean = []
    final_list_kotor = []
    final_label = []

    if jum_minimum == None: jum_minimum = len(df)
    if len(df) < jum_minimum: raise "Jumlah Data Yang Diinginkan melebihi Data yang Ada"
    i = 0
    current = 0
    
    while i < len(df):
      current_kalimat = df.loc[i][col_name]
      current_label = df.loc[i][label_name]

      clean_kalimat = cls.__cleanSentence__(current_kalimat)
      if type(clean_kalimat) != str or clean_kalimat == None or clean_kalimat == "":
        print("Ditemukan string kosong setelah di preprocessed pada kalimat: ", current_kalimat)
        clean_kalimat = "NaN"
      if (len(clean_kalimat.split(' ')) >= minimum_kata):
        final_list_clean.append(str(clean_kalimat))
        final_list_kotor.append(str(current_kalimat))
        if label_mapping != None:
          final_label.append(label_mapping[current_label])
        else:
          final_label.append(current_label)
        current += 1

        if current % 10 == 0:
          print("Memproses {} data".format(current))

      if current == jum_minimum:
        break

      i += 1
    
    data = {
        'raw': final_list_kotor,
        'processed': final_list_clean,
        'label': final_label
    }

    final_df = pd.DataFrame(data)
    if dropna:
      print("NaN Dropped")
      final_df = final_df.dropna(how='any')

    final_df['processed'] = final_df['processed'].astype(str)
    final_df['raw'] = final_df['raw'].astype(str)
    return final_df

  @classmethod
  def __cleanSentence__(cls, text):
    '''
    Melakukan prapemrosesan pada suatu kalimat dengan menghilangkan formatting pada kalimat,
    menghilangkan stopword pada kalimat, mengganti kata alay yang sudah terdefinisikan, serta
    melakukan stemming kalimat tersebut.
    '''

    # #
    # Cleaning Formatted Text Link And Tag (@username) using Regex
    # #
    text = re.sub(r'http\S+', '', text)
    text = re.sub('(@\w+|#\w+)','',text)

    #will replace the html characters with " "
    text = re.sub('<.*?>', '', text)  

    temp_text = list(text)
    for i in range(len(temp_text)):
      if temp_text[i] in string.punctuation:
        temp_text[i] = " "

    text = ''.join(temp_text)

    #will consider only alphabets
    text = re.sub('[^a-zA-Z]',' ',text) 
    #will replace newline with space
    text = re.sub("\n"," ",text)
    #will convert to lower case
    text = text.lower()
    # will replace a word
    text = re.sub("(username|user|url|rt|xf|fx|xe|xa)\s|\s(user|url|rt|xf|fx|xe|xa)","",text)
    # will repalce repated char
    text = re.sub(r'(\w)(\1{2,})', r"\1", text)
    # will replace single word
    text = re.sub(r"\b[a-zA-Z]\b","",text)
    # will replace space more than one
    text = re.sub('(s{2,})',' ',text)
    # will join the words
    text=' '.join(text.split())

    text_split = text.split(' ')
    # #
    # Mengganti kata-kata yang tidak baku
    # aku gapakai try catch lagi, lebih simple malah ini
    # #
    for i in range(len(text_split)):
      if text_split[i] in cls.kamus_alay1.index:
        text_split[i] = cls.kamus_alay1.loc[text_split[i]]['kataBaik']
      elif text_split[i] in cls.kamus_alay2.index:
        text_split[i] = cls.kamus_alay2.loc[text_split[i]]['formal']

    # #
    # Stemming
    # #
    stemmed_text = cls.stemmer.stem(text)

    # #
    # Removing Stopwords and custom word
    # #
    temp_text_split = []
    for i in range(len(text_split)):
      if (type(text_split[i]) == str):
        temp_text_split.append(text_split[i])

    # Memastikan saja
    if len(temp_text_split) == 0:
      return ""
    else:
      final_text = ' '.join(temp_text_split)
    
    return final_text