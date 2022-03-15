# imported from iFest 2021 Data Cleaning Module by Yaudahlah Teams,
# Refactored by Kaenova Mahendra Auditama (Yaudahlah Teams)

import pandas as pd
from tqdm import tqdm
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class DataCleaning:
  def __init__(self, stopword:list = [], slang_word:dict = {}) -> None:
    factory     = StemmerFactory()
    self.stemmer     = factory.create_stemmer()
    self.stopword = stopword
    self.slang_word = slang_word

  def AddKamusAlay(self, new_dict:dict = {}):
    if (type(new_dict) != dict): raise TypeError("Not a valid type")
    self.slang_word = self.slang_word | new_dict
  
  def AddStopWord(self, stopword:list = []):
    if (type(stopword) != list): raise TypeError("Not a valid type")
    self.custom_word = self.custom_word + stopword
    
  def CleanDataFrame(self, df:pd.DataFrame, text_cols:str, label_cols:str, 
                     word_min:int=0, label_mapping:dict=None, dropna:bool=False):
    """
    Using multiprocessing (*if available) to process data from pandas Dataframe.
    Will be outputing a new dataframe with a processed data.
    """
    print("Processing...")
    final_list_clean = []
    final_list_dirty = []
    final_label = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
      sentence = row[text_cols]
      label = row[label_cols]
      
      # Process label
      if label_mapping is not None:
        if label not in label_mapping:
          print(f"Label {label} is not matched any label_mapping you've defined. This label will be ignored")
          continue      
        clean_label = label_mapping[label]
      else:
        clean_label = label  
      
      # Process Text
      clean_sentence = self.__cleanText__(sentence, self.slang_word,
                                          self.stopword, self.stemmer)
      if (clean_sentence is None):
        print(f"Sentence '{sentence}' is empty after processing. This sentence will be ignored")
        continue
      if (len(clean_sentence.split()) < word_min):
        continue
      
      final_list_clean.append(clean_sentence)
      final_list_dirty.append(sentence)
      final_label.append(clean_label)
        
    # Creating pandas dataframe
    data = {
      'raw': final_list_dirty,
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

  def CleanOneText(self, text):
    return self.__cleanText__(text, self.slang_word, self.stopword, self.stemmer)

  def __cleanText__(self, text:str, slangword:dict, stopword:list, stemmer) -> str:
    '''
    Processing a text, deleting some web associated word, removing word from stopword list
    and change defined slang word.
    '''
    # HTML and text annotation removal
    text = re.sub(r'http\S+', '', text)
    text = re.sub('(@\w+|#\w+)','',text)
    text = re.sub('<.*?>', '', text)  
    temp_text = list(text)
    for i in range(len(temp_text)):
      if temp_text[i] in string.punctuation:
        temp_text[i] = " "
    text = ''.join(temp_text)
    text = re.sub('[^a-zA-Z]',' ',text) 
    text = re.sub("\n"," ",text)
    text = text.lower()
    text = re.sub("(username|user|url|rt|xf|fx|xe|xa)\s|\s(user|url|rt|xf|fx|xe|xa)","",text)
    text = re.sub(r'(\w)(\1{2,})', r"\1", text)
    text = re.sub(r"\b[a-zA-Z]\b","",text)
    text = re.sub('(s{2,})',' ',text)
    text=' '.join(text.split())
    text_split = text.split(' ')
    final_text_split = []
    for i in range(len(text_split)):
      if type(text_split[i]) != str:
        continue
      if str(text_split[i]) in stopword:
        continue
      if str(text_split[i]) in slangword:
        text_split[i] = str(slangword[text_split[i]])
      final_text_split.append(text_split[i])
    
    stemmed_text = stemmer.stem(" ".join(final_text_split))
    
    # just to make sure
    if len(stemmed_text) == 0:
      return None   
    
    return stemmed_text