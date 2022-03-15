# Preparing Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from data_cleaning import DataCleaning
import pickle
import sys

text = sys.argv[1]
print(text)

with open('../model/DataCleaner.pkl', 'rb') as f:
	data_cleaner = pickle.load(f)
    
with open("../model/CountVectorizer.pkl", 'rb') as f:
	tokenizer = pickle.load(f)
    
with open("../model/GaussianNB.pkl", 'rb') as f:
	model = pickle.load(f)
 
def predictText(text:str) -> int:
	clean_text = data_cleaner.CleanOneText(text)
	tokenized = tokenizer.transform([clean_text]).toarray()
	return model.predict(tokenized)[0]

print(predictText(text))