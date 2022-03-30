# Preparing Pipeline
from email import message
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from data_cleaning import DataCleaning
import pickle
import sys
import uvicorn
from fastapi import FastAPI, Response, status
from pydantic import BaseModel
import os

with open('./DataCleaner.pkl', 'rb') as f:
  data_cleaner = pickle.load(f)

with open("./CountVectorizer.pkl", 'rb') as f:
  tokenizer = pickle.load(f)

with open("./GaussianNB.pkl", 'rb') as f:
  model = pickle.load(f)

def predictText(text:str) -> int:
  clean_text = data_cleaner.CleanOneText(text)
  print(clean_text)
  tokenized = tokenizer.transform([clean_text]).toarray()
  return model.predict(tokenized)[0]

app = FastAPI()

class RequestText(BaseModel):
  text: str

@app.get("/")
async def index():
  return "Hello from index of API endpoint"

@app.post("/")
async def predict(req: RequestText, response: Response):
  try:
    req.text = req.text.strip()
    if len(req.text) == 0:
      return {
        "status": 200,
        "is_spam": True,
        "prediction": 1
      }
    prediction = predictText(str(req.text))
    is_spam = True if prediction == 1 else False
    return {
      "status": 200,
      "is_spam": is_spam,
      "prediction": int(prediction)
    }
  except Exception as e:
    print(e)
    response.status_code = 500
    return {
      "status": 500,
      "message": "Internal Server Error"
    }

if __name__ == '__main__':
  port = 8001
  print(f"Listening to http://0.0.0.0:{port}")
  uvicorn.run(app, host='0.0.0.0',port=port)