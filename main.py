from fastapi import FastAPI
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle



app = FastAPI()

nltk.download('stopwords')
words = stopwords.words('english')
stemmer = PorterStemmer()

with open('LogisticRegression.pickle', 'rb') as file:
    model = pickle.load(file)



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/news")
def read_item(text: str):
    d = {"predict_news": [text]}
    df = pd.DataFrame(d)
    df['predict_news'] = list(map(lambda x: " ".join([i for i in x.lower().split() if i not in words]), df['predict_news']))
    
    
    df['predict_news'] = df['predict_news'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    
    predict_news_cat = model.predict(df['predict_news'])
    return {"Category": predict_news_cat[0]}