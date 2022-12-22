import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("final.csv",)
df = df[df['soup'].notna()]

count = CountVectorizer(stop_words = 'english')
count_metrics =  count.fit_transform(df['soup'])