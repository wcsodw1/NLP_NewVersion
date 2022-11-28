# python A_NLTK_Sentiment_Analyzer.py


'''
Purpose : classfied the sentiment Analyze by 3 class(positive | Neutral | negative)
Tool : nltk | nltk.sentiment.Sentiment | IntensityAnalyzer polarity_scores | 
'''

import re
import nltk
import string
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# 1.Preprocess
# nltk.download('stopwords')
# nltk_stopwords = nltk.corpus.stopwords.words('english')
# print('NLTK has {} stop words'.format(len(nltk_stopwords)))
# print('The first five stop words are {}'.format(list(nltk_stopwords)[:5]))
# print(nltk_stopwords)


# 2.Sentiment Analyze :
#Input_sentence = "Wow, NLTK is really powerful!"
Input_sentence = "I hate you!"
sia = SentimentIntensityAnalyzer()
sentiment_dict = sia.polarity_scores(Input_sentence)
print("Predict Sentiment is : ", sentiment_dict)
print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
print("Sentence Overall Rated As", end=" ")


# 3.Reuslt :
'''
sentence was rated as  80.0 % Negative
sentence was rated as  20.0 % Neutral
sentence was rated as  0.0 % Positive
'''

# =========================================== # 