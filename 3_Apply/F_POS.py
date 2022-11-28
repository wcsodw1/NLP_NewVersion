
# python POS.py

'''
Purpose : 
Tool : jieba.posseg as pseg : 詞性處理
'''

# 1.PreproccessS
import os
import jieba
import numpy as np
import pandas as pd


print(os.getcwd())  # 抓取現在檔案位置
article = pd.read_csv(
    'C:/Users/user/Desktop/Project/AI/data/NLP/article_practice.csv')  # 讀取檔案絕對位置路徑
# print(article['content'])
# filter rules(依據Domain Knowledge去除不需要的資訊  )

# 2.清理資料 :
# 2.1.將'https?:\/\/\S*' 取代為' '(空值) B.將 ''(空值) 取代為' '(空)
article['content'] = article['content'].str.replace('https?:\/\/\S*##', '')
article['content'] = article['content'].replace('', np.NaN)

# 2.2 String Split : ex: 依照文字之間的空格split
article['content'].str.split(" ", n=4, expand=True)

# 2.3 remove data &  remove NaN
article = article.dropna()

# 2.4 讓index重置成原本的樣子
article = article.reset_index(drop=True)
article['idx'] = article.index
print("len of idx : ", len(article['idx']))

# 3.Jieba(Word Cut Tool) # set dictionary (can define yourself)
os.getcwd()
jieba.set_dictionary(
    'C:/Users/user/Desktop/Project/AI/data/NLP/dict.txt.big')
stop_words = open('C:/Users/user/Desktop/Project/AI/data/NLP/stop_words.txt',
                  encoding="utf-8").read().splitlines()
print(stop_words[:100])

# 4.Data type "Pandas Dataframe" to "list"
data = pd.read_csv(
    'C:/Users/user/Desktop/Project/AI/data/NLP/article_preprocessed.csv')
data = data['content'].tolist()
print(data[:20])

# 5.詞性處理(Posseg) :
for w, f in pseg.cut(data[0]):
    print(w, ' ', f)



# ===================================================== #
# Descibe: TF-IDF為抓取文字當中關鍵字的一種方法(工具)
"""
TF (Term Frequency)「詞頻」指的是一個詞在一個文件中出現的頻率，因此一個詞TF越高，可能就代表他在這篇文章中沒那麼重要
例如英文的 “the”, “a” 還有中文的「的」、「了」等等，這些字都很常出現，但沒什麼意義。
"""

import jieba
import jieba.analyse


# Example1 :
text = '總統蔡英文論文風波延燒後，最新民調今日出爐！據親藍民調公布結果，蔡英文支持度45%，遙遙領先韓國瑜的33%，兩人差距擴大到12個百分點。顯示論文門風波，並未重創小英聲望。'
tags = jieba.analyse.extract_tags(text, topK=20)
# topK 為返回幾個TF/IDF 權重最大的關鍵詞，默認值為20
print(tags)
print("================")


# Example2 :
s = "我出門買早餐, 順便去打籃球"
print("=== Example2 ===")
print(jieba.analyse.extract_tags(s, topK=20, withWeight=False, allowPOS=()))
for x, w in jieba.analyse.extract_tags(s, withWeight=True):
    print('%s %s' % (x, w))



# python jieba_posseg.py

'''
Purpose : 
Tool : jieba.posseg as pseg : 詞性處理
'''

# 1.Preproccess
import os
import jieba
import numpy as np
import pandas as pd


print(os.getcwd())  # 抓取現在檔案位置
article = pd.read_csv(
    'C:/Users/user/Desktop/Project/AI/data/NLP/article_practice.csv')  # 讀取檔案絕對位置路徑
# print(article['content'])
# filter rules(依據Domain Knowledge去除不需要的資訊  )

# 2.清理資料 :
# 2.1.將'https?:\/\/\S*' 取代為' '(空值) B.將 ''(空值) 取代為' '(空)
article['content'] = article['content'].str.replace('https?:\/\/\S*##', '')
article['content'] = article['content'].replace('', np.NaN)

# 2.2 String Split : ex: 依照文字之間的空格split
article['content'].str.split(" ", n=4, expand=True)

# 2.3 remove data &  remove NaN
article = article.dropna()

# 2.4 讓index重置成原本的樣子
article = article.reset_index(drop=True)
article['idx'] = article.index
print("len of idx : ", len(article['idx']))

# 3.Jieba(Word Cut Tool) # set dictionary (can define yourself)
os.getcwd()
jieba.set_dictionary(
    'C:/Users/user/Desktop/Project/AI/data/NLP/dict.txt.big')
stop_words = open('C:/Users/user/Desktop/Project/AI/data/NLP/stop_words.txt',
                  encoding="utf-8").read().splitlines()
print(stop_words[:100])

# 4.Data type "Pandas Dataframe" to "list"
data = pd.read_csv(
    'C:/Users/user/Desktop/Project/AI/data/NLP/article_preprocessed.csv')
data = data['content'].tolist()
print(data[:20])

# 5.詞性處理(Posseg) :
for w, f in pseg.cut(data[0]):
    print(w, ' ', f)

