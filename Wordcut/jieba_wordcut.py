#  python jieba_wordcut.py

#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cd  C:\Users\user\Desktop\Project\AI\New_DeepLearning\MyNewDeepLearning_Project\NLP

'''
Purpose : 
Tool : jieba.posseg as pseg : 詞性處理 | jieba.cut
'''


# 1.import Library : 
import os
import jieba
cwd = os.getcwd() # Get current working directory (CWD)
print("Current working directory:", cwd) # Print current working directory (CWD)

# 2.Input sentence : 
sentence = "獨立音樂需要大家一起來推廣，歡迎加入我們的行列！"
print("Input：", sentence)

# 3.Jieba word cut : 
words = jieba.cut(sentence, cut_all=False)
print("Output 精確模式 Full Mode：")
for word in words: # show result
    print(word)

# 4.Result : 
'''
獨立
音樂
需要
大家
一起
來
推廣
，
歡迎
加入
我們
的
行列
！
'''