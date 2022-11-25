# python Sentiment_analysis.py
# $pip install transformers==2.5.0
# $pip install spacy-transformers==0.6.0
'''
HuggingFace是一个非常流行的 NLP 库。本文包含其主要类和函数的概述以及一些代码示例。可以作为该库的一个入门教程 。
Hugging Face 是一个开源库，用于构建、训练和部署最先进的 NLP 模型。Hugging Face 提供了两个主要的库，用于模型的transformers 和用于数据集的datasets 。可以直接使用 pip 安装它们

transformers库中已经提供了以下的几个任务，例如：

1.text classification 文本分类
2.Q&A 问答
3.text translate 翻译
4.文本摘要
5.text generate 文本生成
6.CV计算机视觉
7.Audio 音频任务
'''

from transformers import pipeline  # 通郭使用pipeline, 可以自動從模型存儲中下載合適的模型
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# A.sentiment-analysis :
classifier = pipeline("sentiment-analysis")  # sentiment-analysis 情緒分類 分類器
# classifier = pipeline("text-classification") # text-classification 語句分類 分類器


# test1 :
# results = classifier("I'm so happy today!")
# print(f"{results[0]['label']} with score {results[0]['score']}")

# test2 : 多句子
results = classifier(
    ["I'm so happy today!", "I hope you don't hate him...", "you suck"])
for result in results:
    print(f"{result['label']} with score {result['score']}")
