
# python transformer_SentimentAnalysis.py

import os
from transformers import pipeline  # 通郭使用pipeline, 可以自動從模型存儲中下載合適的模型
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
3 possible outputs:
LABEL_0 -> negative
LABEL_1 -> neutral
LABEL_2 -> positive
'''

# 1.sentiment-analysis :
classifier = pipeline("sentiment-analysis",
                      model="cardiffnlp/twitter-roberta-base-sentiment",
                      tokenizer="cardiffnlp/twitter-roberta-base-sentiment")


# B.results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
results = classifier(
    ["I'm so happy today!", "I hope you don't hate him...", "you suck"])


for result in results:
    print(f"{result['label']} with score {result['score']}")

# Result :
# LABEL_2 with score 0.9917560815811157
# LABEL_1 with score 0.5936758518218994
# LABEL_0 with score 0.9578036069869995
